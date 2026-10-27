"""Decoding of the high-frequency (HF) sensor frames: Confluent wire format -> Avro -> columns.

The simulator produces one record per operating engine and tick to HF_TOPIC: key = engine_id
(UTF-8), value = magic byte 0x00 + 4-byte big-endian schema-registry id + Avro binary of
``engine_sensor_frame.avsc`` (Confluent wire format). The job validates and strips the 5-byte
header and decodes the body with Spark's ``from_avro`` and the schema file at HF_SCHEMA_PATH,
the same file the producer registers, in PERMISSIVE mode: a payload that does not decode
yields a struct whose fields are all null, never an exception.

:func:`decode_frames` adds ``decode_error``: null for a usable frame, otherwise the reason of
the first failed check (like ``cdc_transforms.parse_cdc_records``):

    null value, value shorter than the header, magic byte other than 0x00
    Avro body that does not decode with the schema (truncated, corrupted, other schema)
    null field the job reads (possible only with an evolved schema)
    negative id or ordinal, non-positive frame period or sample rate, frame_start_ts
    outside the ClickHouse DateTime64 range, more than 65535 channels (UInt16 counters)
    missing or empty fan_frame_vel / core_trf_vel waveform
    a samples_f32le that is not a whole number of float32 values

Rejected records are dead-lettered into ``raw_cdc_dead_letter``, where ``kafka_topic`` tells
them from CDC records. ``raw_key`` is the Kafka key; ``raw_value`` holds the base64 of at most
the first 64 KiB of the value: enough to diagnose a header or schema problem, while frames run
to megabytes at full fidelity. The whole record stays in Kafka for HF_TOPIC_RETENTION_MS and
in the bronze lake until it expires.

``from_avro`` decodes every record with that one writer schema: the schema-registry id of a
record is kept in bronze (``schema_id``) but not resolved, so an evolved producer schema must
be deployed to the job as well. A corrupted length prefix inside an otherwise small value
could make the Avro decoder allocate gigabytes, an OutOfMemoryError that stops the whole
driver; ``streaming_config.AVRO_DECODER_LIMITS`` caps the lengths so that such a record merely
fails to decode.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from functools import reduce
from pathlib import Path

from cdc_schemas import DEAD_LETTER_COLUMNS
from clickhouse_rows import DATETIME64_MAX_MICROS, DATETIME64_MIN_MICROS
from hf_features import (
    FEATURE_COLUMN_NAMES,
    FEATURE_INPUT_COLUMNS,
    FEATURE_SCHEMA_DDL,
    SAMPLE_COLUMNS,
    VIBRATION_PREFIXES,
    feature_batches,
)
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.avro.functions import from_avro

HF_KAFKA_COLUMNS: tuple[str, ...] = (
    "kafka_key",
    "kafka_value",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
)
CONFLUENT_HEADER_BYTES = 5
CONFLUENT_MAGIC_BYTE_HEX = "00"
DEAD_LETTER_VALUE_BYTES = 64 * 1024
UINT16_MAX = 65_535
FRAME_RECORD_NAME = "EngineSensorFrame"
SCHEMA_ID_COLUMN = "schema_id"
FRAME_COLUMN = "frame"
DECODE_ERROR_COLUMN = "decode_error"
BRONZE_PARTITION_COLUMNS: tuple[str, ...] = ("ingest_date", "ingest_hour")
SILVER_PARTITION_COLUMN = "event_date"
# CORRECTED: timestamp-micros values are proleptic Gregorian as the Avro spec defines them, and
# an out-of-range frame_start_ts gets its own decode_error instead of failing the whole decode.
AVRO_OPTIONS: dict[str, str] = {"mode": "PERMISSIVE", "datetimeRebaseMode": "CORRECTED"}

_UNSIGNED_FIELDS = ("lifecycle_id", "cycle_number", "telemetry_id", "frame_seq")
_POSITIVE_FIELDS = ("frame_duration_ms", "vibration_sample_rate_hz")
_FRAME_FIELDS = (
    "engine_id",
    "engine_model",
    "lifecycle_id",
    "cycle_number",
    "telemetry_id",
    "frame_seq",
    "frame_duration_ms",
    "n1_rpm",
    "n2_rpm",
    "bpfo_order",
    "vibration_sample_rate_hz",
)
# Every field the job reads; none is nullable in the schema, so a decoded frame has them all.
REQUIRED_SCHEMA_FIELDS: tuple[str, ...] = (*_FRAME_FIELDS, "frame_start_ts", "vibration", "process")


class FrameSchemaError(ValueError):
    """The HF frame schema file is unreadable or not the record the job decodes."""


def load_frame_schema(path: str) -> str:
    """Avro schema JSON at ``path``, checked to be the frame record with every field the job reads."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        schema = json.loads(text)
    except (OSError, ValueError) as exc:
        raise FrameSchemaError(f"cannot load the HF frame schema {path}: {exc}") from exc
    if not isinstance(schema, dict) or (schema.get("type"), schema.get("name")) != (
        "record",
        FRAME_RECORD_NAME,
    ):
        raise FrameSchemaError(f"{path} is not the Avro record {FRAME_RECORD_NAME}")
    fields = {field.get("name") for field in schema.get("fields", []) if isinstance(field, dict)}
    missing = sorted(set(REQUIRED_SCHEMA_FIELDS) - fields)
    if missing:
        raise FrameSchemaError(f"{path} lacks the field(s) {', '.join(missing)}")
    return text


def hf_kafka_records(source: DataFrame) -> DataFrame:
    """Project the Kafka source onto ``HF_KAFKA_COLUMNS``, keeping the value as bytes."""
    return source.select(
        sf.col("key").cast("string").alias("kafka_key"),
        sf.col("value").alias("kafka_value"),
        sf.col("topic").alias("kafka_topic"),
        sf.col("partition").alias("kafka_partition"),
        sf.col("offset").alias("kafka_offset"),
        sf.col("timestamp").alias("kafka_timestamp"),
    )


def schema_id(value: Column) -> Column:
    """Schema-registry id in the Confluent header of ``value`` (null unless the header is valid)."""
    return sf.when(_header_ok(value), sf.conv(sf.hex(sf.substring(value, 2, 4)), 16, 10).cast("long"))


def bronze_frames(records: DataFrame) -> DataFrame:
    """Every record verbatim plus its schema id and the UTC ingestion date and hour.

    Nothing is dropped, malformed records included: bronze is the replayable copy of what
    Kafka delivered. In a streaming query ``current_timestamp`` is the micro-batch timestamp
    stored in the checkpoint, so a replayed batch lands in the same partitions.
    """
    now = sf.current_timestamp()
    return records.select(
        *HF_KAFKA_COLUMNS,
        schema_id(sf.col("kafka_value")).alias(SCHEMA_ID_COLUMN),
        sf.to_date(now).alias(BRONZE_PARTITION_COLUMNS[0]),
        sf.hour(now).alias(BRONZE_PARTITION_COLUMNS[1]),
    )


def decode_frames(records: DataFrame, schema_json: str) -> DataFrame:
    """``records`` plus ``schema_id``, the decoded ``frame`` struct and ``decode_error``.

    Filter the result on ``decode_error`` only once it is cached: Spark pushes a filter below
    the projections and inlines ``frame`` into every check of the condition, which decodes
    each record once per check (72 ``from_avro`` calls in the optimised plan). A projection
    keeps the single decode, so :func:`feature_input` leaves the rejected rows in and the
    feature workers drop them.
    """
    value = sf.col("kafka_value")
    body = value.substr(sf.lit(CONFLUENT_HEADER_BYTES + 1), sf.length(value) - CONFLUENT_HEADER_BYTES)
    return (
        records.withColumn(SCHEMA_ID_COLUMN, schema_id(value))
        .withColumn(FRAME_COLUMN, sf.when(_header_ok(value), from_avro(body, schema_json, AVRO_OPTIONS)))
        .withColumn(DECODE_ERROR_COLUMN, _decode_error())
    )


def feature_input(decoded: DataFrame) -> DataFrame:
    """Every record in the ``FEATURE_INPUT_COLUMNS`` layout of ``hf_features.feature_batches``.

    Rejected records keep their ``decode_error`` (``feature_batches`` drops them, see
    :func:`decode_frames`). Only the fan and core waveforms travel to the Python workers.
    """
    frame = sf.col(FRAME_COLUMN)
    columns: dict[str, Column] = {name: frame[name] for name in _FRAME_FIELDS}
    columns.update(
        {
            SAMPLE_COLUMNS[prefix]: _waveform(frame["vibration"], channel)
            for prefix, channel in VIBRATION_PREFIXES.items()
        }
    )
    columns.update(
        frame_start_us=sf.unix_micros(frame["frame_start_ts"]),
        vibration_channels=sf.size(frame["vibration"]),
        process_channels=sf.size(frame["process"]),
        process_parameters=frame["process"]["parameter"],
        process_samples=frame["process"]["samples_f32le"],
        frame_bytes=sf.length("kafka_value"),
        kafka_partition=sf.col("kafka_partition"),
        kafka_offset=sf.col("kafka_offset"),
        decode_error=sf.col(DECODE_ERROR_COLUMN),
    )
    return decoded.select(*(columns[name].alias(name) for name in FEATURE_INPUT_COLUMNS))


def frame_features(decoded: DataFrame) -> DataFrame:
    """One ``raw_engine_hf_features`` row (``FEATURE_COLUMNS``) per usable frame."""
    return feature_input(decoded).mapInPandas(feature_batches, FEATURE_SCHEMA_DDL)


def silver_features(features: DataFrame) -> DataFrame:
    """Feature rows plus ``processed_at`` (micro-batch timestamp) and the UTC ``event_date``."""
    return features.select(
        *FEATURE_COLUMN_NAMES,
        sf.current_timestamp().alias("processed_at"),
        sf.to_date("frame_start_ts").alias(SILVER_PARTITION_COLUMN),
    )


def clickhouse_features(features: DataFrame) -> DataFrame:
    """Collect-friendly feature rows: ``frame_start_ts`` as epoch microseconds (see ``hf_rows``)."""
    return features.select(
        *(
            sf.unix_micros(name).alias(name) if name == "frame_start_ts" else sf.col(name)
            for name in FEATURE_COLUMN_NAMES
        )
    )


def dead_letters(decoded: DataFrame) -> DataFrame:
    """Rejected records in ``DEAD_LETTER_COLUMNS`` order, ``kafka_timestamp`` as epoch microseconds.

    Meant for a cached ``decoded`` (see :func:`decode_frames`).
    """
    sources = {
        "kafka_topic": sf.col("kafka_topic"),
        "kafka_partition": sf.col("kafka_partition"),
        "kafka_offset": sf.col("kafka_offset"),
        "kafka_timestamp": sf.unix_micros("kafka_timestamp"),
        "raw_key": sf.col("kafka_key"),
        "raw_value": sf.base64(sf.substring("kafka_value", 1, DEAD_LETTER_VALUE_BYTES)),
        "error": sf.col(DECODE_ERROR_COLUMN),
    }
    rejected = decoded.where(sf.col(DECODE_ERROR_COLUMN).isNotNull())
    return rejected.select(*(sources[name].alias(name) for name in DEAD_LETTER_COLUMNS))


def _magic_byte_hex(value: Column) -> Column:
    return sf.hex(sf.substring(value, 1, 1))


def _header_ok(value: Column) -> Column:
    return (sf.length(value) >= CONFLUENT_HEADER_BYTES) & (_magic_byte_hex(value) == CONFLUENT_MAGIC_BYTE_HEX)


def _waveform(vibration: Column, channel: str) -> Column:
    """Samples of the first vibration channel named ``channel`` (null when absent)."""
    return sf.try_element_at(sf.filter(vibration, lambda item: item["channel"] == channel), sf.lit(1))[
        "samples_f32le"
    ]


def _has_channel(vibration: Column, channel: str) -> Column:
    return sf.exists(vibration, lambda item: item["channel"] == channel)


def _has_empty_waveform(vibration: Column, channel: str) -> Column:
    return sf.exists(
        vibration, lambda item: (item["channel"] == channel) & (sf.length(item["samples_f32le"]) == 0)
    )


def _misaligned(channel: Column) -> Column:
    return sf.length(channel["samples_f32le"]) % 4 != 0


def _named(label: str, failures: Mapping[str, Column]) -> tuple[Column, Column]:
    """Condition "any item fails" plus a reason naming every failing item."""
    names = sf.concat_ws(", ", *(sf.when(failed, sf.lit(name)) for name, failed in failures.items()))
    return reduce(operator.or_, failures.values()), sf.concat(sf.lit(f"{label}: "), names)


def _decode_error() -> Column:
    """Reason of the first failed check, or null when the frame is usable."""
    value, frame = sf.col("kafka_value"), sf.col(FRAME_COLUMN)
    size = sf.length(value).cast("string")
    vibration, process = frame["vibration"], frame["process"]
    misaligned = sf.concat(
        sf.transform(sf.filter(vibration, _misaligned), lambda item: item["channel"]),
        sf.transform(sf.filter(process, _misaligned), lambda item: item["parameter"]),
    )
    required = VIBRATION_PREFIXES.values()
    nulls = {name: frame[name].isNull() for name in REQUIRED_SCHEMA_FIELDS}
    checks: list[tuple[Column, Column]] = [
        (value.isNull(), sf.lit("null value")),
        (
            sf.length(value) < CONFLUENT_HEADER_BYTES,
            sf.concat(sf.lit("value shorter than the 5-byte Confluent header ("), size, sf.lit(" bytes)")),
        ),
        (
            _magic_byte_hex(value) != CONFLUENT_MAGIC_BYTE_HEX,
            sf.concat(sf.lit("unknown magic byte 0x"), _magic_byte_hex(value), sf.lit(" (Confluent: 0x00)")),
        ),
        (
            # PERMISSIVE from_avro turns an undecodable body into a struct of nulls.
            frame.isNull() | reduce(operator.and_, nulls.values()),
            sf.concat(
                sf.lit(f"Avro body does not decode as {FRAME_RECORD_NAME} (schema id "),
                sf.col(SCHEMA_ID_COLUMN).cast("string"),
                sf.lit(", "),
                size,
                sf.lit(" bytes)"),
            ),
        ),
        _named("null field(s)", nulls),
        _named("negative value in unsigned field(s)", {name: frame[name] < 0 for name in _UNSIGNED_FIELDS}),
        _named("non-positive field(s)", {name: frame[name] <= 0 for name in _POSITIVE_FIELDS}),
        (
            ~sf.unix_micros(frame["frame_start_ts"]).between(DATETIME64_MIN_MICROS, DATETIME64_MAX_MICROS),
            sf.lit("frame_start_ts outside the DateTime64 range (1900-2299)"),
        ),
        (
            (sf.size(vibration) > UINT16_MAX) | (sf.size(process) > UINT16_MAX),
            sf.lit(f"more than {UINT16_MAX} vibration or process channels"),
        ),
        _named("missing vibration channel(s)", {name: ~_has_channel(vibration, name) for name in required}),
        _named(
            "empty waveform in channel(s)", {name: _has_empty_waveform(vibration, name) for name in required}
        ),
        (
            sf.size(misaligned) > 0,
            sf.concat(
                sf.lit("samples_f32le is not a whole number of float32 values in channel(s): "),
                sf.array_join(misaligned, ", "),
            ),
        ),
    ]
    return sf.coalesce(*(sf.when(condition, reason) for condition, reason in checks))
