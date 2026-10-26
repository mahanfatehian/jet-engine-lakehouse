"""Spark transformations shared by the lakehouse (Path A) and ClickHouse (Path B) queries.

:func:`parse_cdc_records` is the single source of truth for turning a Debezium envelope
into a typed row, so silver parquet and the ClickHouse raw tables always hold identical
values. Records it rejects (``parse_error`` not null) are left out of silver and routed to
``raw_cdc_dead_letter``. Every check is a null-safe Spark expression: a malformed record
can never raise inside a micro-batch, whether or not ANSI mode is enabled.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from functools import reduce

from cdc_schemas import DEAD_LETTER_COLUMNS, DELETE_OP, SUPPORTED_OPS, FieldKind, FieldSpec, TableSpec
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import DataType, DoubleType, IntegerType, LongType, StringType, StructField, StructType

KAFKA_COLUMNS: tuple[str, ...] = (
    "kafka_key",
    "kafka_value",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
)
BRONZE_PARTITION_COLUMN = "ingest_date"
SILVER_PARTITION_COLUMN = "event_date"
PARSE_ERROR_COLUMN = "parse_error"
DEAD_LETTER_COLUMN = "dead_letter"
CORRUPT_RECORD_FIELD = "_corrupt_record"

_ENVELOPE = "_envelope"
_IMAGE = "_image"


def _epoch_micros(instant: datetime) -> int:
    return (instant - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


# ClickHouse DateTime64 range: instants outside it cannot be stored faithfully.
_MIN_MICROS = _epoch_micros(datetime(1900, 1, 1, tzinfo=UTC))
_MAX_MICROS = _epoch_micros(datetime(2299, 12, 31, 23, 59, 59, 999999, tzinfo=UTC))
_MIN_MILLIS = _MIN_MICROS // 1000
_MAX_MILLIS = _MAX_MICROS // 1000

_JSON_TYPES: dict[FieldKind, DataType] = {
    FieldKind.STRING: StringType(),
    FieldKind.INT: IntegerType(),
    FieldKind.LONG: LongType(),
    FieldKind.DOUBLE: DoubleType(),
    # Read as text and converted explicitly, so a bad value yields a precise dead-letter reason.
    FieldKind.TIMESTAMP: StringType(),
}

_BRONZE_ENVELOPE = StructType(
    [
        StructField("op", StringType()),
        StructField("source", StructType([StructField("ts_ms", LongType())])),
    ]
)


def kafka_records(source: DataFrame) -> DataFrame:
    """Project the Kafka source schema onto the ``kafka_*`` columns used downstream."""
    return source.select(
        sf.col("key").cast("string").alias("kafka_key"),
        sf.col("value").cast("string").alias("kafka_value"),
        sf.col("topic").alias("kafka_topic"),
        sf.col("partition").alias("kafka_partition"),
        sf.col("offset").alias("kafka_offset"),
        sf.col("timestamp").alias("kafka_timestamp"),
    )


def bronze_records(records: DataFrame) -> DataFrame:
    """Raw Kafka records plus ``cdc_op``/``cdc_source_ts_ms`` and the UTC ingestion date.

    Every record is kept verbatim, tombstones and malformed values included: bronze is
    the replayable system of record. In a streaming query ``current_date`` resolves to the
    micro-batch timestamp stored in the checkpoint, so a replayed batch lands in the same
    ``ingest_date`` partition.
    """
    envelope = sf.from_json(sf.col("kafka_value"), _BRONZE_ENVELOPE)
    return records.select(
        *KAFKA_COLUMNS,
        envelope["op"].alias("cdc_op"),
        envelope["source"]["ts_ms"].alias("cdc_source_ts_ms"),
        sf.current_date().alias(BRONZE_PARTITION_COLUMN),
    )


def parse_cdc_records(records: DataFrame, table: TableSpec) -> DataFrame:
    """Parse the Debezium envelopes of one table into typed rows plus ``parse_error``.

    Output columns: ``KAFKA_COLUMNS``, ``parse_error`` (null for a usable record), the
    image columns of ``table`` and the CDC metadata columns. The image is ``after`` for
    ops c/u/r and ``before`` for d (``is_deleted`` = 1). Tombstones (null values) carry
    no image and are dropped.
    """
    envelope = sf.from_json(
        sf.col("kafka_value"),
        _envelope_schema(table),
        {"mode": "PERMISSIVE", "columnNameOfCorruptRecord": CORRUPT_RECORD_FIELD},
    )
    op = sf.col(f"{_ENVELOPE}.op")
    image = sf.when(op == DELETE_OP, sf.col(f"{_ENVELOPE}.before")).otherwise(sf.col(f"{_ENVELOPE}.after"))
    staged = (
        records.where(sf.col("kafka_value").isNotNull())
        .withColumn(_ENVELOPE, envelope)
        .withColumn(_IMAGE, image)
    )
    return staged.select(
        *KAFKA_COLUMNS,
        _parse_error(table).alias(PARSE_ERROR_COLUMN),
        *(_typed_field(field).alias(field.name) for field in table.fields),
        op.alias("cdc_op"),
        sf.coalesce(sf.col(f"{_ENVELOPE}.source.lsn"), sf.lit(0).cast("long")).alias("cdc_lsn"),
        _millis_to_timestamp(sf.col(f"{_ENVELOPE}.source.ts_ms")).alias("cdc_source_ts"),
        _millis_to_timestamp(sf.col(f"{_ENVELOPE}.ts_ms")).alias("cdc_emitted_ts"),
        (op == DELETE_OP).cast("tinyint").alias("is_deleted"),
    )


def silver_records(parsed: DataFrame, table: TableSpec) -> DataFrame:
    """Usable records: the raw ClickHouse columns + ``processed_at`` + ``event_date``.

    Append-only: deletes are kept as rows with ``is_deleted`` = 1 and readers keep the
    highest ``cdc_lsn`` per primary key. ``processed_at`` is the micro-batch timestamp.
    """
    return parsed.where(sf.col(PARSE_ERROR_COLUMN).isNull()).select(
        *table.clickhouse_columns,
        sf.current_timestamp().alias("processed_at"),
        sf.to_date(sf.col(table.event_time_field)).alias(SILVER_PARTITION_COLUMN),
    )


def clickhouse_projection(parsed: DataFrame, table: TableSpec) -> DataFrame:
    """Collect-friendly layout ``(dead_letter, *clickhouse_columns)`` for ``clickhouse_rows``.

    Instants travel as epoch microseconds: ``collect()`` would otherwise convert
    TimestampType through the Python process' local timezone.
    """
    dead_letter_sources = {
        "kafka_topic": sf.col("kafka_topic"),
        "kafka_partition": sf.col("kafka_partition"),
        "kafka_offset": sf.col("kafka_offset"),
        "kafka_timestamp": sf.unix_micros("kafka_timestamp"),
        "raw_key": sf.col("kafka_key"),
        "raw_value": sf.col("kafka_value"),
        "error": sf.col(PARSE_ERROR_COLUMN),
    }
    dead_letter = sf.when(
        sf.col(PARSE_ERROR_COLUMN).isNotNull(),
        sf.struct(*(dead_letter_sources[name].alias(name) for name in DEAD_LETTER_COLUMNS)),
    )
    instants = table.timestamp_columns
    return parsed.select(
        dead_letter.alias(DEAD_LETTER_COLUMN),
        *(
            (sf.unix_micros(name) if name in instants else sf.col(name)).alias(name)
            for name in table.clickhouse_columns
        ),
    )


def _envelope_schema(table: TableSpec) -> StructType:
    image = StructType([StructField(field.name, _JSON_TYPES[field.kind]) for field in table.fields])
    source = StructType([StructField("lsn", LongType()), StructField("ts_ms", LongType())])
    return StructType(
        [
            StructField("before", image),
            StructField("after", image),
            StructField("op", StringType()),
            StructField("ts_ms", LongType()),
            StructField("source", source),
            StructField(CORRUPT_RECORD_FIELD, StringType()),
        ]
    )


def _image_value(name: str) -> Column:
    return sf.col(f"{_IMAGE}.{name}")


def _typed_field(field: FieldSpec) -> Column:
    value = _image_value(field.name)
    return sf.try_to_timestamp(value) if field.kind is FieldKind.TIMESTAMP else value


def _millis_to_timestamp(millis: Column) -> Column:
    # Range-guarded: timestamp_millis overflows (and throws) on absurd inputs.
    return sf.when(millis.between(_MIN_MILLIS, _MAX_MILLIS), sf.timestamp_millis(millis))


def _invalid_timestamp(text: Column) -> Column:
    parsed = sf.try_to_timestamp(text)
    return parsed.isNull() | ~sf.unix_micros(parsed).between(_MIN_MICROS, _MAX_MICROS)


def _failing_fields(
    label: str, fields: Sequence[FieldSpec], fails: Callable[[Column], Column]
) -> tuple[Column, Column]:
    """Condition "any field fails" plus a reason naming every failing field."""
    conditions = [fails(_image_value(field.name)) for field in fields]
    names = [
        sf.when(condition, sf.lit(field.name)) for condition, field in zip(conditions, fields, strict=True)
    ]
    return reduce(operator.or_, conditions, sf.lit(False)), sf.concat(
        sf.lit(f"{label}: "), sf.concat_ws(", ", *names)
    )


def _parse_error(table: TableSpec) -> Column:
    """Reason of the first failed check, or null when the record is usable."""
    envelope = sf.col(_ENVELOPE)
    op = envelope["op"]
    missing_image = sf.concat(
        sf.lit("missing "),
        sf.when(op == DELETE_OP, sf.lit("before")).otherwise(sf.lit("after")),
        sf.lit(" image for op '"),
        op,
        sf.lit("'"),
    )
    checks: list[tuple[Column, Column]] = [
        (envelope.isNull() | envelope[CORRUPT_RECORD_FIELD].isNotNull(), sf.lit("malformed JSON envelope")),
        (op.isNull(), sf.lit("missing op")),
        (~op.isin(*SUPPORTED_OPS), sf.concat(sf.lit("unsupported op '"), op, sf.lit("'"))),
        (envelope["before"].isNull() & envelope["after"].isNull(), sf.lit("both before and after are null")),
        (sf.col(_IMAGE).isNull(), missing_image),
        (
            _millis_to_timestamp(envelope["source"]["ts_ms"]).isNull(),
            sf.lit("missing or invalid source.ts_ms"),
        ),
        (_millis_to_timestamp(envelope["ts_ms"]).isNull(), sf.lit("missing or invalid ts_ms")),
        (envelope["source"]["lsn"] < 0, sf.lit("negative source.lsn")),
        _failing_fields(
            "null required field(s)",
            [field for field in table.fields if not field.nullable],
            lambda value: value.isNull(),
        ),
        _failing_fields(
            "invalid or out-of-range timestamp field(s)",
            [field for field in table.fields if field.kind is FieldKind.TIMESTAMP],
            lambda value: value.isNotNull() & _invalid_timestamp(value),
        ),
        _failing_fields(
            "negative value in unsigned field(s)",
            [field for field in table.fields if field.unsigned],
            lambda value: value < 0,
        ),
    ]
    return sf.coalesce(*(sf.when(condition, reason) for condition, reason in checks))
