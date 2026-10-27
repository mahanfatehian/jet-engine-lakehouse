"""Local-mode Spark tests of the HF decoding: Confluent framing, from_avro, dead letters, features.

Frames are encoded in-test like the edge gateway does (``avro_frames``: magic byte, schema id,
Avro body from Spark's ``to_avro`` with the repository's .avsc), then corrupted on purpose.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyspark")

from clickhouse_rows import dead_letter_row
from hf_decode import (
    DEAD_LETTER_VALUE_BYTES,
    REQUIRED_SCHEMA_FIELDS,
    FrameSchemaError,
    bronze_frames,
    clickhouse_features,
    dead_letters,
    decode_frames,
    feature_input,
    frame_features,
    load_frame_schema,
)
from hf_features import FEATURE_COLUMNS
from pyspark.sql import functions as sf

REPO_SCHEMA = Path(__file__).resolve().parents[2] / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
HF_TOPIC = "engine_hf.sensor_frames"
FS = 2048
F_N1, F_N2 = 3650.0 / 60, 7800.0 / 60  # the default (JT9D) spool speeds of make_frame
F_BPFO = 4.62 * F_N1
SERVER_FILLED = "ingested_at"


def f32le(samples: Any) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


def tone(frequency_hz: float, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency_hz * np.arange(FS) / FS + 1.1)


def varint(value: int) -> bytes:
    """Avro's variable-length encoding of a non-negative zigzag value."""
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


@pytest.fixture(scope="module")
def schema_json() -> str:
    return load_frame_schema(str(REPO_SCHEMA))


def decode_errors(decoded: Any) -> list[str | None]:
    return [row.decode_error for row in decoded.orderBy("kafka_offset").collect()]


def test_the_repository_schema_is_the_record_the_job_decodes(schema_json: str) -> None:
    schema = json.loads(schema_json)
    assert (schema["type"], schema["name"]) == ("record", "EngineSensorFrame")
    fields = {field["name"]: field["type"] for field in schema["fields"]}
    assert set(REQUIRED_SCHEMA_FIELDS) <= set(fields)
    assert fields["frame_start_ts"] == {"type": "long", "logicalType": "timestamp-micros"}
    channel_fields = {field["name"] for field in fields["vibration"]["items"]["fields"]}
    parameter_fields = {field["name"] for field in fields["process"]["items"]["fields"]}
    assert channel_fields == {"channel", "unit", "samples_f32le"}
    assert parameter_fields == {"parameter", "unit", "samples_f32le"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot load the HF frame schema"),
        ("{not json", "cannot load the HF frame schema"),
        ('{"type": "record", "name": "Other", "fields": []}', "is not the Avro record EngineSensorFrame"),
        (
            '{"type": "record", "name": "EngineSensorFrame", "fields": [{"name": "engine_id"}]}',
            "lacks the field",
        ),
    ],
)
def test_unusable_schema_files_are_rejected_with_a_reason(
    tmp_path: Path, content: str | None, message: str
) -> None:
    path = tmp_path / "frame.avsc"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(FrameSchemaError, match=message):
        load_frame_schema(str(path))


def test_confluent_framed_avro_decodes_exactly(schema_json, make_frame, avro_frames, hf_records) -> None:
    frame = make_frame(4242)
    (value,) = avro_frames([frame], schema_id=42)
    assert value[:5] == b"\x00\x00\x00\x00\x2a"
    decoded = decode_frames(hf_records([value]), schema_json)
    (row,) = decoded.select(
        "schema_id", "decode_error", sf.unix_micros("frame.frame_start_ts").alias("start_us"), "frame"
    ).collect()
    assert (row.schema_id, row.decode_error, row.start_us) == (42, None, frame["frame_start_ts"])
    decoded_frame = row.frame.asDict(recursive=True)
    for name in ("engine_id", "engine_model", "lifecycle_id", "cycle_number", "telemetry_id", "frame_seq"):
        assert decoded_frame[name] == frame[name], name
    assert decoded_frame["n1_rpm"] == pytest.approx(frame["n1_rpm"])
    assert [(item["channel"], bytes(item["samples_f32le"])) for item in decoded_frame["vibration"]] == [
        (channel, samples) for channel, _unit, samples in frame["vibration"]
    ]
    assert [item["parameter"] for item in decoded_frame["process"]] == ["EGT_A", "EGT_B", "OIL_T_A"]


def _variants(
    make_frame: Callable[..., dict], avro_frames: Callable[..., list[bytes]]
) -> dict[str, bytes | None]:
    """Corrupted or invalid values, by the reason they must be dead-lettered with."""
    (good,) = avro_frames([make_frame()])
    invalid_frames = {
        "negative value in unsigned field(s): telemetry_id": make_frame(telemetry_id=-5),
        "non-positive field(s): vibration_sample_rate_hz": make_frame(vibration_sample_rate_hz=0),
        "frame_start_ts outside the DateTime64 range": make_frame(frame_start_ts=-2_500_000_000_000_000),
        "missing vibration channel(s): core_trf_vel": make_frame(
            vibration=[("fan_frame_vel", "mm/s", f32le(tone(F_N1, 1.0)))]
        ),
        "empty waveform in channel(s): fan_frame_vel": make_frame(fan=b""),
        "samples_f32le is not a whole number of float32 values in channel(s): OIL_T_A": make_frame(
            process=[("OIL_T_A", "degC", f32le([85.0]) + b"\x01")]
        ),
    }
    encoded = dict(zip(invalid_frames, avro_frames(list(invalid_frames.values())), strict=True))
    header = b"\x00\x00\x00\x00\x01"
    return {
        "null value": None,
        "value shorter than the 5-byte Confluent header (3 bytes)": b"\x00\x00\x01",
        "unknown magic byte 0x7B (Confluent: 0x00)": b'{"engine_id": "JT9D-001"}',
        "unknown magic byte 0x01 (Confluent: 0x00)": b"\x01" + good[1:],
        f"Avro body does not decode as EngineSensorFrame (schema id 1, {len(good) // 2} bytes)": good[
            : len(good) // 2
        ],
        # schema_version 1, then an engine_id claiming 2 GB: the decoder limit rejects it (an OOM otherwise).
        "Avro body does not decode as EngineSensorFrame (schema id 1, 19 bytes)": header
        + b"\x02"
        + varint(2_000_000_000 << 1)
        + b"JT9D-001",
        **encoded,
    }


def test_rejected_records_are_classified(schema_json, make_frame, avro_frames, hf_records) -> None:
    variants = _variants(make_frame, avro_frames)
    errors = decode_errors(decode_frames(hf_records(list(variants.values())), schema_json))
    for expected, error in zip(variants, errors, strict=True):
        assert error is not None and error.startswith(expected), (expected, error)


def test_valid_frames_pass_every_check(schema_json, make_frame, avro_frames, hf_records) -> None:
    extra = [("gearbox_vel", "mm/s", f32le(tone(300.0, 0.1)))]
    frames = [make_frame(4242), make_frame(4243, extra_vibration=extra), make_frame(4244, process=[])]
    assert decode_errors(decode_frames(hf_records(avro_frames(frames)), schema_json)) == [None, None, None]


def test_dead_letters_keep_kafka_metadata_and_a_bounded_value_prefix(schema_json, hf_records) -> None:
    garbage = b"\x00\x00\x00\x00\x07" + bytes(range(256)) * 400  # 102405 bytes, not a frame
    rejected = dead_letters(decode_frames(hf_records([garbage], first_offset=11), schema_json)).collect()
    (row,) = [dead_letter_row(record) for record in rejected]
    topic, partition, offset, timestamp, key, value, error = row
    assert (topic, partition, offset, key) == (HF_TOPIC, 0, 11, "JT9D-001")
    assert timestamp == datetime(2026, 10, 26, 9, 0, 2, 500000, tzinfo=UTC)
    assert base64.b64decode(value) == garbage[:DEAD_LETTER_VALUE_BYTES]
    assert error == "Avro body does not decode as EngineSensorFrame (schema id 7, 102405 bytes)"


def test_dead_letter_of_a_keyless_null_record_has_empty_texts(schema_json, hf_records) -> None:
    rejected = dead_letters(decode_frames(hf_records([None], key=None), schema_json)).collect()
    (row,) = [dead_letter_row(record) for record in rejected]
    assert row[4:] == ("", "", "null value")


def test_bronze_keeps_every_record_verbatim(make_frame, avro_frames, hf_records) -> None:
    (good,) = avro_frames([make_frame()], schema_id=9)
    values = [good, b"\x01garbage", None]
    before = datetime.now(UTC)
    rows = bronze_frames(hf_records(values)).orderBy("kafka_offset").collect()
    after = datetime.now(UTC)
    assert [None if row.kafka_value is None else bytes(row.kafka_value) for row in rows] == values
    assert [row.schema_id for row in rows] == [9, None, None]
    assert {(row.ingest_date, row.ingest_hour) for row in rows} <= {
        (moment.date(), moment.hour) for moment in (before, after)
    }
    assert rows[0].asDict().keys() >= {"kafka_key", "kafka_topic", "kafka_partition", "kafka_offset"}


def test_feature_input_ships_only_the_fan_and_core_waveforms(
    schema_json, make_frame, avro_frames, hf_records
) -> None:
    frame = make_frame(extra_vibration=[("gearbox_vel", "mm/s", f32le(tone(300.0, 0.1)))])
    (value,) = avro_frames([frame])
    row, rejected = feature_input(decode_frames(hf_records([value, None]), schema_json)).collect()
    assert (row.decode_error, rejected.decode_error, rejected.fan_samples) == (None, "null value", None)
    vibration = {channel: samples for channel, _unit, samples in frame["vibration"]}
    assert bytes(row.fan_samples) == vibration["fan_frame_vel"]
    assert bytes(row.core_samples) == vibration["core_trf_vel"]
    assert (row.vibration_channels, row.process_channels) == (3, 3)
    assert row.process_parameters == ["EGT_A", "EGT_B", "OIL_T_A"]
    assert (row.frame_start_us, row.frame_bytes) == (frame["frame_start_ts"], len(value))


def test_feature_schema_is_the_clickhouse_table(
    schema_json, make_frame, avro_frames, hf_records, ddl
) -> None:
    features = frame_features(decode_frames(hf_records(avro_frames([make_frame()])), schema_json))
    columns = [name for name in ddl["raw_engine_hf_features"] if name != SERVER_FILLED]
    assert features.columns == columns
    assert [(field.name, field.dataType.simpleString()) for field in features.schema] == list(FEATURE_COLUMNS)


def test_features_computed_in_spark_match_the_signals(
    spark, schema_json, make_frame, avro_frames, hf_records
) -> None:
    frames = [
        make_frame(
            4242 + index,
            fan=f32le(tone(F_N1, 1.0 + index) + tone(F_BPFO * 1.01, 0.1 * (index + 1))),
            core=f32le(tone(F_N2, 0.5)),
        )
        for index in range(5)
    ]
    values = avro_frames(frames)
    corrupt = values[2][:100]
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "2")  # several Arrow batches per task
    try:
        decoded = decode_frames(hf_records([*values, corrupt]), schema_json)
        rows = clickhouse_features(frame_features(decoded)).orderBy("telemetry_id").collect()
    finally:
        spark.conf.unset("spark.sql.execution.arrow.maxRecordsPerBatch")
    assert [row.telemetry_id for row in rows] == [4242, 4243, 4244, 4245, 4246]  # the corrupt copy is out
    for index, (row, frame, value) in enumerate(zip(rows, frames, values, strict=True)):
        assert row.frame_start_ts == frame["frame_start_ts"]  # epoch micros, exact
        assert (row.frame_bytes, row.kafka_offset, row.vibration_channels) == (len(value), index, 2)
        assert row.fan_amp_1x_n1 == pytest.approx(1.0 + index, rel=0.02)
        assert row.fan_amp_bpfo == pytest.approx(0.1 * (index + 1), rel=0.02)
        assert row.core_amp_1x_n2 == pytest.approx(0.5, rel=0.02)
        assert row.process_means == {"EGT_A": 610.0, "EGT_B": 612.0, "OIL_T_A": 85.0}
        assert row.egt_ab_split == pytest.approx(2.0)
