"""Local-mode Spark tests of the HF feature query's micro-batch handler (ClickHouse is a fake)."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

from cdc_schemas import DEAD_LETTER_COLUMNS, DEAD_LETTER_TABLE
from hf_decode import load_frame_schema
from hf_features import FEATURE_COLUMN_NAMES
from hf_queries import HfFeatureBatchHandler
from hf_rows import HF_BATCH_COLUMNS, HF_BATCHES_TABLE, HF_FEATURES_TABLE

REPO_SCHEMA = Path(__file__).resolve().parents[2] / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
STARTED_AT = datetime(2026, 10, 26, 9, 0, 5, 250000, tzinfo=UTC)
FIRST_FRAME = datetime(2026, 10, 26, 9, 0, 1, 123456, tzinfo=UTC)  # make_frame(4242)


class FakeWriter:
    """Captures inserts per table; optionally fails like an unreachable ClickHouse."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.inserts: dict[str, tuple[Sequence[str], list[Sequence[Any]]]] = {}
        self._fail_on = fail_on

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        if table == self._fail_on:
            raise ConnectionError(f"{table} unavailable")
        if rows:
            self.inserts[table] = (columns, list(rows))

    def as_dicts(self, table: str) -> list[dict[str, Any]]:
        columns, rows = self.inserts[table]
        return [dict(zip(columns, row, strict=True)) for row in rows]


class SerializingWriter(FakeWriter):
    """Serializes every insert like clickhouse-connect does before sending it (Native format with the
    column types of init-db.sql), so a NULL in a non-Nullable column fails here as it does live."""

    def __init__(self, ddl: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._ddl = ddl

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        from clickhouse_connect.datatypes import registry
        from clickhouse_connect.driver.insert import InsertContext
        from clickhouse_connect.driver.transform import NativeTransform

        types = [registry.get_from_name(self._ddl[table][name]) for name in columns]
        context = InsertContext(table, list(columns), types, data=[list(row) for row in rows])
        for _block in NativeTransform.build_insert(context):
            pass
        if context.insert_exception is not None:
            raise context.insert_exception
        super().insert_rows(table, columns, rows)


def fake_clock(*readings: float) -> Iterator[float]:
    return iter(readings)


def handler(writer: FakeWriter) -> HfFeatureBatchHandler:
    clock = fake_clock(100.0, 101.25)
    return HfFeatureBatchHandler(
        writer, load_frame_schema(str(REPO_SCHEMA)), clock=lambda: next(clock), wall_clock=lambda: STARTED_AT
    )


@pytest.fixture
def mixed_batch(make_frame, avro_frames, hf_records) -> tuple[Any, list[bytes]]:
    """Three frames of two engines, then a truncated and a non-Confluent record."""
    frames = [make_frame(4242), make_frame(4243), make_frame(4244, engine_id="GE90-201")]
    values = avro_frames(frames)
    batch = hf_records([*values, values[0][:40], b"not a frame"])
    return batch, values


def test_batch_inserts_features_dead_letters_and_one_summary_row(mixed_batch) -> None:
    batch, values = mixed_batch
    writer = FakeWriter()
    handler(writer)(batch, 7)

    columns, _rows = writer.inserts[HF_FEATURES_TABLE]
    assert tuple(columns) == FEATURE_COLUMN_NAMES
    features = sorted(writer.as_dicts(HF_FEATURES_TABLE), key=lambda row: row["telemetry_id"])
    assert [(row["engine_id"], row["telemetry_id"]) for row in features] == [
        ("JT9D-001", 4242),
        ("JT9D-001", 4243),
        ("GE90-201", 4244),
    ]
    assert features[0]["frame_start_ts"] == FIRST_FRAME
    assert features[0]["fan_amp_1x_n1"] == pytest.approx(2.0, rel=0.02)
    assert features[0]["process_means"] == {"EGT_A": 610.0, "EGT_B": 612.0, "OIL_T_A": 85.0}
    assert isinstance(features[0]["n1_rpm"], float) and isinstance(features[0]["frame_seq"], int)

    columns, _rows = writer.inserts[DEAD_LETTER_TABLE]
    assert tuple(columns) == DEAD_LETTER_COLUMNS
    rejected = writer.as_dicts(DEAD_LETTER_TABLE)
    assert [(row["kafka_topic"], row["kafka_offset"]) for row in rejected] == [
        ("engine_hf.sensor_frames", 3),
        ("engine_hf.sensor_frames", 4),
    ]
    assert rejected[1]["error"] == "unknown magic byte 0x6E (Confluent: 0x00)"

    (summary,) = writer.as_dicts(HF_BATCHES_TABLE)
    assert tuple(writer.inserts[HF_BATCHES_TABLE][0]) == HF_BATCH_COLUMNS
    assert summary == {
        "batch_id": 7,
        "batch_started_at": STARTED_AT,
        "frames": 3,
        "bytes": sum(len(value) for value in values),
        "engines": 2,
        "min_frame_ts": FIRST_FRAME,
        "max_frame_ts": datetime(2026, 10, 26, 9, 0, 3, 123456, tzinfo=UTC),
        "duration_ms": 1250,
    }


def test_frame_times_are_exact_under_a_non_utc_process_timezone(
    make_frame, avro_frames, hf_records, non_utc_process_timezone
) -> None:
    writer = FakeWriter()
    handler(writer)(hf_records(avro_frames([make_frame(4242)])), 1)
    (row,) = writer.as_dicts(HF_FEATURES_TABLE)
    assert row["frame_start_ts"] == FIRST_FRAME
    assert row["frame_start_ts"].utcoffset().total_seconds() == 0
    # clickhouse-connect 1.11 computes DateTime64 ticks from datetime.timestamp():
    ticks = math.floor(row["frame_start_ts"].timestamp()) * 1_000_000 + row["frame_start_ts"].microsecond
    assert ticks == 1_793_005_201_123_456


def test_batch_of_rejected_records_writes_no_summary(hf_records) -> None:
    writer = FakeWriter()
    handler(writer)(hf_records([b"\x00\x00\x00\x00\x01garbage", None]), 3)
    assert set(writer.inserts) == {DEAD_LETTER_TABLE}
    assert len(writer.as_dicts(DEAD_LETTER_TABLE)) == 2


def test_insert_failure_fails_the_batch_and_releases_the_cache(mixed_batch) -> None:
    batch, _values = mixed_batch
    with pytest.raises(ConnectionError, match="raw_engine_hf_features unavailable"):
        handler(FakeWriter(fail_on=HF_FEATURES_TABLE))(batch, 2)
    assert batch.is_cached is False


def test_undefined_features_insert_as_nan_into_the_non_nullable_columns(
    make_frame, avro_frames, hf_records, ddl
) -> None:
    # Arrow turns the NaN of an undefined feature into NULL; inserted as NULL, these frames would
    # fail every replay of the batch, the healthy frame next to them included.
    flat = struct.pack("<2048f", *([0.0] * 2048))
    frames = [
        make_frame(4242),
        make_frame(4243, fan=flat),
        make_frame(4244, process=[]),
        make_frame(4245, process=[("EGT_A", "degC", struct.pack("<8f", *([float("nan")] * 8)))]),
    ]
    writer = SerializingWriter(ddl)
    handler(writer)(hf_records(avro_frames(frames)), 1)

    rows = {row["telemetry_id"]: row for row in writer.as_dicts(HF_FEATURES_TABLE)}
    assert sorted(rows) == [4242, 4243, 4244, 4245]
    floats = [name for name in FEATURE_COLUMN_NAMES if ddl[HF_FEATURES_TABLE][name] == "Float32"]
    assert all(row[name] is not None for row in rows.values() for name in floats)
    assert math.isnan(rows[4243]["fan_kurtosis"]) and rows[4243]["core_rms"] > 0
    assert rows[4244]["process_means"] == {} and math.isnan(rows[4244]["egt_ab_split"])
    assert math.isnan(rows[4245]["process_means"]["EGT_A"])
    assert rows[4242]["fan_amp_1x_n1"] == pytest.approx(2.0, rel=0.02)
