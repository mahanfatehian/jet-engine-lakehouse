"""Contract and unit tests of the HF feature rows: init-db.sql columns and pure conversions.

The ``ddl`` fixture (conftest) parses infrastructure/clickhouse/init-db.sql; no Spark is needed.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from hf_features import FEATURE_COLUMN_NAMES, FEATURE_COLUMNS, FEATURE_SCHEMA_DDL
from hf_rows import (
    HF_BATCH_COLUMNS,
    HF_BATCHES_TABLE,
    HF_FEATURES_TABLE,
    UINT16_MAX,
    BatchStats,
    batch_row,
    batch_stats,
    feature_rows,
)

SERVER_FILLED = "ingested_at"
# ClickHouse type -> the Spark SQL types of FEATURE_COLUMNS that insert into it losslessly
# (unsigned and narrower targets are guarded by hf_decode's checks).
COMPATIBLE_SPARK_TYPES = {
    "LowCardinality(String)": {"string"},
    "UInt16": {"int"},
    "UInt32": {"int"},
    "UInt64": {"bigint"},
    "Int32": {"int"},
    "Int64": {"bigint"},
    "Float32": {"float"},
    "DateTime64(6, 'UTC')": {"timestamp"},
    "Map(LowCardinality(String), Float32)": {"map<string,float>"},
}
# 2026-10-26T09:00:01.123456Z in epoch microseconds, computed independently of the code under test.
ROUND_TRIP_MICROS = 1_793_005_201_123_456
ROUND_TRIP_INSTANT = datetime(2026, 10, 26, 9, 0, 1, 123456, tzinfo=UTC)


def test_feature_columns_are_the_raw_engine_hf_features_insert_columns(ddl: dict) -> None:
    columns = ddl[HF_FEATURES_TABLE]
    assert SERVER_FILLED in columns
    assert tuple(name for name in columns if name != SERVER_FILLED) == FEATURE_COLUMN_NAMES


def test_feature_column_types_insert_losslessly(ddl: dict) -> None:
    columns = ddl[HF_FEATURES_TABLE]
    for name, spark_type in FEATURE_COLUMNS:
        assert spark_type in COMPATIBLE_SPARK_TYPES[columns[name]], name


def test_feature_schema_ddl_lists_every_column_in_order() -> None:
    assert FEATURE_SCHEMA_DDL.split(", ")[:2] == ["engine_id string", "engine_model string"]
    assert [part.split(" ")[0] for part in FEATURE_SCHEMA_DDL.split(", ")] == list(FEATURE_COLUMN_NAMES)


def test_batch_columns_are_the_hf_ingest_batches_insert_columns(ddl: dict) -> None:
    columns = ddl[HF_BATCHES_TABLE]
    assert tuple(name for name in columns if name != SERVER_FILLED) == HF_BATCH_COLUMNS


def collected_row(engine_id: str, micros: int, frame_bytes: int) -> tuple[Any, ...]:
    """A collected feature row: frame_start_ts still in epoch microseconds."""
    values: dict[str, Any] = dict.fromkeys(FEATURE_COLUMN_NAMES, 0.0)
    values.update(engine_id=engine_id, frame_start_ts=micros, frame_bytes=frame_bytes)
    values["process_means"] = {"EGT_A": 610.0}
    return tuple(values[name] for name in FEATURE_COLUMN_NAMES)


def test_feature_rows_carry_exact_timezone_aware_instants() -> None:
    (row,) = feature_rows([collected_row("JT9D-001", ROUND_TRIP_MICROS, 16574)])
    converted = dict(zip(FEATURE_COLUMN_NAMES, row, strict=True))
    assert converted["frame_start_ts"] == ROUND_TRIP_INSTANT
    assert converted["frame_start_ts"].utcoffset().total_seconds() == 0
    assert converted["process_means"] == {"EGT_A": 610.0}
    assert converted["frame_bytes"] == 16574


def test_feature_rows_restore_the_name_order_of_process_means() -> None:
    collected = collected_row("JT9D-001", ROUND_TRIP_MICROS, 1)
    position = FEATURE_COLUMN_NAMES.index("process_means")
    hashed = {"OIL_T_A": 85.0, "EGT_B": 612.0, "N1_A": 3650.0, "EGT_A": 610.0}  # collect() order
    (row,) = feature_rows([(*collected[:position], hashed, *collected[position + 1 :])])
    assert list(row[position]) == ["EGT_A", "EGT_B", "N1_A", "OIL_T_A"]
    assert row[position] == hashed


def test_batch_stats_summarise_volume_engines_and_frame_time() -> None:
    rows = feature_rows(
        [
            collected_row("JT9D-001", ROUND_TRIP_MICROS + 1_000_000, 100),
            collected_row("GE90-201", ROUND_TRIP_MICROS, 250),
            collected_row("JT9D-001", ROUND_TRIP_MICROS + 2_000_000, 150),
        ]
    )
    stats = batch_stats(rows)
    assert stats == BatchStats(
        frames=3,
        bytes=500,
        engines=2,
        min_frame_ts=ROUND_TRIP_INSTANT,
        max_frame_ts=datetime(2026, 10, 26, 9, 0, 3, 123456, tzinfo=UTC),
    )
    assert batch_stats([]) is None


def test_engine_count_saturates_at_the_uint16_maximum() -> None:
    rows = feature_rows(collected_row(f"E-{index}", ROUND_TRIP_MICROS, 1) for index in range(UINT16_MAX + 2))
    stats = batch_stats(rows)
    assert stats is not None
    assert (stats.frames, stats.engines) == (UINT16_MAX + 2, UINT16_MAX)


def test_batch_row_follows_the_column_order() -> None:
    started = datetime(2026, 10, 26, 9, 0, 5, 250000, tzinfo=UTC)
    stats = BatchStats(4, 4096, 2, ROUND_TRIP_INSTANT, ROUND_TRIP_INSTANT)
    row = dict(zip(HF_BATCH_COLUMNS, batch_row(17, started, stats, 1.2346), strict=True))
    assert row == {
        "batch_id": 17,
        "batch_started_at": started,
        "frames": 4,
        "bytes": 4096,
        "engines": 2,
        "min_frame_ts": ROUND_TRIP_INSTANT,
        "max_frame_ts": ROUND_TRIP_INSTANT,
        "duration_ms": 1235,
    }


@pytest.mark.parametrize(("seconds", "expected_ms"), [(0.0, 0), (0.0004, 0), (0.0006, 1), (-0.5, 0)])
def test_batch_duration_is_rounded_to_whole_milliseconds(seconds: float, expected_ms: int) -> None:
    stats = BatchStats(1, 1, 1, ROUND_TRIP_INSTANT, ROUND_TRIP_INSTANT)
    assert batch_row(0, ROUND_TRIP_INSTANT, stats, seconds)[-1] == expected_ms


def test_feature_rows_insert_nan_where_arrow_delivered_null() -> None:
    # mapInPandas' Arrow serializer turns NaN into NULL; the Float32 columns are not Nullable.
    collected = dict(zip(FEATURE_COLUMN_NAMES, collected_row("JT9D-001", ROUND_TRIP_MICROS, 1), strict=True))
    collected.update(fan_kurtosis=None, egt_ab_split=None, process_means={"OIL_T_A": None, "EGT_A": 610.0})
    (row,) = feature_rows([tuple(collected.values())])
    converted = dict(zip(FEATURE_COLUMN_NAMES, row, strict=True))
    assert math.isnan(converted["fan_kurtosis"]) and math.isnan(converted["egt_ab_split"])
    assert list(converted["process_means"]) == ["EGT_A", "OIL_T_A"]
    assert math.isnan(converted["process_means"]["OIL_T_A"]) and converted["process_means"]["EGT_A"] == 610.0
    assert converted["frame_bytes"] == 1 and converted["engine_id"] == "JT9D-001"


def test_feature_rows_insert_an_empty_map_for_a_null_process_means() -> None:
    collected = list(collected_row("JT9D-001", ROUND_TRIP_MICROS, 1))
    collected[FEATURE_COLUMN_NAMES.index("process_means")] = None
    (row,) = feature_rows([collected])
    assert row[FEATURE_COLUMN_NAMES.index("process_means")] == {}
