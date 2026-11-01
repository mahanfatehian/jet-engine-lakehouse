"""Unit tests of the pure-Python ClickHouse row conversion (no Spark, no ClickHouse)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from cdc_schemas import DEAD_LETTER_COLUMNS, ENGINE_TELEMETRY, MAINTENANCE_LOG
from clickhouse_rows import (
    DATETIME64_MAX_MICROS,
    DATETIME64_MIN_MICROS,
    clamp_timestamp,
    convert_row,
    dead_letter_row,
    deduplicate,
    split_collected,
    to_utc_datetime,
)

# 2026-10-26T09:00:01.123456Z in epoch microseconds, computed independently of the code under test.
ROUND_TRIP_MICROS = 1_793_005_201_123_456
ROUND_TRIP_INSTANT = datetime(2026, 10, 26, 9, 0, 1, 123456, tzinfo=UTC)
# The ClickHouse DateTime64 range.
DATETIME64_FIRST = datetime(1900, 1, 1, tzinfo=UTC)
DATETIME64_LAST = datetime(2299, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
# A record timestamp a producer set in microseconds where Kafka expects milliseconds: Spark
# reads it as the year 58,788, which no Python datetime can hold.
MICROS_AS_MILLIS = ROUND_TRIP_MICROS * 1000


def _clickhouse_ticks(value: datetime, precision: int) -> int:
    """DateTime64 tick computation of clickhouse-connect 1.11 for a datetime value."""
    return ((math.floor(value.timestamp()) * 1_000_000 + value.microsecond) * precision) // 1_000_000


def test_to_utc_datetime_is_exact_and_timezone_aware() -> None:
    converted = to_utc_datetime(ROUND_TRIP_MICROS)
    assert converted == ROUND_TRIP_INSTANT
    assert converted is not None and converted.utcoffset() is not None
    assert converted.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    ("micros", "expected"),
    [
        (0, datetime(1970, 1, 1, tzinfo=UTC)),
        (-1, datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)),
        (1_793_005_200_123_456, datetime(2026, 10, 26, 9, 0, 0, 123456, tzinfo=UTC)),
        (1_793_005_566_000_502, datetime(2026, 10, 26, 9, 6, 6, 502, tzinfo=UTC)),
    ],
)
def test_to_utc_datetime_known_instants(micros: int, expected: datetime) -> None:
    assert to_utc_datetime(micros) == expected


def test_to_utc_datetime_keeps_null() -> None:
    assert to_utc_datetime(None) is None


@pytest.mark.parametrize(
    ("micros", "expected"),
    [
        (DATETIME64_MAX_MICROS + 1, datetime(2300, 1, 1, tzinfo=UTC)),
        (DATETIME64_MIN_MICROS - 1, datetime(1899, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)),
        (32_503_680_000_000_000, datetime(3000, 1, 1, tzinfo=UTC)),
    ],
)
def test_to_utc_datetime_is_exact_outside_the_datetime64_range_too(micros: int, expected: datetime) -> None:
    # Valid rows never get here (they are dead-lettered upstream); were one to, it must not be rewritten.
    assert to_utc_datetime(micros) == expected
    assert convert_row((micros,), frozenset({0})) == (expected,)


@pytest.mark.parametrize(
    ("micros", "expected"),
    [
        (ROUND_TRIP_MICROS, ROUND_TRIP_MICROS),
        (0, 0),
        (DATETIME64_MIN_MICROS, DATETIME64_MIN_MICROS),
        (DATETIME64_MAX_MICROS, DATETIME64_MAX_MICROS),
        (DATETIME64_MIN_MICROS - 1, DATETIME64_MIN_MICROS),
        (DATETIME64_MAX_MICROS + 1, DATETIME64_MAX_MICROS),
        (MICROS_AS_MILLIS, DATETIME64_MAX_MICROS),
        (-(2**63), DATETIME64_MIN_MICROS),
    ],
)
def test_clamp_timestamp_moves_only_instants_outside_the_range(micros: int, expected: int) -> None:
    assert clamp_timestamp(micros) == expected


@pytest.mark.parametrize("fraction", [0, 1, 123456, 499999, 500000, 999998, 999999])
def test_aware_datetimes_serialise_to_exact_clickhouse_ticks(fraction: int) -> None:
    # Mirrors clickhouse-connect's float-based tick maths: the instant must survive for every
    # microsecond fraction at DateTime64(6) and truncate correctly at DateTime64(3).
    micros = 1_793_005_201_000_000 + fraction
    value = to_utc_datetime(micros)
    assert value is not None
    assert _clickhouse_ticks(value, 1_000_000) == micros
    assert _clickhouse_ticks(value, 1_000) == micros // 1_000


def test_convert_row_converts_only_timestamp_positions() -> None:
    values = (7, "JT9D-001", ROUND_TRIP_MICROS, None, 1.5)
    converted = convert_row(values, frozenset({2, 3}))
    assert converted == (7, "JT9D-001", ROUND_TRIP_INSTANT, None, 1.5)


def test_dead_letter_row_converts_timestamp_and_defaults_missing_key() -> None:
    record = ("engine_cdc.public.engine_telemetry", 2, 41, ROUND_TRIP_MICROS, None, "not json", "malformed")
    row = dead_letter_row(record)
    assert len(row) == len(DEAD_LETTER_COLUMNS)
    assert row == (
        "engine_cdc.public.engine_telemetry",
        2,
        41,
        ROUND_TRIP_INSTANT,
        "",
        "not json",
        "malformed",
    )


@pytest.mark.parametrize(
    ("kafka_micros", "stored"),
    [
        (DATETIME64_MAX_MICROS, DATETIME64_LAST),
        (32_503_680_000_000_000, DATETIME64_LAST),  # the year 3000
        (MICROS_AS_MILLIS, DATETIME64_LAST),
        (DATETIME64_MIN_MICROS - 1, DATETIME64_FIRST),
        (None, None),  # Spark's Kafka source always sets a timestamp; nothing is made up if not
    ],
)
def test_dead_letter_row_clamps_the_broker_timestamp_into_the_datetime64_range(
    kafka_micros: int | None, stored: datetime | None
) -> None:
    record = ("engine_hf.sensor_frames", 0, 7, kafka_micros, "JT9D-001", "AAAA", "null value")
    row = dead_letter_row(record)
    assert row == ("engine_hf.sensor_frames", 0, 7, stored, "JT9D-001", "AAAA", "null value")


def test_dead_letter_of_a_timestamp_python_cannot_represent_is_still_insertable() -> None:
    with pytest.raises(OverflowError):  # what failed the batch, on every replay, before the clamp
        to_utc_datetime(MICROS_AS_MILLIS)
    topic = "engine_cdc.public.engine_telemetry"
    record = (topic, 2, 41, MICROS_AS_MILLIS, None, "{}", "missing op")
    nulls = (None,) * len(ENGINE_TELEMETRY.clickhouse_columns)
    batch = split_collected(ENGINE_TELEMETRY, [(record, *nulls)])
    assert batch.dead_letters == [(topic, 2, 41, DATETIME64_LAST, "", "{}", "missing op")]


def test_deduplicate_keeps_first_occurrence_per_key() -> None:
    rows = [(1, 100, "a"), (1, 100, "b"), (1, 101, "c"), (2, 100, "d")]
    unique, dropped = deduplicate(rows, [0, 1])
    assert unique == [(1, 100, "a"), (1, 101, "c"), (2, 100, "d")]
    assert dropped == 1


def test_deduplicate_without_duplicates_is_identity() -> None:
    rows = [(1, 1), (2, 2)]
    assert deduplicate(rows, [0]) == (rows, 0)


def _telemetry_values(telemetry_id: int, lsn: int, offset: int) -> tuple[object, ...]:
    by_name: dict[str, object] = {
        "telemetry_id": telemetry_id,
        "engine_id": "JT9D-001",
        "lifecycle_id": 1,
        "cycle_number": 3,
        "event_ts": ROUND_TRIP_MICROS,
        "rpm": 14420.1,
        "core_temp_c": 576.0,
        "egt_c": 652.1,
        "vibration_mm_s": 1.1,
        "fuel_flow_kg_h": 1250.9,
        "oil_pressure_psi": 45.1,
        "wear_factor": 0.0042,
        "status": "HEALTHY",
        "maintenance_id": None,
        "created_ts": 1_793_005_566_004_161,
        "cdc_op": "c",
        "cdc_lsn": lsn,
        "cdc_source_ts": 1_793_005_566_008_000,
        "cdc_emitted_ts": 1_793_005_566_278_000,
        "is_deleted": 0,
        "kafka_partition": 0,
        "kafka_offset": offset,
    }
    return tuple(by_name[name] for name in ENGINE_TELEMETRY.clickhouse_columns)


def test_split_collected_converts_dedups_and_routes_dead_letters() -> None:
    dead = (
        "engine_cdc.public.engine_telemetry",
        1,
        9,
        ROUND_TRIP_MICROS,
        '{"telemetry_id":9}',
        "{}",
        "missing op",
    )
    nulls = (None,) * len(ENGINE_TELEMETRY.clickhouse_columns)
    collected = [
        (None, *_telemetry_values(1, lsn=500, offset=0)),
        (None, *_telemetry_values(1, lsn=500, offset=3)),  # Debezium re-delivery of the same change
        (None, *_telemetry_values(1, lsn=600, offset=4)),
        (dead, *nulls),
    ]
    batch = split_collected(ENGINE_TELEMETRY, collected)

    assert batch.table is ENGINE_TELEMETRY
    assert batch.duplicates == 1
    assert len(batch.rows) == 2
    columns = ENGINE_TELEMETRY.clickhouse_columns
    first = dict(zip(columns, batch.rows[0], strict=True))
    assert first["kafka_offset"] == 0
    assert first["event_ts"] == ROUND_TRIP_INSTANT
    assert first["cdc_source_ts"] == datetime(2026, 10, 26, 9, 6, 6, 8000, tzinfo=UTC)
    assert first["maintenance_id"] is None
    assert [dict(zip(columns, row, strict=True))["cdc_lsn"] for row in batch.rows] == [500, 600]
    assert batch.dead_letters == [
        (
            "engine_cdc.public.engine_telemetry",
            1,
            9,
            ROUND_TRIP_INSTANT,
            '{"telemetry_id":9}',
            "{}",
            "missing op",
        )
    ]


def test_split_collected_of_empty_input() -> None:
    batch = split_collected(MAINTENANCE_LOG, [])
    assert (batch.rows, batch.dead_letters, batch.duplicates) == ([], [], 0)
