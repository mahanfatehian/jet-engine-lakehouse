"""Local-mode Spark tests of the Path B foreachBatch handler (ClickHouse replaced by a fake).

Includes the timezone proof: under a non-UTC *process* timezone a plain ``collect()`` of
a TimestampType shifts the instant, while the rows handed to clickhouse-connect stay the
exact UTC instant ("2026-10-26T09:00:01.123456Z" -> 2026-10-26 09:00:01.123456).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("pyspark")

from cdc_schemas import DEAD_LETTER_COLUMNS, DEAD_LETTER_TABLE, TABLES
from pyspark.sql import functions as sf
from streaming_queries import ClickHouseBatchHandler

TOPICS = {f"engine_cdc.public.{table.name}": table for table in TABLES}
TELEMETRY_TOPIC = "engine_cdc.public.engine_telemetry"
ROUND_TRIP_TEXT = "2026-10-26T09:00:01.123456Z"
ROUND_TRIP_INSTANT = datetime(2026, 10, 26, 9, 0, 1, 123456, tzinfo=UTC)
ROUND_TRIP_MICROS = 1_793_005_201_123_456


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


def round_trip_message(sample_messages: list[Any]) -> Any:
    """A new telemetry insert (telemetry_id 3) whose event_ts has full microsecond precision."""
    base = next(message for message in sample_messages if message.topic == TELEMETRY_TOPIC)
    envelope = json.loads(base.value)
    envelope["after"].update(telemetry_id=3, event_ts=ROUND_TRIP_TEXT)
    envelope["source"]["lsn"] = 26871000
    return base._replace(key='{"telemetry_id":3}', value=json.dumps(envelope), partition=1, offset=0)


@pytest.fixture
def non_utc_process_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the test with the Python process in UTC+03:30 (Iran, no DST)."""
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "IRST-03:30")
        time.tzset()
    if not datetime(2026, 10, 26, 9).astimezone().utcoffset():
        pytest.skip("cannot switch this process to a non-UTC local timezone")
    yield
    monkeypatch.undo()
    if hasattr(time, "tzset"):
        time.tzset()


def test_batch_is_routed_per_table_with_dedup_and_dead_letters(sample_messages, records_frame) -> None:
    telemetry = [message for message in sample_messages if message.topic == TELEMETRY_TOPIC]
    duplicate = telemetry[0]._replace(offset=9)  # same change re-delivered at another offset
    malformed = telemetry[0]._replace(key='{"telemetry_id":4}', value="not json", offset=10)
    tombstone = telemetry[0]._replace(value=None, offset=11)
    writer = FakeWriter()

    handler = ClickHouseBatchHandler(writer, TOPICS)
    handler(records_frame([*sample_messages, duplicate, malformed, tombstone]), 7)

    assert set(writer.inserts) == {
        "raw_engine_telemetry",
        "raw_engine_status_registry",
        "raw_maintenance_log",
        DEAD_LETTER_TABLE,
    }
    for table in TABLES:
        columns, _rows = writer.inserts[table.clickhouse_table]
        assert tuple(columns) == table.clickhouse_columns
        assert "ingested_at" not in columns
    telemetry_rows = writer.as_dicts("raw_engine_telemetry")
    assert [(row["telemetry_id"], row["cdc_op"], row["is_deleted"]) for row in telemetry_rows] == [
        (2, "c", 0),
        (2, "u", 0),
        (1, "c", 0),
        (1, "d", 1),
    ]
    assert len(writer.as_dicts("raw_engine_status_registry")) == 2
    assert [row["maintenance_status"] for row in writer.as_dicts("raw_maintenance_log")] == [
        "OPEN",
        "COMPLETED",
    ]
    (dead_letter,) = writer.as_dicts(DEAD_LETTER_TABLE)
    assert tuple(writer.inserts[DEAD_LETTER_TABLE][0]) == DEAD_LETTER_COLUMNS
    assert dead_letter == {
        "kafka_topic": TELEMETRY_TOPIC,
        "kafka_partition": 2,
        "kafka_offset": 10,
        "kafka_timestamp": datetime(2026, 10, 26, 9, 6, 6, 300000, tzinfo=UTC),
        "raw_key": '{"telemetry_id":4}',
        "raw_value": "not json",
        "error": "malformed JSON envelope",
    }


def test_rows_carry_python_types_clickhouse_connect_expects(sample_messages, records_frame) -> None:
    writer = FakeWriter()
    ClickHouseBatchHandler(writer, TOPICS)(records_frame(sample_messages), 1)
    registry = writer.as_dicts("raw_engine_status_registry")[1]
    assert registry["last_maintenance_ts"] is None
    assert registry["status_changed_ts"] == datetime(2026, 10, 26, 9, 6, 6, 9676, tzinfo=UTC)
    assert registry["cdc_emitted_ts"] == datetime(2026, 10, 26, 9, 6, 6, 280000, tzinfo=UTC)
    assert isinstance(registry["cdc_lsn"], int)
    assert isinstance(registry["wear_factor"], float)
    maintenance = writer.as_dicts("raw_maintenance_log")[1]
    assert maintenance["completed_ts"] == datetime(2026, 10, 26, 9, 0, 21, tzinfo=UTC)
    assert maintenance["downtime_seconds"] == 20


def test_clickhouse_rows_are_exact_under_a_non_utc_process_timezone(
    sample_messages, records_frame, non_utc_process_timezone
) -> None:
    frame = records_frame([round_trip_message(sample_messages)])

    # The pitfall the projection avoids: collect() renders TimestampType in local time.
    naive = frame.select(sf.try_to_timestamp(sf.lit(ROUND_TRIP_TEXT))).collect()[0][0]
    assert naive.tzinfo is None
    assert naive != ROUND_TRIP_INSTANT.replace(tzinfo=None)

    writer = FakeWriter()
    ClickHouseBatchHandler(writer, TOPICS)(frame, 1)
    (row,) = writer.as_dicts("raw_engine_telemetry")
    assert row["event_ts"] == ROUND_TRIP_INSTANT
    assert row["event_ts"].utcoffset().total_seconds() == 0
    # clickhouse-connect 1.11 computes DateTime64 ticks from datetime.timestamp():
    ticks = math.floor(row["event_ts"].timestamp()) * 1_000_000 + row["event_ts"].microsecond
    assert ticks == ROUND_TRIP_MICROS


def test_insert_failure_fails_the_batch_and_releases_the_cache(sample_messages, records_frame) -> None:
    frame = records_frame(sample_messages)
    handler = ClickHouseBatchHandler(FakeWriter(fail_on="raw_engine_status_registry"), TOPICS)
    with pytest.raises(ConnectionError, match="raw_engine_status_registry unavailable"):
        handler(frame, 3)
    assert frame.is_cached is False
