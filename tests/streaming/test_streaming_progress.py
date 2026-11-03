"""Unit tests of the streaming progress rows (``ops_stream_progress``) and of their buffer.

``fixtures/stream_progress.json`` was recorded from Spark 3.5.9: batch 1 of a query shaped like
``clickhouse_raw`` on the live stack's three CDC topics, capped at 20000 offsets per trigger.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from streaming_progress import FLUSH_SECONDS, PROGRESS_COLUMNS, PROGRESS_TABLE, ProgressBuffer, progress_row

FIXTURE = Path(__file__).parent / "fixtures" / "stream_progress.json"
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
NOT_CONNECTED = "ClickHouse is not connected: check_ready() has not succeeded yet"


class FakeWriter:
    """The buffer's writer lookup and writer: fails the lookup (not connected) or the insert (down)."""

    def __init__(self, lookup_error: Exception | None = None) -> None:
        self.inserts: list[tuple[str, Sequence[str], list[Sequence[Any]]]] = []
        self.lookup_error = lookup_error
        self.insert_error: Exception | None = None

    def __call__(self) -> FakeWriter:
        if self.lookup_error is not None:
            raise self.lookup_error
        return self

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        if self.insert_error is not None:
            raise self.insert_error
        self.inserts.append((table, columns, list(rows)))


def test_recorded_progress_becomes_one_row() -> None:
    assert dict(zip(PROGRESS_COLUMNS, progress_row(RECORDED), strict=True)) == {
        "query_name": "clickhouse_raw",
        "run_id": UUID("7318fc90-e360-45a6-b78e-a40b31bfb9de"),
        "batch_id": 1,
        "trigger_ts": datetime(2026, 11, 3, 6, 35, 47, 354000, tzinfo=UTC),
        "input_rows": 19996,
        "input_rows_per_s": 5043.12736443884,
        "processed_rows_per_s": 26948.787061994608,
        "batch_duration_ms": 742,
        "add_batch_ms": 572,
        "latest_offset_ms": 11,
        # Newest minus end offset of the 9 partitions: engine_telemetry 290080 + 290132 + 290341,
        # maintenance_log 403 + 349 + 371, engine_status_registry 22784 + 45597 + 0.
        "kafka_lag_records": 940057,
    }


def test_a_batch_re_run_from_the_offset_log_has_a_null_lag(ddl: dict[str, dict[str, str]]) -> None:
    # Spark 3.5.9 re-runs a failed batch with the offsets of its log: it skips the latestOffset phase
    # and reports no newest offsets. It also leaves out a rate that is NaN or infinite.
    rerun = {key: value for key, value in RECORDED.items() if key != "inputRowsPerSecond"}
    rerun["durationMs"] = {key: ms for key, ms in RECORDED["durationMs"].items() if key != "latestOffset"}
    rerun["sources"] = [{**RECORDED["sources"][0], "latestOffset": None}]
    row = dict(zip(PROGRESS_COLUMNS, progress_row(rerun), strict=True))
    assert (row["input_rows_per_s"], row["latest_offset_ms"], row["kafka_lag_records"]) == (0.0, 0, None)
    assert tuple(ddl[PROGRESS_TABLE]) == PROGRESS_COLUMNS
    assert ddl[PROGRESS_TABLE]["kafka_lag_records"] == "Nullable(UInt64)"


def test_a_full_buffer_drops_and_counts_its_oldest_rows() -> None:
    writer = FakeWriter()
    buffer = ProgressBuffer(writer, capacity=3)
    for batch_id in range(5):
        buffer.add((batch_id,))
    buffer.flush()
    buffer.flush()  # nothing buffered: no insert
    assert writer.inserts == [(PROGRESS_TABLE, PROGRESS_COLUMNS, [(2,), (3,), (4,)])]
    assert buffer.dropped == 2


def test_rows_are_dropped_and_counted_while_clickhouse_is_down(caplog: pytest.LogCaptureFixture) -> None:
    writer = FakeWriter(lookup_error=RuntimeError(NOT_CONNECTED))
    buffer = ProgressBuffer(writer)
    with caplog.at_level(logging.WARNING):
        buffer.add((0,))
        buffer.flush()  # never raises: no query may notice
        writer.lookup_error, writer.insert_error = None, ConnectionError("clickhouse:8123 refused")
        buffer.add((1,))
        buffer.add((2,))
        buffer.flush()
    writer.insert_error = None
    buffer.add((3,))
    buffer.flush()
    assert writer.inserts == [(PROGRESS_TABLE, PROGRESS_COLUMNS, [(3,)])]
    assert buffer.dropped == 3
    assert caplog.messages == [
        f"dropped 1 streaming progress rows (1 since the start): {NOT_CONNECTED}",
        "dropped 2 streaming progress rows (3 since the start): clickhouse:8123 refused",
    ]


def test_the_flush_thread_inserts_every_flush_interval_until_the_stop() -> None:
    writer = FakeWriter()
    buffer = ProgressBuffer(writer)
    timeouts: list[float | None] = []

    class StopInTheThirdWait(threading.Event):  # a micro-batch reports during every wait
        def wait(self, timeout: float | None = None) -> bool:
            timeouts.append(timeout)
            buffer.add((len(timeouts),))
            return len(timeouts) == 3

    buffer.run(StopInTheThirdWait())
    assert timeouts == [FLUSH_SECONDS] * 3
    assert [rows for *_, rows in writer.inserts] == [[(1,)], [(2,)]]  # row 3 waits for the job's last flush


def test_every_micro_batch_of_a_real_query_becomes_one_row(spark: Any, tmp_path: Path) -> None:
    from streaming_queries import ProgressRecorder  # imports PySpark

    writer = FakeWriter()
    buffer = ProgressBuffer(writer)
    recorder = ProgressRecorder(buffer)
    spark.streams.addListener(recorder)
    stream = spark.readStream.format("rate").load().writeStream.queryName("progress_probe").format("noop")
    stream = stream.option("checkpointLocation", str(tmp_path)).trigger(processingTime="200 milliseconds")
    query = stream.start()
    deadline = time.monotonic() + 120
    try:
        while len(writer.inserts) < 3 and time.monotonic() < deadline:
            time.sleep(0.2)
            buffer.flush()
    finally:
        query.stop()
        spark.streams.removeListener(recorder)
    rows = [dict(zip(PROGRESS_COLUMNS, row, strict=True)) for *_, batch in writer.inserts for row in batch]
    assert len(rows) >= 3
    assert [row["batch_id"] for row in rows] == list(range(len(rows)))  # every batch once, in order
    assert {(row["query_name"], row["run_id"], row["kafka_lag_records"]) for row in rows} == {
        ("progress_probe", UUID(query.runId), None)  # the rate source has no Kafka offsets
    }
