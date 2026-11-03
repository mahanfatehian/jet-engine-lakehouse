"""Progress of the streaming queries as ``ops_stream_progress`` rows, written off the micro-batch path.

``streaming_queries.ProgressRecorder`` turns the progress Spark reports for every micro-batch (an
idle trigger reports none) into a row of a :class:`ProgressBuffer`, which the job's ``stream-progress``
thread inserts every ``FLUSH_SECONDS``: ClickHouse never delays a batch or Spark's listener bus.
Progress is monitoring data: rows ClickHouse cannot take after the writer's retries are dropped, as is
the oldest row of a full buffer, and counted in :attr:`ProgressBuffer.dropped` (logged by a failed flush).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from clickhouse_rows import Row, RowWriter
from streaming_supervisor import summarize_error

logger = logging.getLogger(__name__)

PROGRESS_TABLE = "ops_stream_progress"
PROGRESS_COLUMNS: tuple[str, ...] = (
    "query_name",
    "run_id",
    "batch_id",
    "trigger_ts",
    "input_rows",
    "input_rows_per_s",
    "processed_rows_per_s",
    "batch_duration_ms",
    "add_batch_ms",
    "latest_offset_ms",
    "kafka_lag_records",
)
# The buffer only fills while a flush waits for ClickHouse (seconds of retries, minutes for a hung
# server); the ten queries report about 1,500 micro-batches an hour at the default triggers.
BUFFER_ROWS = 1000
FLUSH_SECONDS = 10.0


def progress_row(progress: Mapping[str, Any]) -> Row:
    """The row of one progress report in its JSON form (``StreamingQueryProgress.json``).

    ``timestamp`` is the trigger's start (UTC, milliseconds), ``triggerExecution`` the whole batch.
    Spark leaves out a NaN or infinite rate and a phase the batch skipped; both are stored as 0.
    """
    durations = progress["durationMs"]
    return (
        progress["name"],
        UUID(progress["runId"]),
        progress["batchId"],
        datetime.fromisoformat(progress["timestamp"]),
        progress["numInputRows"],
        progress.get("inputRowsPerSecond", 0.0),
        progress.get("processedRowsPerSecond", 0.0),
        durations["triggerExecution"],
        durations.get("addBatch", 0),
        durations.get("latestOffset", 0),
        _kafka_lag(progress["sources"]),
    )


def _kafka_lag(sources: Sequence[Mapping[str, Any]]) -> int | None:
    """Records left behind the batch: newest minus end offset, summed over every Kafka partition.

    None without newest offsets: a source other than Kafka, or a batch re-run from the offset log.
    """
    lags = [
        sum(
            newest - source["endOffset"][topic][partition]
            for topic, partitions in source["latestOffset"].items()
            for partition, newest in partitions.items()
        )
        for source in sources
        if isinstance(source.get("latestOffset"), dict)
    ]
    return sum(lags) if lags else None


class ProgressBuffer:
    """Rows waiting for ClickHouse: added on Spark's listener thread, inserted by the flush thread.

    ``writer`` is looked up at every flush: the client exists once the ClickHouse group is ready.
    """

    def __init__(self, writer: Callable[[], RowWriter], capacity: int = BUFFER_ROWS) -> None:
        self._writer = writer
        self._rows: deque[Row] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.dropped = 0

    def add(self, row: Row) -> None:
        """Append ``row`` without waiting for ClickHouse; a full buffer drops its oldest row."""
        with self._lock:
            if len(self._rows) == self._rows.maxlen:
                self.dropped += 1
            self._rows.append(row)

    def flush(self) -> None:
        """Insert every buffered row at once. Never raises: rows that fail are dropped and counted."""
        with self._lock:
            rows = list(self._rows)
            self._rows.clear()
        if not rows:
            return
        try:
            self._writer().insert_rows(PROGRESS_TABLE, PROGRESS_COLUMNS, rows)
        except Exception as exc:  # ClickHouse down, not connected yet or without the table
            with self._lock:
                self.dropped += len(rows)
            logger.warning(
                "dropped %d streaming progress rows (%d since the start): %s",
                len(rows),
                self.dropped,
                summarize_error(exc),
            )

    def run(self, stop_event: threading.Event) -> None:
        """Flush every ``FLUSH_SECONDS`` until ``stop_event`` is set (the ``stream-progress`` thread)."""
        while not stop_event.wait(FLUSH_SECONDS):
            self.flush()
