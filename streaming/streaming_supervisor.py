"""Independent supervision of the streaming queries: gated starts, restarts and a heartbeat.

The queries form groups that share dependencies: the lake group (Path A) needs Kafka and
MinIO, the ClickHouse group (Path B) needs Kafka and ClickHouse with its raw tables. Every
round (``POLL_SECONDS``) the supervisor

1. notices queries that terminated on their own, records why and schedules a restart from
   the query's checkpoint after a capped exponential backoff;
2. checks, once per group, the dependencies of the queries that are due to (re)start and
   starts them when they are ready, otherwise backs them off as well;
3. publishes the per-query state to the heartbeat file (``streaming_heartbeat``).

A failing query or an unavailable dependency therefore never stops the other group, nor the
other queries of its own group. The loop only ends on a stop request (SIGTERM/SIGINT: drain,
stop every query, exit 0) or when the Spark driver itself is gone (exit 1, so that Docker's
restart policy starts a fresh process).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from streaming_heartbeat import utc_now_iso
from streaming_retry import capped_backoff

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_DRIVER_STOPPED = 1
POLL_SECONDS = 2.0
RESTART_BASE_DELAY_SECONDS = 2.0
RESTART_MAX_DELAY_SECONDS = 60.0
# docker compose grants 30 s (stop_grace_period) between SIGTERM and SIGKILL.
DRAIN_TIMEOUT_SECONDS = 20.0
ERROR_SUMMARY_CHARS = 500
_DRAIN_POLL_SECONDS = 0.5
# Spark appends the query's offsets and logical plan to the error of a failed query.
_QUERY_DESCRIPTION = "\n=== Streaming Query ==="
_STACK_TRACE_MARKERS = ("\n\tat ", "\nJVM stacktrace:", "\nDriver stacktrace:", _QUERY_DESCRIPTION)
_PYTHON_TRACEBACK = "Traceback (most recent call last):"
# Spark's Kafka source names the option in every data-loss error it raises.
_DATA_LOSS_MARKERS = ("failOnDataLoss", "data may have been lost")


class StreamingQueryLike(Protocol):
    """The part of ``pyspark.sql.streaming.StreamingQuery`` the supervisor relies on."""

    @property
    def name(self) -> str: ...

    @property
    def isActive(self) -> bool: ...  # noqa: N802 - PySpark API

    @property
    def status(self) -> dict[str, Any]: ...

    @property
    def lastProgress(self) -> dict[str, Any] | None: ...  # noqa: N802 - PySpark API

    def exception(self) -> object | None: ...

    def stop(self) -> None: ...


class HeartbeatPublisher(Protocol):
    """Destination of the supervisor state (``streaming_heartbeat.HeartbeatWriter``)."""

    def publish(self, status: Mapping[str, object], *, force: bool = False) -> bool: ...


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """A restartable query: ``start`` builds it and starts it from its checkpoint.

    ``data_loss_recovery`` tells the operator what to do when the query stopped because
    records it had not read yet are gone from Kafka (KAFKA_FAIL_ON_DATA_LOSS=true).
    """

    name: str
    start: Callable[[], StreamingQueryLike]
    data_loss_recovery: str = ""


@dataclass(frozen=True, slots=True)
class QueryGroup:
    """Queries sharing their dependencies; ``check_ready`` raises while one is unavailable."""

    name: str
    check_ready: Callable[[], None]
    queries: tuple[QuerySpec, ...]


def summarize_error(error: object) -> str:
    """One-line digest of an error for the logs and the heartbeat.

    Drops JVM stack traces and Spark's query description. Of a Python traceback (a failing
    ``foreachBatch``) only the final exception is kept, as it names the cause. Whitespace
    is collapsed and the result truncated to ``ERROR_SUMMARY_CHARS``.
    """
    text = str(error)
    for marker in _STACK_TRACE_MARKERS:
        text = text.split(marker, 1)[0]
    prefix, found, _ = text.partition(_PYTHON_TRACEBACK)
    if found:
        text = prefix + _final_exception(text.rpartition(_PYTHON_TRACEBACK)[2])
    text = " ".join(text.split())
    return text if len(text) <= ERROR_SUMMARY_CHARS else text[: ERROR_SUMMARY_CHARS - 3] + "..."


def _final_exception(traceback: str) -> str:
    """The exception lines of a traceback: from its first unindented line (frames are indented)."""
    lines = traceback.splitlines()
    start = next((index for index, line in enumerate(lines) if line[:1].strip()), len(lines))
    return "\n".join(lines[start:])


def is_kafka_data_loss(error: object) -> bool:
    """True when a query failed because offsets it had not read yet are gone from Kafka."""
    text = str(error).split(_QUERY_DESCRIPTION, 1)[0]  # the plan dump is not part of the error
    return any(marker in text for marker in _DATA_LOSS_MARKERS)


@dataclass(slots=True)
class _Supervised:
    """Supervision state of one query across its runs."""

    spec: QuerySpec
    group: str
    query: StreamingQueryLike | None = None
    starts: int = 0
    consecutive_failures: int = 0
    next_attempt_at: float = 0.0
    last_error: str | None = None
    last_error_at: str | None = None
    last_progress_at: str | None = None
    last_batch_id: int | None = None
    progress_marker: tuple[object, ...] | None = None


class QuerySupervisor:
    """Starts, watches and restarts the queries of independent groups (see the module doc)."""

    def __init__(
        self,
        groups: Sequence[QueryGroup],
        stop_event: threading.Event,
        heartbeat: HeartbeatPublisher,
        *,
        driver_alive: Callable[[], bool] = lambda: True,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], str] = utc_now_iso,
        poll_s: float = POLL_SECONDS,
        restart_base_delay_s: float = RESTART_BASE_DELAY_SECONDS,
        restart_max_delay_s: float = RESTART_MAX_DELAY_SECONDS,
        drain_timeout_s: float = DRAIN_TIMEOUT_SECONDS,
    ) -> None:
        self._groups = tuple(groups)
        self._entries = [_Supervised(spec, group.name) for group in self._groups for spec in group.queries]
        self._stop_event = stop_event
        self._heartbeat = heartbeat
        self._driver_alive = driver_alive
        self._clock = clock
        self._wall_clock = wall_clock
        self._poll_s = poll_s
        self._base_delay_s = restart_base_delay_s
        self._max_delay_s = restart_max_delay_s
        self._drain_timeout_s = drain_timeout_s
        self._status = "running"

    def run(self) -> int:
        """Supervise until a stop request (``EXIT_OK``) or a lost driver (``EXIT_DRIVER_STOPPED``)."""
        while not self._stop_event.is_set():
            if not self._driver_alive():
                logger.error("the Spark driver has stopped: exiting so that the job is restarted")
                self._shutdown(drain=False)
                return EXIT_DRIVER_STOPPED
            self.step()
            self._stop_event.wait(self._poll_s)
        logger.info("shutdown requested: draining and stopping %d queries", self._running_count())
        self._shutdown(drain=True)
        return EXIT_OK

    def step(self) -> None:
        """One round: reap terminated queries, (re)start the due ones, publish the heartbeat."""
        changed = False
        for entry in self._entries:
            if entry.query is not None:
                changed |= self._observe(entry, entry.query)
        for group in self._groups:
            changed |= self._start_due(group)
        self._publish(force=changed)

    def snapshot(self) -> dict[str, object]:
        """Supervisor and per-query state, as written to the heartbeat file."""
        now = self._clock()
        queries = {entry.spec.name: self._describe(entry, now) for entry in self._entries}
        return {"status": self._status, "queries": queries}

    def _observe(self, entry: _Supervised, query: StreamingQueryLike) -> bool:
        """Record the progress of a started query; True when it has terminated."""
        active = query.isActive
        self._record_progress(entry, query.lastProgress)
        if active:
            return False
        entry.query = None
        error = query.exception()
        if self._stop_event.is_set():
            reason = "stopped" if error is None else summarize_error(error)
            logger.warning("query %s ended during shutdown: %s", entry.spec.name, reason)
        else:
            self._handle_failure(entry, error)
        return True

    def _record_progress(self, entry: _Supervised, progress: Mapping[str, Any] | None) -> None:
        if not progress:
            return
        marker = (progress.get("runId"), progress.get("batchId"), progress.get("timestamp"))
        if marker == entry.progress_marker:
            return
        entry.progress_marker = marker
        # When the trigger was seen completing: Spark's own timestamp is the trigger's start.
        entry.last_progress_at = self._wall_clock()
        entry.last_batch_id = progress.get("batchId")
        if entry.consecutive_failures:
            logger.info(
                "query %s is healthy again (batch %s) after %d failed attempt(s)",
                entry.spec.name,
                entry.last_batch_id,
                entry.consecutive_failures,
            )
            entry.consecutive_failures = 0

    def _handle_failure(self, entry: _Supervised, error: object | None) -> None:
        text = "stopped without an error" if error is None else str(error)
        delay = self._record_failure(entry, summarize_error(text))
        logger.error(
            "query %s failed (attempt %d), restarting it from its checkpoint in %.0fs: %s",
            entry.spec.name,
            entry.consecutive_failures,
            delay,
            entry.last_error,
        )
        if entry.consecutive_failures == 1 and text != entry.last_error:
            # Full traceback and query description, once per failure streak.
            logger.error("full error of query %s:\n%s", entry.spec.name, text)
        if is_kafka_data_loss(text):
            logger.error(
                "query %s stopped at a gap in the Kafka CDC log: records it had not read yet were "
                "deleted by the topic retention, so it fails until an operator acts. %s",
                entry.spec.name,
                entry.spec.data_loss_recovery,
            )

    def _start_due(self, group: QueryGroup) -> bool:
        """Check the group's dependencies once and start its due queries; True on any change."""
        now = self._clock()
        due = [
            entry
            for entry in self._entries
            if entry.group == group.name and entry.query is None and entry.next_attempt_at <= now
        ]
        if not due or self._stop_event.is_set():
            return False
        try:
            group.check_ready()
        except Exception as exc:  # any probe, connection or schema-check failure means "not ready yet"
            reason = f"dependencies not ready: {summarize_error(exc)}"
            delay = min(self._record_failure(entry, reason) for entry in due)
            logger.warning(
                "%s: %d queries waiting, %s; next check in %.0fs", group.name, len(due), reason, delay
            )
            return True
        logger.info(
            "%s: dependencies ready, starting %s", group.name, ", ".join(entry.spec.name for entry in due)
        )
        for entry in due:
            if self._stop_event.is_set():
                break
            self._start(entry)
            self._publish()  # starting a lake query talks to S3: keep the heartbeat fresh meanwhile
        return True

    def _start(self, entry: _Supervised) -> None:
        try:
            entry.query = entry.spec.start()
        except Exception as exc:  # e.g. a sink path or checkpoint that cannot be opened right now
            delay = self._record_failure(entry, f"start failed: {summarize_error(exc)}")
            logger.error(
                "query %s could not be started (attempt %d), retrying in %.0fs: %s",
                entry.spec.name,
                entry.consecutive_failures,
                delay,
                entry.last_error,
            )
            return
        entry.starts += 1
        if entry.starts > 1:
            logger.warning(
                "query %s restarted from its checkpoint (restart %d)", entry.spec.name, entry.starts - 1
            )

    def _record_failure(self, entry: _Supervised, reason: str) -> float:
        """Count a failed attempt and schedule the next one; returns the backoff delay."""
        entry.consecutive_failures += 1
        entry.last_error = reason
        entry.last_error_at = self._wall_clock()
        delay = capped_backoff(entry.consecutive_failures, self._base_delay_s, self._max_delay_s)
        entry.next_attempt_at = self._clock() + delay
        return delay

    def _shutdown(self, *, drain: bool) -> None:
        """Stop every started query (unless the driver is gone) and publish the final state."""
        self._status = "stopping"
        self._publish(force=True)
        started = [entry.query for entry in self._entries if entry.query is not None]
        if drain:
            stop_queries(started, self._drain_timeout_s)
        for entry in self._entries:
            entry.query = None
        self._status = "stopped"
        self._publish(force=True)

    def _describe(self, entry: _Supervised, now: float) -> dict[str, object]:
        running = entry.query is not None
        waiting = not running and self._status == "running"
        return {
            "group": entry.group,
            "state": self._state(entry),
            "active": running,
            "last_progress_at": entry.last_progress_at,
            "last_batch_id": entry.last_batch_id,
            "consecutive_failures": entry.consecutive_failures,
            "restarts": max(entry.starts - 1, 0),
            "last_error": entry.last_error,
            "last_error_at": entry.last_error_at,
            "next_attempt_in_s": round(max(entry.next_attempt_at - now, 0.0), 1) if waiting else None,
        }

    def _state(self, entry: _Supervised) -> str:
        if entry.query is not None:
            return "running"
        if self._status != "running":
            return "stopped"
        return "restarting" if entry.starts else "starting"

    def _publish(self, *, force: bool = False) -> None:
        self._heartbeat.publish(self.snapshot(), force=force)

    def _running_count(self) -> int:
        return sum(entry.query is not None for entry in self._entries)


def stop_queries(
    queries: Sequence[StreamingQueryLike], drain_timeout_s: float = DRAIN_TIMEOUT_SECONDS
) -> None:
    """Stop every query, first letting in-flight micro-batches finish within a shared deadline.

    An interrupted lake batch is simply re-run after a restart, but the part files it already
    wrote stay behind as orphans that only readers honouring ``_spark_metadata`` ignore
    (see ``streaming_queries``).
    """
    deadline = time.monotonic() + drain_timeout_s
    for query in queries:
        while query.isActive and query.status["isTriggerActive"] and time.monotonic() < deadline:
            time.sleep(_DRAIN_POLL_SECONDS)
        try:
            query.stop()
        except Exception:  # one failing stop must not leave the remaining queries running
            logger.exception("failed to stop query %s", query.name)
