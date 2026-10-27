"""HF frame publisher: a background thread that builds, serializes and produces the frames the fleet commits.

The 1 Hz tick loop only hands lightweight frame requests to the bounded FrameQueue (hf_queue) and never waits
for Kafka, the Schema Registry or the frame synthesis:

    tick thread  FrameTap -> FramePublisher.submit(): a lock and a few list operations, drops when full
    worker       waits until the topic exists and the value schema is registered (backoff 1, 2, 4 .. 30 s;
                 topic auto-creation is disabled), then: take -> FrameBuilder.build -> Avro serializer ->
                 produce -> poll (delivery reports); one metrics line every 60 s
    shutdown     close(): queued live frames are produced and flushed for up to SHUTDOWN_SECONDS; queued
                 backfill frames and whatever Kafka has not acknowledged by then are counted as discarded

Losses are counted, never hidden: ``dropped`` (budget full: Kafka, the registry or the topic unavailable),
``failed`` (build, serialization or delivery error), ``discarded`` (queued or unacknowledged at shutdown).
The frame_seq of a lost frame is never reused, so consumers see each loss as a gap. A fatal producer error
(the idempotent producer cannot continue) replaces the client through the sink factory, so a broken client
never silences the path for good.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol

from hf_queue import FrameQueue

if TYPE_CHECKING:
    from hf_config import HfConfig
    from hf_frames import FrameBuilder, FrameRequest
    from hf_kafka import DeliveryCallback

logger = logging.getLogger("hf_producer")

SHUTDOWN_SECONDS = 5.0
JOIN_MARGIN_SECONDS = 2.0
REPORT_SECONDS = 60.0
READY_TIMEOUT_SECONDS = 5.0
IDLE_POLL_SECONDS = 0.05
BUFFER_RETRY_SECONDS = 0.05
BUFFER_RETRIES = 200
BACKOFF_BASE_SECONDS, BACKOFF_CAP_SECONDS = 1.0, 30.0
FAILURE_LOG_INTERVAL_SECONDS = 60.0


def backoff_delay(attempt: int) -> float:
    """Wait before readiness check ``attempt + 1``: 1, 2, 4, ... seconds, capped at 30 s."""
    return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))


def first_line(exc: BaseException) -> str:
    """First line of an exception message (its type name when the message is empty)."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


class FrameSink(Protocol):
    """The Kafka side the publisher drives (hf_kafka.KafkaSink; fakes in the tests)."""

    topic: str
    schema_id: int | None

    @property
    def fatal_error(self) -> str | None:
        """Set once the client can no longer produce."""

    def check_ready(self, timeout: float) -> str | None:
        """None when frames can be produced, else the reason (an exception counts as one too)."""

    def serialize(self, record: Mapping[str, Any]) -> bytes:
        """Wire-format value of one frame record."""

    def produce(self, key: bytes, value: bytes, on_delivery: DeliveryCallback) -> None:
        """Hand one message to the client (BufferError when its queue is full)."""

    def poll(self, timeout: float) -> int:
        """Serve delivery reports for up to ``timeout`` seconds."""

    def flush(self, timeout: float) -> int:
        """Wait for outstanding messages; returns how many are still pending."""

    def close(self) -> None:
        """Release the client."""


@dataclass
class Counters:
    """Frame counters of the publisher (totals, or the difference of two snapshots)."""

    produced: int = 0
    produced_bytes: int = 0
    serialized: int = 0
    serialized_bytes: int = 0
    dropped: int = 0
    failed: int = 0
    discarded: int = 0

    def minus(self, earlier: Counters) -> Counters:
        """Counts since the ``earlier`` snapshot."""
        return Counters(**{f.name: getattr(self, f.name) - getattr(earlier, f.name) for f in fields(self)})


class PublisherStats:
    """Thread-safe counters: drops are counted on the tick thread, everything else on the worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters = Counters()

    def add(self, **increments: int) -> None:
        """Add to the named counters."""
        with self._lock:
            for name, value in increments.items():
                setattr(self._counters, name, getattr(self._counters, name) + value)

    def snapshot(self) -> Counters:
        """A consistent copy of the counters."""
        with self._lock:
            return Counters(**vars(self._counters))


class FramePublisher:
    """Builds, serializes and produces HF frames on one background thread (see the module doc)."""

    def __init__(
        self,
        config: HfConfig,
        builder: FrameBuilder,
        sink_factory: Callable[[], FrameSink],
        *,
        report_seconds: float = REPORT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stats = PublisherStats()
        self.waiting_for: str | None = "the first readiness check"
        self._config = config
        self._builder = builder
        self._sink_factory = sink_factory
        self._sink: FrameSink | None = sink_factory()
        self._queue = FrameQueue(config.queue_max_bytes, builder.frame_bytes)
        self._clock = clock
        self._report_seconds = report_seconds
        self._report_due = clock() + report_seconds
        self._reported = Counters()
        self._closing = threading.Event()
        self._close_deadline = 0.0
        self._dropping = False
        self._failure_logged_at = -FAILURE_LOG_INTERVAL_SECONDS
        self._thread = threading.Thread(target=self._run, name="hf-publisher", daemon=True)

    @property
    def queue(self) -> FrameQueue:
        """The bounded hand-off (read-only use: depth and budget)."""
        return self._queue

    def start(self) -> None:
        """Start the worker thread."""
        self._thread.start()

    def submit(self, requests: Sequence[FrameRequest]) -> None:
        """Queue the frames of a live tick; frames beyond the budget are dropped and counted (tick thread)."""
        dropped = self._queue.offer(requests)
        if dropped:
            self.stats.add(dropped=dropped)
            if not self._dropping:
                self._dropping = True
                logger.warning(
                    "HF buffer full (%d MiB, waiting for: %s): dropping new frames until Kafka catches up",
                    self._config.queue_max_mb,
                    self.waiting_for or "acknowledgements",
                )
        elif self._dropping and requests:
            self._dropping = False
            logger.info("HF buffer has room again: frames are queued again")

    def submit_backfill(self, requests: Sequence[FrameRequest]) -> None:
        """Queue the frames of a backfilled history (never dropped for a full budget)."""
        self._queue.extend_backfill(requests)

    def close(self, timeout: float = SHUTDOWN_SECONDS) -> None:
        """Stop the worker: flush queued live frames for up to ``timeout`` seconds, then log the totals."""
        self._close_deadline = self._clock() + timeout
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join(timeout + READY_TIMEOUT_SECONDS + JOIN_MARGIN_SECONDS)
            if self._thread.is_alive():
                logger.warning("HF publisher did not stop in time; exiting without it")
        totals = self.stats.snapshot()
        logger.info(
            "HF publisher stopped: %d frames produced (%.1f MB), %d dropped, %d failed, %d discarded",
            totals.produced,
            totals.produced_bytes / 1e6,
            totals.dropped,
            totals.failed,
            totals.discarded,
        )

    # ------------------------------------------------------------------ worker thread
    def _run(self) -> None:
        try:
            while not self._closing.is_set():
                if self._sink is None or self.waiting_for is not None:
                    self.wait_until_ready()
                    continue  # a close may have arrived meanwhile: re-check before producing
                self.pump()
                self.report_if_due()
                if self._sink.fatal_error:
                    self._replace_sink(self._sink.fatal_error)
        except Exception:
            logger.exception(
                "HF publisher stopped by an unexpected error; new frames are dropped from now on"
            )
        finally:
            self._shutdown()

    def wait_until_ready(self) -> bool:
        """Retry (with backoff) until frames can be produced; False when closing first."""
        attempt = 0
        while not self._closing.is_set():
            reason = self._readiness()
            if reason is None:
                logger.info(
                    "HF frames -> %s (schema id %s) on %s",
                    self._sink.topic if self._sink else "?",
                    self._sink.schema_id if self._sink else "?",
                    self._config.bootstrap_servers,
                )
                self.waiting_for = None
                return True
            attempt += 1
            self.waiting_for = reason
            delay = backoff_delay(attempt)
            if attempt == 1 or delay == BACKOFF_CAP_SECONDS:
                logger.warning(
                    "HF frames waiting for Kafka (attempt %d, next in %.0f s; frames queue up to %d MiB, "
                    "then drop): %s",
                    attempt,
                    delay,
                    self._config.queue_max_mb,
                    reason,
                )
            self._closing.wait(delay)
            self.report_if_due()
        return False

    def _readiness(self) -> str | None:
        try:
            if self._sink is None:
                self._sink = self._sink_factory()
            self._sink.poll(0)
            return self._sink.check_ready(READY_TIMEOUT_SECONDS)
        except Exception as exc:
            return first_line(exc)

    def pump(self) -> bool:
        """Produce the next admitted frame (True), or serve delivery reports while none is (False)."""
        assert self._sink is not None
        item = self._queue.take()
        if item is None:
            self._sink.poll(IDLE_POLL_SECONDS)
            return False
        self._send(self._sink, item.request)
        self._sink.poll(0)
        return True

    def _send(self, sink: FrameSink, request: FrameRequest) -> None:
        try:
            value = sink.serialize(self._builder.build(request))
        except Exception as exc:
            self._failed(f"frame of {request.engine_id} could not be built or serialized: {first_line(exc)}")
            return
        size = len(value)
        self.stats.add(serialized=1, serialized_bytes=size)
        self._queue.in_flight(size)
        callback = functools.partial(self._delivered, size)
        for _ in range(BUFFER_RETRIES):
            try:
                sink.produce(request.engine_id.encode(), value, callback)
                return
            except BufferError:
                sink.poll(BUFFER_RETRY_SECONDS)  # the client's local queue is full: let it drain
            except Exception as exc:
                self._queue.settled(size)
                self._failed(f"produce of a frame of {request.engine_id} failed: {first_line(exc)}")
                return
        self._queue.settled(size)
        self._failed(f"the Kafka client queue stayed full; a frame of {request.engine_id} was lost")

    def _delivered(self, size: int, error: Any, _message: Any) -> None:
        self._queue.settled(size)
        if error is None:
            self.stats.add(produced=1, produced_bytes=size)
        else:
            self._failed(f"frame delivery failed: {error}")

    def _failed(self, message: str) -> None:
        self.stats.add(failed=1)
        now = self._clock()
        if now - self._failure_logged_at >= FAILURE_LOG_INTERVAL_SECONDS:
            self._failure_logged_at = now
            logger.warning("HF %s (further failures are counted in the metrics line)", message)

    def _replace_sink(self, reason: str) -> None:
        assert self._sink is not None
        logger.error("Replacing the Kafka producer after a fatal error: %s", reason)
        self._discard_client(self._sink)
        self._sink = None
        self.waiting_for = f"a new producer after: {reason}"

    def _discard_client(self, sink: FrameSink) -> None:
        unsent = sink.flush(0)
        self.stats.add(discarded=unsent)
        self._queue.reset_in_flight()
        sink.close()

    def report_if_due(self) -> None:
        """Log the metrics line when REPORT_SECONDS have passed since the previous one."""
        now = self._clock()
        if now < self._report_due:
            return
        elapsed = now - (self._report_due - self._report_seconds)
        self._report_due = now + self._report_seconds
        totals = self.stats.snapshot()
        interval, self._reported = totals.minus(self._reported), totals
        logger.info(metrics_line(interval, totals, elapsed, self._queue, self.waiting_for))

    def _shutdown(self) -> None:
        backfill = self._queue.discard_backfill()
        if backfill:
            logger.warning(
                "Shutdown: %d queued backfill frames discarded; those pulses stay without HF features",
                backfill,
            )
        self.stats.add(discarded=backfill)
        sink = self._sink
        if sink is None:
            live, _ = self._queue.clear()
            self.stats.add(discarded=live)
            return
        if self.waiting_for is None:
            self._drain(sink)
        live, _ = self._queue.clear()
        unsent = sink.flush(max(0.0, self._close_deadline - self._clock()))
        if live or unsent:
            logger.warning("Shutdown: %d queued and %d unacknowledged frames discarded", live, unsent)
        self.stats.add(discarded=live + unsent)
        sink.close()

    def _drain(self, sink: FrameSink) -> None:
        """Produce the queued live frames until the queue is empty or the close deadline passes."""
        while self._clock() < self._close_deadline and self._queue.depth() != (0, 0):
            item = self._queue.take()
            if item is None:
                sink.poll(IDLE_POLL_SECONDS)
            else:
                self._send(sink, item.request)
                sink.poll(0)


def metrics_line(
    interval: Counters, totals: Counters, seconds: float, queue: FrameQueue, waiting_for: str | None
) -> str:
    """The periodic metrics line: frame counts and volume of the interval plus totals and queue state."""
    rate = interval.serialized_bytes / seconds / 1e6 if seconds > 0 else 0.0
    average = interval.serialized_bytes / interval.serialized / 1e3 if interval.serialized else 0.0
    live, backfill = queue.depth()
    line = (
        f"HF frames, last {seconds:.0f} s: {interval.produced} produced, {interval.dropped} dropped, "
        f"{interval.failed} failed; {rate:.3f} MB/s serialized, {average:.1f} KB per frame; "
        f"queue {live} live + {backfill} backfill, {queue.in_flight_bytes / 1e6:.1f} MB in flight; "
        f"totals: {totals.produced} produced, {totals.dropped} dropped, {totals.failed} failed"
    )
    return line if waiting_for is None else f"{line}; waiting for: {waiting_for}"
