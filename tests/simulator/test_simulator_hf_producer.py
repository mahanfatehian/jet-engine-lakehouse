"""HF frame publisher against a fake Kafka sink: order and keys, backpressure and counted drops while Kafka
is down, readiness with backoff, failure accounting, shutdown flush, metrics line, client replacement, and a
tick thread that never waits for any of it."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import hf_producer
import pytest
from hf_config import HfConfig
from hf_producer import Counters, FramePublisher, backoff_delay, metrics_line
from hf_queue import FrameQueue

FRAME_BYTES = 100_000
CONFIG = HfConfig(queue_max_mb=1)  # a 1 MiB budget holds ten 100 kB frames
TOPIC = "engine_hf.sensor_frames"


@dataclass(frozen=True)
class Request:
    """The part of hf_frames.FrameRequest the publisher reads."""

    engine_id: str
    frame_seq: int


class FakeBuilder:
    """FrameBuilder stand-in: tiny records, an optional failure on chosen frames."""

    frame_bytes = FRAME_BYTES

    def __init__(self, fail_on: frozenset[int] = frozenset()) -> None:
        self.fail_on = fail_on

    def build(self, request: Request) -> dict[str, Any]:
        if request.frame_seq in self.fail_on:
            raise ValueError("NaN in a waveform")
        return {"engine_id": request.engine_id, "frame_seq": request.frame_seq}


class FakeSink:
    """Kafka stand-in: acknowledges on poll while ``up``, holds every delivery while down."""

    topic = TOPIC

    def __init__(self, *, not_ready: int = 0, up: bool = True) -> None:
        self.schema_id: int | None = None
        self.fatal: str | None = None
        self.not_ready = not_ready
        self.up = up
        self.checks = 0
        self.pending: list[tuple[bytes, bytes, Any]] = []
        self.delivered: list[tuple[bytes, bytes]] = []
        self.fail_deliveries = False
        self.buffer_errors = 0
        self.closed = False
        self.check_gate: threading.Event | None = None

    @property
    def fatal_error(self) -> str | None:
        return self.fatal

    def check_ready(self, timeout: float) -> str | None:
        if self.check_gate is not None:
            self.check_gate.wait(5)
        self.checks += 1
        if self.checks <= self.not_ready:
            if self.checks % 2:
                raise ConnectionError("schema-registry:8081: connection refused\n(details)")
            return f"topic {TOPIC} does not exist yet (Broker: Unknown topic or partition)"
        self.schema_id = 7
        return None

    def serialize(self, record: dict[str, Any]) -> bytes:
        return f"{record['engine_id']}#{record['frame_seq']}".encode().ljust(FRAME_BYTES, b".")

    def produce(self, key: bytes, value: bytes, on_delivery: Callable[[Any, Any], None]) -> None:
        if self.buffer_errors:
            self.buffer_errors -= 1
            raise BufferError("Local: Queue full")
        self.pending.append((key, value, on_delivery))

    def poll(self, timeout: float) -> int:
        if not self.up:
            return 0
        served, self.pending = self.pending, []
        for key, value, callback in served:
            if self.fail_deliveries:
                callback("KafkaError{code=_MSG_TIMED_OUT}", None)
            else:
                self.delivered.append((key, value))
                callback(None, None)
        return len(served)

    def flush(self, timeout: float) -> int:
        self.poll(0)
        return len(self.pending)

    def close(self) -> None:
        self.closed = True

    def labels(self) -> list[str]:
        return [value.split(b".")[0].decode() for _, value in self.delivered]


def publisher_with(sink: FakeSink, builder: FakeBuilder | None = None, **kwargs: Any) -> FramePublisher:
    return FramePublisher(CONFIG, builder or FakeBuilder(), lambda: sink, **kwargs)  # type: ignore[arg-type]


def frames(engine_id: str, first: int, count: int) -> list[Request]:
    return [Request(engine_id, seq) for seq in range(first, first + count)]


def pump_all(publisher: FramePublisher) -> None:
    while publisher.pump():
        pass


def wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


def test_frames_are_produced_in_commit_order_keyed_by_engine_id() -> None:
    sink = FakeSink()
    publisher = publisher_with(sink)
    publisher.submit(frames("CFM56-104", 1, 3) + frames("GE90-201", 1, 2))
    assert publisher.wait_until_ready() and sink.schema_id == 7
    pump_all(publisher)
    assert [key for key, _ in sink.delivered] == [b"CFM56-104"] * 3 + [b"GE90-201"] * 2
    assert sink.labels() == ["CFM56-104#1", "CFM56-104#2", "CFM56-104#3", "GE90-201#1", "GE90-201#2"]
    totals = publisher.stats.snapshot()
    assert (totals.produced, totals.produced_bytes, totals.dropped, totals.failed) == (
        5,
        5 * FRAME_BYTES,
        0,
        0,
    )
    assert publisher.queue.used_bytes == 0


def test_kafka_down_fills_the_budget_then_drops_are_counted_until_it_recovers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = FakeSink(up=False)
    publisher = publisher_with(sink)
    assert publisher.wait_until_ready()
    publisher.submit(frames("JT9D-001", 1, 10))
    pump_all(publisher)
    assert len(sink.pending) == 10 and publisher.queue.in_flight_bytes == 10 * FRAME_BYTES
    with caplog.at_level(logging.INFO, logger="hf_producer"):
        publisher.submit(frames("JT9D-001", 11, 3))
        publisher.submit(frames("JT9D-001", 14, 2))
        assert publisher.stats.snapshot().dropped == 5, "the 1 MiB budget is held by unacknowledged frames"
        sink.up = True
        publisher.pump()
        assert publisher.stats.snapshot().produced == 10 and publisher.queue.in_flight_bytes == 0
        publisher.submit(frames("JT9D-001", 16, 2))
    pump_all(publisher)
    assert sink.labels()[-2:] == ["JT9D-001#16", "JT9D-001#17"], "production resumes after the outage"
    assert caplog.text.count("HF buffer full (1 MiB") == 1, "one warning per episode"
    assert "HF buffer has room again" in caplog.text
    assert publisher.stats.snapshot().dropped == 5


def test_the_tick_thread_never_waits_for_kafka_or_the_worker() -> None:
    sink = FakeSink()
    sink.check_gate = threading.Event()  # the worker hangs in its readiness check, like a slow broker
    publisher = publisher_with(sink)
    publisher.start()
    started = time.perf_counter()
    for seq in range(1, 2001):
        publisher.submit([Request("CFM56-104", seq)])
    elapsed = time.perf_counter() - started
    sink.check_gate.set()
    publisher.close(timeout=2.0)
    assert elapsed < 0.5, f"2000 submits took {elapsed:.3f} s"
    totals = publisher.stats.snapshot()
    assert (totals.produced, totals.dropped, totals.discarded) == (10, 1990, 0)
    assert sink.labels() == [f"CFM56-104#{seq}" for seq in range(1, 11)]


def test_readiness_is_retried_with_backoff_until_topic_and_schema_are_there(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[int] = []
    monkeypatch.setattr(hf_producer, "backoff_delay", lambda attempt: attempts.append(attempt) or 0.0)
    sink = FakeSink(not_ready=3)
    publisher = publisher_with(sink)
    publisher.submit(frames("GE90-201", 1, 12))
    with caplog.at_level(logging.INFO, logger="hf_producer"):
        assert publisher.wait_until_ready()
    assert sink.checks == 4 and attempts == [1, 2, 3] and publisher.waiting_for is None
    assert "attempt 1" in caplog.text and "schema-registry:8081: connection refused" in caplog.text
    assert "HF frames -> engine_hf.sensor_frames (schema id 7)" in caplog.text
    assert publisher.queue.depth() == (10, 0) and publisher.stats.snapshot().dropped == 2
    assert [backoff_delay(attempt) for attempt in range(1, 8)] == [1, 2, 4, 8, 16, 30, 30]


def test_build_produce_and_delivery_failures_are_counted_never_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(hf_producer, "BUFFER_RETRIES", 5)
    monkeypatch.setattr(hf_producer, "BUFFER_RETRY_SECONDS", 0.0)
    sink = FakeSink()
    publisher = publisher_with(sink, FakeBuilder(fail_on=frozenset({2})))
    assert publisher.wait_until_ready()
    with caplog.at_level(logging.WARNING, logger="hf_producer"):
        publisher.submit(frames("JT9D-002", 1, 3))
        pump_all(publisher)
    sink.buffer_errors = 3  # the client's queue is full for a moment: polled and retried
    publisher.submit(frames("JT9D-002", 4, 1))
    pump_all(publisher)
    sink.buffer_errors = 99  # ... and for good: the frame is lost
    publisher.submit(frames("JT9D-002", 5, 1))
    pump_all(publisher)
    sink.buffer_errors, sink.fail_deliveries = 0, True
    publisher.submit(frames("JT9D-002", 6, 1))
    pump_all(publisher)
    totals = publisher.stats.snapshot()
    assert (totals.produced, totals.failed, totals.serialized) == (3, 3, 5)
    assert sink.labels() == ["JT9D-002#1", "JT9D-002#3", "JT9D-002#4"]
    assert publisher.queue.in_flight_bytes == 0, "failed frames release their budget"
    assert "frame of JT9D-002 could not be built or serialized: NaN in a waveform" in caplog.text


def test_close_produces_queued_live_frames_and_discards_unsent_backfill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = FakeSink()
    sink.check_gate = threading.Event()
    publisher = publisher_with(sink)
    publisher.start()
    publisher.submit_backfill(frames("GE90-202", 1, 50))
    publisher.submit(frames("CFM56-105", 1, 5))
    with caplog.at_level(logging.INFO, logger="hf_producer"):
        closing = threading.Thread(target=publisher.close, kwargs={"timeout": 2.0})
        closing.start()
        time.sleep(0.05)
        sink.check_gate.set()
        closing.join(10)
    totals = publisher.stats.snapshot()
    assert sink.labels() == [f"CFM56-105#{seq}" for seq in range(1, 6)] and sink.closed
    assert (totals.produced, totals.discarded) == (5, 50)
    assert "Shutdown: 50 queued backfill frames discarded" in caplog.text
    assert (
        "HF publisher stopped: 5 frames produced (0.5 MB), 0 dropped, 0 failed, 50 discarded" in caplog.text
    )


def test_close_while_kafka_stays_unreachable_returns_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_producer, "backoff_delay", lambda attempt: 0.01)
    sink = FakeSink(not_ready=10**9)
    publisher = publisher_with(sink)
    publisher.start()
    publisher.submit(frames("JT9D-001", 1, 4))
    started = time.perf_counter()
    publisher.close(timeout=0.3)
    assert time.perf_counter() - started < 2.0
    totals = publisher.stats.snapshot()
    assert (totals.produced, totals.discarded) == (0, 4) and sink.closed and sink.pending == []


def test_a_metrics_line_reports_every_interval_with_totals_and_queue_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [1000.0]
    sink = FakeSink()
    publisher = publisher_with(sink, report_seconds=60.0, clock=lambda: now[0])
    assert publisher.wait_until_ready()
    publisher.submit(frames("CFM56-104", 1, 3))
    pump_all(publisher)
    publisher.stats.add(dropped=2)
    with caplog.at_level(logging.INFO, logger="hf_producer"):
        now[0] += 59.0
        publisher.report_if_due()
        assert "HF frames, last" not in caplog.text
        now[0] += 1.0
        publisher.report_if_due()
        publisher.submit(frames("CFM56-104", 4, 1))
        pump_all(publisher)
        now[0] += 60.0
        publisher.report_if_due()
    assert (
        "HF frames, last 60 s: 3 produced, 2 dropped, 0 failed; 0.005 MB/s serialized, 100.0 KB per frame; "
        "queue 0 live + 0 backfill, 0.0 MB in flight; totals: 3 produced, 2 dropped, 0 failed"
    ) in caplog.text
    assert "HF frames, last 60 s: 1 produced, 0 dropped, 0 failed" in caplog.text
    assert "totals: 4 produced, 2 dropped, 0 failed" in caplog.text


def test_metrics_line_names_what_the_path_is_waiting_for() -> None:
    queue = FrameQueue(budget_bytes=10, frame_bytes=1)
    queue.offer([1, 2])
    queue.extend_backfill([3])
    line = metrics_line(Counters(), Counters(dropped=4), 60.0, queue, "topic engine_hf.sensor_frames missing")
    assert "0 produced, 0 dropped" in line and "queue 2 live + 1 backfill" in line
    assert line.endswith(
        "totals: 0 produced, 4 dropped, 0 failed; waiting for: topic engine_hf.sensor_frames missing"
    )
    assert Counters(produced=5, failed=2).minus(Counters(produced=3)) == Counters(produced=2, failed=2)


def test_a_fatal_producer_error_replaces_the_client(caplog: pytest.LogCaptureFixture) -> None:
    first, second = FakeSink(up=False), FakeSink()
    created: list[FakeSink] = []

    def factory() -> FakeSink:
        created.append((first, second)[len(created)])
        return created[-1]

    publisher = FramePublisher(CONFIG, FakeBuilder(), factory)  # type: ignore[arg-type]
    first.fatal = "Broker: Producer has been fenced"
    publisher.submit(frames("GE90-201", 1, 1))
    with caplog.at_level(logging.INFO, logger="hf_producer"):
        publisher.start()
        wait_for(lambda: len(created) == 2 and second.schema_id == 7)
        publisher.submit(frames("GE90-201", 2, 1))
        wait_for(lambda: len(second.delivered) == 1)
        publisher.close(timeout=1.0)
    totals = publisher.stats.snapshot()
    assert first.closed and second.labels() == ["GE90-201#2"]
    assert (totals.produced, totals.discarded) == (1, 1), "the frame stuck in the broken client is counted"
    assert "Replacing the Kafka producer after a fatal error: Broker: Producer has been fenced" in caplog.text


def test_an_unexpected_worker_error_never_reaches_the_tick_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sink = FakeSink()
    publisher = publisher_with(sink)

    def broken() -> None:
        raise RuntimeError("bug in the metrics code")

    monkeypatch.setattr(publisher, "report_if_due", broken)
    with caplog.at_level(logging.ERROR, logger="hf_producer"):
        publisher.start()
        wait_for(lambda: sink.closed)
    publisher.submit(frames("JT9D-001", 1, 12))
    publisher.close(timeout=0.1)
    assert "HF publisher stopped by an unexpected error" in caplog.text
    assert publisher.stats.snapshot().dropped == 2, "frames queue up to the budget, then drop"
