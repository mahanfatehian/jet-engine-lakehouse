"""Unit tests of query supervision and graceful shutdown, using fake query objects."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("pyspark")

import streaming_queries
from pyspark.errors import StreamingQueryException
from streaming_queries import EXIT_OK, EXIT_QUERY_FAILED, stop_queries, supervise


class FakeQuery:
    def __init__(
        self, name: str, *, active: bool = True, error: object = None, triggers: list[bool] | None = None
    ) -> None:
        self.name = name
        self.isActive = active
        self.stop_calls = 0
        self._error = error
        self._triggers = list(triggers or [])
        self.fail_stop = False

    @property
    def status(self) -> dict[str, Any]:
        return {"isTriggerActive": self._triggers.pop(0) if self._triggers else False}

    def exception(self) -> object:
        return self._error

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("py4j connection lost")
        self.isActive = False


class FakeStreams:
    def __init__(self, await_any: Callable[[float], bool]) -> None:
        self.awaitAnyTermination = await_any


class FakeSpark:
    def __init__(self, await_any: Callable[[float], bool]) -> None:
        self.streams = FakeStreams(await_any)


def query_exception() -> StreamingQueryException:
    # The real constructor needs a running JVM; supervise() only needs the type.
    return StreamingQueryException.__new__(StreamingQueryException)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(streaming_queries.time, "sleep", sleeps.append)
    return sleeps


def test_requested_shutdown_stops_every_query() -> None:
    stop_event = threading.Event()
    queries = [FakeQuery("a"), FakeQuery("b")]

    def await_any(_timeout: float) -> bool:
        stop_event.set()  # SIGTERM arrives while the supervisor waits
        return False

    assert supervise(FakeSpark(await_any), queries, stop_event) == EXIT_OK
    assert [query.stop_calls for query in queries] == [1, 1]


def test_failed_query_stops_the_others_and_exits_non_zero(caplog: pytest.LogCaptureFixture) -> None:
    failed = FakeQuery("clickhouse_raw", active=False, error="insert into raw_engine_telemetry failed")
    healthy = FakeQuery("silver_engine_telemetry")

    def await_any(_timeout: float) -> bool:
        raise query_exception()

    with caplog.at_level(logging.ERROR):
        code = supervise(FakeSpark(await_any), [failed, healthy], threading.Event())
    assert code == EXIT_QUERY_FAILED
    assert healthy.stop_calls == 1
    assert "query clickhouse_raw failed: insert into raw_engine_telemetry failed" in caplog.text


def test_query_stopped_without_error_is_still_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    stopped = FakeQuery("bronze_maintenance_log", active=False)
    code = supervise(FakeSpark(lambda _timeout: True), [stopped], threading.Event())
    assert code == EXIT_QUERY_FAILED
    assert "query bronze_maintenance_log stopped unexpectedly" in caplog.text


def test_failure_during_shutdown_exits_cleanly() -> None:
    stop_event = threading.Event()
    failed = FakeQuery("clickhouse_raw", active=False, error="shutdown requested while retrying")

    def await_any(_timeout: float) -> bool:
        stop_event.set()
        raise query_exception()

    assert supervise(FakeSpark(await_any), [failed], stop_event) == EXIT_OK


def test_stop_waits_for_in_flight_micro_batches(no_sleep: list[float]) -> None:
    query = FakeQuery("silver_engine_telemetry", triggers=[True, True, False])
    stop_queries([query], drain_timeout_s=30.0)
    assert query.stop_calls == 1
    assert len(no_sleep) == 2


def test_stop_does_not_wait_past_the_deadline(no_sleep: list[float]) -> None:
    query = FakeQuery("bronze_engine_telemetry", triggers=[True] * 100)
    stop_queries([query], drain_timeout_s=0.0)
    assert query.stop_calls == 1
    assert no_sleep == []


def test_one_failing_stop_does_not_leave_others_running() -> None:
    broken, healthy = FakeQuery("a"), FakeQuery("b")
    broken.fail_stop = True
    stop_queries([broken, healthy])
    assert healthy.stop_calls == 1
    assert healthy.isActive is False
