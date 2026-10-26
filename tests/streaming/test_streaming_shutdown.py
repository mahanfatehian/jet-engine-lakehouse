"""Unit tests of the graceful stop of the queries (drain in-flight micro-batches, then stop)."""

from __future__ import annotations

from typing import Any

import pytest
import streaming_supervisor
from streaming_supervisor import stop_queries


class DrainingQuery:
    """A query whose trigger stays active for the given number of status polls."""

    def __init__(self, name: str, triggers: list[bool] | None = None) -> None:
        self.name = name
        self.isActive = True
        self.stop_calls = 0
        self.fail_stop = False
        self._triggers = list(triggers or [])

    @property
    def status(self) -> dict[str, Any]:
        return {"isTriggerActive": self._triggers.pop(0) if self._triggers else False}

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("py4j connection lost")
        self.isActive = False


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(streaming_supervisor.time, "sleep", sleeps.append)
    return sleeps


def test_stop_waits_for_in_flight_micro_batches(no_sleep: list[float]) -> None:
    query = DrainingQuery("silver_engine_telemetry", triggers=[True, True, False])
    stop_queries([query], drain_timeout_s=30.0)
    assert query.stop_calls == 1
    assert len(no_sleep) == 2


def test_stop_does_not_wait_past_the_deadline(no_sleep: list[float]) -> None:
    query = DrainingQuery("bronze_engine_telemetry", triggers=[True] * 100)
    stop_queries([query], drain_timeout_s=0.0)
    assert query.stop_calls == 1
    assert no_sleep == []


def test_one_failing_stop_does_not_leave_others_running() -> None:
    broken, healthy = DrainingQuery("a"), DrainingQuery("b")
    broken.fail_stop = True
    stop_queries([broken, healthy])
    assert healthy.stop_calls == 1
    assert healthy.isActive is False
