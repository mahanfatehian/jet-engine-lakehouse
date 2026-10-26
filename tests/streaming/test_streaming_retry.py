"""Unit tests of the bounded exponential-backoff retry helper."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from streaming_retry import RetryAbortedError, RetryPolicy, call_with_retry


class TransientError(Exception):
    pass


class FakeWait:
    """Records requested delays instead of sleeping; can simulate a shutdown request."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.delays: list[float] = []
        self._stop_after = stop_after

    def __call__(self, seconds: float) -> bool:
        self.delays.append(seconds)
        return self._stop_after is not None and len(self.delays) >= self._stop_after


def flaky(failures: int, error: Exception) -> tuple[list[int], Callable[[], str]]:
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return "done"

    return calls, operation


def test_policy_delays_grow_exponentially_and_are_capped() -> None:
    assert RetryPolicy(attempts=6, base_delay_s=1.0, max_delay_s=10.0).delays() == (1.0, 2.0, 4.0, 8.0, 10.0)
    assert RetryPolicy(attempts=1, base_delay_s=1.0, max_delay_s=1.0).delays() == ()


@pytest.mark.parametrize(("attempts", "base", "cap"), [(0, 1.0, 1.0), (3, 0.0, 1.0), (3, 2.0, 1.0)])
def test_policy_rejects_invalid_values(attempts: int, base: float, cap: float) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(attempts=attempts, base_delay_s=base, max_delay_s=cap)


def test_retries_transient_errors_until_success() -> None:
    calls, operation = flaky(2, TransientError("boom"))
    wait = FakeWait()
    result = call_with_retry(
        operation,
        policy=RetryPolicy(attempts=5, base_delay_s=0.5, max_delay_s=8.0),
        is_retryable=lambda exc: isinstance(exc, TransientError),
        description="op",
        wait=wait,
    )
    assert result == "done"
    assert len(calls) == 3
    assert wait.delays == [0.5, 1.0]


def test_raises_last_error_when_attempts_are_exhausted() -> None:
    calls, operation = flaky(10, TransientError("still down"))
    wait = FakeWait()
    with pytest.raises(TransientError, match="still down"):
        call_with_retry(
            operation,
            policy=RetryPolicy(attempts=3, base_delay_s=1.0, max_delay_s=1.0),
            is_retryable=lambda exc: True,
            description="op",
            wait=wait,
        )
    assert len(calls) == 3
    assert wait.delays == [1.0, 1.0]


def test_non_retryable_errors_propagate_immediately() -> None:
    calls, operation = flaky(1, ValueError("bad input"))
    wait = FakeWait()
    with pytest.raises(ValueError, match="bad input"):
        call_with_retry(
            operation,
            policy=RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=1.0),
            is_retryable=lambda exc: isinstance(exc, TransientError),
            description="op",
            wait=wait,
        )
    assert len(calls) == 1
    assert wait.delays == []


def test_shutdown_request_aborts_the_backoff() -> None:
    calls, operation = flaky(10, TransientError("down"))
    with pytest.raises(RetryAbortedError) as raised:
        call_with_retry(
            operation,
            policy=RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=1.0),
            is_retryable=lambda exc: True,
            description="op",
            wait=FakeWait(stop_after=1),
        )
    assert len(calls) == 1
    assert isinstance(raised.value.__cause__, TransientError)
