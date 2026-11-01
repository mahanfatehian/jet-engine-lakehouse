"""Unit tests of the bounded exponential-backoff retry helper and the opt-in equal jitter."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
from streaming_retry import RetryAbortedError, RetryPolicy, call_with_retry, capped_backoff, equal_jitter


class TransientError(Exception):
    pass


class FixedRandom(random.Random):
    """A generator whose every ``random()`` draw is ``value``."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def random(self) -> float:
        return self.value


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


def test_capped_backoff_doubles_up_to_the_cap() -> None:
    assert [capped_backoff(failures, 2.0, 60.0) for failures in range(1, 9)] == [2, 4, 8, 16, 32, 60, 60, 60]


def test_capped_backoff_stays_finite_for_endless_failures() -> None:
    assert capped_backoff(10**6, 2.0, 60.0) == 60.0
    assert capped_backoff(0, 2.0, 60.0) == 2.0


@pytest.mark.parametrize(
    ("draw", "expected"),
    [(0.0, 8.0), (0.25, 7.0), (0.5, 6.0), (0.75, 5.0), (1.0 - 2**-53, 4.0)],
)
def test_equal_jitter_keeps_half_of_the_delay_and_draws_the_other_half(draw: float, expected: float) -> None:
    assert equal_jitter(8.0, FixedRandom(draw)) == pytest.approx(expected, abs=1e-12)


def test_equal_jitter_stays_within_half_and_all_of_the_delay() -> None:
    rng = random.Random(2026)
    waits = [equal_jitter(60.0, rng) for _ in range(5000)]
    assert all(30.0 <= wait <= 60.0 for wait in waits)
    # Spread over the whole upper half, not bunched: what de-synchronises the restarts.
    assert min(waits) < 30.5 and max(waits) > 59.5
    assert sum(waits) / len(waits) == pytest.approx(45.0, abs=1.0)


def test_equal_jitter_is_reproducible_from_its_seed() -> None:
    def schedule(seed: int) -> list[float]:
        rng = random.Random(seed)
        return [equal_jitter(capped_backoff(failures, 2.0, 60.0), rng) for failures in range(1, 12)]

    assert schedule(7) == schedule(7)
    assert schedule(7) != schedule(8)


def test_jittered_backoff_never_shrinks_while_it_doubles() -> None:
    rng = random.Random(3)
    for _ in range(200):
        waits = [equal_jitter(capped_backoff(failures, 2.0, 60.0), rng) for failures in range(1, 6)]
        assert waits == sorted(waits)  # 2, 4, 8, 16, 32 s steps: no overlap between them


@pytest.mark.parametrize(("attempts", "base", "cap"), [(0, 1.0, 1.0), (3, 0.0, 1.0), (3, 2.0, 1.0)])
def test_policy_rejects_invalid_values(attempts: int, base: float, cap: float) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(attempts=attempts, base_delay_s=base, max_delay_s=cap)


@pytest.mark.parametrize(
    ("policy", "total"),
    [
        (RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=16.0), 15.0),  # the ClickHouse inserts
        (RetryPolicy(attempts=6, base_delay_s=1.0, max_delay_s=10.0), 25.0),  # 1+2+4+8, then the cap
        (RetryPolicy(attempts=1, base_delay_s=1.0, max_delay_s=1.0), 0.0),
    ],
)
def test_max_total_delay_is_what_an_always_failing_call_waits(policy: RetryPolicy, total: float) -> None:
    assert policy.max_total_delay_s == total
    calls, operation = flaky(policy.attempts, TransientError("down"))
    wait = FakeWait()
    with pytest.raises(TransientError):
        call_with_retry(operation, policy=policy, is_retryable=lambda exc: True, description="op", wait=wait)
    assert len(calls) == policy.attempts
    assert sum(wait.delays) == policy.max_total_delay_s


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
