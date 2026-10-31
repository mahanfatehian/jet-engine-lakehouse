"""Bounded retries with exponential backoff for the job's network I/O.

Waiting is delegated to an injected ``wait(seconds) -> bool`` callable (normally
``threading.Event.wait`` of the shutdown event) so that a SIGTERM interrupts a backoff
immediately instead of keeping the container alive until the retries are exhausted.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

WaitFn = Callable[[float], bool]
"""Sleeps up to the given seconds; returns True when shutdown was requested meanwhile."""

# Far beyond any cap; bounds the arithmetic for an operation that keeps failing for weeks.
_MAX_BACKOFF_EXPONENT = 32


class RetryAbortedError(RuntimeError):
    """Raised when shutdown is requested while waiting for the next attempt."""


def capped_backoff(failures: int, base_delay_s: float, max_delay_s: float, *, jitter: bool = True) -> float:
    """Wait after ``failures`` consecutive failures: ``base_delay_s * 2**(failures - 1)``, capped.

    When ``jitter`` is True (the default) the delay is multiplied by a uniform random
    factor in [0.5, 1.0) so that concurrent queries do not retry in lockstep against a
    recovering broker (thundering-herd avoidance).
    """
    exponent = min(max(failures, 1) - 1, _MAX_BACKOFF_EXPONENT)
    delay = min(base_delay_s * 2**exponent, max_delay_s)
    return delay * random.uniform(0.5, 1.0) if jitter else delay


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff: ``base_delay_s * 2**n`` capped at ``max_delay_s``."""

    attempts: int
    base_delay_s: float
    max_delay_s: float

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_delay_s <= 0 or self.max_delay_s < self.base_delay_s:
            raise ValueError("delays must satisfy 0 < base_delay_s <= max_delay_s")

    def delays(self) -> tuple[float, ...]:
        """Waits between consecutive attempts (one fewer than ``attempts``), without jitter."""
        return tuple(
            capped_backoff(failures, self.base_delay_s, self.max_delay_s, jitter=False)
            for failures in range(1, self.attempts)
        )

    @property
    def max_total_delay_s(self) -> float:
        """Worst-case total wait (no jitter) before the policy is exhausted."""
        return sum(self.delays())


def call_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    description: str,
    wait: WaitFn,
) -> T:
    """Run ``operation`` until it succeeds or the policy is exhausted.

    Non-retryable exceptions propagate immediately; after the last attempt the final
    exception propagates unchanged. ``RetryAbortedError`` is raised when ``wait``
    reports a shutdown request.
    """
    delays = policy.delays()
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt == policy.attempts or not is_retryable(exc):
                raise
            delay = delays[attempt - 1]
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                policy.attempts,
                exc,
                delay,
            )
            if wait(delay):
                raise RetryAbortedError(f"{description}: shutdown requested while retrying") from exc
            attempt += 1
