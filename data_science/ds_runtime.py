"""Process plumbing shared by the data-science CLIs.

Logging set-up, the platform's exit-code convention (0 = done, 3 = skipped
because preconditions are not met), cooperative SIGTERM handling and bounded
retries with exponential backoff for network I/O.
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import NoReturn, TypeVar

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_SKIPPED = 3
# 128 + SIGTERM: the conventional status of a process stopped by that signal.
EXIT_INTERRUPTED = 143

_T = TypeVar("_T")
_NOISY_LOGGERS = ("botocore", "boto3", "s3transfer", "urllib3")

logger = logging.getLogger(__name__)


class PreconditionsNotMetError(Exception):
    """The job's preconditions are not met; the CLI exits with EXIT_SKIPPED."""


class TerminationRequestedError(Exception):
    """SIGTERM/SIGINT arrived and the job stopped at a safe point."""


def configure_logging(level_name: str = "INFO") -> None:
    """Send single-line, UTC-stamped log records to stderr at the given level."""
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.getLevelNamesMapping().get(level_name.upper(), logging.INFO))
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class StopToken:
    """Cooperative cancellation flag, set by SIGTERM/SIGINT once handlers are installed."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def install_signal_handlers(self) -> StopToken:
        """Route SIGTERM and SIGINT to this token; must be called from the main thread."""
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self._on_signal)
        return self

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        logger.warning("received %s; stopping at the next safe point", signal.Signals(signum).name)
        self._event.set()

    def request_stop(self) -> None:
        """Ask the job to stop at its next safe point."""
        self._event.set()

    @property
    def stop_requested(self) -> bool:
        """True once a stop was requested."""
        return self._event.is_set()

    def wait(self, timeout_seconds: float) -> bool:
        """Sleep up to ``timeout_seconds``; return True early when a stop is requested."""
        return self._event.wait(timeout_seconds)

    def raise_if_stopped(self) -> None:
        """Raise TerminationRequestedError when a stop was requested (call between job phases)."""
        if self._event.is_set():
            raise TerminationRequestedError("termination requested")


def retry_call(
    operation: Callable[[], _T],
    *,
    description: str,
    is_retryable: Callable[[Exception], bool],
    attempts: int = 5,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 30.0,
) -> _T:
    """Run ``operation``, retrying transient failures with capped exponential backoff.

    Non-retryable errors and the error of the last attempt propagate unchanged.
    """
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = min(max_delay_seconds, base_delay_seconds * 2 ** (attempt - 1))
            # Jitter keeps concurrent jobs from retrying against a recovering service in lockstep.
            delay *= random.uniform(0.5, 1.0)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a value >= 1, got {value}")
    return value


def positive_float(text: str) -> float:
    """argparse type: a number > 0."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not value > 0:
        raise argparse.ArgumentTypeError(f"expected a value > 0, got {value}")
    return value


def run_cli(main: Callable[[], int]) -> NoReturn:
    """Run a CLI entry point and exit with the platform's exit-code convention."""
    try:
        code = main()
    except PreconditionsNotMetError as exc:
        logger.info("skipped: %s", exc)
        code = EXIT_SKIPPED
    except TerminationRequestedError:
        logger.warning("terminated before completion; nothing further was written")
        code = EXIT_INTERRUPTED
    except Exception:
        logger.exception("job failed")
        code = EXIT_FAILURE
    sys.exit(code)
