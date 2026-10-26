"""Cooperative shutdown: SIGTERM/SIGINT set a flag that every wait in the runner polls."""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType

logger = logging.getLogger(__name__)

HANDLED_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)


class ShutdownFlag:
    """Stop request shared by the scheduler, the task runner and the startup wait.

    The first request asks for a graceful stop (a running child receives SIGTERM and the
    loop ends between tasks); a repeated request marks the stop as forced, so a child that
    is still shutting down gets killed right away.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._requests = 0
        self._reason = ""

    def request(self, reason: str) -> None:
        """Record a stop request; ``reason`` names its origin, e.g. the signal name."""
        # Deliberately lock-free: signal handlers run on the main thread between bytecodes
        # and could otherwise deadlock on a lock the interrupted code already holds.
        self._requests += 1
        if not self._reason:
            self._reason = reason
        self._event.set()

    @property
    def requested(self) -> bool:
        """True once a stop has been requested."""
        return self._event.is_set()

    @property
    def forced(self) -> bool:
        """True when a stop has been requested more than once."""
        return self._requests > 1

    @property
    def reason(self) -> str:
        """Origin of the first stop request, empty while running."""
        return self._reason

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds; return True early as soon as a stop is requested."""
        return self._event.wait(timeout)


def install_signal_handlers(
    flag: ShutdownFlag, signals: tuple[signal.Signals, ...] = HANDLED_SIGNALS
) -> None:
    """Turn ``signals`` into stop requests on ``flag`` instead of abrupt termination.

    Must be called from the main thread (a Python restriction on ``signal.signal``).
    """

    def _handle(signum: int, _frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        if flag.requested:
            logger.warning("received %s again: forcing the running task to stop", name)
        else:
            logger.warning(
                "received %s: stopping gracefully (a running task gets SIGTERM; repeat to force)", name
            )
        flag.request(name)

    for sig in signals:
        signal.signal(sig, _handle)
