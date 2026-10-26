"""SIGTERM/SIGINT handling: graceful first request, forced second request."""

from __future__ import annotations

import signal
from collections.abc import Iterator

import pytest
from pipeline_signals import HANDLED_SIGNALS, ShutdownFlag, install_signal_handlers


@pytest.fixture
def restore_signal_handlers() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in HANDLED_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_flag_starts_clear() -> None:
    flag = ShutdownFlag()

    assert not flag.requested
    assert not flag.forced
    assert flag.reason == ""
    assert flag.wait(0.0) is False


def test_first_request_is_graceful_and_a_second_one_forces() -> None:
    flag = ShutdownFlag()

    flag.request("SIGTERM")
    assert flag.requested and not flag.forced
    assert flag.wait(10.0) is True

    flag.request("SIGINT")
    assert flag.forced
    assert flag.reason == "SIGTERM"


@pytest.mark.usefixtures("restore_signal_handlers")
@pytest.mark.parametrize("sig", HANDLED_SIGNALS)
def test_signals_become_stop_requests(sig: signal.Signals) -> None:
    flag = ShutdownFlag()
    install_signal_handlers(flag)
    handler = signal.getsignal(sig)
    assert callable(handler)

    handler(sig, None)
    assert flag.requested and not flag.forced
    assert flag.reason == sig.name

    handler(sig, None)
    assert flag.forced
