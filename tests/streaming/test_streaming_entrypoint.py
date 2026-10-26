"""Unit tests of the job entrypoint that need no running services."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator

import pytest

pytest.importorskip("pyspark")
pytest.importorskip("clickhouse_connect")

import spark_kafka_to_lakehouse as job


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def restore_signal_handlers() -> Iterator[None]:
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    yield
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def test_invalid_configuration_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    monkeypatch.setenv("MAX_OFFSETS_PER_TRIGGER", "lots")
    assert job.main() == job.EXIT_CONFIG_ERROR


def test_sigterm_only_requests_a_stop(restore_signal_handlers: None) -> None:
    stop_event = threading.Event()
    job.install_signal_handlers(stop_event)
    signal.raise_signal(signal.SIGTERM)
    assert stop_event.is_set()


def test_logging_uses_utc_iso_timestamps(restore_logging: None) -> None:
    job.configure_logging("DEBUG")
    (handler,) = logging.getLogger().handlers
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "message", None, None)
    record.created, record.msecs = 0.0, 0.0
    assert handler.format(record) == "1970-01-01T00:00:00.000Z INFO t: message"
    assert logging.getLogger("py4j").level == logging.WARNING
