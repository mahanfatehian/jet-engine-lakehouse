"""Unit tests of the job entrypoint that need no running services."""

from __future__ import annotations

import json
import logging
import signal
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def heartbeat_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restore_signal_handlers: None
) -> Iterator[Path]:
    """Run ``main`` with a temporary heartbeat file and logging left to pytest's capture."""
    path = tmp_path / "streaming.heartbeat"
    monkeypatch.setenv("STREAMING_HEARTBEAT_FILE", str(path))
    monkeypatch.setattr(job, "configure_logging", lambda _level: None)
    yield path


class FakeContext:
    def __init__(self, stopped: bool) -> None:
        self.stopped = stopped

    def isStopped(self) -> bool:  # noqa: N802 - JVM API
        return self.stopped


def fake_spark(context: FakeContext | None) -> SimpleNamespace:
    def sc() -> FakeContext:
        if context is None:
            raise ConnectionRefusedError("py4j gateway is gone")
        return context

    return SimpleNamespace(sparkContext=SimpleNamespace(_jsc=SimpleNamespace(sc=sc)))


def test_invalid_configuration_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    monkeypatch.setenv("MAX_OFFSETS_PER_TRIGGER", "lots")
    assert job.main() == job.EXIT_CONFIG_ERROR


def test_heartbeat_is_fresh_before_the_driver_starts(
    monkeypatch: pytest.MonkeyPatch, heartbeat_file: Path
) -> None:
    seen: list[str] = []

    def fake_run(_settings: object, _stop_event: object, _heartbeat: object) -> int:
        seen.append(json.loads(heartbeat_file.read_text(encoding="utf-8"))["status"])
        return job.EXIT_FAILURE

    monkeypatch.setattr(job, "run", fake_run)
    assert job.main() == job.EXIT_FAILURE  # the supervisor's exit code is passed through
    assert seen == ["starting"]


def test_tolerated_data_loss_is_announced(
    monkeypatch: pytest.MonkeyPatch, heartbeat_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KAFKA_FAIL_ON_DATA_LOSS", "false")
    monkeypatch.setattr(job, "run", lambda *_args: 0)
    with caplog.at_level(logging.WARNING):
        assert job.main() == 0
    assert "KAFKA_FAIL_ON_DATA_LOSS=false" in caplog.text


def test_unexpected_errors_exit_with_code_1(monkeypatch: pytest.MonkeyPatch, heartbeat_file: Path) -> None:
    def broken_run(*_args: object) -> int:
        raise RuntimeError("cannot launch the JVM")

    monkeypatch.setattr(job, "run", broken_run)
    assert job.main() == job.EXIT_FAILURE


def test_driver_alive_follows_the_spark_context() -> None:
    assert job.driver_alive(fake_spark(FakeContext(stopped=False))) is True
    assert job.driver_alive(fake_spark(FakeContext(stopped=True))) is False
    assert job.driver_alive(fake_spark(None)) is False


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
