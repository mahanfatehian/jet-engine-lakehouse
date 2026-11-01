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
from streaming_config import StreamingSettings
from streaming_readiness import DependencyNotReadyError, Probe

MINIO_DOWN = "http://minio:9000/minio/health/live is unreachable: <urlopen error [Errno 111] refused>"


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

    def isStopped(self) -> bool:
        return self.stopped


def fake_spark(context: FakeContext | None) -> SimpleNamespace:
    def sc() -> FakeContext:
        if context is None:
            raise ConnectionRefusedError("py4j gateway is gone")
        return context

    return SimpleNamespace(sparkContext=SimpleNamespace(_jsc=SimpleNamespace(sc=sc)))


class FakeProbe:
    """A readiness probe that answers, or fails with ``error`` like the real probes do."""

    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise DependencyNotReadyError(self.error)


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


@pytest.mark.parametrize(("value", "announced"), [(None, True), ("true", False)])
def test_best_effort_hf_mode_is_announced(
    monkeypatch: pytest.MonkeyPatch,
    heartbeat_file: Path,
    caplog: pytest.LogCaptureFixture,
    value: str | None,
    announced: bool,
) -> None:
    if value is not None:
        monkeypatch.setenv("HF_FAIL_ON_DATA_LOSS", value)
    monkeypatch.setattr(job, "run", lambda *_args: 0)
    with caplog.at_level(logging.INFO):
        assert job.main() == 0
    assert ("HF_FAIL_ON_DATA_LOSS=false (best effort)" in caplog.text) is announced


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


def test_dependency_check_reports_every_dependency_that_does_not_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kafka, minio = FakeProbe(), FakeProbe(MINIO_DOWN)
    clickhouse = FakeProbe("http://clickhouse:8123/ping answered HTTP 503")
    with caplog.at_level(logging.INFO):
        exit_code = job.check_dependencies((("Kafka", kafka), ("MinIO", minio), ("ClickHouse", clickhouse)))
    assert exit_code == job.EXIT_FAILURE
    assert (kafka.calls, minio.calls, clickhouse.calls) == (1, 1, 1)  # a failure does not end the check
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.ERROR, f"dependency not ready: MinIO: {MINIO_DOWN}"),
        (logging.ERROR, "dependency not ready: ClickHouse: http://clickhouse:8123/ping answered HTTP 503"),
    ]


def test_dependency_check_passes_when_every_dependency_answers(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert job.check_dependencies((("Kafka", FakeProbe()), ("MinIO", FakeProbe()))) == job.EXIT_OK
    assert caplog.messages == ["dependencies answer: Kafka, MinIO"]


@pytest.mark.parametrize(("minio_error", "exit_code"), [(None, 0), (MINIO_DOWN, 1)])
def test_check_dependencies_option_probes_once_and_leaves_the_job_alone(
    monkeypatch: pytest.MonkeyPatch, heartbeat_file: Path, minio_error: str | None, exit_code: int
) -> None:
    probes = {"Kafka": FakeProbe(), "MinIO": FakeProbe(minio_error), "ClickHouse": FakeProbe()}
    configured: list[StreamingSettings] = []
    runs: list[object] = []

    def fake_dependency_probes(settings: StreamingSettings) -> tuple[Probe, ...]:
        configured.append(settings)
        return tuple(probes.items())

    monkeypatch.setattr(job, "dependency_probes", fake_dependency_probes)
    monkeypatch.setattr(job, "run", lambda *args: runs.append(args) or 0)
    assert job.main(["--check-dependencies"]) == exit_code
    assert [probe.calls for probe in probes.values()] == [1, 1, 1]
    assert [settings.heartbeat_file for settings in configured] == [str(heartbeat_file)]  # the job's settings
    assert runs == []  # no Spark driver
    assert not heartbeat_file.exists()  # a job running next to the check keeps its own heartbeat


def test_unknown_option_exits_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        job.main(["--verbose"])
    assert exited.value.code == job.EXIT_CONFIG_ERROR
    assert "unrecognized arguments: --verbose" in capsys.readouterr().err
