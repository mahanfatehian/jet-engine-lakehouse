"""SubprocessTaskRunner against a fully simulated child process (no real subprocess is started)."""

from __future__ import annotations

import io
import logging
import math
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pipeline_process
import pytest
from pipeline_process import SubprocessTaskRunner
from pipeline_signals import ShutdownFlag
from pipeline_tasks import TaskName, TaskSpec, TaskStatus

SPEC = TaskSpec(TaskName.PREDICT, ("python", "predict_rul.py", "--mode", "batch"), Path("/app"), 10.0)
GRACE_SECONDS = 3.0


class FakeProcess:
    """Stands in for ``subprocess.Popen``.

    It exits by itself once ``runtime`` seconds of fake time have passed, and on SIGTERM
    when ``honours_sigterm`` is set; ``wait`` lets fake time pass like a real timeout would.
    """

    def __init__(
        self,
        clock: Any,
        *,
        output: str = "",
        returncode: int = 0,
        runtime: float = 0.0,
        honours_sigterm: bool = True,
    ) -> None:
        self._clock = clock
        self._final_returncode = returncode
        self._exit_at = clock() + runtime
        self._honours_sigterm = honours_sigterm
        self.stdout = io.StringIO(output)
        self.pid = 4242
        self.returncode: int | None = None
        self.signals: list[str] = []

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None and self._clock() >= self._exit_at:
            self.returncode = self._final_returncode
        if self.returncode is None:
            assert timeout is not None
            self._clock.advance(timeout)
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("SIGTERM")
        if self._honours_sigterm and self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.signals.append("SIGKILL")
        if self.returncode is None:
            self.returncode = -9


class FakePopen:
    """Popen factory recording how the child would have been started."""

    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((argv, kwargs))
        return self.process


def _runner(
    clock: Any, popen: Any, shutdown: ShutdownFlag | None = None, **kwargs: Any
) -> SubprocessTaskRunner:
    return SubprocessTaskRunner(
        shutdown or ShutdownFlag(),
        popen=popen,
        clock=clock,
        base_env={"PATH": "/usr/bin"},
        grace_seconds=GRACE_SECONDS,
        **kwargs,
    )


def test_success_streams_output_with_task_prefix(clock: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_process")
    popen = FakePopen(FakeProcess(clock, output="loading model\nscored 6 engines\n", runtime=2.0))

    result = _runner(clock, popen).run(SPEC)

    assert (result.status, result.exit_code) == (TaskStatus.SUCCESS, 0)
    assert result.output_tail == ("loading model", "scored 6 engines")
    assert result.duration_seconds == pytest.approx(2.0)
    assert "[predict] loading model" in caplog.messages
    assert "[predict] scored 6 engines" in caplog.messages


def test_exit_code_3_is_logged_as_skipped_not_as_failure(
    clock: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_process")
    popen = FakePopen(FakeProcess(clock, output="no production model\n", returncode=3))

    result = _runner(clock, popen).run(SPEC)

    assert (result.status, result.exit_code) == (TaskStatus.SKIPPED, 3)
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("SKIPPED" in message for message in caplog.messages)


def test_failure_logs_the_tail_of_the_output(clock: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_process")
    output = "".join(f"line {number}\n" for number in range(50))
    popen = FakePopen(FakeProcess(clock, output=output, returncode=2))

    result = _runner(clock, popen, tail_lines=5).run(SPEC)

    assert (result.status, result.exit_code) == (TaskStatus.FAILED, 2)
    assert result.output_tail == tuple(f"line {number}" for number in range(45, 50))
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "exit code 2" in errors[0]
    assert errors[0].endswith("line 45\nline 46\nline 47\nline 48\nline 49")


def test_missing_executable_fails_with_127(clock: Any) -> None:
    def broken_popen(argv: list[str], **kwargs: Any) -> FakeProcess:
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    result = _runner(clock, broken_popen).run(SPEC)

    assert (result.status, result.exit_code) == (TaskStatus.FAILED, 127)
    assert "could not start 'python'" in result.output_tail[0]


def test_timeout_sends_sigterm_and_reports_124(clock: Any) -> None:
    process = FakeProcess(clock, runtime=math.inf)

    result = _runner(clock, FakePopen(process)).run(SPEC)

    assert process.signals == ["SIGTERM"]
    assert (result.status, result.exit_code, result.timed_out) == (TaskStatus.FAILED, 124, True)
    assert result.duration_seconds == pytest.approx(SPEC.timeout_seconds, abs=1.0)


def test_timeout_kills_a_child_that_ignores_sigterm(clock: Any) -> None:
    process = FakeProcess(clock, runtime=math.inf, honours_sigterm=False)

    result = _runner(clock, FakePopen(process)).run(SPEC)

    assert process.signals == ["SIGTERM", "SIGKILL"]
    assert (result.exit_code, result.timed_out) == (124, True)
    assert result.duration_seconds == pytest.approx(SPEC.timeout_seconds + GRACE_SECONDS, abs=1.0)


def test_stop_request_forwards_sigterm_to_the_child(clock: Any) -> None:
    shutdown = ShutdownFlag()
    process = FakeProcess(clock, runtime=math.inf)
    shutdown.request("SIGTERM")

    result = _runner(clock, FakePopen(process), shutdown).run(SPEC)

    assert process.signals == ["SIGTERM"]
    assert (result.status, result.exit_code) == (TaskStatus.FAILED, 143)
    assert result.interrupted and not result.timed_out


def test_forced_stop_kills_without_waiting_for_the_grace_period(clock: Any) -> None:
    shutdown = ShutdownFlag()
    process = FakeProcess(clock, runtime=math.inf, honours_sigterm=False)
    shutdown.request("SIGTERM")
    shutdown.request("SIGTERM")

    result = _runner(clock, FakePopen(process), shutdown).run(SPEC)

    assert process.signals == ["SIGTERM", "SIGKILL"]
    assert result.exit_code == 137
    assert result.duration_seconds < GRACE_SECONDS


def test_child_environment_and_pipes(clock: Any) -> None:
    popen = FakePopen(FakeProcess(clock))
    spec = TaskSpec(
        TaskName.DBT_RUN,
        ("dbt", "run"),
        Path("/app/analytics"),
        60.0,
        (("DBT_USE_COLORS", "false"), ("DBT_SEND_ANONYMOUS_USAGE_STATS", "false")),
    )
    runner = SubprocessTaskRunner(
        ShutdownFlag(), popen=popen, clock=clock, base_env={"DBT_USE_COLORS": "true", "PATH": "/usr/bin"}
    )

    runner.run(spec)

    argv, kwargs = popen.calls[0]
    assert argv == ["dbt", "run"]
    assert kwargs["cwd"] == Path("/app/analytics")
    assert kwargs["env"] == {
        "PATH": "/usr/bin",
        "DBT_USE_COLORS": "true",
        "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
        "PYTHONUNBUFFERED": "1",
    }
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["text"] is True


def test_poll_hook_runs_while_the_child_is_busy(clock: Any) -> None:
    polls: list[float] = []
    popen = FakePopen(FakeProcess(clock, runtime=3.0))

    _runner(clock, popen, on_poll=lambda: polls.append(clock())).run(SPEC)

    assert len(polls) >= 5


class HeldOpenPipe(io.StringIO):
    """Output pipe kept open by an orphaned grandchild: blocks after its first line."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def __iter__(self) -> Iterator[str]:
        yield "started worker\n"
        self.release.wait(10)


def test_a_pipe_held_open_by_a_grandchild_does_not_block_the_runner(
    clock: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_process")
    monkeypatch.setattr(pipeline_process, "READER_JOIN_SECONDS", 0.2)
    process = FakeProcess(clock)
    pipe = HeldOpenPipe()
    process.stdout = pipe

    try:
        result = _runner(clock, FakePopen(process)).run(SPEC)
    finally:
        pipe.release.set()

    assert (result.status, result.output_tail) == (TaskStatus.SUCCESS, ("started worker",))
    assert any("output pipe is still open" in message for message in caplog.messages)
