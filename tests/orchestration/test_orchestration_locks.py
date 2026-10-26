"""Cross-process task locks: serialisation, waiting, timeouts and fallbacks."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pipeline_locks
import pytest
from pipeline_locks import LOCK_GROUPS, TaskLocks, lock_path, lock_wait_limit
from pipeline_signals import ShutdownFlag
from pipeline_tasks import EXIT_LOCK_TIMEOUT, TaskName, TaskResult, TaskSpec, TaskStatus

ORCHESTRATION_DIR = Path(pipeline_locks.__file__).resolve().parent
needs_flock = pytest.mark.skipif(pipeline_locks.fcntl is None, reason="flock is POSIX-only")

HOLDER_CHILD = """
import sys
from pathlib import Path
from pipeline_locks import TaskLocks
from pipeline_signals import ShutdownFlag
from pipeline_tasks import TaskName, TaskResult, TaskSpec, TaskStatus

def hold(spec):
    print("locked", flush=True)
    sys.stdin.readline()  # returns when the parent closes our stdin
    return TaskResult(spec.name, TaskStatus.SUCCESS, 0, 0.0)

spec = TaskSpec(TaskName.DBT_RUN, ("dbt", "run"), Path("."), 60.0)
TaskLocks(Path(sys.argv[1]), ShutdownFlag()).run_locked(spec, hold)
"""


def _spec(task: TaskName) -> TaskSpec:
    return TaskSpec(task, ("true",), Path("."), 60.0)


class RecordingRun:
    """Task stand-in that records its calls and what the lock file said while it ran."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls: list[TaskName] = []
        self.holder_files: list[str] = []

    def __call__(self, spec: TaskSpec) -> TaskResult:
        self.calls.append(spec.name)
        self.holder_files.append(
            lock_path(self.directory, LOCK_GROUPS[spec.name]).read_text(encoding="utf-8")
        )
        return TaskResult(spec.name, TaskStatus.SUCCESS, 0, 0.0)


class OtherHolder:
    """Another open file description holding the lock: conflicts exactly like another process.

    Never blocks: raises BlockingIOError when the lock is already taken.
    """

    def __init__(self, path: Path, info: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            pipeline_locks.fcntl.flock(self.fd, pipeline_locks.fcntl.LOCK_EX | pipeline_locks.fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise
        if info is not None:
            os.write(self.fd, json.dumps(info).encode("utf-8"))
        self.released = False

    def release(self) -> None:
        if not self.released:
            os.close(self.fd)
            self.released = True


def _fake_sleep(clock: Any, shutdown: ShutdownFlag, on_sleep: Any = None) -> Any:
    def sleep(seconds: float) -> bool:
        clock.advance(seconds)
        if on_sleep is not None:
            on_sleep()
        return shutdown.requested

    return sleep


def test_dbt_tasks_share_one_lock_and_the_others_have_their_own() -> None:
    assert set(LOCK_GROUPS) == set(TaskName)
    assert {LOCK_GROUPS[task] for task in (TaskName.DBT_RUN, TaskName.DBT_TEST, TaskName.DBT_BUILD)} == {
        "dbt"
    }
    assert [LOCK_GROUPS[task] for task in (TaskName.TRAIN, TaskName.SNAPSHOT, TaskName.PREDICT)] == [
        "train",
        "snapshot",
        "predict",
    ]
    assert lock_path(Path("/locks"), "dbt") == Path("/locks") / "pipeline-dbt.lock"


def test_wait_limit_outlasts_the_slowest_task_of_the_group() -> None:
    assert lock_wait_limit("dbt") == 30 * 60 + 20 + 60  # dbt build timeout + grace + margin
    assert lock_wait_limit("predict") == 5 * 60 + 20 + 60


@needs_flock
def test_task_runs_while_holding_its_lock_and_names_the_holder(tmp_path: Path) -> None:
    directory = tmp_path / "locks"
    path = lock_path(directory, "dbt")
    run = RecordingRun(directory)

    def run_and_probe(spec: TaskSpec) -> TaskResult:
        with pytest.raises(BlockingIOError):
            OtherHolder(path)
        return run(spec)

    result = TaskLocks(directory, ShutdownFlag()).run_locked(_spec(TaskName.DBT_BUILD), run_and_probe)

    assert result.status is TaskStatus.SUCCESS
    holder = json.loads(run.holder_files[0])
    assert (holder["task"], holder["pid"]) == ("dbt-build", os.getpid())
    assert path.read_text(encoding="utf-8") == ""  # holder record cleared on release
    OtherHolder(path).release()  # and the lock itself is free again


@needs_flock
def test_a_busy_lock_times_out_without_running_the_task(
    clock: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_locks")
    directory = tmp_path / "locks"
    info = {"task": "dbt-run", "pid": 4242, "host": "loop-container", "since": "2026-10-26T12:00:00+00:00"}
    other = OtherHolder(lock_path(directory, "dbt"), info)
    shutdown = ShutdownFlag()
    locks = TaskLocks(
        directory, shutdown, clock=clock, sleep=_fake_sleep(clock, shutdown), max_wait_seconds=10
    )
    run = RecordingRun(directory)

    try:
        result = locks.run_locked(_spec(TaskName.DBT_TEST), run)
    finally:
        other.release()

    assert (result.status, result.exit_code, result.interrupted) == (
        TaskStatus.FAILED,
        EXIT_LOCK_TIMEOUT,
        False,
    )
    assert result.duration_seconds == pytest.approx(10.0)
    assert run.calls == []
    assert "held by dbt-run (pid 4242 on host loop-container" in result.output_tail[0]
    assert any(message.startswith("task dbt-test waits for the dbt lock") for message in caplog.messages)


@needs_flock
def test_the_task_starts_as_soon_as_the_holder_releases_the_lock(
    clock: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_locks")
    directory = tmp_path / "locks"
    other = OtherHolder(lock_path(directory, "train"))
    sleeps: list[int] = []
    polls: list[float] = []

    def release_after_three_polls() -> None:
        sleeps.append(1)
        if len(sleeps) == 3:
            other.release()

    shutdown = ShutdownFlag()
    locks = TaskLocks(
        directory,
        shutdown,
        on_poll=lambda: polls.append(clock()),
        clock=clock,
        sleep=_fake_sleep(clock, shutdown, release_after_three_polls),
    )
    run = RecordingRun(directory)

    result = locks.run_locked(_spec(TaskName.TRAIN), run)

    assert result.status is TaskStatus.SUCCESS
    assert run.calls == [TaskName.TRAIN]
    assert len(polls) == 3  # the heartbeat keeps ticking while the task waits
    assert "task train got the train lock after 3.0s" in caplog.messages


@needs_flock
def test_a_stop_request_ends_the_wait(clock: Any, tmp_path: Path) -> None:
    directory = tmp_path / "locks"
    other = OtherHolder(lock_path(directory, "predict"))
    shutdown = ShutdownFlag()
    locks = TaskLocks(
        directory,
        shutdown,
        clock=clock,
        sleep=_fake_sleep(clock, shutdown, lambda: shutdown.request("SIGTERM")),
    )
    run = RecordingRun(directory)

    try:
        result = locks.run_locked(_spec(TaskName.PREDICT), run)
    finally:
        other.release()

    assert (result.status, result.exit_code, result.interrupted) == (TaskStatus.FAILED, 1, True)
    assert run.calls == []


@needs_flock
def test_locks_of_different_groups_do_not_block_each_other(tmp_path: Path) -> None:
    directory = tmp_path / "locks"
    other = OtherHolder(lock_path(directory, "train"))
    run = RecordingRun(directory)

    try:
        result = TaskLocks(directory, ShutdownFlag(), max_wait_seconds=0).run_locked(
            _spec(TaskName.PREDICT), run
        )
    finally:
        other.release()

    assert result.status is TaskStatus.SUCCESS


@needs_flock
def test_a_second_process_waits_for_the_lock_and_times_out(tmp_path: Path) -> None:
    directory = tmp_path / "locks"
    python_path = [str(ORCHESTRATION_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}
    child = subprocess.Popen(
        [sys.executable, "-c", HOLDER_CHILD, str(directory)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline().strip() == "locked"
        run = RecordingRun(directory)

        busy = TaskLocks(directory, ShutdownFlag(), poll_interval=0.05, max_wait_seconds=0.5)
        blocked = busy.run_locked(_spec(TaskName.DBT_BUILD), run)

        assert (blocked.status, blocked.exit_code) == (TaskStatus.FAILED, EXIT_LOCK_TIMEOUT)
        assert f"dbt-run (pid {child.pid} on host" in blocked.output_tail[0]
        assert run.calls == []

        child.stdin.close()  # the holder finishes its task and releases the lock
        assert child.wait(timeout=30) == 0
        free = TaskLocks(directory, ShutdownFlag(), poll_interval=0.05, max_wait_seconds=10)
        assert free.run_locked(_spec(TaskName.DBT_BUILD), run).status is TaskStatus.SUCCESS
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_an_unusable_lock_directory_runs_tasks_unlocked_and_reports_it_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="pipeline_locks")
    not_a_directory = tmp_path / "locks"
    not_a_directory.write_text("occupied", encoding="utf-8")
    locks = TaskLocks(not_a_directory, ShutdownFlag())
    calls: list[TaskName] = []

    def run(spec: TaskSpec) -> TaskResult:
        calls.append(spec.name)
        return TaskResult(spec.name, TaskStatus.SUCCESS, 0, 0.0)

    results = [locks.run_locked(_spec(TaskName.DBT_RUN), run) for _ in range(2)]

    assert [result.status for result in results] == [TaskStatus.SUCCESS] * 2
    assert calls == [TaskName.DBT_RUN] * 2
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "dbt-run runs WITHOUT the dbt lock" in errors[0]


def test_platforms_without_flock_run_tasks_unlocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(pipeline_locks, "fcntl", None)
    calls: list[TaskName] = []

    def run(spec: TaskSpec) -> TaskResult:
        calls.append(spec.name)
        return TaskResult(spec.name, TaskStatus.SUCCESS, 0, 0.0)

    result = TaskLocks(tmp_path, ShutdownFlag()).run_locked(_spec(TaskName.SNAPSHOT), run)

    assert result.status is TaskStatus.SUCCESS
    assert calls == [TaskName.SNAPSHOT]
    assert "no fcntl.flock" in caplog.text
    assert not lock_path(tmp_path, "snapshot").exists()
