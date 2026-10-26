"""Cross-process task locks: a manual run and the orchestrator loop never run the same job at once.

Every task holds an exclusive ``flock`` on ``<PIPELINE_LOCK_DIR>/pipeline-<group>.lock`` while it
runs. dbt run, test and build share one lock (they rebuild the same relations, and
dbt-clickhouse swaps ``table`` models in through fixed ``<model>__dbt_backup`` tables that two
concurrent runs drop under each other); train, snapshot and predict have one lock each.
Compose mounts one named volume at the lock directory in the orchestrator service and in every
``docker compose run orchestrator`` container, so the kernel arbitrates across containers too.
The kernel drops a flock when the process holding it exits, so a crashed or killed run never
leaves a stale lock behind. Lock files are opened non-inheritable: the runner holds the lock,
the task's child process does not.

A waiting run polls instead of blocking: it keeps the heartbeat fresh, honours a stop request
within one poll interval, and gives up after the longest time a holder can legitimately keep
the lock (the slowest task of the group plus its termination grace and a margin).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pipeline_process import TERMINATE_GRACE_SECONDS
from pipeline_signals import ShutdownFlag
from pipeline_tasks import (
    EXIT_FAILURE,
    EXIT_LOCK_TIMEOUT,
    TASK_TIMEOUT_SECONDS,
    TaskName,
    TaskResult,
    TaskSpec,
    TaskStatus,
)

try:
    import fcntl
except ImportError:  # Windows: no flock. The orchestrator itself only runs in Linux containers.
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LOCK_GROUPS: dict[TaskName, str] = {
    TaskName.DBT_RUN: "dbt",
    TaskName.DBT_TEST: "dbt",
    TaskName.DBT_BUILD: "dbt",
    TaskName.SNAPSHOT: "snapshot",
    TaskName.TRAIN: "train",
    TaskName.PREDICT: "predict",
}
LOCK_POLL_SECONDS = 1.0
WAIT_REPORT_SECONDS = 30.0
LOCK_WAIT_MARGIN_SECONDS = 60.0

TaskRun = Callable[[TaskSpec], TaskResult]


class _Wait(Enum):
    ACQUIRED = "acquired"
    TIMED_OUT = "timed out"
    STOPPED = "stopped"


def lock_path(directory: Path, group: str) -> Path:
    """Location of the lock file of ``group`` inside ``directory``."""
    return directory / f"pipeline-{group}.lock"


def lock_wait_limit(group: str) -> float:
    """Longest a holder of ``group``'s lock can legitimately keep it, in seconds."""
    slowest = max(TASK_TIMEOUT_SECONDS[task] for task, name in LOCK_GROUPS.items() if name == group)
    return slowest + TERMINATE_GRACE_SECONDS + LOCK_WAIT_MARGIN_SECONDS


class TaskLocks:
    """Runs task specs while holding the lock of their group."""

    def __init__(
        self,
        directory: Path,
        shutdown: ShutdownFlag,
        *,
        on_poll: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], bool] | None = None,
        poll_interval: float = LOCK_POLL_SECONDS,
        max_wait_seconds: float | None = None,
    ) -> None:
        self.directory = directory
        self._shutdown = shutdown
        self._on_poll = on_poll
        self._clock = clock
        self._sleep = shutdown.wait if sleep is None else sleep
        self._poll_interval = poll_interval
        self._max_wait_seconds = max_wait_seconds
        self._reported_unavailable: set[str] = set()

    def run_locked(self, spec: TaskSpec, run: TaskRun) -> TaskResult:
        """Run ``spec`` through ``run`` while holding its group's lock.

        Returns a FAILED result without running the task when the lock stays busy beyond the
        wait limit (exit code 75) or when a stop is requested while waiting. A lock directory
        that cannot be used is reported once and the task runs unlocked: a misconfigured
        mount must not stop the whole pipeline.
        """
        group = LOCK_GROUPS[spec.name]
        fd = self._open(group, spec.name)
        if fd is None:
            return run(spec)
        try:
            started = self._clock()
            try:
                outcome = self._wait_for_lock(fd, group, spec.name)
            except OSError as exc:
                self._report_unavailable(
                    group, spec.name, f"cannot lock {lock_path(self.directory, group)}: {exc}"
                )
                return run(spec)
            if outcome is not _Wait.ACQUIRED:
                return self._not_acquired(spec, group, outcome, self._clock() - started)
            _write_holder(fd, spec.name)
            try:
                return run(spec)
            finally:
                _clear_holder(fd)
        finally:
            os.close(fd)  # closing the last descriptor releases the flock

    def _open(self, group: str, task: TaskName) -> int | None:
        if fcntl is None:
            self._report_unavailable(group, task, "this platform has no fcntl.flock")
            return None
        path = lock_path(self.directory, group)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            return os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        except OSError as exc:
            self._report_unavailable(group, task, f"cannot open {path}: {exc}")
            return None

    def _wait_for_lock(self, fd: int, group: str, task: TaskName) -> _Wait:
        started = self._clock()
        limit = self._max_wait_seconds if self._max_wait_seconds is not None else lock_wait_limit(group)
        next_report: float | None = None
        while not _try_lock(fd):
            if self._shutdown.requested:
                return _Wait.STOPPED
            now = self._clock()
            waited = now - started
            if waited >= limit:
                return _Wait.TIMED_OUT
            if next_report is None:
                logger.warning(
                    "task %s waits for the %s lock (held by %s; giving up after %.0fs)",
                    task,
                    group,
                    self._holder(group),
                    limit,
                )
                next_report = now + WAIT_REPORT_SECONDS
            elif now >= next_report:
                logger.info("task %s still waiting for the %s lock after %.0fs", task, group, waited)
                next_report = now + WAIT_REPORT_SECONDS
            if self._on_poll is not None:
                self._on_poll()
            self._sleep(min(self._poll_interval, limit - waited))
        if next_report is not None:
            logger.info("task %s got the %s lock after %.1fs", task, group, self._clock() - started)
        return _Wait.ACQUIRED

    def _not_acquired(self, spec: TaskSpec, group: str, outcome: _Wait, waited: float) -> TaskResult:
        if outcome is _Wait.STOPPED:
            message = f"stop requested while waiting for the {group} lock: {spec.name} was not started"
            logger.warning("task %s: %s", spec.name, message)
            return TaskResult(
                spec.name, TaskStatus.FAILED, EXIT_FAILURE, waited, interrupted=True, output_tail=(message,)
            )
        message = (
            f"the {group} lock stayed busy for {waited:.0f}s (held by {self._holder(group)}): "
            f"{spec.name} was not started"
        )
        logger.error("task %s FAILED (exit code %d): %s", spec.name, EXIT_LOCK_TIMEOUT, message)
        return TaskResult(spec.name, TaskStatus.FAILED, EXIT_LOCK_TIMEOUT, waited, output_tail=(message,))

    def _holder(self, group: str) -> str:
        try:
            info = json.loads(lock_path(self.directory, group).read_text(encoding="utf-8"))
            return f"{info['task']} (pid {info['pid']} on host {info['host']}, since {info['since']})"
        except (OSError, ValueError, KeyError, TypeError):
            return "another process"

    def _report_unavailable(self, group: str, task: TaskName, reason: str) -> None:
        if group in self._reported_unavailable:
            logger.debug("task %s runs without the %s lock: %s", task, group, reason)
            return
        self._reported_unavailable.add(group)
        logger.error(
            "task locks unavailable (%s): %s runs WITHOUT the %s lock and may collide with a "
            "concurrent manual or scheduled run; check PIPELINE_LOCK_DIR=%s",
            reason,
            task,
            group,
            self.directory,
        )


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _write_holder(fd: int, task: TaskName) -> None:
    """Record who holds the lock, so a waiting run can say what it waits for (best effort)."""
    info = {
        "task": task.value,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "since": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(info).encode("utf-8"))
    except OSError as exc:
        logger.debug("cannot record the lock holder: %s", exc)


def _clear_holder(fd: int) -> None:
    try:
        os.ftruncate(fd, 0)
    except OSError as exc:
        logger.debug("cannot clear the lock holder: %s", exc)
