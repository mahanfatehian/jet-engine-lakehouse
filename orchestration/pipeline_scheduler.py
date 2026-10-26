"""Scheduling policy of the ``loop`` mode (contract section 8).

A cycle starts every DBT_INTERVAL_SECONDS with ``dbt run``; when that succeeds, dbt test,
predict, train and snapshot follow, each only when it is due. A failed ``dbt run`` skips
the rest of its cycle. Training is attempted after every dbt run while no production model
exists (bootstrap), afterwards every TRAIN_INTERVAL_SECONDS.

All interval arithmetic uses the start time of the cycle, so a task whose interval equals
the dbt interval runs in every cycle no matter how long the tasks before it took. The
clock, the sleep and the task execution are injected, which keeps every decision testable.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pipeline_config import ScheduleIntervals
from pipeline_health import HeartbeatWriter, utc_now
from pipeline_signals import ShutdownFlag
from pipeline_tasks import EXIT_FAILURE, TaskName, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

TaskExecutor = Callable[[TaskName], TaskResult]

MAX_BACKOFF_SECONDS = 1800.0
IDLE_POLL_SECONDS = 5.0
_MAX_BACKOFF_EXPONENT = 32  # 2**32 cadences is far beyond any cap; avoids float overflow


def backoff_delay(
    consecutive_failures: int, base_seconds: float, cap_seconds: float = MAX_BACKOFF_SECONDS
) -> float:
    """Seconds to wait before the next attempt of a task.

    After ``k`` consecutive failures the wait is ``base_seconds * 2**(k - 1)``: the first
    failure keeps the regular cadence, every further one doubles it, up to ``cap_seconds``
    (or the cadence itself when that is longer than the cap).
    """
    exponent = min(max(consecutive_failures - 1, 0), _MAX_BACKOFF_EXPONENT)
    return min(base_seconds * 2**exponent, max(cap_seconds, base_seconds))


def dependent_tasks(bootstrap: bool) -> tuple[TaskName, ...]:
    """Tasks that follow a successful ``dbt run`` within one cycle, in execution order.

    While no production model exists, training precedes prediction so that a freshly promoted
    first model serves predictions in the same cycle. Afterwards prediction goes first: it is
    the latency-sensitive output of the pipeline, retraining is not.
    """
    if bootstrap:
        return (TaskName.DBT_TEST, TaskName.TRAIN, TaskName.PREDICT, TaskName.SNAPSHOT)
    return (TaskName.DBT_TEST, TaskName.PREDICT, TaskName.TRAIN, TaskName.SNAPSHOT)


def model_available_after(result: TaskResult, available: bool) -> bool:
    """Update the belief "a production model exists" from one task result.

    A successful predict or train proves that a production model exists; predict exiting 3
    means there is none (contract section 8). Any other outcome carries no information.
    """
    if result.status is TaskStatus.SUCCESS and result.name in (TaskName.PREDICT, TaskName.TRAIN):
        return True
    if result.status is TaskStatus.SKIPPED and result.name is TaskName.PREDICT:
        return False
    return available


@dataclass
class TaskTracker:
    """Scheduling state of one task."""

    runs: int = 0
    consecutive_failures: int = 0
    last_cycle_at: float | None = None
    last_result: TaskResult | None = None
    last_finished_at: str | None = None

    def record(self, result: TaskResult, cycle_at: float, finished_at: str) -> None:
        """Account for one execution that belonged to the cycle started at ``cycle_at``."""
        self.runs += 1
        self.last_cycle_at = cycle_at
        self.last_result = result
        self.last_finished_at = finished_at
        failed = result.status is TaskStatus.FAILED
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0

    def next_due_at(self, base_seconds: float) -> float:
        """Earliest cycle start at which the task may run again (cadence plus failure backoff)."""
        if self.last_cycle_at is None:
            return -math.inf
        return self.last_cycle_at + backoff_delay(self.consecutive_failures, base_seconds)

    def as_status(self) -> dict[str, object]:
        """JSON-serialisable view for the heartbeat file."""
        result = self.last_result
        return {
            "status": result.status.value if result else None,
            "exit_code": result.exit_code if result else None,
            "duration_seconds": round(result.duration_seconds, 1) if result else None,
            "finished_at": self.last_finished_at,
            "runs": self.runs,
            "consecutive_failures": self.consecutive_failures,
        }


class PipelineScheduler:
    """Runs pipeline cycles until a stop is requested."""

    def __init__(
        self,
        execute: TaskExecutor,
        intervals: ScheduleIntervals,
        shutdown: ShutdownFlag,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], bool] | None = None,
        wall_clock: Callable[[], datetime] = utc_now,
        heartbeat: HeartbeatWriter | None = None,
    ) -> None:
        self._execute = execute
        self._intervals = intervals
        self._shutdown = shutdown
        self._clock = clock
        self._sleep = shutdown.wait if sleep is None else sleep
        self._wall_clock = wall_clock
        self._heartbeat = heartbeat
        self._trackers = {task: TaskTracker() for task in TaskName}
        self._current_task: TaskName | None = None
        self.model_available = False
        self.cycles_completed = 0

    def tracker(self, task: TaskName) -> TaskTracker:
        """Scheduling state of ``task``."""
        return self._trackers[task]

    def base_interval(self, task: TaskName) -> float:
        """Regular cadence of ``task`` in seconds, as seen by this cycle-driven loop."""
        intervals = self._intervals
        bootstrap_training = task is TaskName.TRAIN and not self.model_available
        if task in (TaskName.DBT_RUN, TaskName.DBT_TEST) or bootstrap_training:
            return intervals.dbt_seconds
        own = {
            TaskName.PREDICT: intervals.predict_seconds,
            TaskName.SNAPSHOT: intervals.snapshot_seconds,
            TaskName.TRAIN: intervals.train_seconds,
        }[task]
        # Dependent tasks get one chance per cycle; anchoring their backoff on the cycle
        # length keeps "doubling the wait" meaningful for intervals shorter than a cycle.
        return max(own, intervals.dbt_seconds)

    def is_due(self, task: TaskName, cycle_at: float) -> bool:
        """Whether ``task`` should run in the cycle that started at ``cycle_at``."""
        return cycle_at >= self._trackers[task].next_due_at(self.base_interval(task))

    def seconds_until_next_cycle(self) -> float:
        """Seconds until the next ``dbt run`` is due (0 when it is due now)."""
        due_at = self._trackers[TaskName.DBT_RUN].next_due_at(self.base_interval(TaskName.DBT_RUN))
        return max(0.0, due_at - self._clock())

    def run_cycle(self) -> list[TaskResult]:
        """Run one cycle: ``dbt run``, then every dependent task that is due."""
        cycle_at = self._clock()
        bootstrap = not self.model_available
        mode = "bootstrap: no production model confirmed yet" if bootstrap else "production model available"
        logger.info("cycle %d starting (%s)", self.cycles_completed + 1, mode)
        results = [self._run(TaskName.DBT_RUN, cycle_at)]
        if results[0].status is TaskStatus.FAILED:
            logger.warning("dbt run failed: skipping dbt test, train, predict and snapshot in this cycle")
        else:
            for task in dependent_tasks(bootstrap):
                if self._shutdown.requested:
                    break
                if self.is_due(task, cycle_at):
                    results.append(self._run(task, cycle_at))
        self.cycles_completed += 1
        self._publish("idle")
        return results

    def run_forever(self) -> None:
        """Run cycles until a stop is requested; a stop interrupts the idle wait immediately."""
        intervals = self._intervals
        logger.info(
            "pipeline loop started: dbt every %.0fs, predict every %.0fs, snapshot every %.0fs, "
            "train every %.0fs (after every dbt run while no production model exists)",
            intervals.dbt_seconds,
            intervals.predict_seconds,
            intervals.snapshot_seconds,
            intervals.train_seconds,
        )
        while not self._shutdown.requested:
            wait = self.seconds_until_next_cycle()
            if wait > 0:
                self._publish("idle")
                self._sleep(min(wait, IDLE_POLL_SECONDS))
                continue
            self.run_cycle()
        logger.info(
            "pipeline loop stopped (%s) after %d cycle(s)",
            self._shutdown.reason or "stop requested",
            self.cycles_completed,
        )
        self._publish("stopped")

    def status(self, state: str) -> dict[str, object]:
        """Snapshot of the scheduler state, as written to the heartbeat file."""
        return {
            "state": state,
            "pid": os.getpid(),
            "cycles_completed": self.cycles_completed,
            "production_model_available": self.model_available,
            "current_task": self._current_task.value if self._current_task else None,
            "next_cycle_in_seconds": round(self.seconds_until_next_cycle(), 1),
            "tasks": {task.value: tracker.as_status() for task, tracker in self._trackers.items()},
        }

    def _run(self, task: TaskName, cycle_at: float) -> TaskResult:
        self._current_task = task
        self._publish("running")
        started = self._clock()
        try:
            result = self._execute(task)
        except Exception:
            # A broken launcher (not a failing task) must not take the whole loop down either.
            logger.exception("task %s could not be executed", task)
            result = TaskResult(task, TaskStatus.FAILED, EXIT_FAILURE, self._clock() - started)
        finally:
            self._current_task = None
        tracker = self._trackers[task]
        tracker.record(result, cycle_at, self._wall_clock().isoformat(timespec="seconds"))
        self._update_model_state(result)
        if tracker.consecutive_failures > 1:
            delay = backoff_delay(tracker.consecutive_failures, self.base_interval(task))
            logger.warning(
                "task %s failed %d times in a row: backing off, next attempt in about %.0fs",
                task,
                tracker.consecutive_failures,
                delay,
            )
        return result

    def _update_model_state(self, result: TaskResult) -> None:
        available = model_available_after(result, self.model_available)
        if available and not self.model_available:
            logger.info(
                "production model available: training now runs every %.0fs with --skip-if-unchanged",
                self._intervals.train_seconds,
            )
        elif self.model_available and not available:
            logger.warning("no production model: training is attempted after every dbt run (bootstrap)")
        self.model_available = available

    def _publish(self, state: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat.publish(self.status(state))
