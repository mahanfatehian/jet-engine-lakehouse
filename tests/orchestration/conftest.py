"""Fixtures shared by the orchestration tests: time and child processes are always simulated."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pipeline_config import RunnerSettings, ScheduleIntervals
from pipeline_scheduler import PipelineScheduler
from pipeline_signals import ShutdownFlag
from pipeline_tasks import TaskName, TaskResult, classify_exit_code


class FakeClock:
    """Monotonic clock that only moves when a test (or a fake) advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedExecutor:
    """Fake task execution: scripted exit codes per task (the last one repeats), 1 s per task."""

    def __init__(self, clock: FakeClock, script: dict[TaskName, list[int]] | None = None) -> None:
        self.clock = clock
        self.script = {task: list(codes) for task, codes in (script or {}).items()}
        self.calls: list[TaskName] = []

    def __call__(self, task: TaskName) -> TaskResult:
        self.calls.append(task)
        codes = self.script.get(task, [0])
        code = codes.pop(0) if len(codes) > 1 else codes[0]
        self.clock.advance(1.0)
        return TaskResult(task, classify_exit_code(code), code, 1.0)


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake monotonic clock."""
    return FakeClock()


@pytest.fixture
def scripted(clock: FakeClock) -> Callable[..., ScriptedExecutor]:
    """Factory of scripted executors bound to the fake clock."""

    def build(script: dict[TaskName, list[int]] | None = None) -> ScriptedExecutor:
        return ScriptedExecutor(clock, script)

    return build


@pytest.fixture
def make_scheduler(clock: FakeClock) -> Callable[..., PipelineScheduler]:
    """Factory of schedulers on the fake clock (contract default intervals unless given)."""

    def build(
        executor: Any,
        intervals: ScheduleIntervals | None = None,
        shutdown: ShutdownFlag | None = None,
        **kwargs: Any,
    ) -> PipelineScheduler:
        return PipelineScheduler(
            executor, intervals or ScheduleIntervals(), shutdown or ShutdownFlag(), clock=clock, **kwargs
        )

    return build


@pytest.fixture
def run_cycles(clock: FakeClock) -> Callable[[PipelineScheduler, ScriptedExecutor, int], list[list[str]]]:
    """Run N cycles, each as soon as it is due; returns the tasks executed per cycle."""

    def run(scheduler: PipelineScheduler, executor: ScriptedExecutor, count: int) -> list[list[str]]:
        history = []
        for _ in range(count):
            clock.advance(scheduler.seconds_until_next_cycle())
            first_call = len(executor.calls)
            scheduler.run_cycle()
            history.append([task.value for task in executor.calls[first_call:]])
        return history

    return run


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Repository root the task commands are resolved against."""
    return tmp_path / "repo"


@pytest.fixture
def settings(repo_root: Path) -> RunnerSettings:
    """Runner settings with distinct, recognisable paths and the contract default intervals."""
    return RunnerSettings(
        repo_root=repo_root,
        dbt_project_dir=repo_root / "analytics",
        dbt_profiles_dir=repo_root / "profiles",
        clickhouse_host="clickhouse",
        clickhouse_port=8123,
        intervals=ScheduleIntervals(),
        python_executable="/opt/venv/bin/python",
        dbt_executable="/opt/venv/bin/dbt",
        lock_dir=repo_root / "locks",
    )
