"""Fixtures shared by the orchestration tests: time and child processes are always simulated."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline_config import RunnerSettings, ScheduleIntervals


class FakeClock:
    """Monotonic clock that only moves when a test (or a fake) advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake monotonic clock."""
    return FakeClock()


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
    )
