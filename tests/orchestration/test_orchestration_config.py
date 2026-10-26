"""Environment parsing of the pipeline runner settings."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pipeline_config
import pytest
from pipeline_config import (
    REPO_ROOT,
    ConfigError,
    ScheduleIntervals,
    load_settings,
    resolve_dbt_executable,
)

PYTHON = "/opt/venv/bin/python"


def test_defaults_apply_without_any_environment(tmp_path: Path) -> None:
    settings = load_settings({}, repo_root=tmp_path, python_executable=PYTHON)

    assert settings.repo_root == tmp_path
    assert settings.dbt_project_dir == tmp_path / "analytics"
    assert settings.dbt_profiles_dir == tmp_path / "analytics"
    assert settings.clickhouse_ping_url == "http://localhost:8123/ping"
    assert settings.intervals == ScheduleIntervals(120.0, 60.0, 900.0, 1800.0)
    assert settings.python_executable == PYTHON
    assert settings.lock_dir == Path(tempfile.gettempdir())
    assert settings.gate_on_dbt_tests is True


def test_environment_overrides_every_setting(tmp_path: Path) -> None:
    project_dir = tmp_path / "elsewhere" / "analytics"
    env = {
        "PIPELINE_LOCK_DIR": str(tmp_path / "locks"),
        "PIPELINE_GATE_ON_DBT_TESTS": "false",
        "DBT_PROJECT_DIR": str(project_dir),
        "DBT_PROFILES_DIR": "profiles",
        "CLICKHOUSE_HOST": "clickhouse",
        "CLICKHOUSE_PORT": "18123",
        "DBT_INTERVAL_SECONDS": "30",
        "PREDICT_INTERVAL_SECONDS": "15.5",
        "SNAPSHOT_INTERVAL_SECONDS": "600",
        "TRAIN_INTERVAL_SECONDS": "3600",
    }

    settings = load_settings(env, repo_root=tmp_path, python_executable=PYTHON)

    assert settings.dbt_project_dir == project_dir
    assert settings.dbt_profiles_dir == tmp_path / "profiles"
    assert settings.clickhouse_ping_url == "http://clickhouse:18123/ping"
    assert settings.intervals == ScheduleIntervals(30.0, 15.5, 600.0, 3600.0)
    assert settings.lock_dir == tmp_path / "locks"
    assert settings.gate_on_dbt_tests is False


def test_blank_values_fall_back_to_defaults(tmp_path: Path) -> None:
    env = {
        "CLICKHOUSE_HOST": " ",
        "CLICKHOUSE_PORT": "",
        "DBT_INTERVAL_SECONDS": "",
        "DBT_PROJECT_DIR": "",
        "PIPELINE_LOCK_DIR": "",
        "PIPELINE_GATE_ON_DBT_TESTS": " ",
    }

    settings = load_settings(env, repo_root=tmp_path, python_executable=PYTHON)

    assert settings.clickhouse_ping_url == "http://localhost:8123/ping"
    assert settings.intervals.dbt_seconds == 120.0
    assert settings.dbt_project_dir == tmp_path / "analytics"
    assert settings.lock_dir == Path(tempfile.gettempdir())
    assert settings.gate_on_dbt_tests is True


def test_relative_lock_dir_is_resolved_against_the_repository(tmp_path: Path) -> None:
    settings = load_settings({"PIPELINE_LOCK_DIR": "run/locks"}, repo_root=tmp_path, python_executable=PYTHON)

    assert settings.lock_dir == tmp_path / "run" / "locks"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("YES", True), ("On", True), ("false", False), ("0", False), ("no", False)],
)
def test_dbt_test_gate_flag_spellings(tmp_path: Path, value: str, expected: bool) -> None:
    env = {"PIPELINE_GATE_ON_DBT_TESTS": value}

    assert load_settings(env, repo_root=tmp_path, python_executable=PYTHON).gate_on_dbt_tests is expected


def test_invalid_gate_flag_names_the_variable(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="PIPELINE_GATE_ON_DBT_TESTS"):
        load_settings({"PIPELINE_GATE_ON_DBT_TESTS": "maybe"}, repo_root=tmp_path, python_executable=PYTHON)


@pytest.mark.parametrize("value", ["abc", "0", "-5", "inf", "nan"])
def test_invalid_interval_names_the_variable(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="TRAIN_INTERVAL_SECONDS"):
        load_settings({"TRAIN_INTERVAL_SECONDS": value}, repo_root=tmp_path, python_executable=PYTHON)


@pytest.mark.parametrize("value", ["http", "0", "70000", "8123.5"])
def test_invalid_port_names_the_variable(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="CLICKHOUSE_PORT"):
        load_settings({"CLICKHOUSE_PORT": value}, repo_root=tmp_path, python_executable=PYTHON)


def test_repo_root_is_resolved_relative_to_the_module() -> None:
    assert (REPO_ROOT / "orchestration" / "pipeline_runner.py").is_file()
    assert Path(pipeline_config.__file__).resolve().parent == REPO_ROOT / "orchestration"


def test_dbt_next_to_the_interpreter_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    searched: list[str | None] = []

    def fake_which(command: str, path: str | None = None) -> str | None:
        searched.append(path)
        return f"{path}/{command}" if path else "/usr/bin/dbt"

    monkeypatch.setattr(shutil, "which", fake_which)
    interpreter_dir = tmp_path / "venv" / "bin"

    assert resolve_dbt_executable(str(interpreter_dir / "python")) == f"{interpreter_dir}/dbt"
    assert searched == [str(interpreter_dir)]


def test_dbt_falls_back_to_path_then_bare_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda command, path=None: None if path else "/usr/bin/dbt")
    assert resolve_dbt_executable(PYTHON) == "/usr/bin/dbt"

    monkeypatch.setattr(shutil, "which", lambda command, path=None: None)
    assert resolve_dbt_executable(PYTHON) == "dbt"
