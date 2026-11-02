"""Exit-code mapping, command construction and aggregate exit codes of the task catalogue."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from pipeline_config import REPO_ROOT, RunnerSettings
from pipeline_scheduler import SCHEDULED_TASKS, dependent_tasks
from pipeline_tasks import (
    CLI_TASK_CHOICES,
    EXIT_DATA_MISMATCH,
    GATED_BY_DBT_TESTS,
    ONCE_SEQUENCE,
    TaskName,
    TaskResult,
    TaskStatus,
    build_task_spec,
    classify_exit_code,
    describe_exit_code,
    expand_task,
    normalize_returncode,
    once_exit_code,
    task_exit_code,
)

PYTHON = "/opt/venv/bin/python"


def _result(task: TaskName, exit_code: int) -> TaskResult:
    return TaskResult(task, classify_exit_code(exit_code), exit_code, 1.0)


@pytest.mark.parametrize(
    ("exit_code", "status"),
    [
        (0, TaskStatus.SUCCESS),
        (3, TaskStatus.SKIPPED),
        (1, TaskStatus.FAILED),
        (2, TaskStatus.FAILED),
        (65, TaskStatus.FAILED),
        (75, TaskStatus.FAILED),
        (124, TaskStatus.FAILED),
        (127, TaskStatus.FAILED),
        (143, TaskStatus.FAILED),
    ],
)
def test_exit_code_classification(exit_code: int, status: TaskStatus) -> None:
    assert classify_exit_code(exit_code) is status


@pytest.mark.parametrize(
    ("exit_code", "description"),
    [
        (65, "exit code 65: a data layer disagrees with the source"),
        (127, "exit code 127: the command could not be started"),
        (1, "exit code 1"),
        (2, "exit code 2"),
    ],
)
def test_exit_codes_with_a_fixed_meaning_are_spelled_out(exit_code: int, description: str) -> None:
    assert describe_exit_code(exit_code) == description


def test_the_reconciliation_job_signals_mismatches_with_the_runners_exit_code() -> None:
    script = REPO_ROOT / "data_science" / "reconcile_lakehouse.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert constants["EXIT_DATA_MISMATCH"] == EXIT_DATA_MISMATCH == 65


def test_the_restore_job_reuses_the_reconciliations_exit_codes() -> None:
    # restore_from_lake imports EXIT_DATA_MISMATCH and the skip convention instead of its own.
    tree = ast.parse((REPO_ROOT / "data_science" / "restore_from_lake.py").read_text(encoding="utf-8"))
    imported = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("reconcile_lakehouse", "EXIT_DATA_MISMATCH") in imported
    assert ("ds_runtime", "PreconditionsNotMetError") in imported
    assert ("ds_runtime", "run_cli") in imported


def test_restore_raw_runs_on_demand_only() -> None:
    restore = TaskName.RESTORE_RAW

    assert restore not in SCHEDULED_TASKS
    assert restore not in ONCE_SEQUENCE
    assert restore not in dependent_tasks(bootstrap=True) and restore not in dependent_tasks(bootstrap=False)
    assert restore not in GATED_BY_DBT_TESTS


def test_make_restore_raw_runs_the_task_in_a_one_off_orchestrator_with_its_arguments() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    recipe = re.search(r"^restore-raw: ## .+\n\t(.+)$", makefile, re.MULTILINE)
    assert recipe and recipe.group(1) == "$(RUN_TASK) restore-raw $(TASK_ARGS)"
    phony = re.search(r"^\.PHONY:(.*?)\n\n", makefile, re.MULTILINE | re.DOTALL)
    assert phony and "restore-raw" in phony.group(1).split()


@pytest.mark.parametrize(("returncode", "expected"), [(0, 0), (2, 2), (3, 3), (-15, 143), (-9, 137)])
def test_signal_deaths_map_to_shell_exit_codes(returncode: int, expected: int) -> None:
    assert normalize_returncode(returncode) == expected


def test_dbt_commands_pass_project_and_profiles_dirs(settings: RunnerSettings) -> None:
    run = build_task_spec(TaskName.DBT_RUN, settings)
    test = build_task_spec(TaskName.DBT_TEST, settings)
    build = build_task_spec(TaskName.DBT_BUILD, settings)

    dirs = ("--project-dir", str(settings.dbt_project_dir), "--profiles-dir", str(settings.dbt_profiles_dir))
    assert run.argv == ("/opt/venv/bin/dbt", "run", *dirs)
    assert test.argv == ("/opt/venv/bin/dbt", "test", *dirs)
    assert build.argv == ("/opt/venv/bin/dbt", "build", *dirs)
    assert run.cwd == test.cwd == build.cwd == settings.dbt_project_dir
    assert dict(run.env_defaults) == {"DBT_USE_COLORS": "false", "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"}
    assert build.env_defaults == run.env_defaults


def test_python_tasks_run_the_data_science_scripts(settings: RunnerSettings, repo_root: Path) -> None:
    scripts = repo_root / "data_science"

    snapshot = build_task_spec(TaskName.SNAPSHOT, settings)
    train = build_task_spec(TaskName.TRAIN, settings)
    predict = build_task_spec(TaskName.PREDICT, settings)
    reconcile = build_task_spec(TaskName.RECONCILE, settings)
    restore = build_task_spec(TaskName.RESTORE_RAW, settings)

    assert snapshot.argv == (PYTHON, str(scripts / "feature_pipeline.py"), "snapshot")
    assert train.argv == (PYTHON, str(scripts / "train_rul_model.py"))
    assert predict.argv == (PYTHON, str(scripts / "predict_rul.py"), "--mode", "batch")
    assert reconcile.argv == (PYTHON, str(scripts / "reconcile_lakehouse.py"))
    assert restore.argv == (PYTHON, str(scripts / "restore_from_lake.py"))
    assert (REPO_ROOT / "data_science" / "reconcile_lakehouse.py").is_file()
    assert (REPO_ROOT / "data_science" / "restore_from_lake.py").is_file()
    for spec in (snapshot, train, predict, reconcile, restore):
        assert spec.cwd == repo_root
        assert spec.env_defaults == ()


@pytest.mark.parametrize("task", list(TaskName))
def test_extra_arguments_are_appended_to_every_command(settings: RunnerSettings, task: TaskName) -> None:
    plain = build_task_spec(task, settings).argv
    extended = build_task_spec(task, settings, extra_args=("--flag", "value")).argv

    assert extended == (*plain, "--flag", "value")


def test_loop_mode_keeps_extra_arguments_after_skip_if_unchanged(settings: RunnerSettings) -> None:
    argv = build_task_spec(
        TaskName.TRAIN, settings, loop_mode=True, extra_args=("--min-lifecycles", "8")
    ).argv

    assert argv[-3:] == ("--skip-if-unchanged", "--min-lifecycles", "8")


def test_loop_mode_only_adds_skip_if_unchanged_to_training(settings: RunnerSettings) -> None:
    for task in TaskName:
        once_argv = build_task_spec(task, settings).argv
        loop_argv = build_task_spec(task, settings, loop_mode=True).argv
        if task is TaskName.TRAIN:
            assert loop_argv == (*once_argv, "--skip-if-unchanged")
        else:
            assert loop_argv == once_argv


@pytest.mark.parametrize(
    ("task", "minutes"),
    [
        (TaskName.DBT_RUN, 15),
        (TaskName.DBT_TEST, 15),
        (TaskName.DBT_BUILD, 30),
        (TaskName.SNAPSHOT, 10),
        (TaskName.TRAIN, 30),
        (TaskName.PREDICT, 5),
        (TaskName.RECONCILE, 10),
        (TaskName.RESTORE_RAW, 60),
    ],
)
def test_task_timeouts(settings: RunnerSettings, task: TaskName, minutes: int) -> None:
    assert build_task_spec(task, settings).timeout_seconds == minutes * 60


def test_cli_task_names_are_the_contract_ones_plus_dbt_build_reconcile_and_restore_raw() -> None:
    assert CLI_TASK_CHOICES == (
        "dbt",
        "dbt-run",
        "dbt-test",
        "dbt-build",
        "snapshot",
        "train",
        "predict",
        "reconcile",
        "restore-raw",
    )


def test_failed_dbt_tests_gate_training_and_snapshots_only() -> None:
    # Not the reconciliation: it only reads the layers, so it also audits a cycle whose tests failed.
    assert set(GATED_BY_DBT_TESTS) == {TaskName.TRAIN, TaskName.SNAPSHOT}


def test_dbt_expands_to_run_then_test() -> None:
    assert expand_task("dbt") == (TaskName.DBT_RUN, TaskName.DBT_TEST)
    assert expand_task("train") == (TaskName.TRAIN,)
    assert expand_task("dbt-test") == (TaskName.DBT_TEST,)
    assert expand_task("dbt-build") == (TaskName.DBT_BUILD,)


def test_once_sequence_is_dbt_snapshot_train_predict() -> None:
    assert ONCE_SEQUENCE == (
        TaskName.DBT_RUN,
        TaskName.DBT_TEST,
        TaskName.SNAPSHOT,
        TaskName.TRAIN,
        TaskName.PREDICT,
    )


@pytest.mark.parametrize(
    ("exit_codes", "expected"),
    [
        ([0, 0, 0, 0, 0], 0),
        ([0, 0, 3, 3, 3], 0),
        ([0, 1, 0, 0, 0], 1),
        ([0, 0, 0, 2, 0], 1),
        ([0, 0, 0, 0, 124], 1),
        ([2], 1),
        ([0, 0, 0], 1),
    ],
    ids=[
        "all-success",
        "skips-are-fine",
        "dbt-test-failed",
        "train-failed",
        "timeout",
        "dbt-run-failed",
        "cut-short",
    ],
)
def test_once_aggregate_exit_code(exit_codes: list[int], expected: int) -> None:
    results = [_result(task, code) for task, code in zip(ONCE_SEQUENCE, exit_codes, strict=False)]
    assert once_exit_code(results, len(ONCE_SEQUENCE)) == expected


@pytest.mark.parametrize(
    ("exit_codes", "planned", "expected"),
    [
        ([0], 1, 0),
        ([3], 1, 3),
        ([2], 1, 2),
        ([124], 1, 124),
        ([0, 0], 2, 0),
        ([0, 3], 2, 0),
        ([3, 3], 2, 3),
        ([1], 2, 1),
        ([0, 1], 2, 1),
        ([0], 2, 1),
    ],
)
def test_task_exit_code_is_the_tasks_own_code(exit_codes: list[int], planned: int, expected: int) -> None:
    tasks = (TaskName.DBT_RUN, TaskName.DBT_TEST)
    results = [_result(task, code) for task, code in zip(tasks, exit_codes, strict=False)]
    assert task_exit_code(results, planned) == expected
