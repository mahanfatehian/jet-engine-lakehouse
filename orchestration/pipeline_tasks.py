"""Task catalogue of the pipeline runner: names, command lines, timeouts and exit codes.

Every task runs as a separate process, so a crash, memory leak or hung query in dbt or in
the data-science scripts can never take the scheduler down with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pipeline_config import RunnerSettings

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_SKIPPED = 3  # contract: "skipped: preconditions not met", never a crash
EXIT_LOCK_TIMEOUT = 75  # sysexits EX_TEMPFAIL: another run held the task's lock for too long
EXIT_TIMEOUT = 124  # same convention as coreutils `timeout`
EXIT_SPAWN_FAILED = 127  # shell convention for "command not found / not executable"


class TaskName(StrEnum):
    """Atomic tasks the runner executes as subprocesses."""

    DBT_RUN = "dbt-run"
    DBT_TEST = "dbt-test"
    DBT_BUILD = "dbt-build"
    SNAPSHOT = "snapshot"
    TRAIN = "train"
    PREDICT = "predict"


class TaskStatus(StrEnum):
    """Classified outcome of one task execution."""

    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


DBT_COMPOSITE = "dbt"
CLI_TASK_CHOICES: tuple[str, ...] = (DBT_COMPOSITE, *(task.value for task in TaskName))
ONCE_SEQUENCE: tuple[TaskName, ...] = (
    TaskName.DBT_RUN,
    TaskName.DBT_TEST,
    TaskName.SNAPSHOT,
    TaskName.TRAIN,
    TaskName.PREDICT,
)
# Tasks that must not consume dbt output whose tests failed: training would learn from it
# (and could promote the result), a snapshot would publish it as the newest lake dataset.
# Prediction keeps serving the existing production model.
GATED_BY_DBT_TESTS: frozenset[TaskName] = frozenset({TaskName.TRAIN, TaskName.SNAPSHOT})

DBT_SUBCOMMANDS: dict[TaskName, str] = {
    TaskName.DBT_RUN: "run",
    TaskName.DBT_TEST: "test",
    TaskName.DBT_BUILD: "build",
}

TASK_TIMEOUT_SECONDS: dict[TaskName, float] = {
    TaskName.DBT_RUN: 15 * 60,
    TaskName.DBT_TEST: 15 * 60,
    TaskName.DBT_BUILD: 30 * 60,
    TaskName.SNAPSHOT: 10 * 60,
    TaskName.TRAIN: 30 * 60,
    TaskName.PREDICT: 5 * 60,
}

# Applied with setdefault, so an operator can still override them: ANSI colour codes only
# clutter `docker logs`, and usage tracking would try to reach the internet on every run.
DBT_ENV_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("DBT_USE_COLORS", "false"),
    ("DBT_SEND_ANONYMOUS_USAGE_STATS", "false"),
)


@dataclass(frozen=True)
class TaskSpec:
    """A fully resolved command for one task."""

    name: TaskName
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: float
    env_defaults: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one task execution."""

    name: TaskName
    status: TaskStatus
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
    interrupted: bool = False
    output_tail: tuple[str, ...] = ()


def classify_exit_code(exit_code: int) -> TaskStatus:
    """Map a process exit code to a task status: 0 success, 3 skipped, anything else failed."""
    if exit_code == EXIT_SUCCESS:
        return TaskStatus.SUCCESS
    if exit_code == EXIT_SKIPPED:
        return TaskStatus.SKIPPED
    return TaskStatus.FAILED


def normalize_returncode(returncode: int) -> int:
    """Convert a Popen return code to a shell exit code (killed by signal N -> 128 + N)."""
    return 128 - returncode if returncode < 0 else returncode


def expand_task(name: str) -> tuple[TaskName, ...]:
    """Resolve a CLI task name; ``dbt`` stands for ``dbt run`` followed by ``dbt test``."""
    if name == DBT_COMPOSITE:
        return (TaskName.DBT_RUN, TaskName.DBT_TEST)
    return (TaskName(name),)


def build_task_spec(
    task: TaskName,
    settings: RunnerSettings,
    *,
    loop_mode: bool = False,
    extra_args: Sequence[str] = (),
) -> TaskSpec:
    """Build the command line of ``task``.

    ``loop_mode`` adds ``--skip-if-unchanged`` to training, so the scheduled retraining is a
    cheap no-op (exit 3) while no new lifecycle has completed since the last run.
    ``extra_args`` are appended verbatim (``task train -- --promotion-policy always``).
    """
    timeout = TASK_TIMEOUT_SECONDS[task]
    if task in DBT_SUBCOMMANDS:
        argv = (
            settings.dbt_executable,
            DBT_SUBCOMMANDS[task],
            "--project-dir",
            str(settings.dbt_project_dir),
            "--profiles-dir",
            str(settings.dbt_profiles_dir),
            *extra_args,
        )
        # dbt resolves target/ and logs/ against the working directory in some versions.
        return TaskSpec(task, argv, settings.dbt_project_dir, timeout, DBT_ENV_DEFAULTS)
    if task is TaskName.SNAPSHOT:
        argv = _data_science_command(settings, "feature_pipeline.py", "snapshot", *extra_args)
    elif task is TaskName.TRAIN:
        skip = ("--skip-if-unchanged",) if loop_mode else ()
        argv = _data_science_command(settings, "train_rul_model.py", *skip, *extra_args)
    else:
        argv = _data_science_command(settings, "predict_rul.py", "--mode", "batch", *extra_args)
    return TaskSpec(task, argv, settings.repo_root, timeout)


def once_exit_code(results: Sequence[TaskResult], planned: int) -> int:
    """Exit code of ``once``: 0 when all ``planned`` tasks ran and none failed, else 1."""
    completed = len(results) == planned
    failed = any(result.status is TaskStatus.FAILED for result in results)
    return EXIT_SUCCESS if completed and not failed else EXIT_FAILURE


def task_exit_code(results: Sequence[TaskResult], planned: int) -> int:
    """Exit code of ``task``: the task's own code, so callers can tell skipped (3) from failed.

    For the composite ``dbt`` task the first failure wins; a sequence cut short by a stop
    request exits 1, and a sequence in which everything was skipped exits 3.
    """
    for result in results:
        if result.status is TaskStatus.FAILED:
            return result.exit_code
    if len(results) < planned:
        return EXIT_FAILURE
    if results and all(result.status is TaskStatus.SKIPPED for result in results):
        return EXIT_SKIPPED
    return EXIT_SUCCESS


def _data_science_command(settings: RunnerSettings, script: str, *args: str) -> tuple[str, ...]:
    script_path = settings.repo_root / "data_science" / script
    return (settings.python_executable, str(script_path), *args)
