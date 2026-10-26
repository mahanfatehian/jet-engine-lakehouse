"""Lightweight scheduler of the lakehouse pipeline: dbt models -> feature snapshots -> RUL model.

Run from the repository root, or from /app inside the data-platform image::

    python orchestration/pipeline_runner.py loop
    python orchestration/pipeline_runner.py once
    python orchestration/pipeline_runner.py task {dbt,dbt-run,dbt-test,snapshot,train,predict}

Every task is a subprocess. Exit code 3 from a task means "skipped: preconditions not met"
and is logged as SKIPPED, not as a failure. Settings come from the environment variables
of contract section 2 (DBT_*, *_INTERVAL_SECONDS, CLICKHOUSE_HOST/PORT, LOG_LEVEL).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pipeline_config import ConfigError, RunnerSettings, load_settings
from pipeline_health import (
    CLICKHOUSE_WAIT_SECONDS,
    ClickHouseUnavailableError,
    HeartbeatWriter,
    default_heartbeat_path,
    wait_for_clickhouse,
)
from pipeline_process import SubprocessTaskRunner
from pipeline_scheduler import PipelineScheduler, TaskExecutor
from pipeline_signals import ShutdownFlag, install_signal_handlers
from pipeline_tasks import (
    CLI_TASK_CHOICES,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    ONCE_SEQUENCE,
    TaskName,
    TaskResult,
    TaskStatus,
    build_task_spec,
    expand_task,
    once_exit_code,
    task_exit_code,
)

logger = logging.getLogger("pipeline_runner")

EXIT_CONFIG_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the runner."""
    parser = argparse.ArgumentParser(
        prog="pipeline_runner.py",
        description="Schedules dbt, feature snapshots and RUL model training/inference.",
    )
    parser.add_argument(
        "--clickhouse-wait-seconds",
        type=float,
        default=CLICKHOUSE_WAIT_SECONDS,
        help="how long to wait for ClickHouse /ping at startup (default %(default).0f; 0 skips the check)",
    )
    modes = parser.add_subparsers(dest="mode", required=True, metavar="{loop,once,task}")
    loop = modes.add_parser("loop", help="run the schedule until SIGTERM/SIGINT")
    loop.add_argument(
        "--heartbeat-file",
        type=Path,
        default=default_heartbeat_path(),
        help="JSON liveness file rewritten every iteration (default %(default)s)",
    )
    modes.add_parser("once", help="run dbt -> snapshot -> train -> predict once; exit 1 if any task failed")
    task = modes.add_parser("task", help="run one task and exit with its exit code (3 = skipped)")
    task.add_argument("task", choices=CLI_TASK_CHOICES, help="'dbt' runs dbt run, then dbt test")
    return parser


def configure_logging(level_name: str) -> None:
    """Log to stdout with UTC ISO-8601 timestamps; unknown level names fall back to INFO."""
    level = logging.getLevelNamesMapping().get(level_name.strip().upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])


def make_executor(settings: RunnerSettings, runner: SubprocessTaskRunner, *, loop_mode: bool) -> TaskExecutor:
    """Bind task names to their concrete command lines for the given mode."""

    def execute(task: TaskName) -> TaskResult:
        return runner.run(build_task_spec(task, settings, loop_mode=loop_mode))

    return execute


def run_sequence(
    tasks: Sequence[TaskName], execute: TaskExecutor, shutdown: ShutdownFlag
) -> list[TaskResult]:
    """Run ``tasks`` in order; a failed ``dbt run`` or a stop request ends the sequence early."""
    results: list[TaskResult] = []
    for index, task in enumerate(tasks):
        if shutdown.requested:
            logger.warning("stop requested: not starting %s", _names(tasks[index:]))
            break
        result = execute(task)
        results.append(result)
        remaining = tasks[index + 1 :]
        if task is TaskName.DBT_RUN and result.status is TaskStatus.FAILED and remaining:
            logger.error("dbt run failed: skipping %s", _names(remaining))
            break
    return results


def run_loop(settings: RunnerSettings, shutdown: ShutdownFlag, heartbeat_path: Path) -> int:
    """``loop`` mode: schedule tasks until SIGTERM/SIGINT, then exit 0."""
    heartbeat = HeartbeatWriter(heartbeat_path)
    runner = SubprocessTaskRunner(shutdown, on_poll=heartbeat.tick)
    execute = make_executor(settings, runner, loop_mode=True)
    PipelineScheduler(execute, settings.intervals, shutdown, heartbeat=heartbeat).run_forever()
    return EXIT_SUCCESS


def run_once(settings: RunnerSettings, shutdown: ShutdownFlag) -> int:
    """``once`` mode: dbt run + test, snapshot, train, predict; 0 only if nothing failed."""
    runner = SubprocessTaskRunner(shutdown)
    results = run_sequence(ONCE_SEQUENCE, make_executor(settings, runner, loop_mode=False), shutdown)
    summary = ", ".join(f"{result.name}={result.status}" for result in results) or "no task ran"
    logger.info("once finished: %s", summary)
    return once_exit_code(results, len(ONCE_SEQUENCE))


def run_task(settings: RunnerSettings, shutdown: ShutdownFlag, task_name: str) -> int:
    """``task`` mode: run one (or, for ``dbt``, two) tasks and return the task's exit code."""
    tasks = expand_task(task_name)
    runner = SubprocessTaskRunner(shutdown)
    results = run_sequence(tasks, make_executor(settings, runner, loop_mode=False), shutdown)
    return task_exit_code(results, len(tasks))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        settings = load_settings(os.environ)
    except ConfigError as exc:
        logger.error("invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    if not _clickhouse_ready(settings, shutdown, args.clickhouse_wait_seconds):
        # A stop during the startup wait is a clean shutdown for the long-running loop only.
        stopped_cleanly = args.mode == "loop" and shutdown.requested
        return EXIT_SUCCESS if stopped_cleanly else EXIT_FAILURE
    if args.mode == "loop":
        return run_loop(settings, shutdown, args.heartbeat_file)
    if args.mode == "once":
        return run_once(settings, shutdown)
    return run_task(settings, shutdown, args.task)


def _clickhouse_ready(settings: RunnerSettings, shutdown: ShutdownFlag, max_wait_seconds: float) -> bool:
    if max_wait_seconds <= 0:
        return True
    try:
        if wait_for_clickhouse(settings.clickhouse_ping_url, shutdown, max_wait_seconds=max_wait_seconds):
            return True
    except ClickHouseUnavailableError as exc:
        logger.error("%s", exc)
        return False
    logger.warning("stop requested while waiting for ClickHouse")
    return False


def _names(tasks: Sequence[TaskName]) -> str:
    return ", ".join(task.value for task in tasks)


if __name__ == "__main__":
    sys.exit(main())
