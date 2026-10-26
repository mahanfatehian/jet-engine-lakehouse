"""Command-line behaviour of pipeline_runner with signals, waits and subprocesses replaced by fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pipeline_runner
import pytest
from pipeline_signals import ShutdownFlag
from pipeline_tasks import TaskName, TaskResult, TaskSpec, classify_exit_code

ENV_VARS = (
    "CLICKHOUSE_HOST",
    "CLICKHOUSE_PORT",
    "DBT_PROJECT_DIR",
    "DBT_PROFILES_DIR",
    "DBT_INTERVAL_SECONDS",
    "PREDICT_INTERVAL_SECONDS",
    "SNAPSHOT_INTERVAL_SECONDS",
    "TRAIN_INTERVAL_SECONDS",
)


class CliHarness:
    """Records what main() would have executed; exit codes are scripted per task."""

    def __init__(self) -> None:
        self.exit_codes: dict[TaskName, int] = {}
        self.specs: list[TaskSpec] = []
        self.wait_calls: list[tuple[str, float]] = []
        self.wait_outcome: Any = True
        self.runner_kwargs: list[dict[str, Any]] = []

    def wait_for_clickhouse(self, url: str, shutdown: ShutdownFlag, *, max_wait_seconds: float) -> bool:
        self.wait_calls.append((url, max_wait_seconds))
        if isinstance(self.wait_outcome, Exception):
            raise self.wait_outcome
        if self.wait_outcome is False:
            shutdown.request("SIGTERM")
        return bool(self.wait_outcome)

    def runner_factory(self, shutdown: ShutdownFlag, **kwargs: Any) -> CliHarness:
        self.runner_kwargs.append(kwargs)
        return self

    def run(self, spec: TaskSpec) -> TaskResult:
        self.specs.append(spec)
        code = self.exit_codes.get(spec.name, 0)
        return TaskResult(spec.name, classify_exit_code(code), code, 0.1)

    @property
    def executed(self) -> list[str]:
        return [spec.name.value for spec in self.specs]


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> CliHarness:
    harness = CliHarness()
    monkeypatch.setattr(pipeline_runner, "install_signal_handlers", lambda flag: None)
    monkeypatch.setattr(pipeline_runner, "configure_logging", lambda level: None)
    monkeypatch.setattr(pipeline_runner, "wait_for_clickhouse", harness.wait_for_clickhouse)
    monkeypatch.setattr(pipeline_runner, "SubprocessTaskRunner", harness.runner_factory)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return harness


@pytest.mark.parametrize("exit_code", [0, 3, 2, 124])
def test_task_mode_exits_with_the_tasks_exit_code(cli: CliHarness, exit_code: int) -> None:
    cli.exit_codes[TaskName.PREDICT] = exit_code

    assert pipeline_runner.main(["task", "predict"]) == exit_code
    assert cli.executed == ["predict"]


def test_task_train_does_not_skip_unchanged_data(cli: CliHarness) -> None:
    assert pipeline_runner.main(["task", "train"]) == 0
    assert "--skip-if-unchanged" not in cli.specs[0].argv


def test_task_dbt_runs_dbt_run_then_dbt_test(cli: CliHarness) -> None:
    assert pipeline_runner.main(["task", "dbt"]) == 0
    assert cli.executed == ["dbt-run", "dbt-test"]
    assert [spec.argv[1] for spec in cli.specs] == ["run", "test"]


def test_task_dbt_stops_after_a_failed_dbt_run(cli: CliHarness) -> None:
    cli.exit_codes[TaskName.DBT_RUN] = 2

    assert pipeline_runner.main(["task", "dbt"]) == 2
    assert cli.executed == ["dbt-run"]


def test_once_runs_the_whole_pipeline_in_order(cli: CliHarness) -> None:
    assert pipeline_runner.main(["once"]) == 0
    assert cli.executed == ["dbt-run", "dbt-test", "snapshot", "train", "predict"]


def test_once_treats_skips_as_success(cli: CliHarness) -> None:
    cli.exit_codes.update({TaskName.SNAPSHOT: 3, TaskName.TRAIN: 3, TaskName.PREDICT: 3})

    assert pipeline_runner.main(["once"]) == 0


def test_once_exits_1_when_any_task_fails_but_still_runs_the_rest(cli: CliHarness) -> None:
    cli.exit_codes[TaskName.SNAPSHOT] = 1

    assert pipeline_runner.main(["once"]) == 1
    assert cli.executed == ["dbt-run", "dbt-test", "snapshot", "train", "predict"]


def test_once_skips_everything_after_a_failed_dbt_run(cli: CliHarness) -> None:
    cli.exit_codes[TaskName.DBT_RUN] = 1

    assert pipeline_runner.main(["once"]) == 1
    assert cli.executed == ["dbt-run"]


def test_startup_waits_for_clickhouse_at_the_configured_address(
    cli: CliHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse")

    pipeline_runner.main(["task", "predict"])

    assert cli.wait_calls == [("http://clickhouse:8123/ping", 300.0)]


def test_zero_wait_skips_the_clickhouse_check(cli: CliHarness) -> None:
    assert pipeline_runner.main(["--clickhouse-wait-seconds", "0", "task", "predict"]) == 0
    assert cli.wait_calls == []


def test_unreachable_clickhouse_fails_without_running_tasks(cli: CliHarness) -> None:
    cli.wait_outcome = pipeline_runner.ClickHouseUnavailableError("ClickHouse did not answer")

    assert pipeline_runner.main(["once"]) == 1
    assert cli.specs == []


@pytest.mark.parametrize(("argv", "expected"), [(["loop"], 0), (["once"], 1), (["task", "train"], 1)])
def test_stop_during_the_startup_wait(cli: CliHarness, argv: list[str], expected: int) -> None:
    cli.wait_outcome = False

    assert pipeline_runner.main(argv) == expected
    assert cli.specs == []


def test_invalid_configuration_exits_2(cli: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBT_INTERVAL_SECONDS", "soon")

    assert pipeline_runner.main(["once"]) == 2
    assert cli.specs == []


def test_unknown_task_is_rejected_by_the_parser(cli: CliHarness) -> None:
    with pytest.raises(SystemExit) as excinfo:
        pipeline_runner.main(["task", "deploy"])

    assert excinfo.value.code == 2


def test_loop_wires_scheduler_heartbeat_and_skip_if_unchanged(
    cli: CliHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: dict[str, Any] = {}

    class FakeScheduler:
        def __init__(self, execute: Any, intervals: Any, shutdown: ShutdownFlag, *, heartbeat: Any) -> None:
            built.update(execute=execute, intervals=intervals, heartbeat=heartbeat)

        def run_forever(self) -> None:
            built["execute"](TaskName.TRAIN)

    monkeypatch.setattr(pipeline_runner, "PipelineScheduler", FakeScheduler)
    monkeypatch.setenv("DBT_INTERVAL_SECONDS", "45")
    heartbeat_file = tmp_path / "hb.json"

    assert pipeline_runner.main(["loop", "--heartbeat-file", str(heartbeat_file)]) == 0
    assert cli.specs[0].argv[-1] == "--skip-if-unchanged"
    assert built["heartbeat"].path == heartbeat_file
    assert built["intervals"].dbt_seconds == 45.0
    assert cli.runner_kwargs == [{"on_poll": built["heartbeat"].tick}]


def test_run_sequence_does_not_start_tasks_after_a_stop_request(cli: CliHarness) -> None:
    shutdown = ShutdownFlag()

    def execute(task: TaskName) -> TaskResult:
        shutdown.request("SIGTERM")
        return TaskResult(task, classify_exit_code(0), 0, 0.1)

    results = pipeline_runner.run_sequence(pipeline_runner.ONCE_SEQUENCE, execute, shutdown)

    assert [result.name for result in results] == [TaskName.DBT_RUN]
