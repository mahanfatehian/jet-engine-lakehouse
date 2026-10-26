"""Scheduling decisions of the loop over a fake clock with scripted task outcomes."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pipeline_config import ScheduleIntervals
from pipeline_health import HeartbeatWriter
from pipeline_scheduler import (
    IDLE_POLL_SECONDS,
    backoff_delay,
    dependent_tasks,
    model_available_after,
)
from pipeline_signals import ShutdownFlag
from pipeline_tasks import TaskName, TaskResult, TaskStatus, classify_exit_code

RUN, TEST = TaskName.DBT_RUN.value, TaskName.DBT_TEST.value
TRAIN, PREDICT, SNAPSHOT = TaskName.TRAIN.value, TaskName.PREDICT.value, TaskName.SNAPSHOT.value


def test_bootstrap_trains_after_every_dbt_run_until_a_model_exists(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.TRAIN: [3], TaskName.PREDICT: [3]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 3)

    # The snapshot waits for the second successful dbt run (see the snapshot tests).
    assert history == [
        [RUN, TEST, TRAIN, PREDICT],
        [RUN, TEST, TRAIN, PREDICT, SNAPSHOT],
        [RUN, TEST, TRAIN, PREDICT],
    ]
    assert not scheduler.model_available


def test_first_successful_training_ends_bootstrap_and_serves_predictions_at_once(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.TRAIN: [3, 0], TaskName.PREDICT: [3, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 2)

    assert history[1] == [RUN, TEST, TRAIN, PREDICT, SNAPSHOT]
    last_predict = scheduler.tracker(TaskName.PREDICT).last_result
    assert last_predict is not None and last_predict.status is TaskStatus.SUCCESS
    assert scheduler.model_available
    assert run_cycles(scheduler, executor, 1) == [[RUN, TEST, PREDICT]]


def test_steady_state_predicts_before_training_on_their_own_intervals(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted()
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 17)

    # Cycle i starts 120*i s after the first one. Training is due again 1800 s after cycle 0
    # (cycle 15). The first snapshot runs in cycle 1 and then every 900 s rounded up to whole
    # cycles, i.e. 960 s (cycle 9). Predictions run in every cycle.
    assert history[0] == [RUN, TEST, TRAIN, PREDICT]
    assert history[1] == [RUN, TEST, PREDICT, SNAPSHOT]
    assert history[2] == [RUN, TEST, PREDICT]
    assert history[15] == [RUN, TEST, PREDICT, TRAIN]
    assert [index for index, tasks in enumerate(history) if TRAIN in tasks] == [0, 15]
    assert [index for index, tasks in enumerate(history) if SNAPSHOT in tasks] == [1, 9]
    assert all(PREDICT in tasks for tasks in history)


def test_predict_skip_without_model_restarts_bootstrap_in_the_same_cycle(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.PREDICT: [0, 3], TaskName.TRAIN: [0, 3]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 3)

    assert history[0] == [RUN, TEST, TRAIN, PREDICT]
    # predict finds no production model: training is attempted right away, then goes first.
    assert history[1] == [RUN, TEST, PREDICT, TRAIN, SNAPSHOT]
    assert history[2] == [RUN, TEST, TRAIN, PREDICT]
    assert not scheduler.model_available


def test_predict_interval_longer_than_the_dbt_interval(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    intervals = ScheduleIntervals(
        dbt_seconds=120, predict_seconds=300, snapshot_seconds=900, train_seconds=1800
    )
    executor = scripted()
    scheduler = make_scheduler(executor, intervals)

    history = run_cycles(scheduler, executor, 7)

    assert [index for index, tasks in enumerate(history) if PREDICT in tasks] == [0, 3, 6]


def test_failed_dbt_run_skips_dependents_and_the_loop_carries_on(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.DBT_RUN: [1, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 3)

    assert history == [[RUN], [RUN, TEST, TRAIN, PREDICT], [RUN, TEST, PREDICT, SNAPSHOT]]
    assert scheduler.tracker(TaskName.DBT_RUN).consecutive_failures == 0


def test_repeated_dbt_failures_back_off_exponentially_up_to_the_cap(
    clock: Any, scripted: Any, make_scheduler: Any
) -> None:
    executor = scripted({TaskName.DBT_RUN: [1]})
    scheduler = make_scheduler(executor)

    waits = []
    for _ in range(7):
        clock.advance(scheduler.seconds_until_next_cycle())
        cycle_start = clock()
        scheduler.run_cycle()
        waits.append(scheduler.seconds_until_next_cycle() + (clock() - cycle_start))

    assert waits == [120, 240, 480, 960, 1800, 1800, 1800]
    assert executor.calls == [TaskName.DBT_RUN] * 7


def test_a_failing_task_backs_off_without_blocking_the_others(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.PREDICT: [1]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 9)

    assert [index for index, tasks in enumerate(history) if PREDICT in tasks] == [0, 1, 3, 7]
    assert all(tasks[:2] == [RUN, TEST] for tasks in history)
    assert history[0] == [RUN, TEST, TRAIN, PREDICT]
    assert scheduler.tracker(TaskName.PREDICT).consecutive_failures == 4


def test_a_skip_resets_the_failure_streak(scripted: Any, make_scheduler: Any, run_cycles: Any) -> None:
    executor = scripted({TaskName.PREDICT: [1, 1, 3]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 5)

    assert [index for index, tasks in enumerate(history) if PREDICT in tasks] == [0, 1, 3, 4]
    assert scheduler.tracker(TaskName.PREDICT).consecutive_failures == 0


def test_an_exception_while_launching_a_task_counts_as_failure(
    make_scheduler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def execute(task: TaskName) -> TaskResult:
        if task is TaskName.PREDICT:
            raise RuntimeError("cannot fork")
        return TaskResult(task, TaskStatus.SUCCESS, 0, 0.0)

    scheduler = make_scheduler(execute)

    results = scheduler.run_cycle()

    assert results[-1].name is TaskName.PREDICT
    assert (results[-1].status, results[-1].exit_code) == (TaskStatus.FAILED, 1)
    assert "task predict could not be executed" in caplog.text
    assert scheduler.cycles_completed == 1


def test_stop_request_ends_the_cycle_between_tasks(
    clock: Any, scripted: Any, make_scheduler: Any, tmp_path: Path
) -> None:
    shutdown = ShutdownFlag()
    executor = scripted()

    def execute(task: TaskName) -> TaskResult:
        if task is TaskName.DBT_TEST:
            shutdown.request("SIGTERM")
        return executor(task)

    heartbeat = HeartbeatWriter(tmp_path / "heartbeat.json", clock=clock)
    scheduler = make_scheduler(execute, shutdown=shutdown, sleep=lambda seconds: False, heartbeat=heartbeat)

    scheduler.run_forever()

    assert executor.calls == [TaskName.DBT_RUN, TaskName.DBT_TEST]
    assert json.loads(heartbeat.path.read_text(encoding="utf-8"))["state"] == "stopped"


def test_idle_wait_is_chunked_and_interruptible(clock: Any, scripted: Any, make_scheduler: Any) -> None:
    shutdown = ShutdownFlag()
    executor = scripted()
    sleeps: list[float] = []

    def sleep(seconds: float) -> bool:
        sleeps.append(seconds)
        clock.advance(seconds)
        if len(executor.calls) > 5:
            shutdown.request("SIGTERM")
        return shutdown.requested

    make_scheduler(executor, shutdown=shutdown, sleep=sleep).run_forever()

    # 116 s of idle time between the cycles (the first one ran 4 tasks of 1 s each), then
    # one more chunk during which the stop arrives.
    assert executor.calls.count(TaskName.DBT_RUN) == 2
    assert max(sleeps) <= IDLE_POLL_SECONDS
    assert sum(sleeps) == pytest.approx(116 + IDLE_POLL_SECONDS)


def test_heartbeat_reports_task_states(
    clock: Any, scripted: Any, make_scheduler: Any, tmp_path: Path
) -> None:
    def noon() -> datetime:
        return datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    heartbeat = HeartbeatWriter(tmp_path / "heartbeat.json", clock=clock, wall_clock=noon)
    executor = scripted({TaskName.TRAIN: [3], TaskName.PREDICT: [3]})
    scheduler = make_scheduler(executor, heartbeat=heartbeat, wall_clock=noon)

    scheduler.run_cycle()

    payload = json.loads(heartbeat.path.read_text(encoding="utf-8"))
    assert payload["timestamp"] == "2026-10-26T12:00:00+00:00"
    assert payload["state"] == "idle"
    assert payload["cycles_completed"] == 1
    assert payload["production_model_available"] is False
    assert payload["dbt_test_gate"] == {"enabled": True, "blocking": False}
    assert payload["current_task"] is None
    assert payload["next_cycle_in_seconds"] == pytest.approx(116.0)
    assert set(payload["tasks"]) == {RUN, TEST, SNAPSHOT, TRAIN, PREDICT}
    assert payload["tasks"]["dbt-run"]["status"] == "SUCCESS"
    assert payload["tasks"]["snapshot"]["runs"] == 0
    assert payload["tasks"]["train"] == {
        "status": "SKIPPED",
        "exit_code": 3,
        "duration_seconds": 1.0,
        "finished_at": "2026-10-26T12:00:00+00:00",
        "runs": 1,
        "consecutive_failures": 0,
    }


def test_running_task_is_visible_in_the_heartbeat(clock: Any, make_scheduler: Any, tmp_path: Path) -> None:
    heartbeat = HeartbeatWriter(tmp_path / "heartbeat.json", clock=clock)
    seen: dict[str, object] = {}

    def execute(task: TaskName) -> TaskResult:
        seen.setdefault(task.value, json.loads(heartbeat.path.read_text(encoding="utf-8"))["current_task"])
        return TaskResult(task, TaskStatus.SUCCESS, 0, 0.0)

    scheduler = make_scheduler(execute, heartbeat=heartbeat)
    scheduler.run_cycle()
    scheduler.run_cycle()

    assert seen == {task: task for task in (RUN, TEST, TRAIN, PREDICT, SNAPSHOT)}


@pytest.mark.parametrize(
    ("failures", "base", "expected"),
    [
        (0, 120, 120),
        (1, 120, 120),
        (2, 120, 240),
        (3, 120, 480),
        (5, 120, 1800),
        (2, 1800, 1800),
        (3, 3600, 3600),
        (10_000, 120, 1800),
    ],
)
def test_backoff_delay(failures: int, base: float, expected: float) -> None:
    assert backoff_delay(failures, base) == expected


@pytest.mark.parametrize(
    ("task", "exit_code", "before", "after"),
    [
        (TaskName.PREDICT, 0, False, True),
        (TaskName.TRAIN, 0, False, True),
        (TaskName.PREDICT, 3, True, False),
        (TaskName.TRAIN, 3, False, False),
        (TaskName.TRAIN, 3, True, True),
        (TaskName.PREDICT, 1, True, True),
        (TaskName.PREDICT, 1, False, False),
        (TaskName.DBT_RUN, 0, False, False),
    ],
)
def test_model_availability_transitions(task: TaskName, exit_code: int, before: bool, after: bool) -> None:
    result = TaskResult(task, classify_exit_code(exit_code), exit_code, 1.0)
    assert model_available_after(result, before) is after


def test_dependent_task_order() -> None:
    assert [task.value for task in dependent_tasks(bootstrap=True)] == [TEST, TRAIN, PREDICT, SNAPSHOT]
    assert [task.value for task in dependent_tasks(bootstrap=False)] == [TEST, PREDICT, TRAIN, SNAPSHOT]


def test_backoff_warning_is_logged(
    scripted: Any, make_scheduler: Any, run_cycles: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pipeline_scheduler")
    executor = scripted({TaskName.DBT_RUN: [1]})
    scheduler = make_scheduler(executor)

    run_cycles(scheduler, executor, 2)

    assert "task dbt-run failed 2 times in a row: backing off, next attempt in about 240s" in caplog.messages


@pytest.mark.parametrize(
    ("seconds", "expected"), [(60, 120), (120, 120), (121, 240), (900, 960), (1800, 1800)]
)
def test_effective_cadence_is_rounded_up_to_whole_dbt_cycles(
    make_scheduler: Any, seconds: float, expected: float
) -> None:
    assert make_scheduler(lambda task: None).effective_seconds(seconds) == expected


def test_startup_log_states_the_effective_cadences(
    make_scheduler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_scheduler")
    shutdown = ShutdownFlag()
    shutdown.request("SIGTERM")

    make_scheduler(lambda task: None, shutdown=shutdown).run_forever()

    started = next(message for message in caplog.messages if message.startswith("pipeline loop started"))
    assert "predict every 120s, snapshot every 960s, train every 1800s" in started
    assert "=60/900/1800 rounded up to whole dbt cycles" in started
    assert "a failed dbt test skips train and snapshot of its cycle" in started
