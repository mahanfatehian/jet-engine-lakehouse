"""Loop gates: failed dbt tests block train and snapshot; snapshots wait for data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pipeline_health import HeartbeatWriter
from pipeline_tasks import TaskName

RUN, TEST = TaskName.DBT_RUN.value, TaskName.DBT_TEST.value
TRAIN, PREDICT, SNAPSHOT = TaskName.TRAIN.value, TaskName.PREDICT.value, TaskName.SNAPSHOT.value


def test_failed_dbt_test_skips_train_and_snapshot_but_still_predicts(
    scripted: Any, make_scheduler: Any, run_cycles: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_scheduler")
    executor = scripted({TaskName.DBT_TEST: [0, 1, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 3)

    # Cycle 1 would snapshot (never done yet), but its dbt test failed.
    assert history == [
        [RUN, TEST, TRAIN, PREDICT],
        [RUN, TEST, PREDICT],
        [RUN, TEST, PREDICT, SNAPSHOT],
    ]
    assert any(
        message.startswith("dbt test failed (1 time(s) in a row): skipping train and snapshot")
        for message in caplog.messages
    )
    assert "dbt test passed again: train and snapshot are no longer blocked" in caplog.messages
    assert not scheduler.dbt_tests_blocking


def test_failed_dbt_test_blocks_bootstrap_training(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    # No champion yet: the first model would be promoted unconditionally, so this matters most.
    executor = scripted({TaskName.DBT_TEST: [1], TaskName.PREDICT: [3]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 3)

    assert history == [[RUN, TEST, PREDICT]] * 3
    assert TaskName.TRAIN not in executor.calls
    assert not scheduler.model_available


def test_due_training_waits_for_passing_dbt_tests(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    # Training is due in cycle 15 (1800 s), but the tests fail in cycles 15 and 16.
    executor = scripted({TaskName.DBT_TEST: [0] * 15 + [1, 1, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 18)

    assert history[15] == [RUN, TEST, PREDICT]
    assert history[16] == [RUN, TEST, PREDICT]
    assert history[17] == [RUN, TEST, PREDICT, TRAIN, SNAPSHOT]


def test_dbt_test_runs_after_every_dbt_run_without_backoff(
    scripted: Any, make_scheduler: Any, run_cycles: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pipeline_scheduler")
    executor = scripted({TaskName.DBT_TEST: [1]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 5)

    assert all(tasks[:2] == [RUN, TEST] for tasks in history)
    assert scheduler.tracker(TaskName.DBT_TEST).consecutive_failures == 5
    assert not any("dbt-test failed" in message and "backing off" in message for message in caplog.messages)


def test_gate_can_be_disabled(
    scripted: Any, make_scheduler: Any, run_cycles: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pipeline_scheduler")
    executor = scripted({TaskName.DBT_TEST: [1]})
    scheduler = make_scheduler(executor, gate_on_dbt_tests=False)

    history = run_cycles(scheduler, executor, 2)

    assert history == [[RUN, TEST, TRAIN, PREDICT], [RUN, TEST, PREDICT, SNAPSHOT]]
    assert not scheduler.dbt_tests_blocking
    assert any("PIPELINE_GATE_ON_DBT_TESTS=false" in message for message in caplog.messages)


def test_heartbeat_shows_the_blocking_gate(
    clock: Any, scripted: Any, make_scheduler: Any, tmp_path: Path
) -> None:
    heartbeat = HeartbeatWriter(tmp_path / "heartbeat.json", clock=clock)
    executor = scripted({TaskName.DBT_TEST: [1]})

    make_scheduler(executor, heartbeat=heartbeat).run_cycle()

    payload = json.loads(heartbeat.path.read_text(encoding="utf-8"))
    assert payload["dbt_test_gate"] == {"enabled": True, "blocking": True}
    assert payload["tasks"]["dbt-test"]["status"] == "FAILED"


def test_no_snapshot_in_the_cycle_of_the_first_successful_dbt_run(
    scripted: Any, make_scheduler: Any, run_cycles: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="pipeline_scheduler")
    executor = scripted({TaskName.DBT_RUN: [1, 1, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 4)

    assert [index for index, tasks in enumerate(history) if SNAPSHOT in tasks] == [3]
    assert history[2] == [RUN, TEST, TRAIN, PREDICT]
    deferrals = [message for message in caplog.messages if message.startswith("snapshot deferred")]
    assert len(deferrals) == 1


def test_a_skipped_snapshot_is_retried_after_the_next_dbt_run(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    # Exit 3: the dbt models have no rows to export yet.
    executor = scripted({TaskName.SNAPSHOT: [3, 3, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 12)

    # Cycle 3 exports the first snapshot; the next one is 960 s (8 cycles) later.
    assert [index for index, tasks in enumerate(history) if SNAPSHOT in tasks] == [1, 2, 3, 11]


def test_a_failed_snapshot_keeps_its_own_cadence_and_backoff(
    scripted: Any, make_scheduler: Any, run_cycles: Any
) -> None:
    executor = scripted({TaskName.SNAPSHOT: [1, 0]})
    scheduler = make_scheduler(executor)

    history = run_cycles(scheduler, executor, 10)

    assert [index for index, tasks in enumerate(history) if SNAPSHOT in tasks] == [1, 9]
