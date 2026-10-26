"""Local-mode Spark test of the supervisor with real streaming queries.

Two rate-source queries stand in for a lake query and ``clickhouse_raw``: the second one's
sink fails once (like a ClickHouse outage). The supervisor must restart only that query,
which replays the failed micro-batch from its checkpoint, while the first keeps committing.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

from streaming_heartbeat import HeartbeatWriter
from streaming_supervisor import EXIT_OK, QueryGroup, QuerySpec, QuerySupervisor

TIMEOUT_S = 120.0
FAILING_BATCH = 2


class BatchRecorder:
    """``foreachBatch`` sink recording (batch_id, time) per attempt; fails once on ``fail_on``."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.attempts: list[tuple[int, float]] = []
        self.committed: list[tuple[int, float]] = []
        self.failed_at: float | None = None
        self._fail_on = fail_on

    def __call__(self, _batch: Any, batch_id: int) -> None:
        now = time.monotonic()
        self.attempts.append((batch_id, now))
        if batch_id == self._fail_on and self.failed_at is None:
            self.failed_at = now
            raise RuntimeError("simulated ClickHouse outage")
        self.committed.append((batch_id, now))


def rate_query(spark: Any, name: str, sink: BatchRecorder, checkpoint: Path) -> Callable[[], Any]:
    def start() -> Any:
        return (
            spark.readStream.format("rate")
            .option("rowsPerSecond", 20)
            .load()
            .writeStream.queryName(name)
            .foreachBatch(sink)
            .option("checkpointLocation", str(checkpoint))
            .trigger(processingTime="200 milliseconds")
            .start()
        )

    return start


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.2)
    return False


def test_failed_query_restarts_from_its_checkpoint_while_the_other_keeps_running(
    spark: Any, tmp_path: Path
) -> None:
    steady, flaky = BatchRecorder(), BatchRecorder(fail_on=FAILING_BATCH)
    groups = [
        QueryGroup(
            "lake", lambda: None, (QuerySpec("steady", rate_query(spark, "steady", steady, tmp_path / "s")),)
        ),
        QueryGroup(
            "clickhouse",
            lambda: None,
            (QuerySpec("flaky", rate_query(spark, "flaky", flaky, tmp_path / "f")),),
        ),
    ]
    heartbeat_file = tmp_path / "streaming.heartbeat"
    stop_event = threading.Event()
    supervisor = QuerySupervisor(
        groups,
        stop_event,
        HeartbeatWriter(heartbeat_file, min_interval_s=0.0),
        poll_s=0.2,
        restart_base_delay_s=1.0,
        restart_max_delay_s=1.0,
    )
    exit_codes: list[int] = []
    thread = threading.Thread(target=lambda: exit_codes.append(supervisor.run()), daemon=True)
    thread.start()
    try:
        recovered = wait_until(lambda: sum(batch_id > FAILING_BATCH for batch_id, _ in flaky.committed) >= 2)
    finally:
        stop_event.set()
        thread.join(TIMEOUT_S)
    assert recovered, f"flaky never recovered: attempts={flaky.attempts}"
    assert exit_codes == [EXIT_OK]

    attempted = [batch_id for batch_id, _ in flaky.attempts]
    retry = attempted.index(FAILING_BATCH) + 1
    assert attempted[retry] == FAILING_BATCH  # the restarted run replayed the failed batch
    assert attempted[retry:] == sorted(set(attempted[retry:]))  # then continued in order
    assert flaky.failed_at is not None
    restarted_at = flaky.attempts[retry][1]
    assert any(flaky.failed_at < committed_at < restarted_at for _, committed_at in steady.committed)

    status = json.loads(heartbeat_file.read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    flaky_state, steady_state = status["queries"]["flaky"], status["queries"]["steady"]
    assert (flaky_state["restarts"], flaky_state["consecutive_failures"]) == (1, 0)
    assert "RuntimeError: simulated ClickHouse outage" in flaky_state["last_error"]
    assert (steady_state["restarts"], steady_state["last_error"]) == (0, None)
    assert steady_state["last_batch_id"] >= 1
