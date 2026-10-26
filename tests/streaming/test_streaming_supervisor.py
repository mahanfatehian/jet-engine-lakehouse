"""Unit tests of the query supervisor with fake queries, gates and clock (no JVM needed).

They cover the independence of the lake and ClickHouse groups, restarts with capped
exponential backoff, the failure count, data-loss reporting, the heartbeat and shutdown.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

import pytest
import streaming_supervisor
from streaming_readiness import DependencyNotReadyError
from streaming_supervisor import (
    EXIT_DRIVER_STOPPED,
    EXIT_OK,
    QueryGroup,
    QuerySpec,
    QuerySupervisor,
    is_kafka_data_loss,
    summarize_error,
)

NOW_TEXT = "2026-10-26T13:00:00.000Z"
QUERY_NAMES = ("clickhouse_raw", "bronze_t", "silver_t")
# Error of a query whose unread offsets were deleted by the retention (Spark 3.5.9 wording).
DATA_LOSS_ERROR = (
    "Job aborted due to stage failure: Task 0 in stage 3.0 failed 1 times, most recent failure: "
    "Lost task 0.0 in stage 3.0 (TID 3) (spark executor driver): java.lang.IllegalStateException: "
    "Cannot fetch offset 100 (GroupId: spark-kafka-source-1, TopicPartition: engine_cdc.public.t-0). \n"
    "Some data may have been lost because they are not available in Kafka any more; either the\n"
    " data was aged out by Kafka or the topic may have been deleted before all the data in the\n"
    " topic was processed. If you don't want your streaming query to fail on such cases, set the\n"
    ' source option "failOnDataLoss" to "false".\n'
    "\tat org.apache.spark.sql.kafka010.KafkaDataConsumer.reportDataLoss(KafkaDataConsumer.scala:1)\n"
)
# What PySpark 3.5 reports for an exception raised inside foreachBatch.
FOREACH_BATCH_ERROR = (
    "An exception was raised by the Python Proxy. Return Message: Traceback (most recent call last):\n"
    '  File "/usr/local/lib/python3.11/site-packages/pyspark/sql/utils.py", line 117, in call\n'
    "    self.func(DataFrame(jdf, wrapped_session_jdf), batch_id)\n"
    "clickhouse_connect.driver.exceptions.OperationalError: connection refused\n"
    "\n=== Streaming Query ===\nIdentifier: clickhouse_raw [id = 1, runId = 2]\n"
    "Logical Plan:\nWriteToMicroBatchDataSourceV1 ForeachBatchSink, failOnDataLoss=true\n"
)


class FakeQuery:
    """One run of a streaming query."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.isActive = True
        self.lastProgress: dict[str, Any] | None = None
        self.error: object = None
        self.stop_calls = 0

    @property
    def status(self) -> dict[str, Any]:
        return {"isTriggerActive": False}

    def exception(self) -> object:
        return self.error

    def stop(self) -> None:
        self.stop_calls += 1
        self.isActive = False

    def fail(self, error: object) -> None:
        self.isActive = False
        self.error = error

    def progress(self, batch_id: int) -> None:
        timestamp = f"2026-10-26T13:00:{batch_id:02d}.000Z"
        self.lastProgress = {"runId": f"run-{id(self)}", "batchId": batch_id, "timestamp": timestamp}


class Starter:
    """``QuerySpec.start`` fake: a new FakeQuery run per start, after ``failures`` failed starts."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.runs: list[FakeQuery] = []
        self.failures = 0

    def __call__(self) -> FakeQuery:
        if self.failures:
            self.failures -= 1
            raise RuntimeError(f"cannot open the checkpoint of {self.name}")
        self.runs.append(FakeQuery(self.name))
        return self.runs[-1]

    @property
    def current(self) -> FakeQuery:
        return self.runs[-1]


class Gate:
    """``QueryGroup.check_ready`` fake."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if not self.ready:
            raise DependencyNotReadyError("ClickHouse: http://clickhouse:8123/ping is unreachable")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHeartbeat:
    def __init__(self) -> None:
        self.published: list[tuple[dict[str, Any], bool]] = []

    def publish(self, status: Mapping[str, object], *, force: bool = False) -> bool:
        self.published.append((copy.deepcopy(dict(status)), force))
        return True

    @property
    def last(self) -> dict[str, Any]:
        return self.published[-1][0]


class StopAfterRounds(threading.Event):
    """A stop request that arrives while the supervisor sleeps after ``rounds`` rounds."""

    def __init__(self, rounds: int) -> None:
        super().__init__()
        self._rounds = rounds

    def wait(self, timeout: float | None = None) -> bool:
        self._rounds -= 1
        if self._rounds <= 0:
            self.set()
        return self.is_set()


class Platform:
    """The real job's shape: group ``clickhouse`` with one query, group ``lake`` with two."""

    def __init__(
        self,
        *,
        clickhouse_ready: bool = True,
        stop_event: threading.Event | None = None,
        driver_alive: Callable[[], bool] = lambda: True,
    ) -> None:
        self.clock = FakeClock()
        self.heartbeat = FakeHeartbeat()
        self.stop_event = stop_event or threading.Event()
        self.gates = {"clickhouse": Gate(clickhouse_ready), "lake": Gate(True)}
        self.starters = {name: Starter(name) for name in QUERY_NAMES}
        groups = [
            QueryGroup("clickhouse", self.gates["clickhouse"], (self._spec("clickhouse_raw"),)),
            QueryGroup("lake", self.gates["lake"], (self._spec("bronze_t"), self._spec("silver_t"))),
        ]
        self.supervisor = QuerySupervisor(
            groups,
            self.stop_event,
            self.heartbeat,
            driver_alive=driver_alive,
            clock=self.clock,
            wall_clock=self.wall_clock,
            poll_s=0.0,
        )

    def wall_clock(self) -> str:
        """UTC text of the fake clock: NOW_TEXT plus the elapsed fake seconds."""
        return f"2026-10-26T13:00:{self.clock.now - 1000.0:06.3f}Z"

    def _spec(self, name: str) -> QuerySpec:
        return QuerySpec(name, self.starters[name], f"recover {name} like this")

    def state(self, name: str) -> dict[str, Any]:
        queries = self.supervisor.snapshot()["queries"]
        assert isinstance(queries, dict)
        return queries[name]

    def runs(self) -> list[int]:
        return [len(self.starters[name].runs) for name in QUERY_NAMES]

    def restart_due(self, name: str) -> None:
        """Advance the clock to the next attempt of ``name`` and run a round."""
        self.clock.advance(self.state(name)["next_attempt_in_s"])
        self.supervisor.step()


def test_each_group_starts_as_soon_as_its_own_dependencies_are_ready() -> None:
    platform = Platform(clickhouse_ready=False)
    platform.supervisor.step()
    assert platform.runs() == [0, 1, 1]
    waiting = platform.state("clickhouse_raw")
    assert (waiting["state"], waiting["active"], waiting["consecutive_failures"]) == ("starting", False, 1)
    assert (
        waiting["last_error"]
        == "dependencies not ready: ClickHouse: http://clickhouse:8123/ping is unreachable"
    )
    assert (waiting["last_error_at"], waiting["next_attempt_in_s"]) == (NOW_TEXT, 2.0)
    assert platform.state("bronze_t")["state"] == "running"


def test_unready_dependencies_are_rechecked_with_capped_exponential_backoff() -> None:
    platform = Platform(clickhouse_ready=False)
    delays = []
    for _ in range(8):
        platform.supervisor.step()
        delays.append(platform.state("clickhouse_raw")["next_attempt_in_s"])
        platform.clock.advance(delays[-1] - 0.1)
        platform.supervisor.step()  # not due yet: no check
        platform.clock.advance(0.1)
    assert delays == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]
    assert platform.gates["clickhouse"].calls == 8
    assert platform.gates["lake"].calls == 1
    platform.gates["clickhouse"].ready = True
    platform.supervisor.step()
    assert platform.state("clickhouse_raw")["state"] == "running"


def test_failed_query_restarts_from_its_checkpoint_while_the_others_keep_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    platform = Platform()
    platform.supervisor.step()
    lake_runs = [platform.starters[name].current for name in ("bronze_t", "silver_t")]
    platform.starters["clickhouse_raw"].current.fail("insert into raw_engine_telemetry failed")
    with caplog.at_level(logging.ERROR):
        platform.supervisor.step()
    assert (
        "query clickhouse_raw failed (attempt 1), restarting it from its checkpoint in 2s: "
        "insert into raw_engine_telemetry failed"
    ) in caplog.text
    assert all(run.isActive and run.stop_calls == 0 for run in lake_runs)
    failed = platform.state("clickhouse_raw")
    assert (failed["state"], failed["active"], failed["consecutive_failures"]) == ("restarting", False, 1)
    platform.clock.advance(1.9)
    platform.supervisor.step()
    assert platform.runs() == [1, 1, 1]  # still backing off
    platform.clock.advance(0.1)
    platform.supervisor.step()
    assert platform.runs() == [2, 1, 1]
    assert (platform.state("clickhouse_raw")["state"], platform.state("clickhouse_raw")["restarts"]) == (
        "running",
        1,
    )


def test_restart_waits_until_the_group_dependencies_are_back() -> None:
    platform = Platform()
    platform.supervisor.step()
    platform.starters["clickhouse_raw"].current.fail("ClickHouse went away")
    platform.gates["clickhouse"].ready = False
    platform.supervisor.step()
    platform.restart_due("clickhouse_raw")
    waiting = platform.state("clickhouse_raw")
    assert (waiting["consecutive_failures"], waiting["next_attempt_in_s"]) == (2, 4.0)
    assert waiting["last_error"].startswith("dependencies not ready: ClickHouse")
    platform.gates["clickhouse"].ready = True
    platform.restart_due("clickhouse_raw")
    assert platform.runs() == [2, 1, 1]


def test_progress_after_a_restart_resets_the_failure_count(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.supervisor.step()
    for _ in range(3):
        platform.starters["bronze_t"].current.fail("S3 write failed")
        platform.supervisor.step()
        platform.restart_due("bronze_t")
    assert platform.state("bronze_t")["consecutive_failures"] == 3
    platform.starters["bronze_t"].current.progress(7)
    with caplog.at_level(logging.INFO):
        platform.supervisor.step()
    healthy = platform.state("bronze_t")
    assert (healthy["consecutive_failures"], healthy["restarts"], healthy["last_batch_id"]) == (0, 3, 7)
    # When the supervisor saw the batch complete, not Spark's trigger start (13:00:07).
    assert healthy["last_progress_at"] == "2026-10-26T13:00:14.000Z"
    assert healthy["last_error"] == "S3 write failed"  # kept for the record
    assert "query bronze_t is healthy again (batch 7) after 3 failed attempt(s)" in caplog.text
    platform.starters["bronze_t"].current.fail("S3 write failed")
    platform.supervisor.step()
    assert platform.state("bronze_t")["next_attempt_in_s"] == 2.0  # the backoff starts over


def test_start_failures_are_retried_with_backoff(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.starters["silver_t"].failures = 2
    with caplog.at_level(logging.ERROR):
        platform.supervisor.step()
    assert (
        "query silver_t could not be started (attempt 1), retrying in 2s: "
        "start failed: cannot open the checkpoint of silver_t"
    ) in caplog.text
    assert platform.runs() == [1, 1, 0]  # its sibling started anyway
    platform.restart_due("silver_t")
    assert platform.state("silver_t")["next_attempt_in_s"] == 4.0
    platform.restart_due("silver_t")
    assert platform.state("silver_t")["state"] == "running"


def test_kafka_data_loss_is_reported_with_its_recovery_steps(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.supervisor.step()
    platform.starters["bronze_t"].current.fail(DATA_LOSS_ERROR)
    with caplog.at_level(logging.ERROR):
        platform.supervisor.step()
    assert "query bronze_t stopped at a gap in the Kafka CDC log" in caplog.text
    assert "recover bronze_t like this" in caplog.text
    assert "Cannot fetch offset 100" in platform.state("bronze_t")["last_error"]
    platform.restart_due("bronze_t")  # retried: e.g. an operator reset the checkpoint meanwhile
    assert platform.runs() == [1, 2, 1]


def test_full_error_is_logged_once_per_failure_streak(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.supervisor.step()
    with caplog.at_level(logging.ERROR):
        for _ in range(2):
            platform.starters["clickhouse_raw"].current.fail(FOREACH_BATCH_ERROR)
            platform.supervisor.step()
            platform.restart_due("clickhouse_raw")
    digest = "Return Message: clickhouse_connect.driver.exceptions.OperationalError: connection refused"
    assert [record.getMessage().endswith(digest) for record in caplog.records].count(True) == 2
    details = [
        record.getMessage() for record in caplog.records if "full error of query" in record.getMessage()
    ]
    assert len(details) == 1
    assert "=== Streaming Query ===" in details[0]


def test_ordinary_failures_do_not_mention_data_loss(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.supervisor.step()
    platform.starters["clickhouse_raw"].current.fail(FOREACH_BATCH_ERROR)
    with caplog.at_level(logging.ERROR):
        platform.supervisor.step()
    assert "gap in the Kafka CDC log" not in caplog.text
    assert platform.state("clickhouse_raw")["last_error"] == (
        "An exception was raised by the Python Proxy. Return Message: "
        "clickhouse_connect.driver.exceptions.OperationalError: connection refused"
    )


def test_every_round_publishes_the_heartbeat_and_state_changes_force_it() -> None:
    platform = Platform(clickhouse_ready=False)
    for _ in range(3):
        platform.supervisor.step()
        platform.clock.advance(0.5)
    assert [force for _status, force in platform.heartbeat.published[-3:]] == [True, False, False]
    status = platform.heartbeat.last
    assert status["status"] == "running"
    assert set(status["queries"]) == set(QUERY_NAMES)
    assert status["queries"]["bronze_t"] == {
        "group": "lake",
        "state": "running",
        "active": True,
        "last_progress_at": None,
        "last_batch_id": None,
        "consecutive_failures": 0,
        "restarts": 0,
        "last_error": None,
        "last_error_at": None,
        "next_attempt_in_s": None,
    }
    assert status["queries"]["clickhouse_raw"]["next_attempt_in_s"] == 1.0


def test_requested_shutdown_drains_and_stops_every_query(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform(stop_event=StopAfterRounds(1))
    with caplog.at_level(logging.INFO):
        assert platform.supervisor.run() == EXIT_OK
    assert [platform.starters[name].current.stop_calls for name in QUERY_NAMES] == [1, 1, 1]
    assert "shutdown requested: draining and stopping 3 queries" in caplog.text
    final = platform.heartbeat.last
    assert final["status"] == "stopped"
    assert {query["state"] for query in final["queries"].values()} == {"stopped"}
    assert {query["next_attempt_in_s"] for query in final["queries"].values()} == {None}


def test_query_ending_during_shutdown_is_not_restarted(caplog: pytest.LogCaptureFixture) -> None:
    platform = Platform()
    platform.supervisor.step()
    platform.starters["clickhouse_raw"].current.fail("insert: shutdown requested while retrying")
    platform.stop_event.set()
    with caplog.at_level(logging.WARNING):
        platform.supervisor.step()
    platform.clock.advance(120.0)
    platform.supervisor.step()
    assert platform.runs() == [1, 1, 1]
    assert (
        "query clickhouse_raw ended during shutdown: insert: shutdown requested while retrying" in caplog.text
    )
    assert platform.state("clickhouse_raw")["consecutive_failures"] == 0


def test_lost_driver_ends_supervision_with_a_failure_code(caplog: pytest.LogCaptureFixture) -> None:
    alive = iter([True, False])
    platform = Platform(driver_alive=lambda: next(alive))
    with caplog.at_level(logging.ERROR):
        assert platform.supervisor.run() == EXIT_DRIVER_STOPPED
    assert "the Spark driver has stopped" in caplog.text
    assert [platform.starters[name].current.stop_calls for name in QUERY_NAMES] == [0, 0, 0]
    assert platform.heartbeat.last["status"] == "stopped"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DATA_LOSS_ERROR, True),
        (
            "Partition t-0's offset was changed from 10 to 0, some data may have been missed. "
            "Some data may have been lost because they are not available in Kafka any more",
            True,
        ),
        ("insert into raw_engine_telemetry failed", False),
        (FOREACH_BATCH_ERROR, False),  # "failOnDataLoss" only appears in the plan dump
    ],
)
def test_kafka_data_loss_detection(error: str, expected: bool) -> None:
    assert is_kafka_data_loss(error) is expected


def test_error_summary_keeps_the_final_exception_of_a_chained_traceback() -> None:
    error = (
        "An exception was raised by the Python Proxy. Return Message: Traceback (most recent call last):\n"
        '  File "socket.py", line 974, in getaddrinfo\n'
        "socket.gaierror: [Errno -5] No address associated with hostname\n"
        "\nThe above exception was the direct cause of the following exception:\n\n"
        "Traceback (most recent call last):\n"
        '  File "streaming_queries.py", line 108, in _collect\n'
        "    return split_collected(table, clickhouse_projection(parsed, table).collect())\n"
        "py4j.protocol.Py4JJavaError: An error occurred while calling o437.collectToPython.\n"
        ": java.lang.IllegalStateException: Cannot fetch offset 2 (GroupId: g, TopicPartition: t-0). \n"
        "Some data may have been lost because they are not available in Kafka any more; either the\n"
        ' source option "failOnDataLoss" to "false".\n'
        "\tat org.apache.spark.sql.kafka010.KafkaDataConsumer.reportDataLoss(KafkaDataConsumer.scala:1)\n"
    )
    assert summarize_error(error) == (
        "An exception was raised by the Python Proxy. Return Message: py4j.protocol.Py4JJavaError: "
        "An error occurred while calling o437.collectToPython. : java.lang.IllegalStateException: "
        "Cannot fetch offset 2 (GroupId: g, TopicPartition: t-0). Some data may have been lost because "
        'they are not available in Kafka any more; either the source option "failOnDataLoss" to "false".'
    )


def test_error_summary_drops_stack_traces_and_is_bounded() -> None:
    data_loss = summarize_error(DATA_LOSS_ERROR)
    assert "java.lang.IllegalStateException: Cannot fetch offset 100" in data_loss
    assert "KafkaDataConsumer" not in data_loss
    assert len(data_loss) <= streaming_supervisor.ERROR_SUMMARY_CHARS
    assert summarize_error("java.lang.IllegalStateException: boom\n\tat Foo.bar(Foo.java:1)") == (
        "java.lang.IllegalStateException: boom"
    )
    assert summarize_error("multi\n  line\tmessage") == "multi line message"
    long = summarize_error("x" * 2000)
    assert len(long) == streaming_supervisor.ERROR_SUMMARY_CHARS
    assert long.endswith("...")
