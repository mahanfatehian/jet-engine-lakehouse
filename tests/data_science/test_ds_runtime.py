"""Exit-code convention, bounded retries, transient-error classification, the ClickHouse gateway's
probe, statements and inserts, and settings."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import ds_runtime
import lakehouse_io
import pytest
from artifact_store import _is_transient_s3_error
from botocore.exceptions import ClientError, EndpointConnectionError
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from ds_runtime import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_SKIPPED,
    PreconditionsNotMetError,
    StopToken,
    TerminationRequestedError,
    retry_call,
    run_cli,
)
from ds_settings import ClickHouseSettings, ConfigError, PromotionSettings, load_settings
from lakehouse_io import ClickHouseGateway, _is_transient, final_modifier
from model_registry import ML_MODEL_RUNS_TABLE
from predict_rul import PREDICTIONS_TABLE
from reconcile_lakehouse import RESULTS_TABLE
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProtocolError, ReadTimeoutError

CLICKHOUSE = ClickHouseSettings(
    host="clickhouse", port=8123, user="default", password="", database="iot_engine"
)
CLICKHOUSE_DDL = Path(__file__).resolve().parents[2] / "infrastructure" / "clickhouse" / "init-db.sql"
DEDUPLICATION_WINDOW = r"non_replicated_deduplication_window = [1-9]\d*"


def _raise(exc: Exception) -> int:
    raise exc


def _transport_error(reason: Exception) -> OperationalError:
    """An HTTP failure as clickhouse-connect raises it: an OperationalError chained to urllib3's error."""
    error = OperationalError("Error executing HTTP request")
    error.__cause__ = reason
    return error


def _connection_refused() -> OperationalError:
    # urllib3 gave up connecting: nothing was sent.
    return _transport_error(MaxRetryError(None, "/", NewConnectionError(None, "Connection refused")))


class FakeClickHouseClient:
    """clickhouse-connect client stand-in: scripted ping() answers and statement outcomes, calls recorded."""

    def __init__(self, pings: list[bool] | None = None, outcomes: list[object] | None = None) -> None:
        self.pings = list(pings or [])
        self.outcomes = list(outcomes or [])
        self.ping_calls = 0
        self.commands: list[tuple[str, dict]] = []
        self.inserts: list[tuple[str, list, dict]] = []

    def ping(self) -> bool:
        # Like clickhouse-connect: a failed probe returns False, it never raises.
        self.ping_calls += 1
        return self.pings.pop(0)

    def command(self, cmd: str, parameters: dict | None = None) -> object:
        self.commands.append((cmd, parameters))
        return self._outcome()

    def insert(self, table: str, data: list, column_names: list[str], settings: dict | None = None) -> None:
        self.inserts.append((table, data, settings))
        self._outcome()

    def _outcome(self) -> object:
        outcome = self.outcomes.pop(0) if self.outcomes else ""
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        pass


@pytest.fixture
def backoffs(monkeypatch) -> list[float]:
    """The waits of retry_call, recorded instead of slept."""
    waits: list[float] = []
    monkeypatch.setattr(ds_runtime.time, "sleep", waits.append)
    return waits


@pytest.fixture
def gateway_over(monkeypatch) -> Callable[[FakeClickHouseClient], ClickHouseGateway]:
    """A ClickHouseGateway (3 attempts) whose clickhouse-connect client is the given fake."""

    def build(client: FakeClickHouseClient) -> ClickHouseGateway:
        monkeypatch.setattr(lakehouse_io.clickhouse_connect, "get_client", lambda **_settings: client)
        return ClickHouseGateway(CLICKHOUSE, attempts=3)

    return build


def test_exit_codes_follow_the_contract() -> None:
    assert (EXIT_OK, EXIT_FAILURE, EXIT_SKIPPED, EXIT_INTERRUPTED) == (0, 1, 3, 143)


@pytest.mark.parametrize(
    ("main", "code"),
    [
        (lambda: EXIT_OK, EXIT_OK),
        (lambda: _raise(PreconditionsNotMetError("no data yet")), EXIT_SKIPPED),
        (lambda: _raise(TerminationRequestedError("stop")), EXIT_INTERRUPTED),
        (lambda: _raise(RuntimeError("boom")), EXIT_FAILURE),
    ],
)
def test_run_cli_maps_outcomes_to_exit_codes(main, code: int) -> None:
    with pytest.raises(SystemExit) as exited:
        run_cli(main)

    assert exited.value.code == code


def test_retry_call_retries_transient_errors_a_bounded_number_of_times() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "ok"

    assert retry_call(flaky, description="flaky", is_retryable=lambda _: True, base_delay_seconds=0.0) == "ok"
    assert len(calls) == 3

    calls.clear()
    with pytest.raises(ConnectionError):
        retry_call(
            flaky, description="flaky", is_retryable=lambda _: True, attempts=2, base_delay_seconds=0.0
        )
    assert len(calls) == 2


def test_retry_call_does_not_retry_permanent_errors() -> None:
    calls: list[int] = []

    def broken() -> None:
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        retry_call(broken, description="broken", is_retryable=lambda _: False, base_delay_seconds=0.0)
    assert len(calls) == 1


def test_transient_error_classification() -> None:
    assert _is_transient(OperationalError("connection refused"))
    assert _is_transient(DatabaseError("too many queries", code=202))
    assert not _is_transient(DatabaseError("unknown table", code=60))
    assert _is_transient_s3_error(EndpointConnectionError(endpoint_url="http://minio:9000"))
    slow_down = {"Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503}}
    assert _is_transient_s3_error(ClientError(slow_down, "PutObject"))
    denied = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}
    assert not _is_transient_s3_error(ClientError(denied, "PutObject"))


def test_ping_retries_an_unanswered_probe_and_reports_false_after_the_last_attempt(
    gateway_over, backoffs: list[float]
) -> None:
    restarting = FakeClickHouseClient(pings=[False, False, True])
    down = FakeClickHouseClient(pings=[False, False, False])

    assert gateway_over(restarting).ping() is True
    assert restarting.ping_calls == 3 and len(backoffs) == 2
    assert gateway_over(down).ping() is False
    assert down.ping_calls == 3 and len(backoffs) == 4


def test_command_passes_the_statement_through_and_retries_what_never_ran(
    gateway_over, backoffs: list[float]
) -> None:
    statement = "ALTER TABLE ml_model_runs DELETE WHERE model_version = {version:String}"
    client = FakeClickHouseClient(
        outcomes=[_connection_refused(), DatabaseError("too many simultaneous queries", code=202), ""]
    )
    gateway = gateway_over(client)

    assert gateway.command(statement, {"version": "rul-1"}) is None
    gateway.command("OPTIMIZE TABLE ml_model_runs FINAL")

    # Refused connections and admission control never started the statement: safe to send again.
    retried = [(statement, {"version": "rul-1"})] * 3
    assert client.commands == [*retried, ("OPTIMIZE TABLE ml_model_runs FINAL", {})]
    assert len(backoffs) == 2


@pytest.mark.parametrize(
    "failure",
    [
        _transport_error(ReadTimeoutError(None, "/", "Read timed out. (read timeout=600)")),
        _transport_error(ProtocolError("Connection aborted.", ConnectionResetError(104, "reset by peer"))),
        DatabaseError("memory limit exceeded", code=241),
        DatabaseError("timeout exceeded", code=159),
        DatabaseError("unknown table", code=60),
    ],
    ids=["read-timeout", "connection-reset", "memory-limit", "timeout-exceeded", "unknown-table"],
)
def test_command_never_reruns_a_statement_that_may_have_run(
    gateway_over, backoffs: list[float], failure: Exception
) -> None:
    client = FakeClickHouseClient(outcomes=[failure, ""])

    with pytest.raises(type(failure)):
        gateway_over(client).command("INSERT INTO gold_engine_rul_predictions SELECT * FROM staged")

    # The server may already have applied it: a second run would insert every row twice.
    assert len(client.commands) == 1 and backoffs == []


def test_an_insert_retried_after_a_read_timeout_keeps_its_deduplication_token(
    gateway_over, backoffs: list[float]
) -> None:
    timed_out = _transport_error(ReadTimeoutError(None, "/", "Read timed out. (read timeout=600)"))
    client = FakeClickHouseClient(outcomes=[timed_out])
    gateway = gateway_over(client)
    rows, columns = [("JT9D-001", 42.0)], ["engine_id", "predicted_rul_cycles"]

    gateway.insert_rows(PREDICTIONS_TABLE, rows, columns)
    gateway.insert_rows(PREDICTIONS_TABLE, rows, columns)

    # The timed-out attempt may have stored the batch: its retry carries the same token, so the table
    # drops the copy. The second call is a new batch with identical rows and gets a token of its own.
    tokens = [settings["insert_deduplication_token"] for _, _, settings in client.inserts]
    assert len(tokens) == 3 and tokens[0] == tokens[1] != tokens[2]
    assert all(data == rows for _, data, _ in client.inserts) and len(backoffs) == 1


def test_every_table_the_jobs_insert_into_has_a_deduplication_window() -> None:
    ddl = CLICKHOUSE_DDL.read_text(encoding="utf-8")

    for table in (PREDICTIONS_TABLE, ML_MODEL_RUNS_TABLE, RESULTS_TABLE):
        create = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\b.*?;", ddl, re.DOTALL)
        assert create and re.search(DEDUPLICATION_WINDOW, create.group(0)), f"{table}: no window on CREATE"
        # init-db.sql runs on every ClickHouse start: tables created before the window get it too.
        assert re.search(rf"ALTER TABLE {table} MODIFY SETTING {DEDUPLICATION_WINDOW};", ddl), table


def test_final_is_only_applied_to_engines_that_merge_versions() -> None:
    assert final_modifier("ReplacingMergeTree") == "FINAL"
    assert final_modifier("ReplicatedReplacingMergeTree") == "FINAL"
    assert final_modifier("MergeTree") == ""
    assert final_modifier("View") == ""


def test_stop_token_wakes_waiters_and_raises_at_safe_points() -> None:
    stop = StopToken()
    assert not stop.wait(0.01)
    stop.raise_if_stopped()

    stop.request_stop()

    assert stop.stop_requested and stop.wait(10)
    with pytest.raises(TerminationRequestedError):
        stop.raise_if_stopped()


def test_settings_defaults_follow_the_contract() -> None:
    settings = load_settings({})

    assert (settings.clickhouse.host, settings.clickhouse.port) == ("localhost", 8123)
    assert (settings.clickhouse.user, settings.clickhouse.database) == ("clickhouse", "iot_engine")
    assert settings.s3.endpoint_url == "http://localhost:9000"
    assert (settings.s3.bucket, settings.s3.region) == ("lakehouse", "us-east-1")
    model = settings.model
    assert (model.failure_horizon_cycles, model.rul_cap_cycles, model.min_training_lifecycles) == (25, 125, 6)
    assert (model.flight_hours_per_cycle, model.promotion_policy) == (2.0, "better")
    assert model.artifact_prefix == "ml-artifacts/rul_model"
    assert model.training_max_rows == 600_000
    assert (model.promotion.min_r2, model.promotion.min_roc_auc, model.promotion.f1_tolerance) == (
        0.5,
        0.75,
        0.05,
    )
    assert settings.snapshot_retention == 5
    assert settings.snapshot_window_days == 7
    assert model.spectral_max_lag_seconds == 60.0
    assert "password" not in repr(settings.clickhouse)
    assert "secret_access_key" not in repr(settings.s3)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RUL_CAP_CYCLES": "abc"}, "integer"),
        ({"MIN_TRAINING_LIFECYCLES": "0"}, ">= 1"),
        ({"FLIGHT_HOURS_PER_CYCLE": "-1"}, "> 0"),
        ({"MODEL_PROMOTION_POLICY": "sometimes"}, "better, always"),
        ({"RUL_CAP_CYCLES": "20", "RUL_FAILURE_HORIZON_CYCLES": "25"}, "must exceed"),
        ({"ML_ARTIFACT_PREFIX": "///"}, "must not be empty"),
        ({"TRAINING_MAX_ROWS": "999"}, ">= 1000"),
        ({"SNAPSHOT_WINDOW_DAYS": "0"}, ">= 1"),
        ({"PROMOTION_MIN_R2": "1.5"}, r"PROMOTION_MIN_R2 must be a finite number in \[-inf, 1\]"),
        ({"PROMOTION_MIN_R2": "-inf"}, "finite"),
        ({"PROMOTION_MIN_ROC_AUC": "nan"}, "PROMOTION_MIN_ROC_AUC"),
        ({"PROMOTION_MIN_ROC_AUC": "high"}, "must be a number"),
        ({"PROMOTION_F1_TOLERANCE": "-0.1"}, r"\[0, 1\]"),
        (
            {"SPECTRAL_MAX_LAG_SECONDS": "-1"},
            r"SPECTRAL_MAX_LAG_SECONDS must be a finite number in \[0, 3600\]",
        ),
        ({"SPECTRAL_MAX_LAG_SECONDS": "7200"}, r"\[0, 3600\]"),
        ({"SPECTRAL_MAX_LAG_SECONDS": "nan"}, "SPECTRAL_MAX_LAG_SECONDS"),
    ],
)
def test_invalid_settings_are_rejected(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings(env)


def test_promotion_and_window_settings_are_read_from_the_environment() -> None:
    settings = load_settings(
        {
            "TRAINING_MAX_ROWS": "50000",
            "SNAPSHOT_WINDOW_DAYS": "2",
            "PROMOTION_MIN_R2": "-0.5",
            "PROMOTION_MIN_ROC_AUC": "0.9",
            "PROMOTION_F1_TOLERANCE": "0",
            "SPECTRAL_MAX_LAG_SECONDS": "0",
        }
    )

    assert settings.model.training_max_rows == 50_000 and settings.snapshot_window_days == 2
    assert settings.model.promotion == PromotionSettings(min_r2=-0.5, min_roc_auc=0.9, f1_tolerance=0.0)
    assert settings.model.spectral_max_lag_seconds == 0.0
    assert load_settings({"SPECTRAL_MAX_LAG_SECONDS": "120"}).model.spectral_max_lag_seconds == 120.0
