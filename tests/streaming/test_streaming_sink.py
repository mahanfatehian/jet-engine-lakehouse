"""Unit tests of the ClickHouse sink: retry classification, inserts and the schema check."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("clickhouse_connect")

import clickhouse_sink
from cdc_schemas import DEAD_LETTER_COLUMNS, DEAD_LETTER_TABLE, ENGINE_TELEMETRY, TABLES
from clickhouse_connect.driver.exceptions import (
    DatabaseError,
    DataError,
    OperationalError,
    ProgrammingError,
)
from streaming_config import load_settings
from streaming_retry import RetryPolicy

FAST_POLICY = RetryPolicy(attempts=3, base_delay_s=0.01, max_delay_s=0.02)


def no_shutdown(_seconds: float) -> bool:
    return False


class FakeClient:
    """Records inserts; fails the first ``failures`` calls with ``error``."""

    def __init__(self, failures: int = 0, error: Exception | None = None, tables: list[str] | None = None):
        self.inserts: list[dict[str, Any]] = []
        self.calls = 0
        self.queries = 0
        self.closed = False
        self._failures = failures
        self._error = error
        self._tables = tables or []

    def insert(self, table: str, data: Any, column_names: Any, database: str) -> None:
        self.calls += 1
        if self.calls <= self._failures:
            assert self._error is not None
            raise self._error
        self.inserts.append(
            {"table": table, "rows": list(data), "columns": column_names, "database": database}
        )

    def query(self, sql: str, parameters: dict[str, str]) -> Any:
        assert "{database:String}" in sql
        assert parameters == {"database": "iot_engine"}
        self.queries += 1
        return type("Result", (), {"result_rows": [(name,) for name in self._tables]})()

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (OperationalError("connection refused"), True),
        (DatabaseError("Code: 252. TOO_MANY_PARTS"), True),
        (ConnectionResetError("reset"), True),
        (DataError("value out of range"), False),
        (ProgrammingError("unknown column"), False),
        (ValueError("bug"), False),
    ],
)
def test_transient_error_classification(error: Exception, transient: bool) -> None:
    assert clickhouse_sink.is_transient_clickhouse_error(error) is transient


def test_insert_uses_explicit_columns_without_ingested_at() -> None:
    client = FakeClient()
    writer = clickhouse_sink.ClickHouseWriter(client, "iot_engine", no_shutdown, FAST_POLICY)
    rows = [tuple(range(len(ENGINE_TELEMETRY.clickhouse_columns)))]
    writer.insert_rows(ENGINE_TELEMETRY.clickhouse_table, ENGINE_TELEMETRY.clickhouse_columns, rows)
    (insert,) = client.inserts
    assert insert["table"] == "raw_engine_telemetry"
    assert insert["database"] == "iot_engine"
    assert insert["columns"] == list(ENGINE_TELEMETRY.clickhouse_columns)
    assert "ingested_at" not in insert["columns"]
    assert insert["rows"] == rows


def test_empty_batches_are_not_sent() -> None:
    client = FakeClient()
    clickhouse_sink.ClickHouseWriter(client, "iot_engine", no_shutdown, FAST_POLICY).insert_rows(
        DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, []
    )
    assert client.calls == 0


def test_transient_failures_are_retried() -> None:
    client = FakeClient(failures=2, error=OperationalError("timeout"))
    waits: list[float] = []
    writer = clickhouse_sink.ClickHouseWriter(
        client, "iot_engine", lambda seconds: waits.append(seconds) or False, FAST_POLICY
    )
    writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, [("t", 0, 1, None, "", "x", "bad")])
    assert client.calls == 3
    assert len(client.inserts) == 1
    assert waits == [0.01, 0.02]


def test_persistent_failure_is_raised_to_fail_the_query() -> None:
    client = FakeClient(failures=99, error=OperationalError("server down"))
    writer = clickhouse_sink.ClickHouseWriter(client, "iot_engine", no_shutdown, FAST_POLICY)
    with pytest.raises(OperationalError, match="server down"):
        writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, [("t", 0, 1, None, "", "x", "bad")])
    assert client.calls == FAST_POLICY.attempts


def test_data_errors_are_not_retried() -> None:
    client = FakeClient(failures=99, error=DataError("cannot serialise"))
    writer = clickhouse_sink.ClickHouseWriter(client, "iot_engine", no_shutdown, FAST_POLICY)
    with pytest.raises(DataError):
        writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, [("t", 0, 1, None, "", "x", "bad")])
    assert client.calls == 1


def test_verify_tables_accepts_complete_schema() -> None:
    required = [table.clickhouse_table for table in TABLES] + [DEAD_LETTER_TABLE]
    client = FakeClient(tables=[*required, "gold_engine_rul_predictions"])
    clickhouse_sink.verify_tables(client, "iot_engine", required, no_shutdown, FAST_POLICY)


def test_verify_tables_names_missing_tables() -> None:
    client = FakeClient(tables=["raw_engine_telemetry"])
    with pytest.raises(clickhouse_sink.MissingTablesError, match="raw_cdc_dead_letter, raw_maintenance_log"):
        clickhouse_sink.verify_tables(
            client,
            "iot_engine",
            ["raw_engine_telemetry", "raw_maintenance_log", DEAD_LETTER_TABLE],
            no_shutdown,
        )


def test_connect_retries_until_the_server_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[dict[str, Any]] = []
    sentinel = object()

    def fake_get_client(**kwargs: Any) -> object:
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise OperationalError("connection refused")
        return sentinel

    monkeypatch.setattr(clickhouse_sink.clickhouse_connect, "get_client", fake_get_client)
    settings = load_settings({"CLICKHOUSE_HOST": "clickhouse", "CLICKHOUSE_PASSWORD": "pw"})
    assert clickhouse_sink.connect(settings, no_shutdown, FAST_POLICY) is sentinel
    assert len(attempts) == 3
    assert attempts[-1]["host"] == "clickhouse"
    assert attempts[-1]["port"] == 8123
    assert attempts[-1]["username"] == "clickhouse"
    assert attempts[-1]["password"] == "pw"
    assert attempts[-1]["database"] == "iot_engine"
    assert attempts[-1]["autogenerate_session_id"] is False


def test_required_tables_are_the_raw_tables_and_the_dead_letters() -> None:
    assert clickhouse_sink.REQUIRED_TABLES == (
        "raw_engine_telemetry",
        "raw_engine_status_registry",
        "raw_maintenance_log",
        DEAD_LETTER_TABLE,
    )


def test_target_connects_once_and_checks_the_tables_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(tables=list(clickhouse_sink.REQUIRED_TABLES))
    connects: list[dict[str, Any]] = []
    monkeypatch.setattr(
        clickhouse_sink.clickhouse_connect, "get_client", lambda **kwargs: connects.append(kwargs) or client
    )
    target = clickhouse_sink.ClickHouseTarget(load_settings({}), no_shutdown)
    target.check_ready()
    target.check_ready()  # e.g. before the restart that follows a ClickHouse outage
    assert (len(connects), client.queries) == (1, 2)
    target.writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, [("t", 0, 1, None, "", "x", "bad")])
    assert [insert["table"] for insert in client.inserts] == [DEAD_LETTER_TABLE]
    target.close()
    assert client.closed


def test_target_makes_one_attempt_per_check(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def refused(**_kwargs: Any) -> object:
        attempts.append(1)
        raise OperationalError("connection refused")

    monkeypatch.setattr(clickhouse_sink.clickhouse_connect, "get_client", refused)
    waits: list[float] = []
    target = clickhouse_sink.ClickHouseTarget(
        load_settings({}), lambda seconds: waits.append(seconds) or False
    )
    with pytest.raises(OperationalError):
        target.check_ready()
    assert (attempts, waits) == ([1], [])  # the supervisor owns the backoff
    with pytest.raises(RuntimeError, match="not connected"):
        _ = target.writer
    target.close()  # nothing to release


def test_target_reports_missing_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(tables=["raw_engine_telemetry"])
    monkeypatch.setattr(clickhouse_sink.clickhouse_connect, "get_client", lambda **_kwargs: client)
    target = clickhouse_sink.ClickHouseTarget(load_settings({}), no_shutdown)
    with pytest.raises(clickhouse_sink.MissingTablesError, match="raw_cdc_dead_letter"):
        target.check_ready()


def test_target_checks_extra_tables_apart_from_the_raw_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(tables=[*clickhouse_sink.REQUIRED_TABLES, "raw_engine_hf_features"])
    monkeypatch.setattr(clickhouse_sink.clickhouse_connect, "get_client", lambda **_kwargs: client)
    target = clickhouse_sink.ClickHouseTarget(load_settings({}), no_shutdown)
    with pytest.raises(RuntimeError, match="not connected"):
        target.check_tables(["raw_engine_hf_features"])
    target.check_ready()  # the CDC tables exist: the CDC query may start
    target.check_tables(["raw_engine_hf_features"])
    with pytest.raises(clickhouse_sink.MissingTablesError, match="hf_ingest_batches"):
        target.check_tables(["raw_engine_hf_features", "hf_ingest_batches"])
    assert client.queries == 3
