"""ClickHouse access for Path B: client creation, schema check and retried inserts.

Delivery is at-least-once: when a micro-batch still fails after its retries, the error
fails the streaming query, the container restarts and Spark replays the batch from its
checkpoint. The raw tables are ReplacingMergeTree versioned by ``cdc_lsn`` (or keyed by
it), so replayed rows collapse into one version on merge / ``FINAL``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import (
    DatabaseError,
    DataError,
    InterfaceError,
    NotSupportedError,
    ProgrammingError,
)
from streaming_config import StreamingSettings
from streaming_retry import RetryPolicy, WaitFn, call_with_retry

CLICKHOUSE_RETRY_POLICY = RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=16.0)  # waits 1+2+4+8 s
CONNECT_TIMEOUT_S = 10
SEND_RECEIVE_TIMEOUT_S = 60
CLIENT_NAME = "iot-engine-streaming"

# Client-side data/programming errors fail identically on every attempt.
_PERMANENT_ERRORS = (DataError, ProgrammingError, NotSupportedError, InterfaceError)


class MissingTablesError(RuntimeError):
    """The target database lacks tables created by infrastructure/clickhouse/init-db.sql."""


def is_transient_clickhouse_error(exc: Exception) -> bool:
    """True for errors worth retrying: transport failures and server-side errors."""
    if isinstance(exc, _PERMANENT_ERRORS):
        return False
    return isinstance(exc, (DatabaseError, OSError))


def connect(
    settings: StreamingSettings, wait: WaitFn, policy: RetryPolicy = CLICKHOUSE_RETRY_POLICY
) -> Client:
    """Create an HTTP client; its constructor already queries the server, hence the retries."""
    return call_with_retry(
        lambda: clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_db,
            connect_timeout=CONNECT_TIMEOUT_S,
            send_receive_timeout=SEND_RECEIVE_TIMEOUT_S,
            client_name=CLIENT_NAME,
            # Inserts are sequential and stateless; a server session would only add locking.
            autogenerate_session_id=False,
        ),
        policy=policy,
        is_retryable=is_transient_clickhouse_error,
        description="connecting to ClickHouse",
        wait=wait,
    )


def verify_tables(
    client: Client,
    database: str,
    tables: Iterable[str],
    wait: WaitFn,
    policy: RetryPolicy = CLICKHOUSE_RETRY_POLICY,
) -> None:
    """Fail fast with a clear message when a target table is missing.

    Without this check every micro-batch would fail its inserts and the container would
    crash-loop with a much less obvious error.
    """
    result = call_with_retry(
        lambda: client.query(
            "SELECT name FROM system.tables WHERE database = {database:String}",
            parameters={"database": database},
        ),
        policy=policy,
        is_retryable=is_transient_clickhouse_error,
        description="listing ClickHouse tables",
        wait=wait,
    )
    existing = {row[0] for row in result.result_rows}
    missing = sorted(set(tables) - existing)
    if missing:
        raise MissingTablesError(
            f"ClickHouse database {database!r} lacks table(s) {', '.join(missing)}; "
            "apply infrastructure/clickhouse/init-db.sql"
        )


class ClickHouseWriter:
    """Inserts rows with an explicit column list and bounded, interruptible retries."""

    def __init__(
        self,
        client: Client,
        database: str,
        wait: WaitFn,
        policy: RetryPolicy = CLICKHOUSE_RETRY_POLICY,
    ) -> None:
        self._client = client
        self._database = database
        self._wait = wait
        self._policy = policy

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Insert ``rows`` into ``table`` (no-op when empty); raises once retries are exhausted."""
        if not rows:
            return
        call_with_retry(
            lambda: self._client.insert(table, rows, column_names=list(columns), database=self._database),
            policy=self._policy,
            is_retryable=is_transient_clickhouse_error,
            description=f"insert of {len(rows)} rows into {self._database}.{table}",
            wait=self._wait,
        )
