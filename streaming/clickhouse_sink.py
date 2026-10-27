"""ClickHouse access for Path B: client creation, schema check and retried inserts.

Delivery is at-least-once: when a micro-batch still fails after its retries, the error
fails the ``clickhouse_raw`` query and the supervisor restarts it once ClickHouse answers
again; Spark then replays the batch from the query's checkpoint. The raw tables are
ReplacingMergeTree versioned by ``cdc_lsn`` (or keyed by it), so replayed rows collapse
into one version on merge / ``FINAL``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import clickhouse_connect
from cdc_schemas import DEAD_LETTER_TABLE, TABLES
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
# Readiness checks make one attempt: the supervisor retries them with its own backoff.
SINGLE_ATTEMPT = RetryPolicy(attempts=1, base_delay_s=1.0, max_delay_s=1.0)
REQUIRED_TABLES: tuple[str, ...] = (*(table.clickhouse_table for table in TABLES), DEAD_LETTER_TABLE)
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


class ClickHouseTarget:
    """Destination of Path B: one client, created by the first successful readiness check.

    The HTTP client is stateless, so the same client and writer serve every run of the
    ``clickhouse_raw`` query, including runs after a ClickHouse restart.
    """

    def __init__(
        self, settings: StreamingSettings, wait: WaitFn, tables: Iterable[str] = REQUIRED_TABLES
    ) -> None:
        self._settings = settings
        self._wait = wait
        self._tables = tuple(tables)
        self._client: Client | None = None
        self._writer: ClickHouseWriter | None = None

    def check_ready(self) -> None:
        """Connect when not connected yet, then check that every raw table exists (one attempt each)."""
        if self._client is None:
            self._client = connect(self._settings, self._wait, SINGLE_ATTEMPT)
            self._writer = ClickHouseWriter(self._client, self._settings.clickhouse_db, self._wait)
        verify_tables(self._client, self._settings.clickhouse_db, self._tables, self._wait, SINGLE_ATTEMPT)

    def check_tables(self, tables: Iterable[str]) -> None:
        """Check, in one attempt, that ``tables`` exist too (the HF feature tables).

        Kept apart from :meth:`check_ready` so that a database missing only these tables
        holds back the queries that need them, not the CDC path.
        """
        if self._client is None:
            raise RuntimeError("ClickHouse is not connected: check_ready() has not succeeded yet")
        verify_tables(self._client, self._settings.clickhouse_db, tables, self._wait, SINGLE_ATTEMPT)

    @property
    def writer(self) -> ClickHouseWriter:
        """Writer of the connected client; :meth:`check_ready` must have succeeded once."""
        if self._writer is None:
            raise RuntimeError("ClickHouse is not connected: check_ready() has not succeeded yet")
        return self._writer

    def close(self) -> None:
        """Release the client's connection pool."""
        if self._client is not None:
            self._client.close()
