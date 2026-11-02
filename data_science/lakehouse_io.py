"""ClickHouse access for the data-science jobs, with bounded retries on network I/O."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from ds_runtime import retry_call
from ds_settings import ClickHouseSettings
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError

_T = TypeVar("_T")

# Engines whose FINAL modifier merges row versions at read time; ClickHouse
# rejects FINAL on plain MergeTree tables and on views.
_FINAL_CAPABLE_ENGINE_MARKERS = ("Replacing", "Collapsing", "Summing", "Aggregating", "Coalescing")
# Server error codes that clear up on their own: TIMEOUT_EXCEEDED, TOO_MANY_SIMULTANEOUS_QUERIES,
# SOCKET_TIMEOUT, NETWORK_ERROR, MEMORY_LIMIT_EXCEEDED, TOO_MANY_PARTS.
_TRANSIENT_SERVER_CODES = frozenset({159, 202, 209, 210, 241, 252})
# The subset raised before a statement starts: TOO_MANY_SIMULTANEOUS_QUERIES (admission control).
_NOT_STARTED_SERVER_CODES = frozenset({202})

_PARQUET_EXPORT_SETTINGS = {
    "output_format_parquet_compression_method": "zstd",
    "output_format_parquet_string_as_string": 1,
}
_COPY_BUFFER_BYTES = 1 << 20

logger = logging.getLogger(__name__)


def final_modifier(engine: str) -> str:
    """Return ``FINAL`` when reads of a table with this engine must merge row versions."""
    return "FINAL" if any(marker in engine for marker in _FINAL_CAPABLE_ENGINE_MARKERS) else ""


def _is_transient(exc: Exception) -> bool:
    # OperationalError covers connection failures and timeouts; any other server
    # error (bad SQL, unknown table, ...) would fail identically on every retry.
    if isinstance(exc, OperationalError):
        return True
    return isinstance(exc, DatabaseError) and exc.code in _TRANSIENT_SERVER_CODES


def _never_ran(exc: Exception) -> bool:
    """True when a failed statement certainly did not run, so running it again cannot apply it twice.

    clickhouse-connect raises transport failures as an OperationalError without a server
    code, chained to urllib3's error (a MaxRetryError whose reason is the connection error).
    ConnectTimeoutError - NewConnectionError and NameResolutionError are subclasses - means
    no connection, so nothing was sent; a read timeout or a dropped connection can come after
    the server executed the statement. A server error can only be retried if the server
    refused the statement before starting it.
    """
    if isinstance(exc, DatabaseError) and exc.code is not None:
        return exc.code in _NOT_STARTED_SERVER_CODES
    if not isinstance(exc, OperationalError):
        return False
    cause = exc.__cause__
    return isinstance(cause.reason if isinstance(cause, MaxRetryError) else cause, ConnectTimeoutError)


class ClickHouseGateway:
    """Retrying facade over a clickhouse-connect HTTP client bound to CLICKHOUSE_DB."""

    def __init__(self, settings: ClickHouseSettings, *, attempts: int = 5) -> None:
        self._settings = settings
        self._attempts = attempts
        self._client: Client = self._retry(
            f"connect to ClickHouse at {settings.host}:{settings.port}", self._connect
        )

    def _connect(self) -> Client:
        return clickhouse_connect.get_client(
            host=self._settings.host,
            port=self._settings.port,
            username=self._settings.user,
            password=self._settings.password,
            database=self._settings.database,
            connect_timeout=10,
            send_receive_timeout=600,
            # Retries live in retry_call only, so attempts stay bounded and logged.
            query_retries=0,
            autogenerate_session_id=False,
        )

    def _retry(
        self,
        description: str,
        operation: Callable[[], _T],
        is_retryable: Callable[[Exception], bool] = _is_transient,
    ) -> _T:
        return retry_call(
            operation,
            description=description,
            is_retryable=is_retryable,
            attempts=self._attempts,
        )

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> ClickHouseGateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ping(self) -> bool:
        """Return True when the server answers its /ping endpoint, with the gateway's bounded retries.

        clickhouse-connect's ping() never raises: a failed probe returns False, which retry_call
        would pass straight through. So the False is raised as an OperationalError inside the
        retried operation, a server that is restarting gets the same backoff as every other
        call, and False comes back only after the last attempt.
        """

        def probe() -> bool:
            if not self._client.ping():
                raise OperationalError("ClickHouse did not answer /ping")
            return True

        try:
            return self._retry("ping ClickHouse", probe)
        except OperationalError:
            return False

    def command(self, statement: str, parameters: Mapping[str, Any] | None = None) -> None:
        """Execute a statement that returns no rows (CREATE TABLE, INSERT ... SELECT, ALTER, ...).

        Retried only when it certainly did not run (see _never_ran): such a statement may
        change data, and a timeout or a dropped connection can come after ClickHouse executed
        it, so those failures propagate instead of running it twice.
        """
        self._retry(
            "execute ClickHouse command",
            lambda: self._client.command(statement, parameters=dict(parameters or {})),
            is_retryable=_never_ran,
        )

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        """Map each existing table (or view) among ``tables`` to its engine name."""
        names = sorted(set(tables))
        result = self._retry(
            "look up table engines",
            lambda: self._client.query(
                "SELECT name, engine FROM system.tables "
                "WHERE database = currentDatabase() AND name IN {names:Array(String)}",
                parameters={"names": names},
            ),
        )
        return {str(name): str(engine) for name, engine in result.result_rows}

    def query_df(self, sql: str, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Run a SELECT and return the result as a DataFrame."""
        return self._retry(
            "query ClickHouse", lambda: self._client.query_df(sql, parameters=dict(parameters or {}))
        )

    def query_rows(self, sql: str, parameters: Mapping[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a SELECT and return the result rows as tuples."""
        result = self._retry(
            "query ClickHouse", lambda: self._client.query(sql, parameters=dict(parameters or {}))
        )
        return [tuple(row) for row in result.result_rows]

    def insert_rows(self, table: str, rows: Sequence[Sequence[Any]], column_names: Sequence[str]) -> None:
        """Insert ``rows`` (in ``column_names`` order) into ``table`` as one batch."""
        self._retry(
            f"insert into {table}",
            lambda: self._client.insert(table, list(rows), column_names=list(column_names)),
        )

    def export_parquet(self, sql: str, destination: Path) -> int:
        """Stream the result of ``sql`` as one Parquet file to ``destination``; return its size."""

        def _export() -> int:
            stream = self._client.raw_stream(sql, settings=_PARQUET_EXPORT_SETTINGS, fmt="Parquet")
            try:
                with destination.open("wb") as sink:
                    shutil.copyfileobj(stream, sink, _COPY_BUFFER_BYTES)
            finally:
                stream.close()
            return destination.stat().st_size

        return self._retry(f"export {destination.name}", _export)
