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

_T = TypeVar("_T")

# Engines whose FINAL modifier merges row versions at read time; ClickHouse
# rejects FINAL on plain MergeTree tables and on views.
_FINAL_CAPABLE_ENGINE_MARKERS = ("Replacing", "Collapsing", "Summing", "Aggregating", "Coalescing")
# Server error codes that clear up on their own: TIMEOUT_EXCEEDED, TOO_MANY_SIMULTANEOUS_QUERIES,
# SOCKET_TIMEOUT, NETWORK_ERROR, MEMORY_LIMIT_EXCEEDED, TOO_MANY_PARTS.
_TRANSIENT_SERVER_CODES = frozenset({159, 202, 209, 210, 241, 252})

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

    def _retry(self, description: str, operation: Callable[[], _T]) -> _T:
        return retry_call(
            operation,
            description=description,
            is_retryable=_is_transient,
            attempts=self._attempts,
        )

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> ClickHouseGateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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
