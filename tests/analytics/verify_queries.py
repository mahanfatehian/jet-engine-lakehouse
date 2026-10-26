"""Read helpers shared by the analytics verification modules (verify_analytics.py, verify_gold.py)."""

from __future__ import annotations

import os
from typing import Any

from clickhouse_connect.driver.client import Client
from seed_engine_raw import with_retries

DATABASE = os.environ.get("CLICKHOUSE_DB", "iot_engine")


def rows(client: Client, sql: str) -> list[tuple[Any, ...]]:
    """Run a query (with retries) and return its rows."""
    return with_retries(lambda: client.query(sql).result_rows, "query ClickHouse")


def scalar(client: Client, sql: str) -> Any:
    """Run a query that returns one value."""
    return rows(client, sql)[0][0]
