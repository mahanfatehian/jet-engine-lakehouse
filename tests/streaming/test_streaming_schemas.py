"""Contract tests: the table specs must match infrastructure/clickhouse/init-db.sql exactly."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cdc_schemas import (
    DEAD_LETTER_COLUMNS,
    DEAD_LETTER_TABLE,
    TABLES,
    FieldKind,
    FieldSpec,
    TableSpec,
)

DDL_FILE = Path(__file__).resolve().parents[2] / "infrastructure" / "clickhouse" / "init-db.sql"
_TABLE_BLOCK = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\)", re.DOTALL)
_COLUMN_LINE = re.compile(r"^\s+(\w+)\s+(.+?),?$")
SERVER_FILLED = "ingested_at"


def _ddl_columns() -> dict[str, dict[str, str]]:
    """Map table -> ordered {column: type} parsed from the DDL file (indexes skipped)."""
    tables: dict[str, dict[str, str]] = {}
    for name, body in _TABLE_BLOCK.findall(DDL_FILE.read_text(encoding="utf-8")):
        columns: dict[str, str] = {}
        for line in body.splitlines():
            match = _COLUMN_LINE.match(line)
            if match and match.group(1) != "INDEX":
                columns[match.group(1)] = match.group(2)
        tables[name] = columns
    return tables


@pytest.fixture(scope="module")
def ddl() -> dict[str, dict[str, str]]:
    if not DDL_FILE.exists():
        pytest.skip(f"{DDL_FILE} not available")
    return _ddl_columns()


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_insert_columns_match_ddl_order_without_server_default(table: TableSpec, ddl: dict) -> None:
    columns = ddl[table.clickhouse_table]
    assert SERVER_FILLED in columns
    assert tuple(name for name in columns if name != SERVER_FILLED) == table.clickhouse_columns


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_nullability_matches_ddl(table: TableSpec, ddl: dict) -> None:
    columns = ddl[table.clickhouse_table]
    for field in table.fields:
        assert field.nullable == columns[field.name].startswith("Nullable("), field.name


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_unsigned_flags_match_ddl(table: TableSpec, ddl: dict) -> None:
    columns = ddl[table.clickhouse_table]
    for field in table.fields:
        base_type = columns[field.name].removeprefix("Nullable(")
        assert field.unsigned == base_type.startswith("UInt"), field.name


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_timestamp_columns_are_exactly_the_datetime64_columns(table: TableSpec, ddl: dict) -> None:
    columns = ddl[table.clickhouse_table]
    datetime64 = {
        name for name, ch_type in columns.items() if "DateTime64" in ch_type and name != SERVER_FILLED
    }
    assert datetime64 == table.timestamp_columns


def test_dead_letter_columns_match_ddl(ddl: dict) -> None:
    columns = ddl[DEAD_LETTER_TABLE]
    assert tuple(name for name in columns if name != SERVER_FILLED) == DEAD_LETTER_COLUMNS


def test_dedup_key_is_primary_key_plus_lsn() -> None:
    assert [table.dedup_key_columns for table in TABLES] == [
        ("telemetry_id", "cdc_lsn"),
        ("engine_id", "cdc_lsn"),
        ("maintenance_id", "cdc_lsn"),
    ]


def test_silver_partition_sources_follow_the_contract() -> None:
    assert {table.name: table.event_time_field for table in TABLES} == {
        "engine_telemetry": "event_ts",
        "engine_status_registry": "updated_ts",
        "maintenance_log": "failure_ts",
    }


def test_table_spec_rejects_unknown_key_column() -> None:
    with pytest.raises(ValueError, match="unknown"):
        TableSpec("t", "raw_t", ("missing",), "ts", (FieldSpec("ts", FieldKind.TIMESTAMP),))


def test_table_spec_rejects_nullable_event_time() -> None:
    fields = (FieldSpec("id", FieldKind.LONG), FieldSpec("ts", FieldKind.TIMESTAMP, nullable=True))
    with pytest.raises(ValueError, match="event-time"):
        TableSpec("t", "raw_t", ("id",), "ts", fields)
