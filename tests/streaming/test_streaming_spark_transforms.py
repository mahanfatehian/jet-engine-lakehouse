"""Local-mode Spark tests of the CDC parsing, bronze and silver transformations.

Fed with the real Debezium messages captured from the stack (c/u/d for every table) plus
malformed variants. Skipped automatically when PySpark or a JVM is unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("pyspark")

from cdc_schemas import (
    ENGINE_STATUS_REGISTRY,
    ENGINE_TELEMETRY,
    MAINTENANCE_LOG,
    TABLES,
    FieldKind,
    TableSpec,
)
from cdc_transforms import PARSE_ERROR_COLUMN, bronze_records, parse_cdc_records, silver_records
from pyspark.sql import functions as sf

TOPICS = {table.name: f"engine_cdc.public.{table.name}" for table in TABLES}
SPARK_TYPES = {
    FieldKind.STRING: "string",
    FieldKind.INT: "int",
    FieldKind.LONG: "bigint",
    FieldKind.DOUBLE: "double",
    FieldKind.TIMESTAMP: "timestamp",
}
METADATA_TYPES = [
    ("cdc_op", "string"),
    ("cdc_lsn", "bigint"),
    ("cdc_source_ts", "timestamp"),
    ("cdc_emitted_ts", "timestamp"),
    ("is_deleted", "tinyint"),
    ("kafka_partition", "int"),
    ("kafka_offset", "bigint"),
]


def micros(text: str) -> int:
    instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (instant - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def of_table(messages: list[Any], table: TableSpec) -> list[Any]:
    return [message for message in messages if message.topic == TOPICS[table.name]]


def edited(change: Callable[[dict[str, Any]], object]) -> Callable[[str], str]:
    def apply(value: str) -> str:
        envelope = json.loads(value)
        change(envelope)
        return json.dumps(envelope)

    return apply


def exact_rows(parsed: Any, table: TableSpec) -> list[dict[str, Any]]:
    """Collect with instants as epoch microseconds, immune to the process timezone."""
    columns = [
        sf.unix_micros(name).alias(name) if name in table.timestamp_columns else sf.col(name)
        for name in (PARSE_ERROR_COLUMN, *table.clickhouse_columns)
    ]
    return [row.asDict() for row in parsed.select(*columns).collect()]


def test_real_telemetry_events_parse_with_correct_images(sample_messages, records_frame) -> None:
    rows = exact_rows(
        parse_cdc_records(records_frame(of_table(sample_messages, ENGINE_TELEMETRY)), ENGINE_TELEMETRY),
        ENGINE_TELEMETRY,
    )
    assert [row[PARSE_ERROR_COLUMN] for row in rows] == [None] * 4
    assert [(row["telemetry_id"], row["cdc_op"], row["is_deleted"]) for row in rows] == [
        (2, "c", 0),
        (2, "u", 0),
        (1, "c", 0),
        (1, "d", 1),
    ]
    assert [row["maintenance_id"] for row in rows] == [None, 1, None, None]
    created = rows[2]
    assert created["event_ts"] == micros("2026-10-26T09:00:00.123456Z")
    assert created["created_ts"] == micros("2026-10-26T09:06:06.004161Z")
    assert created["cdc_lsn"] == 26867296
    assert created["cdc_source_ts"] == 1793005566008 * 1000
    assert created["cdc_emitted_ts"] == 1793005566277 * 1000
    assert (created["kafka_partition"], created["kafka_offset"]) == (0, 0)
    assert (created["rpm"], created["wear_factor"], created["status"]) == (14412.37, 0.0, "HEALTHY")
    deleted = rows[3]  # the delete carries the before image
    assert (deleted["cycle_number"], deleted["cdc_lsn"], deleted["engine_id"]) == (0, 26870264, "FIXTURE-001")


def test_real_registry_and_maintenance_events_parse(sample_messages, records_frame) -> None:
    registry = exact_rows(
        parse_cdc_records(
            records_frame(of_table(sample_messages, ENGINE_STATUS_REGISTRY)), ENGINE_STATUS_REGISTRY
        ),
        ENGINE_STATUS_REGISTRY,
    )
    assert [row[PARSE_ERROR_COLUMN] for row in registry] == [None, None]
    assert [row["operational_status"] for row in registry] == ["HEALTHY", "WARNING"]
    assert registry[0]["last_telemetry_ts"] is None
    assert registry[1]["last_telemetry_ts"] == micros("2026-10-26T09:00:01.000000Z")
    assert registry[1]["last_maintenance_ts"] is None
    assert registry[1]["updated_ts"] == micros("2026-10-26T09:06:06.009676Z")

    maintenance = exact_rows(
        parse_cdc_records(records_frame(of_table(sample_messages, MAINTENANCE_LOG)), MAINTENANCE_LOG),
        MAINTENANCE_LOG,
    )
    assert [row[PARSE_ERROR_COLUMN] for row in maintenance] == [None, None]
    opened, completed = maintenance
    assert (opened["maintenance_status"], opened["action_taken"], opened["completed_ts"]) == (
        "OPEN",
        None,
        None,
    )
    assert completed["action_taken"] == "BEARING_REPLACEMENT"
    assert completed["completed_ts"] == micros("2026-10-26T09:00:21Z")
    assert (completed["downtime_seconds"], completed["warning_rows_flagged"]) == (20, 1)


ERROR_CASES: dict[str, tuple[Callable[[str], str], str]] = {
    "not-json": (lambda _value: "not json", "malformed JSON envelope"),
    "empty-value": (lambda _value: "", "malformed JSON envelope"),
    "truncated-json": (lambda value: value[: len(value) // 2], "malformed JSON envelope"),
    "json-array": (lambda value: f"[{value}]", "malformed JSON envelope"),
    "type-mismatch": (edited(lambda e: e["after"].update(rpm="fast")), "malformed JSON envelope"),
    "missing-op": (edited(lambda e: e.pop("op")), "missing op"),
    "unsupported-op": (edited(lambda e: e.update(op="t")), "unsupported op 't'"),
    "no-images": (edited(lambda e: e.update(before=None, after=None)), "both before and after are null"),
    "create-without-after": (
        edited(lambda e: e.update(before=e["after"], after=None)),
        "missing after image for op 'c'",
    ),
    "delete-without-before": (edited(lambda e: e.update(op="d")), "missing before image for op 'd'"),
    "missing-source-ts": (edited(lambda e: e["source"].pop("ts_ms")), "missing or invalid source.ts_ms"),
    "absurd-ts-ms": (edited(lambda e: e.update(ts_ms=9_000_000_000_000_000_000)), "missing or invalid ts_ms"),
    "negative-lsn": (edited(lambda e: e["source"].update(lsn=-5)), "negative source.lsn"),
    "null-required": (
        edited(lambda e: e["after"].update(engine_id=None, status=None)),
        "null required field(s): engine_id, status",
    ),
    "bad-timestamp": (
        edited(lambda e: e["after"].update(event_ts="yesterday")),
        "invalid or out-of-range timestamp field(s): event_ts",
    ),
    "out-of-range-timestamp": (
        edited(lambda e: e["after"].update(created_ts="1800-01-01T00:00:00Z")),
        "invalid or out-of-range timestamp field(s): created_ts",
    ),
    "negative-unsigned": (
        edited(lambda e: e["after"].update(cycle_number=-1)),
        "negative value in unsigned field(s): cycle_number",
    ),
}


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_unusable_records_get_a_precise_error_instead_of_failing(
    case, sample_messages, records_frame
) -> None:
    transform, expected = ERROR_CASES[case]
    base = of_table(sample_messages, ENGINE_TELEMETRY)[2]  # op "c" of telemetry_id 1
    broken = base._replace(value=transform(base.value), offset=99)
    parsed = parse_cdc_records(records_frame([base, broken]), ENGINE_TELEMETRY)
    errors = [row[0] for row in parsed.orderBy("kafka_offset").select(PARSE_ERROR_COLUMN).collect()]
    assert errors == [None, expected]


def test_tombstones_are_skipped(sample_messages, records_frame) -> None:
    base = of_table(sample_messages, MAINTENANCE_LOG)[0]
    parsed = parse_cdc_records(records_frame([base, base._replace(value=None, offset=5)]), MAINTENANCE_LOG)
    assert [row["kafka_offset"] for row in parsed.collect()] == [0]


@pytest.mark.parametrize(
    "text",
    [
        "2026-10-26T09:00:01Z",
        "2026-10-26T09:00:01.5Z",
        "2026-10-26T09:00:01.123Z",
        "2026-10-26T09:00:01.123456Z",
        "2026-10-26T12:30:01.123456+03:30",
    ],
)
def test_timestamptz_strings_parse_exactly_whatever_the_precision(
    text, sample_messages, records_frame
) -> None:
    base = of_table(sample_messages, ENGINE_TELEMETRY)[0]
    message = base._replace(value=edited(lambda e: e["after"].update(event_ts=text))(base.value))
    (row,) = exact_rows(parse_cdc_records(records_frame([message]), ENGINE_TELEMETRY), ENGINE_TELEMETRY)
    assert row[PARSE_ERROR_COLUMN] is None
    assert row["event_ts"] == micros(text)


def test_snapshot_reads_and_missing_lsn_are_accepted(sample_messages, records_frame) -> None:
    base = of_table(sample_messages, ENGINE_STATUS_REGISTRY)[0]
    snapshot = base._replace(
        value=edited(lambda e: (e.update(op="r"), e["source"].update(lsn=None)))(base.value)
    )
    (row,) = exact_rows(
        parse_cdc_records(records_frame([snapshot]), ENGINE_STATUS_REGISTRY), ENGINE_STATUS_REGISTRY
    )
    assert (row[PARSE_ERROR_COLUMN], row["cdc_op"], row["is_deleted"], row["cdc_lsn"]) == (None, "r", 0, 0)


def test_bronze_keeps_every_record_with_light_metadata(sample_messages, records_frame) -> None:
    telemetry = of_table(sample_messages, ENGINE_TELEMETRY)
    messages = [
        *telemetry,
        telemetry[0]._replace(value="not json", offset=7),
        telemetry[0]._replace(value=None, offset=8),
    ]
    before = datetime.now(UTC).date()
    bronze = bronze_records(records_frame(messages))
    rows = bronze.collect()
    after = datetime.now(UTC).date()

    assert bronze.dtypes == [
        ("kafka_key", "string"),
        ("kafka_value", "string"),
        ("kafka_topic", "string"),
        ("kafka_partition", "int"),
        ("kafka_offset", "bigint"),
        ("kafka_timestamp", "timestamp"),
        ("cdc_op", "string"),
        ("cdc_source_ts_ms", "bigint"),
        ("ingest_date", "date"),
    ]
    assert [row["cdc_op"] for row in rows] == ["c", "u", "c", "d", None, None]
    assert rows[0]["cdc_source_ts_ms"] == 1793005566008
    assert rows[0]["kafka_value"] == telemetry[0].value
    assert rows[5]["kafka_value"] is None
    assert {row["ingest_date"] for row in rows} <= {before, after}


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_silver_schema_is_raw_columns_plus_processing_columns(table, sample_messages, records_frame) -> None:
    messages = of_table(sample_messages, table)
    broken = messages[0]._replace(value="not json", offset=42)
    silver = silver_records(parse_cdc_records(records_frame([*messages, broken]), table), table)

    expected = [(field.name, SPARK_TYPES[field.kind]) for field in table.fields] + METADATA_TYPES
    assert silver.dtypes == [*expected, ("processed_at", "timestamp"), ("event_date", "date")]
    rows = silver.collect()
    assert len(rows) == len(messages)  # the malformed record is excluded
    assert all(row["processed_at"] is not None for row in rows)
    assert {row["event_date"].isoformat() for row in rows} == {"2026-10-26"}
