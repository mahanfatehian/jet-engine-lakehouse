"""Load the synthetic CDC history of synthetic_cdc.py into the ClickHouse raw_* landing tables.

Usage (ClickHouse settings come from CLICKHOUSE_HOST/PORT/USER/PASSWORD/DB, contract defaults):
    python tests/analytics/seed_engine_raw.py --phase base [--manifest /tmp/seed_manifest.json]
    python tests/analytics/seed_engine_raw.py --phase increment

Rows are written the way the Spark job writes them: CDC metadata filled from the envelope,
ingested_at left to the server default, several inserts per table like micro-batches.
Increment pulses flagged late_visible go first, with an explicit ingested_at 90 s older than
the newest raw row: a slow concurrent insert that becomes visible behind the dbt watermark.
The increment phase also appends a few RUL predictions (normally written by predict_rul.py).

The fixture reuses real fleet engine ids, so writing it into the platform's ClickHouse would
corrupt real data: every write path refuses to run unless ANALYTICS_IT_ALLOW_WRITES=1, which
run_dbt_integration.sh sets for its throwaway server only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError, OperationalError
from synthetic_cdc import CdcEvent, History, build_history

LOGGER = logging.getLogger("seed_engine_raw")
T = TypeVar("T")

WRITE_GUARD_ENV = "ANALYTICS_IT_ALLOW_WRITES"
EXIT_REFUSED = 2
PHASES = ("base", "increment")
KAFKA_PARTITIONS = 3
BATCH_ROWS = 1500
LATE_VISIBLE_LAG = timedelta(seconds=90)
CDC_COLUMNS = (
    "cdc_op",
    "cdc_lsn",
    "cdc_source_ts",
    "cdc_emitted_ts",
    "is_deleted",
    "kafka_partition",
    "kafka_offset",
)
TABLE_COLUMNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "engine_telemetry": (
        "raw_engine_telemetry",
        (
            "telemetry_id",
            "engine_id",
            "lifecycle_id",
            "cycle_number",
            "event_ts",
            "rpm",
            "core_temp_c",
            "egt_c",
            "vibration_mm_s",
            "fuel_flow_kg_h",
            "oil_pressure_psi",
            "wear_factor",
            "status",
            "maintenance_id",
            "created_ts",
        ),
    ),
    "engine_status_registry": (
        "raw_engine_status_registry",
        (
            "engine_id",
            "engine_model",
            "operational_status",
            "current_lifecycle_id",
            "current_cycle",
            "wear_factor",
            "cycles_since_new",
            "lifecycles_completed",
            "last_maintenance_ts",
            "status_changed_ts",
            "last_telemetry_ts",
            "updated_ts",
            "registered_ts",
        ),
    ),
    "maintenance_log": (
        "raw_maintenance_log",
        (
            "maintenance_id",
            "engine_id",
            "lifecycle_id",
            "failure_cycle",
            "failure_ts",
            "failure_mode",
            "maintenance_status",
            "action_taken",
            "opened_ts",
            "completed_ts",
            "downtime_seconds",
            "warning_rows_flagged",
            "notes",
        ),
    ),
}
PRIMARY_KEYS = {
    "engine_telemetry": "telemetry_id",
    "engine_status_registry": "engine_id",
    "maintenance_log": "maintenance_id",
}
PREDICTION_COLUMNS = (
    "predicted_at",
    "engine_id",
    "engine_model",
    "lifecycle_id",
    "cycle_number",
    "telemetry_id",
    "feature_event_ts",
    "operational_status",
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "predicted_rul_cycles",
    "predicted_rul_hours",
    "failure_probability",
    "failure_within_horizon",
    "horizon_cycles",
    "risk_band",
    "model_version",
)


def with_retries(action: Callable[[], T], what: str, attempts: int = 5, base_delay_s: float = 0.5) -> T:
    """Run ``action``, retrying transient connection errors with exponential backoff."""
    attempt = 1
    while True:
        try:
            return action()
        except OperationalError as exc:
            if attempt >= attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            LOGGER.warning("%s failed (%s); retry %d/%d in %.1fs", what, exc, attempt, attempts - 1, delay)
            time.sleep(delay)
            attempt += 1


def connect() -> Client:
    """Open an HTTP client on CLICKHOUSE_DB using the contract's env vars and defaults."""
    settings = {
        "host": os.environ.get("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        "username": os.environ.get("CLICKHOUSE_USER", "clickhouse"),
        "password": os.environ.get("CLICKHOUSE_PASSWORD", "clickhouse"),
        "database": os.environ.get("CLICKHOUSE_DB", "iot_engine"),
    }
    return with_retries(lambda: clickhouse_connect.get_client(**settings), "connect to ClickHouse")


def _partition(event: CdcEvent) -> int:
    key = event.image[PRIMARY_KEYS[event.table]]
    # Kafka hashes the record key; a stable hash is enough to spread rows like the real topics
    return zlib.crc32(str(key).encode()) % KAFKA_PARTITIONS


def assign_offsets(history: History) -> dict[int, tuple[int, int]]:
    """Kafka (partition, offset) per event, in delivery order: base first, then increment."""
    counters: defaultdict[tuple[str, int], int] = defaultdict(int)
    placement: dict[int, tuple[int, int]] = {}
    ordered = sorted(history.events, key=lambda event: (PHASES.index(event.phase), event.lsn))
    for event in ordered:
        partition = _partition(event)
        placement[id(event)] = (partition, counters[(event.table, partition)])
        counters[(event.table, partition)] += 1
    return placement


def encode(event: CdcEvent, placement: tuple[int, int]) -> list[Any]:
    """Raw-table row of one change event: the row image followed by the CDC metadata columns."""
    _, columns = TABLE_COLUMNS[event.table]
    partition, offset = placement
    cdc = [
        event.op,
        event.lsn,
        event.commit_ts,
        event.commit_ts + timedelta(milliseconds=250),
        int(event.op == "d"),
        partition,
        offset,
    ]
    return [event.image[column] for column in columns] + cdc


def _insert(client: Client, table: str, columns: Sequence[str], rows: list[list[Any]]) -> None:
    for start in range(0, len(rows), BATCH_ROWS):
        chunk = rows[start : start + BATCH_ROWS]
        with_retries(
            lambda chunk=chunk: client.insert(table, chunk, column_names=list(columns)),
            f"insert {len(chunk)} rows into {table}",
        )


def writes_allowed() -> bool:
    """True when the caller opted in to writing fixture data (see the module docstring)."""
    if os.environ.get(WRITE_GUARD_ENV) == "1":
        return True
    LOGGER.error(
        "refusing to write fixture data to ClickHouse %s:%s/%s: set %s=1 for a throwaway server only",
        os.environ.get("CLICKHOUSE_HOST", "localhost"),
        os.environ.get("CLICKHOUSE_PORT", "8123"),
        os.environ.get("CLICKHOUSE_DB", "iot_engine"),
        WRITE_GUARD_ENV,
    )
    return False


def _newest_ingestion(client: Client) -> datetime:
    result = with_retries(
        lambda: client.query("SELECT max(ingested_at) FROM raw_engine_telemetry"),
        "read the newest ingested_at",
    )
    return result.result_rows[0][0]


def load_phase(client: Client, history: History, phase: str) -> dict[str, int]:
    """Insert every event of ``phase``; returns the number of rows written per raw table."""
    placement = assign_offsets(history)
    events = [event for event in history.events if event.phase == phase]
    written: dict[str, int] = {}
    late = [event for event in events if event.late_visible]
    if late:
        visible_at = _newest_ingestion(client) - LATE_VISIBLE_LAG
        table, columns = TABLE_COLUMNS["engine_telemetry"]
        rows = [[*encode(event, placement[id(event)]), visible_at] for event in late]
        _insert(client, table, (*columns, *CDC_COLUMNS, "ingested_at"), rows)
        LOGGER.info("inserted %d late-visible pulses with ingested_at=%s", len(rows), visible_at.isoformat())
        written["late_visible"] = len(rows)
    for source_table, (table, columns) in TABLE_COLUMNS.items():
        rows = [
            encode(event, placement[id(event)])
            for event in events
            if event.table == source_table and not event.late_visible
        ]
        _insert(client, table, (*columns, *CDC_COLUMNS), rows)
        written[table] = len(rows)
        LOGGER.info("inserted %d %s rows into %s", len(rows), phase, table)
    if phase == "increment":
        rows = [[prediction[column] for column in PREDICTION_COLUMNS] for prediction in history.predictions]
        _insert(client, "gold_engine_rul_predictions", PREDICTION_COLUMNS, rows)
        written["gold_engine_rul_predictions"] = len(rows)
    return written


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed synthetic CDC data into the ClickHouse raw tables.")
    parser.add_argument("--phase", choices=PHASES, required=True, help="delivery phase to load")
    parser.add_argument("--seed", type=int, default=42, help="simulation seed (default 42)")
    parser.add_argument("--manifest", type=Path, help="write the scenario manifest (JSON) to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if not writes_allowed():
        return EXIT_REFUSED
    history = build_history(args.seed)
    if args.manifest:
        args.manifest.write_text(json.dumps(history.manifest, indent=2), encoding="utf-8")
        LOGGER.info("wrote scenario manifest to %s", args.manifest)
    try:
        client = connect()
        written = load_phase(client, history, args.phase)
    except ClickHouseError:
        LOGGER.exception("seeding phase %s failed", args.phase)
        return 1
    LOGGER.info("phase %s loaded: %s", args.phase, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
