"""Verification of the dbt analytics models against the synthetic fixture of seed_engine_raw.py.

    python tests/analytics/verify_analytics.py snapshot incremental
    python tests/analytics/verify_analytics.py compare incremental full_refresh
    python tests/analytics/verify_analytics.py check /tmp/seed_manifest.json [--incremental]

snapshot  copies the FINAL contents of every model into analytics_verify.<tag>__<model>
          (build-time columns excluded)
compare   row counts, per-column checksums and row-level EXCEPT diffs of two snapshots
check     CDC dedup, late / deleted / corrected pulses, registry lag, RUL labels, finite
          features, fct grain and predictions, and an independent Python recomputation of the
          rolling features of selected lifecycles (feature_reference.py)
Exit code 0 when every check passes, 1 otherwise; snapshot (the only write) needs
ANALYTICS_IT_ALLOW_WRITES=1 like seed_engine_raw.py and exits 2 without it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError
from feature_reference import ROLLED_SENSORS, SENSORS, WINDOWS, Reading, expected_features
from seed_engine_raw import EXIT_REFUSED, connect, with_retries, writes_allowed

LOGGER = logging.getLogger("verify_analytics")
DATABASE = os.environ.get("CLICKHOUSE_DB", "iot_engine")
VERIFY_DATABASE = "analytics_verify"
MODELS = (
    "stg_engine_telemetry",
    "stg_engine_status_registry",
    "stg_maintenance_log",
    "int_engine_lifecycles",
    "int_sensor_rolling_features",
    "mart_rul_training_features",
    "fct_engine_health_status",
)
BUILD_TIME_COLUMNS = frozenset({"feature_computed_at", "refreshed_at", "telemetry_staleness_seconds"})
FLOAT_FEATURES = (
    *SENSORS,
    *(
        f"{s}_{stat}_{label}"
        for s in ROLLED_SENSORS
        for label, _ in WINDOWS
        for stat in ("mean", "std", "min", "max")
    ),
    *(
        name
        for s in ROLLED_SENSORS
        for name in (f"{s}_delta_1", *(f"{s}_roc_{label}" for label, _ in WINDOWS))
    ),
    "egt_c_drift",
    "core_temp_c_drift",
    "vibration_mm_s_ratio",
    "oil_pressure_psi_drop",
    "fuel_flow_kg_h_drift_pct",
    "rpm_drift_pct",
)
REFERENCE_LIFECYCLES = (("CFM56-104", 2), ("GE90-202", 4), ("JT9D-001", 2), ("GE90-201", 4), ("JT9D-001", 3))
BASELINE_CYCLES = 10
RUL_CAP_CYCLES = int(os.environ.get("RUL_CAP_CYCLES", "125"))
RUL_HORIZON_CYCLES = int(os.environ.get("RUL_FAILURE_HORIZON_CYCLES", "25"))


def rows(client: Client, sql: str) -> list[tuple[Any, ...]]:
    """Run a query (with retries) and return its rows."""
    return with_retries(lambda: client.query(sql).result_rows, "query ClickHouse")


def scalar(client: Client, sql: str) -> Any:
    """Run a query that returns one value."""
    return rows(client, sql)[0][0]


def model_columns(client: Client, database: str, table: str) -> list[tuple[str, str]]:
    """(name, type) of a table's columns in physical order."""
    return [
        (name, kind)
        for name, kind in rows(
            client,
            (
                f"SELECT name, type FROM system.columns WHERE database = '{database}' AND table = '{table}' "
                "ORDER BY position"
            ),
        )
    ]


def snapshot(client: Client, tag: str) -> list[str]:
    """Copy the FINAL view of every model into the verify database under ``tag``."""
    client.command(f"CREATE DATABASE IF NOT EXISTS {VERIFY_DATABASE}")
    for model in MODELS:
        columns = [
            name for name, _ in model_columns(client, DATABASE, model) if name not in BUILD_TIME_COLUMNS
        ]
        target = f"{VERIFY_DATABASE}.{tag}__{model}"
        client.command(f"DROP TABLE IF EXISTS {target}")
        client.command(
            f"CREATE TABLE {target} ENGINE = MergeTree ORDER BY tuple() AS "
            f"SELECT {', '.join(columns)} FROM {DATABASE}.{model} FINAL"
        )
        LOGGER.info("snapshot %s: %d rows", target, scalar(client, f"SELECT count() FROM {target}"))
    return []


def _column_checksum(name: str, kind: str) -> str:
    value = f"round({name}, 9)" if "Float" in kind else name
    return f"(count(), countIf(isNull({name})), sum(cityHash64(ifNull(toString({value}), 'NULL'))))"


def compare(client: Client, left: str, right: str) -> list[str]:
    """Row counts, per-column checksums and EXCEPT diffs (floats rounded to 1e-9) of two snapshots."""
    failures: list[str] = []
    for model in MODELS:
        left_table, right_table = f"{VERIFY_DATABASE}.{left}__{model}", f"{VERIFY_DATABASE}.{right}__{model}"
        columns = model_columns(client, VERIFY_DATABASE, f"{left}__{model}")
        rounded = ", ".join(f"round({name}, 9)" if "Float" in kind else name for name, kind in columns)
        counts = [scalar(client, f"SELECT count() FROM {table}") for table in (left_table, right_table)]
        missing = scalar(
            client,
            f"SELECT count() FROM (SELECT {rounded} FROM {left_table} "
            f"EXCEPT SELECT {rounded} FROM {right_table})",
        )
        extra = scalar(
            client,
            f"SELECT count() FROM (SELECT {rounded} FROM {right_table} "
            f"EXCEPT SELECT {rounded} FROM {left_table})",
        )
        bit_exact = (
            scalar(
                client, f"SELECT count() FROM (SELECT * FROM {left_table} EXCEPT SELECT * FROM {right_table})"
            )
            == 0
        )
        checksum_sql = ", ".join(_column_checksum(name, kind) for name, kind in columns)
        left_sums, right_sums = (
            rows(client, f"SELECT {checksum_sql} FROM {table}")[0] for table in (left_table, right_table)
        )
        differing = [name for (name, _), a, b in zip(columns, left_sums, right_sums, strict=True) if a != b]
        LOGGER.info(
            "%s: rows %d vs %d, row diffs %d/%d, %d columns checksummed, mismatching %s, bit-exact %s",
            model,
            counts[0],
            counts[1],
            missing,
            extra,
            len(columns),
            differing or "none",
            bit_exact,
        )
        if counts[0] != counts[1] or missing or extra or differing:
            failures.append(f"{model}: {left} and {right} differ")
    return failures


def check_dedup(client: Client, manifest: dict[str, Any]) -> list[str]:
    """stg FINAL holds exactly the newest live version of every pulse; old versions are physical only."""
    failures = []
    final_rows, unique_ids = rows(
        client, f"SELECT count(), uniqExact(telemetry_id) FROM {DATABASE}.stg_engine_telemetry FINAL"
    )[0]
    physical = scalar(client, f"SELECT count() FROM {DATABASE}.stg_engine_telemetry")
    expected_live = scalar(
        client,
        (
            "SELECT countIf(latest_deleted = 0) FROM (SELECT telemetry_id, "
            f"argMax(is_deleted, cdc_lsn) AS latest_deleted FROM {DATABASE}.raw_engine_telemetry "
            "GROUP BY telemetry_id)"
        ),
    )
    LOGGER.info(
        "stg_engine_telemetry: %d physical versions, %d under FINAL, %d expected live pulses",
        physical,
        final_rows,
        expected_live,
    )
    if final_rows != unique_ids or final_rows != expected_live:
        failures.append(
            f"stg_engine_telemetry FINAL has {final_rows} rows / {unique_ids} ids, expected {expected_live}"
        )
    unlinked = scalar(
        client,
        (
            f"SELECT count() FROM {DATABASE}.stg_engine_telemetry FINAL WHERE status != 'HEALTHY' "
            f"AND maintenance_id IS NULL AND (engine_id, lifecycle_id) IN (SELECT engine_id, lifecycle_id "
            f"FROM {DATABASE}.int_engine_lifecycles FINAL WHERE has_failed = 1)"
        ),
    )
    if unlinked:
        failures.append(f"{unlinked} failed-lifecycle pulses lost their maintenance_id update")
    orders = rows(
        client,
        "SELECT count(), countIf(maintenance_status = 'COMPLETED') "
        f"FROM {DATABASE}.stg_maintenance_log FINAL",
    )[0]
    if orders != (len(manifest["completed_lifecycles"]),) * 2:
        failures.append(f"stg_maintenance_log has (rows, completed) = {orders}")
    return failures


def check_scenarios(client: Client, manifest: dict[str, Any], incremental: bool) -> list[str]:
    """Late, late-visible, deleted and corrected pulses and the lagging registry are all resolved."""
    failures = []
    for model in ("stg_engine_telemetry", "int_sensor_rolling_features"):
        present = {
            row[0]
            for row in rows(
                client,
                f"SELECT telemetry_id FROM {DATABASE}.{model} FINAL WHERE telemetry_id IN "
                f"({', '.join(map(str, manifest['withheld_ids'] + manifest['deleted_ids']))})",
            )
        }
        if set(manifest["withheld_ids"]) - present:
            failures.append(f"{model}: late pulses missing {sorted(set(manifest['withheld_ids']) - present)}")
        if present & set(manifest["deleted_ids"]):
            failures.append(
                f"{model}: deleted pulses still visible {sorted(present & set(manifest['deleted_ids']))}"
            )
        egt = scalar(
            client,
            f"SELECT egt_c FROM {DATABASE}.{model} FINAL WHERE telemetry_id = {manifest['corrected_id']}",
        )
        if not math.isclose(egt, manifest["corrected_egt_c"], abs_tol=1e-9):
            failures.append(f"{model}: corrected egt_c is {egt}, expected {manifest['corrected_egt_c']}")
    unknown = scalar(
        client, f"SELECT countIf(engine_model = 'UNKNOWN') FROM {DATABASE}.int_sensor_rolling_features FINAL"
    )
    if unknown:
        failures.append(f"{unknown} feature rows still carry engine_model UNKNOWN")
    retractions = scalar(
        client,
        f"SELECT count() FROM {DATABASE}.int_sensor_rolling_features WHERE is_deleted = 1 "
        f"AND telemetry_id = {manifest['deleted_ids'][0]}",
    )
    LOGGER.info("retraction rows for the late-deleted pulse: %d", retractions)
    if incremental and not retractions:
        failures.append("the late delete of a published pulse produced no retraction row")
    return failures


def check_labels(client: Client) -> list[str]:
    """Mart labels recomputed from stg FINAL; only completed lifecycles, every pulse up to the failure."""
    bad = scalar(
        client,
        f"""
        SELECT count() FROM (
            SELECT m.rul_cycles AS rul, m.rul_cycles_capped AS capped, m.failure_within_horizon AS horizon,
                   m.failure_cycle AS failure_cycle, f.failure_cycle AS expected_fc,
                   m.cycle_number AS cycle_number, m.split_group AS split_group,
                   m.engine_id AS engine_id, m.lifecycle_id AS lifecycle_id, f.failures AS failures
            FROM (SELECT * FROM {DATABASE}.mart_rul_training_features FINAL) AS m
            LEFT JOIN (
                SELECT engine_id, lifecycle_id, countIf(status = 'FAILURE') AS failures,
                       minIf(cycle_number, status = 'FAILURE') AS failure_cycle
                FROM {DATABASE}.stg_engine_telemetry FINAL
                GROUP BY engine_id, lifecycle_id
            ) AS f ON f.engine_id = m.engine_id AND f.lifecycle_id = m.lifecycle_id
            WHERE failures = 0 OR failure_cycle != expected_fc
               OR rul != toInt32(expected_fc) - toInt32(cycle_number) OR rul < 0
               OR capped != least(rul, {RUL_CAP_CYCLES}) OR horizon != (rul <= {RUL_HORIZON_CYCLES})
               OR split_group != concat(engine_id, '#', toString(lifecycle_id)))""",
    )
    expected_rows = scalar(
        client,
        f"""
        SELECT count()
        FROM (SELECT engine_id, lifecycle_id, cycle_number FROM {DATABASE}.stg_engine_telemetry FINAL) AS t
        INNER JOIN (SELECT engine_id, lifecycle_id, minIf(cycle_number, status = 'FAILURE') AS failure_cycle
                    FROM {DATABASE}.stg_engine_telemetry FINAL GROUP BY engine_id, lifecycle_id
                    HAVING countIf(status = 'FAILURE') > 0) AS f
            ON t.engine_id = f.engine_id AND t.lifecycle_id = f.lifecycle_id
        WHERE t.cycle_number <= f.failure_cycle""",
    )
    mart_rows = scalar(client, f"SELECT count() FROM {DATABASE}.mart_rul_training_features FINAL")
    LOGGER.info("mart rows %d (expected %d), rows with inconsistent labels %d", mart_rows, expected_rows, bad)
    return (
        [f"mart labels: {bad} inconsistent rows, {mart_rows} rows vs {expected_rows} expected"]
        if bad or expected_rows != mart_rows
        else []
    )


def check_finite(client: Client) -> list[str]:
    """No NULL/NaN/Inf in any float feature of the feature table and the mart."""
    condition = " OR ".join(f"NOT isFinite({column})" for column in FLOAT_FEATURES)
    failures = []
    for model in ("int_sensor_rolling_features", "mart_rul_training_features"):
        bad = scalar(client, f"SELECT count() FROM {DATABASE}.{model} FINAL WHERE {condition}")
        if bad:
            failures.append(f"{model}: {bad} rows with non-finite features")
    return failures


def check_board(client: Client, manifest: dict[str, Any]) -> list[str]:
    """One fct row per engine; predictions joined for the current lifecycle only."""
    board = {
        row[0]: row[1:]
        for row in rows(
            client,
            (
                f"SELECT engine_id, predicted_rul_cycles, operational_status, engine_model "
                f"FROM {DATABASE}.fct_engine_health_status FINAL"
            ),
        )
    }
    count = scalar(client, f"SELECT count() FROM {DATABASE}.fct_engine_health_status FINAL")
    failures = []
    if count != len(manifest["engines"]) or set(board) != set(manifest["engines"]):
        failures.append(f"fct has {count} rows for engines {sorted(board)}")
    for engine, expected in manifest["expected_predicted_rul"].items():
        actual = board.get(engine, (None,))[0]
        if actual != expected:
            failures.append(f"fct {engine}: predicted_rul_cycles {actual}, expected {expected}")
    lagging = board.get(manifest["lagging_registry_engine"], (None, "UNKNOWN", "UNKNOWN"))
    if "UNKNOWN" in lagging[1:]:
        failures.append(f"fct {manifest['lagging_registry_engine']}: registry state not applied {lagging}")
    return failures


def check_feature_values(client: Client) -> list[str]:
    """Recompute the features of REFERENCE_LIFECYCLES in Python and compare every value."""
    failures = []
    for engine_id, lifecycle_id in REFERENCE_LIFECYCLES:
        where = f"engine_id = '{engine_id}' AND lifecycle_id = {lifecycle_id}"
        readings = [
            Reading(row[0], row[1], row[2], dict(zip(SENSORS, row[3:], strict=True)))
            for row in rows(
                client,
                f"SELECT telemetry_id, cycle_number, toUnixTimestamp64Micro(event_ts), {', '.join(SENSORS)} "
                f"FROM {DATABASE}.stg_engine_telemetry FINAL WHERE {where}",
            )
        ]
        expected = expected_features(readings, BASELINE_CYCLES)
        actual = {
            row[0]: dict(zip(FLOAT_FEATURES, row[1:], strict=True))
            for row in rows(
                client,
                f"SELECT telemetry_id, {', '.join(FLOAT_FEATURES)} "
                f"FROM {DATABASE}.int_sensor_rolling_features FINAL WHERE {where}",
            )
        }
        mismatches = [
            (tid, name, actual[tid][name], value)
            for tid, features in expected.items()
            for name, value in features.items()
            if tid not in actual or not math.isclose(actual[tid][name], value, rel_tol=1e-9, abs_tol=1e-6)
        ]
        LOGGER.info(
            "%s#%d: %d pulses, %d values compared, %d mismatches",
            engine_id,
            lifecycle_id,
            len(expected),
            len(expected) * len(FLOAT_FEATURES),
            len(mismatches),
        )
        if mismatches or set(actual) != set(expected):
            failures.append(f"{engine_id}#{lifecycle_id}: feature mismatches {mismatches[:3]}")
    return failures


def check(client: Client, manifest_path: Path, incremental: bool) -> list[str]:
    """Run every invariant check against the current model contents."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        *check_dedup(client, manifest),
        *check_scenarios(client, manifest, incremental),
        *check_labels(client),
        *check_finite(client),
        *check_board(client, manifest),
        *check_feature_values(client),
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the dbt analytics models.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot", help="copy model contents").add_argument("tag")
    compare_parser = commands.add_parser("compare", help="diff two snapshots")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    check_parser = commands.add_parser("check", help="run the invariant checks")
    check_parser.add_argument("manifest", type=Path)
    check_parser.add_argument("--incremental", action="store_true", help="also expect retraction rows")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if args.command == "snapshot" and not writes_allowed():
        return EXIT_REFUSED
    try:
        client = connect()
        if args.command == "snapshot":
            failures = snapshot(client, args.tag)
        elif args.command == "compare":
            failures = compare(client, args.left, args.right)
        else:
            failures = check(client, args.manifest, args.incremental)
    except ClickHouseError:
        LOGGER.exception("verification could not run")
        return 1
    for failure in failures:
        LOGGER.error("FAILED: %s", failure)
    LOGGER.info("%s: %s", args.command, "FAILED" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
