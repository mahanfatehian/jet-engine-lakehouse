"""Checks of the gold models against the fixture's scenario and the dbt vars (see verify_analytics.py).

check_board     fct_engine_health_status: one row per engine, the newest prediction of the current
                lifecycle as scored, that prediction aged to the board's current_cycle
                (synthetic_cdc.expected_board_predictions derives the expectations from the
                simulated registry, independently of the SQL), and an engine that never reported
                telemetry reading as unknown and stale
check_mart_cap  mart_rul_training_features holds exactly the newest N completed lifecycles of
                every engine, each with every pulse up to its failure cycle
"""

from __future__ import annotations

import logging
import math
from typing import Any

from clickhouse_connect.driver.client import Client
from verify_queries import DATABASE, rows, scalar

LOGGER = logging.getLogger("verify_analytics")
ADJUSTED_COLUMNS = (
    "prediction_cycle_number",
    "cycles_since_prediction",
    "predicted_rul_cycles_adjusted",
    "predicted_rul_hours_adjusted",
    "risk_band_adjusted",
)
UNKNOWN_WITHOUT_TELEMETRY = ("telemetry_staleness_seconds", "degradation_index", "health_score")
BOARD_COLUMNS = (
    "predicted_rul_cycles",
    "operational_status",
    "engine_model",
    "is_telemetry_stale",
    "prediction_age_seconds",
    *UNKNOWN_WITHOUT_TELEMETRY,
    *ADJUSTED_COLUMNS,
)


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def _aging_failures(engine: str, actual: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    """Differences between a board row's prediction aging columns and the expectation."""
    if expected is None:
        present = {name: actual[name] for name in ("prediction_age_seconds", *ADJUSTED_COLUMNS)}
        unexpected = {name: value for name, value in present.items() if value is not None}
        return [f"fct {engine}: no prediction expected, got {unexpected}"] if unexpected else []
    failures = [
        f"fct {engine}: {name} {actual[name]}, expected {value}"
        for name, value in expected.items()
        if not _matches(actual[name], value)
    ]
    age = actual["prediction_age_seconds"]
    if age is None or age < 0:
        failures.append(f"fct {engine}: prediction_age_seconds {age}, expected >= 0")
    return failures


def _silent_engine_failures(engine: str, row: dict[str, Any]) -> list[str]:
    """An engine that never reported telemetry reads as unknown and stale, never as healthy."""
    measured = {name: row[name] for name in UNKNOWN_WITHOUT_TELEMETRY if row[name] is not None}
    failures = [f"fct {engine}: no telemetry, yet {measured}"] if measured else []
    if row["is_telemetry_stale"] != 1:
        failures.append(f"fct {engine}: no telemetry, yet is_telemetry_stale {row['is_telemetry_stale']}")
    return failures


def check_board(client: Client, manifest: dict[str, Any]) -> list[str]:
    """One fct row per engine; predictions joined for the current lifecycle only, then aged."""
    board = {
        row[0]: dict(zip(BOARD_COLUMNS, row[1:], strict=True))
        for row in rows(
            client,
            f"SELECT engine_id, {', '.join(BOARD_COLUMNS)} FROM {DATABASE}.fct_engine_health_status FINAL",
        )
    }
    count = scalar(client, f"SELECT count() FROM {DATABASE}.fct_engine_health_status FINAL")
    failures = []
    if count != len(manifest["engines"]) or set(board) != set(manifest["engines"]):
        failures.append(f"fct has {count} rows for engines {sorted(board)}")
    for engine, expected in manifest["expected_predicted_rul"].items():
        actual = board.get(engine, {}).get("predicted_rul_cycles")
        if actual != expected:
            failures.append(f"fct {engine}: predicted_rul_cycles {actual}, expected {expected}")
    for engine, expected in manifest["expected_board_predictions"].items():
        if engine in board:
            LOGGER.info(
                "fct %s prediction aging: %s", engine, {k: board[engine][k] for k in ADJUSTED_COLUMNS}
            )
            failures += _aging_failures(engine, board[engine], expected)
    lagging = board.get(manifest["lagging_registry_engine"], {})
    if "UNKNOWN" in (lagging.get("operational_status", "UNKNOWN"), lagging.get("engine_model", "UNKNOWN")):
        failures.append(f"fct {manifest['lagging_registry_engine']}: registry state not applied {lagging}")
    if manifest["silent_engine"] in board:
        failures += _silent_engine_failures(manifest["silent_engine"], board[manifest["silent_engine"]])
    return failures


def check_mart_cap(client: Client, max_lifecycles: int) -> list[str]:
    """The mart holds the newest ``max_lifecycles`` completed lifecycles per engine, in full."""
    completed: dict[str, list[int]] = {}
    for engine_id, lifecycle_id in rows(
        client,
        f"SELECT engine_id, lifecycle_id FROM {DATABASE}.stg_engine_telemetry FINAL "
        "GROUP BY engine_id, lifecycle_id HAVING countIf(status = 'FAILURE') > 0",
    ):
        completed.setdefault(engine_id, []).append(lifecycle_id)
    expected = {
        (engine_id, lifecycle_id)
        for engine_id, lifecycle_ids in completed.items()
        for lifecycle_id in sorted(lifecycle_ids, reverse=True)[:max_lifecycles]
    }
    actual = set(
        rows(
            client,
            f"SELECT DISTINCT engine_id, lifecycle_id FROM {DATABASE}.mart_rul_training_features FINAL",
        )
    )
    in_window = ", ".join(f"('{engine_id}', {lifecycle_id})" for engine_id, lifecycle_id in sorted(expected))
    expected_rows = scalar(
        client,
        f"""
        SELECT count()
        FROM (SELECT engine_id, lifecycle_id, cycle_number FROM {DATABASE}.stg_engine_telemetry FINAL) AS t
        INNER JOIN (SELECT engine_id, lifecycle_id, minIf(cycle_number, status = 'FAILURE') AS failure_cycle
                    FROM {DATABASE}.stg_engine_telemetry FINAL GROUP BY engine_id, lifecycle_id) AS f
            ON t.engine_id = f.engine_id AND t.lifecycle_id = f.lifecycle_id
        WHERE (t.engine_id, t.lifecycle_id) IN ({in_window or "('', 0)"})
          AND t.cycle_number <= f.failure_cycle""",
    )
    mart_rows = scalar(client, f"SELECT count() FROM {DATABASE}.mart_rul_training_features FINAL")
    capped = sum(max(len(ids) - max_lifecycles, 0) for ids in completed.values())
    LOGGER.info(
        "mart window of %d lifecycles per engine: %d kept (%d older left out), %d rows (expected %d)",
        max_lifecycles,
        len(actual),
        capped,
        mart_rows,
        expected_rows,
    )
    failures = []
    if actual != expected:
        failures.append(
            f"mart lifecycles: missing {sorted(expected - actual)}, extra {sorted(actual - expected)}"
        )
    if mart_rows != expected_rows:
        failures.append(f"mart has {mart_rows} rows, expected {expected_rows}")
    if not capped:
        failures.append(
            f"max {max_lifecycles} lifecycles per engine leaves out no lifecycle: nothing was tested"
        )
    return failures
