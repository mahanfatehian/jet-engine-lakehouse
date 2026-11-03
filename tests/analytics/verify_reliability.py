"""Checks of the reliability models against reliability_reference.py (see verify_analytics.py).

check_reliability  reads the CDC versions of engine_status_registry, work orders and pulses back from
                   the raw tables exactly as they landed (replays, an out-of-order version, a delete
                   and a re-registration, a registration dated after the history written behind it),
                   rebuilds every reliability model in Python and compares every row and column of
                   int_platform_outages, int_engine_status_history, fct_engine_failures,
                   fct_engine_lifecycle_status_time, rpt_fleet_reliability_daily and
                   rpt_failure_mode_pareto; then checks the fixture's own truth (synthetic_cdc.py,
                   manifest key ``reliability``): the failures it simulated with their return to
                   service and operating run, each registry scenario, and that every failure is a
                   failed lifecycle of int_engine_lifecycles at the same cycle and instant
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

from clickhouse_connect.driver.client import Client
from reliability_reference import (
    Failure,
    RegistryVersion,
    StatusInterval,
    WorkOrder,
    as_of_us,
    daily_reliability,
    failure_mode_pareto,
    failures,
    lifecycle_status_time,
    platform_outages,
    status_intervals,
)
from verify_queries import DATABASE, rows

LOGGER = logging.getLogger("verify_analytics")
OUTAGE_MIN_GAP_US = 60_000_000  # dbt var outage_min_gap_seconds
Table = dict[tuple[Any, ...], dict[str, Any]]
STATUS_SECONDS = (
    "healthy_seconds",
    "warning_seconds",
    "critical_seconds",
    "failure_seconds",
    "maintenance_reset_seconds",
)
OUTAGE_COLUMNS = {"outage_end_us": "toUnixTimestamp64Micro(outage_end)"}
HISTORY_COLUMNS = {
    "engine_model": "toString(engine_model)",
    "operational_status": "toString(operational_status)",
    "is_available": "is_available",
    "lifecycle_id": "lifecycle_id",
    "cycle_at_entry": "cycle_at_entry",
    "valid_to_us": "toUnixTimestamp64Micro(valid_to)",
    "is_current": "is_current",
    "duration_seconds": "duration_seconds",
    "closed_by_delete": "closed_by_delete",
    "entry_cdc_op": "toString(entry_cdc_op)",
    "entry_cdc_lsn": "entry_cdc_lsn",
}
FAILURE_COLUMNS = {
    "engine_model": "toString(engine_model)",
    "lifecycle_id": "lifecycle_id",
    "failure_cycle": "failure_cycle",
    "returned_us": "toUnixTimestamp64Micro(returned_to_service_at)",
    "repair_seconds": "repair_seconds",
    "operating_since_us": "toUnixTimestamp64Micro(operating_since)",
    "operating_seconds": "operating_seconds",
    "operating_cycles": "operating_cycles",
    "maintenance_id": "maintenance_id",
    "failure_mode": "toString(failure_mode)",
    "work_order_downtime_seconds": "work_order_downtime_seconds",
}
LIFECYCLE_COLUMNS = {
    "engine_model": "toString(engine_model)",
    "observed_from_us": "toUnixTimestamp64Micro(observed_from)",
    "observed_to_us": "toUnixTimestamp64Micro(observed_to)",
    "is_current": "is_current",
    "status_intervals": "status_intervals",
    "failures": "failures",
    **{name: name for name in STATUS_SECONDS},
    "available_seconds": "available_seconds",
    "unavailable_seconds": "unavailable_seconds",
    "observed_seconds": "observed_seconds",
    "availability": "availability",
}
DAILY_COLUMNS = {
    name: name
    for name in (
        "engines",
        "observed_seconds",
        *STATUS_SECONDS,
        "available_seconds",
        "availability",
        "failures",
        "repairs",
        "mttr_seconds",
        "operating_periods",
        "mtbf_seconds",
        "mtbf_cycles",
    )
}
PARETO_COLUMNS = {
    **{
        name: name
        for name in (
            "pareto_rank",
            "work_orders",
            "completed_work_orders",
            "share_of_work_orders",
            "cumulative_share",
            "engines_affected",
            "mean_failure_cycle",
            "total_downtime_seconds",
            "mean_downtime_seconds",
        )
    },
    "first_failure_us": "toUnixTimestamp64Micro(first_failure_ts)",
    "last_failure_us": "toUnixTimestamp64Micro(last_failure_ts)",
}


def _iso_micros(value: str | None) -> int | None:
    return None if value is None else round(datetime.fromisoformat(value).timestamp() * 1_000_000)


def read_registry_versions(client: Client) -> list[RegistryVersion]:
    """Every row of raw_engine_status_registry (replays included), its status cleaned like staging."""
    return [
        RegistryVersion(
            engine_id=row[0],
            cdc_lsn=row[1],
            cdc_op=row[2],
            is_deleted=bool(row[3]),
            operational_status=row[4].strip().upper(),
            engine_model=row[5],
            lifecycle_id=row[6],
            cycle=row[7],
            status_changed_us=row[8],
            last_telemetry_us=row[9],
            committed_us=row[10],
        )
        for row in rows(
            client,
            f"""
            SELECT engine_id, cdc_lsn, toString(cdc_op), is_deleted, toString(operational_status),
                   toString(engine_model), current_lifecycle_id, current_cycle,
                   toUnixTimestamp64Micro(status_changed_ts), toUnixTimestamp64Micro(last_telemetry_ts),
                   toUnixTimestamp64Micro(toDateTime64(cdc_source_ts, 6, 'UTC'))
            FROM {DATABASE}.raw_engine_status_registry""",
        )
    ]


def read_pulse_times(client: Client) -> list[int]:
    """Event time of every pulse version that landed: a retracted pulse still proves the platform ran."""
    sql = f"SELECT DISTINCT toUnixTimestamp64Micro(event_ts) FROM {DATABASE}.raw_engine_telemetry"
    return [row[0] for row in rows(client, sql)]


def read_work_orders(client: Client) -> list[WorkOrder]:
    """Newest version (cdc_lsn, then ingested_at) of every work order, deleted ones removed."""
    newest: dict[tuple[str, int], tuple[Any, ...]] = {}
    for row in rows(
        client,
        f"""
        SELECT engine_id, maintenance_id, lifecycle_id, toString(failure_mode), toString(maintenance_status),
               failure_cycle, toUnixTimestamp64Micro(failure_ts), downtime_seconds, is_deleted, cdc_lsn,
               toUnixTimestamp64Micro(ingested_at)
        FROM {DATABASE}.raw_maintenance_log""",
    ):
        key = (row[0], row[1])
        if key not in newest or row[9:11] > newest[key][9:11]:
            newest[key] = row
    return [
        WorkOrder(
            maintenance_id=row[1],
            engine_id=row[0],
            lifecycle_id=row[2],
            failure_mode=row[3].strip().upper(),
            maintenance_status=row[4].strip().upper(),
            failure_cycle=row[5],
            failure_us=row[6],
            downtime_seconds=row[7],
        )
        for row in newest.values()
        if not row[8]
    ]


def read_model(client: Client, model: str, keys: tuple[str, ...], columns: dict[str, str]) -> Table:
    """A model under FINAL by key; ``columns`` maps result names to SQL expressions."""
    selected = ", ".join((*keys, *columns.values()))
    return {
        tuple(row[: len(keys)]): dict(zip(columns, row[len(keys) :], strict=True))
        for row in rows(client, f"SELECT {selected} FROM {DATABASE}.{model} FINAL")
    }


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and actual is not None:
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
    return actual == expected


def compare_rows(model: str, actual: Table, expected: Table) -> list[str]:
    """Missing and unexpected keys and every differing value of a model against its reference."""
    found = []
    if missing := sorted(expected.keys() - actual.keys(), key=str):
        found.append(f"{model}: {len(missing)} reference rows missing, e.g. {missing[:3]}")
    if extra := sorted(actual.keys() - expected.keys(), key=str):
        found.append(f"{model}: {len(extra)} rows not in the reference, e.g. {extra[:3]}")
    shared = sorted(expected.keys() & actual.keys(), key=str)
    mismatches = [
        (key, name, actual[key][name], value)
        for key in shared
        for name, value in expected[key].items()
        if not _same(actual[key][name], value)
    ]
    if mismatches:
        found.append(f"{model}: {len(mismatches)} values differ from the reference, e.g. {mismatches[:3]}")
    LOGGER.info(
        "%s: %d rows (reference %d), %d values compared, %d differ",
        model,
        len(actual),
        len(expected),
        sum(len(expected[key]) for key in shared),
        len(mismatches),
    )
    return found


def _history_reference(intervals: list[StatusInterval]) -> Table:
    return {
        (i.engine_id, i.valid_from_us): {
            "engine_model": i.engine_model,
            "operational_status": i.operational_status,
            "is_available": int(i.is_available),
            "lifecycle_id": i.lifecycle_id,
            "cycle_at_entry": i.cycle_at_entry,
            "valid_to_us": i.valid_to_us,
            "is_current": int(i.valid_to_us is None),
            "duration_seconds": None if i.valid_to_us is None else (i.valid_to_us - i.valid_from_us) / 1e6,
            "closed_by_delete": int(i.closed_by_delete),
            "entry_cdc_op": i.entry_cdc_op,
            "entry_cdc_lsn": i.entry_cdc_lsn,
        }
        for i in intervals
    }


def _failure_reference(failure_list: list[Failure], orders: list[WorkOrder]) -> Table:
    work_order = {(o.engine_id, o.lifecycle_id): o for o in sorted(orders, key=lambda o: o.maintenance_id)}
    expected = {}
    for f in failure_list:
        order = work_order.get((f.engine_id, f.lifecycle_id))
        expected[(f.engine_id, f.failed_at_us)] = {
            "engine_model": f.engine_model,
            "lifecycle_id": f.lifecycle_id,
            "failure_cycle": f.failure_cycle,
            "returned_us": f.returned_us,
            "repair_seconds": None if f.repair_us is None else f.repair_us / 1e6,
            "operating_since_us": f.operating_since_us,
            "operating_seconds": None if f.operating_us is None else f.operating_us / 1e6,
            "operating_cycles": f.operating_cycles,
            "maintenance_id": None if order is None else order.maintenance_id,
            "failure_mode": "UNKNOWN" if order is None else order.failure_mode,
            "work_order_downtime_seconds": None if order is None else order.downtime_seconds,
        }
    return expected


def check_reference(client: Client) -> tuple[list[str], list[StatusInterval]]:
    """Every reliability model equals its Python reference, row by row and column by column."""
    versions = read_registry_versions(client)
    orders = read_work_orders(client)
    intervals = status_intervals(versions)
    outages = platform_outages(read_pulse_times(client), OUTAGE_MIN_GAP_US)
    failure_list = failures(intervals)
    horizon = as_of_us(versions)
    found = compare_rows(
        "int_platform_outages",
        read_model(client, "int_platform_outages", ("toUnixTimestamp64Micro(outage_start)",), OUTAGE_COLUMNS),
        {(start,): {"outage_end_us": end} for start, end in outages},
    )
    found += compare_rows(
        "int_engine_status_history",
        read_model(
            client,
            "int_engine_status_history",
            ("toString(engine_id)", "toUnixTimestamp64Micro(valid_from)"),
            HISTORY_COLUMNS,
        ),
        _history_reference(intervals),
    )
    found += compare_rows(
        "fct_engine_failures",
        read_model(
            client,
            "fct_engine_failures",
            ("toString(engine_id)", "toUnixTimestamp64Micro(failed_at)"),
            FAILURE_COLUMNS,
        ),
        _failure_reference(failure_list, orders),
    )
    found += compare_rows(
        "fct_engine_lifecycle_status_time",
        read_model(
            client,
            "fct_engine_lifecycle_status_time",
            ("toString(engine_id)", "lifecycle_id"),
            LIFECYCLE_COLUMNS,
        ),
        lifecycle_status_time(intervals, failure_list, horizon),
    )
    daily = read_model(
        client,
        "rpt_fleet_reliability_daily",
        ("day", "toString(engine_id)"),
        {**DAILY_COLUMNS, "as_of_us": "toUnixTimestamp64Micro(as_of_ts)"},
    )
    horizons = {row.pop("as_of_us") for row in daily.values()}
    if horizons - {horizon}:
        found.append(f"rpt_fleet_reliability_daily: as_of_ts {sorted(horizons)}, expected {horizon}")
    found += compare_rows(
        "rpt_fleet_reliability_daily", daily, daily_reliability(intervals, failure_list, horizon)
    )
    found += compare_rows(
        "rpt_failure_mode_pareto",
        read_model(
            client,
            "rpt_failure_mode_pareto",
            ("toString(engine_id)", "toString(failure_mode)"),
            PARETO_COLUMNS,
        ),
        failure_mode_pareto(orders),
    )
    return found, intervals


def _simulated_failures(client: Client, scenario: dict[str, Any]) -> list[str]:
    """fct_engine_failures holds exactly the simulated failures, as int_engine_lifecycles saw them."""
    simulated = {
        (f["engine_id"], f["lifecycle_id"]): (
            f["failure_cycle"],
            _iso_micros(f["failed_at"]),
            _iso_micros(f["returned_to_service_at"]),
            _iso_micros(f["operating_since"]),
            f["operating_cycles"],
            f["failure_mode"],
        )
        for f in scenario["failures"]
    }
    published = {
        (row[0], row[1]): tuple(row[2:])
        for row in rows(
            client,
            f"""
            SELECT toString(engine_id), lifecycle_id, failure_cycle, toUnixTimestamp64Micro(failed_at),
                   toUnixTimestamp64Micro(returned_to_service_at), toUnixTimestamp64Micro(operating_since),
                   operating_cycles, toString(failure_mode)
            FROM {DATABASE}.fct_engine_failures FINAL""",
        )
    }
    failed_lifecycles = {
        (row[0], row[1]): (row[2], row[3])
        for row in rows(
            client,
            "SELECT toString(engine_id), lifecycle_id, failure_cycle, toUnixTimestamp64Micro(failure_ts) "
            f"FROM {DATABASE}.int_engine_lifecycles FINAL WHERE has_failed = 1",
        )
    }
    LOGGER.info(
        "fct_engine_failures: %d failures, %d simulated, %d failed lifecycles in int_engine_lifecycles",
        len(published),
        len(simulated),
        len(failed_lifecycles),
    )
    found = []
    if published != simulated:
        keys = sorted(k for k in simulated.keys() | published.keys() if simulated.get(k) != published.get(k))
        found.append(f"fct_engine_failures differs from the simulated failures at {keys[:4]}")
    if {key: value[:2] for key, value in published.items()} != failed_lifecycles:
        found.append("fct_engine_failures and the failed lifecycles of int_engine_lifecycles disagree")
    return found


def _backfilled_engine(intervals: list[StatusInterval], scenario: dict[str, Any]) -> list[str]:
    """History written behind the registration starts at its first change; the registration is dropped."""
    first = min(intervals, key=lambda i: i.valid_from_us)
    if first.valid_from_us != _iso_micros(scenario["first_change_at"]) or first.entry_cdc_op != "u":
        return [f"{scenario['engine_id']}: history starts at {first}, not at its first status change"]
    if any(i.valid_from_us == _iso_micros(scenario["registered_at"]) for i in intervals):
        return [f"{scenario['engine_id']}: the registration dated after its history still opens an interval"]
    return []


def _deregistered_engine(intervals: list[StatusInterval], scenario: dict[str, Any]) -> list[str]:
    """Registered, deleted, registered again: two HEALTHY intervals with the deregistration between."""
    shape = [
        (i.operational_status, i.valid_from_us, i.valid_to_us, i.closed_by_delete, i.entry_cdc_op)
        for i in sorted(intervals, key=lambda i: i.valid_from_us)
    ]
    expected = [
        (
            "HEALTHY",
            _iso_micros(scenario["registered_at"]),
            _iso_micros(scenario["deregistered_at"]),
            True,
            "c",
        ),
        ("HEALTHY", _iso_micros(scenario["registered_again_at"]), None, False, "c"),
    ]
    return [] if shape == expected else [f"{scenario['engine_id']}: history {shape}, expected {expected}"]


def check_scenarios(client: Client, scenario: dict[str, Any], intervals: list[StatusInterval]) -> list[str]:
    """The fixture's truth: the simulated failures and every registry scenario it delivers."""
    by_engine: dict[str, list[StatusInterval]] = {}
    for interval in intervals:
        by_engine.setdefault(interval.engine_id, []).append(interval)
    backfilled, deregistered = scenario["backfilled_engine"], scenario["deregistered_engine"]
    found = [
        *_simulated_failures(client, scenario),
        *_backfilled_engine(by_engine.get(backfilled["engine_id"], []), backfilled),
        *_deregistered_engine(by_engine.get(deregistered["engine_id"], []), deregistered),
    ]
    late, replayed = scenario["out_of_order"], scenario["replayed"]
    opened = {(i.engine_id, i.entry_cdc_lsn): i for i in intervals}.get((late["engine_id"], late["cdc_lsn"]))
    if opened is None or (opened.operational_status, opened.lifecycle_id) != (
        late["operational_status"],
        late["lifecycle_id"],
    ):
        found.append(f"the out-of-order registry version {late} opens no interval of its own: {opened}")
    copies = rows(
        client,
        f"SELECT count() FROM {DATABASE}.raw_engine_status_registry "
        f"WHERE engine_id = '{replayed['engine_id']}' AND cdc_lsn = {int(replayed['cdc_lsn'])}",
    )[0][0]
    if copies < 2:
        found.append(f"the replayed registry version landed {copies} time(s): nothing was replayed")
    LOGGER.info(
        "registry scenarios: backfilled %s, deregistered %s, out-of-order lsn %d, lsn %d landed %d times",
        backfilled["engine_id"],
        deregistered["engine_id"],
        late["cdc_lsn"],
        replayed["cdc_lsn"],
        copies,
    )
    return found


def check_reliability(client: Client, manifest: dict[str, Any]) -> list[str]:
    """The reliability models against the Python reference and the fixture's scenario."""
    found, intervals = check_reference(client)
    return found + check_scenarios(client, manifest["reliability"], intervals)
