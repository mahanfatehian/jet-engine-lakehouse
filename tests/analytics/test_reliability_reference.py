"""Unit tests of reliability_reference.py, the pure-Python reference of the reliability models.

The integration harness trusts this reference to judge the dbt models, so the status model is
pinned here on hand-made change logs: folding of progress updates, replays and log order, the
backdated registration of a backfilled history, deregistration, failures with their repair and
operating times, the cut at UTC midnight, the time per lifecycle, outages and the failure-mode Pareto.
The fixture's own truth (synthetic_cdc.py) must agree with the reference as well.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from reliability_reference import (
    DAY_US,
    FLEET,
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
from synthetic_cdc import build_history

S = 1_000_000  # one second in microseconds
ENGINE = "JT9D-001"


def version(lsn: int, status: str, changed_s: float, **overrides: Any) -> RegistryVersion:
    """A registry version of ENGINE in ``status`` since ``changed_s`` seconds after the epoch."""
    fields: dict[str, Any] = {
        "engine_id": ENGINE,
        "cdc_lsn": lsn,
        "cdc_op": "u",
        "is_deleted": False,
        "operational_status": status,
        "engine_model": "JT9D-7R4",
        "lifecycle_id": 1,
        "cycle": 0,
        "status_changed_us": round(changed_s * S),
        "last_telemetry_us": None,
        "committed_us": round(changed_s * S),
    }
    fields.update(overrides)
    return RegistryVersion(**fields)


def shape(intervals: list[StatusInterval]) -> list[tuple[str, float, float | None, bool]]:
    """(status, valid_from s, valid_to s, closed_by_delete) of every interval."""
    return [
        (
            i.operational_status,
            i.valid_from_us / S,
            None if i.valid_to_us is None else i.valid_to_us / S,
            i.closed_by_delete,
        )
        for i in intervals
    ]


LIFECYCLE = [
    version(1, "HEALTHY", 0, cdc_op="c"),
    version(2, "HEALTHY", 0, cycle=1, last_telemetry_us=10 * S),
    version(3, "WARNING", 50, cycle=5),
    version(4, "WARNING", 50, cycle=6, last_telemetry_us=60 * S),
    version(5, "CRITICAL", 80, cycle=8),
    version(6, "FAILURE", 100, cycle=9),
    version(7, "MAINTENANCE_RESET", 120, lifecycle_id=2),
    version(8, "HEALTHY", 121, lifecycle_id=2),
    version(9, "HEALTHY", 121, lifecycle_id=2, cycle=1, last_telemetry_us=130 * S),
    version(10, "WARNING", 170, lifecycle_id=2, cycle=5),
    version(11, "FAILURE", 200, lifecycle_id=2, cycle=7),
]


def test_progress_updates_fold_into_the_interval_of_their_status() -> None:
    intervals = status_intervals(LIFECYCLE)
    assert shape(intervals) == [
        ("HEALTHY", 0, 50, False),
        ("WARNING", 50, 80, False),
        ("CRITICAL", 80, 100, False),
        ("FAILURE", 100, 120, False),
        ("MAINTENANCE_RESET", 120, 121, False),
        ("HEALTHY", 121, 170, False),
        ("WARNING", 170, 200, False),
        ("FAILURE", 200, None, False),
    ]
    assert [(i.entry_cdc_lsn, i.lifecycle_id, i.cycle_at_entry) for i in intervals[:3]] == [
        (1, 1, 0),
        (3, 1, 5),
        (5, 1, 8),
    ]


def test_replays_and_arrival_order_change_nothing() -> None:
    delivered = [*reversed(LIFECYCLE), LIFECYCLE[2], LIFECYCLE[5]]
    assert status_intervals(delivered) == status_intervals(LIFECYCLE)


def test_a_late_status_change_opens_its_interval_instead_of_the_next_version() -> None:
    without = status_intervals([v for v in LIFECYCLE if v.cdc_lsn != 3])
    warning = next(i for i in without if i.operational_status == "WARNING")
    # the next version carries the same status and status_changed_ts: same interval, later entry
    assert (warning.valid_from_us, warning.entry_cdc_lsn, warning.cycle_at_entry) == (50 * S, 4, 6)


def test_history_written_behind_its_registration_drops_the_registration() -> None:
    # the first-boot backfill: registered now (t=500), history backdated before it
    backfilled = [version(1, "HEALTHY", 500, cdc_op="c"), *LIFECYCLE[2:]]
    intervals = status_intervals(backfilled)
    assert shape(intervals)[0] == ("WARNING", 50, 80, False)
    assert all(i.valid_from_us != 500 * S for i in intervals)


def test_a_backdated_change_supersedes_the_changes_it_predates() -> None:
    log = [version(1, "HEALTHY", 10), version(2, "WARNING", 30), version(3, "CRITICAL", 20)]
    assert shape(status_intervals(log)) == [("HEALTHY", 10, 20, False), ("CRITICAL", 20, None, False)]


def test_a_delete_closes_the_interval_until_the_engine_is_registered_again() -> None:
    log = [
        version(1, "HEALTHY", 0, cdc_op="c"),
        version(2, "HEALTHY", 0, cdc_op="d", is_deleted=True, committed_us=40 * S),
        version(3, "HEALTHY", 90, cdc_op="c"),
    ]
    assert shape(status_intervals(log)) == [("HEALTHY", 0, 40, True), ("HEALTHY", 90, None, False)]
    gone = status_intervals(log[:2])
    assert shape(gone) == [("HEALTHY", 0, 40, True)]


def test_a_change_at_the_instant_of_the_delete_still_closes_by_deregistration() -> None:
    log = [
        version(1, "HEALTHY", 0, cdc_op="c"),
        version(2, "WARNING", 40),
        version(3, "WARNING", 40, cdc_op="d", is_deleted=True, committed_us=40 * S),
    ]
    assert shape(status_intervals(log)) == [("HEALTHY", 0, 40, True)]


def test_failures_measure_the_repair_and_the_operating_run_before_them() -> None:
    first, second = failures(status_intervals(LIFECYCLE))
    # the first run starts at the registration: a known start of service
    assert (first.failed_at_us, first.returned_us, first.repair_us) == (100 * S, 121 * S, 21 * S)
    assert (first.operating_since_us, first.operating_us, first.operating_cycles) == (0, 100 * S, 10)
    assert (second.lifecycle_id, second.failure_cycle, second.operating_cycles) == (2, 7, 8)
    assert (second.operating_us, second.returned_us, second.repair_us) == (79 * S, None, None)


def test_an_operating_run_older_than_the_history_has_no_operating_time() -> None:
    snapshot = [version(1, "WARNING", 50, cdc_op="r", cycle=6), *LIFECYCLE[4:]]
    first = failures(status_intervals(snapshot))[0]
    assert (first.operating_since_us, first.operating_cycles, first.repair_us) == (None, None, 21 * S)


def test_a_failure_deregistered_before_its_return_is_never_repaired() -> None:
    log = [*LIFECYCLE[:6], version(12, "FAILURE", 100, cdc_op="d", is_deleted=True, committed_us=110 * S)]
    (failure,) = failures(status_intervals(log))
    assert (failure.returned_us, failure.repair_us) == (None, None)


def test_open_intervals_are_measured_up_to_the_horizon_of_the_log() -> None:
    log = [version(1, "HEALTHY", 0, cdc_op="c", last_telemetry_us=45 * S), version(2, "WARNING", 30)]
    horizon = as_of_us(log)
    assert horizon == 45 * S
    rows = lifecycle_status_time(status_intervals(log), [], horizon)
    assert rows[(ENGINE, 1)]["warning_seconds"] == 15
    assert rows[(ENGINE, 1)]["observed_to_us"] == 45 * S
    deleted = [*log, version(3, "WARNING", 30, cdc_op="d", is_deleted=True, committed_us=60 * S)]
    assert as_of_us(deleted) == 60 * S


def test_days_split_at_utc_midnight_and_the_fleet_row_sums_its_engines() -> None:
    midnight = DAY_US // S
    log = [
        version(1, "HEALTHY", midnight - 30, cdc_op="c"),
        version(2, "FAILURE", midnight + 30),
        version(3, "HEALTHY", midnight + 40),
        version(
            4,
            "HEALTHY",
            midnight - 10,
            engine_id="GE90-201",
            cdc_op="c",
            last_telemetry_us=(midnight + 50) * S,
        ),
    ]
    intervals = status_intervals(log)
    rows = daily_reliability(intervals, failures(intervals), as_of_us(log))
    first_day, second_day = date(1970, 1, 1), date(1970, 1, 2)
    assert rows[(first_day, ENGINE)]["healthy_seconds"] == 30
    assert rows[(second_day, ENGINE)]["observed_seconds"] == 50
    assert rows[(second_day, ENGINE)]["availability"] == pytest.approx(40 / 50)
    assert (rows[(second_day, ENGINE)]["failures"], rows[(second_day, ENGINE)]["mttr_seconds"]) == (1, 10)
    assert rows[(second_day, ENGINE)]["mtbf_seconds"] == 60
    fleet = rows[(second_day, FLEET)]
    assert (fleet["engines"], fleet["observed_seconds"], fleet["failures"]) == (2, 100, 1)
    assert rows[(first_day, FLEET)]["observed_seconds"] == 40


def test_lifecycle_time_counts_each_status_once() -> None:
    intervals = status_intervals(LIFECYCLE)
    rows = lifecycle_status_time(intervals, failures(intervals), as_of_us(LIFECYCLE))
    first, second = rows[(ENGINE, 1)], rows[(ENGINE, 2)]
    assert (first["healthy_seconds"], first["warning_seconds"], first["critical_seconds"]) == (50, 30, 20)
    assert (first["failure_seconds"], first["failures"], first["is_current"]) == (20, 1, 0)
    assert (second["maintenance_reset_seconds"], second["failure_seconds"], second["is_current"]) == (1, 0, 1)
    assert first["availability"] == pytest.approx(100 / 120)


def test_only_pauses_of_the_fleet_longer_than_the_gap_are_outages() -> None:
    # ticks, the 21 s pause of every engine under repair at once, a pause of exactly the gap, a stop
    pulses = [s * S for s in (0, 1, 1, 22, 82, 83, 200)]
    assert platform_outages(pulses, 60 * S) == [(83 * S, 200 * S)]


def test_the_pareto_ranks_modes_and_accumulates_their_shares() -> None:
    def order(number: int, engine: str, mode: str, downtime: int | None) -> WorkOrder:
        status = "OPEN" if downtime is None else "COMPLETED"
        return WorkOrder(number, engine, number, mode, status, 10 * number, number * S, downtime)

    orders = [
        order(1, ENGINE, "OIL_SYSTEM_LEAK", 20),
        order(2, ENGINE, "BEARING_DEGRADATION", 22),
        order(3, "GE90-201", "OIL_SYSTEM_LEAK", None),
        order(4, "GE90-201", "HPT_BLADE_EROSION", 20),
    ]
    rows = failure_mode_pareto(orders)
    fleet = [rows[(FLEET, mode)] for mode in ("OIL_SYSTEM_LEAK", "BEARING_DEGRADATION", "HPT_BLADE_EROSION")]
    assert [row["pareto_rank"] for row in fleet] == [1, 2, 3]
    assert [row["cumulative_share"] for row in fleet] == [0.5, 0.75, 1.0]
    assert (fleet[0]["engines_affected"], fleet[0]["completed_work_orders"]) == (2, 1)
    assert (fleet[0]["total_downtime_seconds"], fleet[0]["mean_downtime_seconds"]) == (20, 20)
    assert rows[("GE90-201", "OIL_SYSTEM_LEAK")]["total_downtime_seconds"] is None


def _micros(value: datetime) -> int:
    return round(value.timestamp() * S)


def test_the_fixture_failures_follow_from_its_registry_log() -> None:
    history = build_history(42)
    versions = [
        RegistryVersion(
            engine_id=e.image["engine_id"],
            cdc_lsn=e.lsn,
            cdc_op=e.op,
            is_deleted=e.op == "d",
            operational_status=e.image["operational_status"],
            engine_model=e.image["engine_model"],
            lifecycle_id=e.image["current_lifecycle_id"],
            cycle=e.image["current_cycle"],
            status_changed_us=_micros(e.image["status_changed_ts"]),
            last_telemetry_us=None
            if e.image["last_telemetry_ts"] is None
            else _micros(e.image["last_telemetry_ts"]),
            committed_us=_micros(e.commit_ts),
        )
        for e in history.events
        if e.table == "engine_status_registry"
    ]
    derived = [
        (
            f.engine_id,
            f.lifecycle_id,
            f.failure_cycle,
            f.failed_at_us,
            f.returned_us,
            f.operating_since_us,
            f.operating_cycles,
        )
        for f in failures(status_intervals(versions))
    ]
    simulated = [
        (
            f["engine_id"],
            f["lifecycle_id"],
            f["failure_cycle"],
            _micros(datetime.fromisoformat(f["failed_at"])),
            _micros(datetime.fromisoformat(f["returned_to_service_at"])),
            None if f["operating_since"] is None else _micros(datetime.fromisoformat(f["operating_since"])),
            f["operating_cycles"],
        )
        for f in history.manifest["reliability"]["failures"]
    ]
    assert derived == simulated
    # the spare engine's first interval spans a UTC midnight, so the daily report cuts it
    spare = history.manifest["reliability"]["deregistered_engine"]
    registered, deregistered = (
        datetime.fromisoformat(spare[key]) for key in ("registered_at", "deregistered_at")
    )
    assert registered.tzinfo == UTC and registered.date() < deregistered.date()
