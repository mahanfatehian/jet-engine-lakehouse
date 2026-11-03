"""Independent Python reference of the reliability models (int_engine_status_history,
int_platform_outages and the gold reliability models) for verify_reliability.py and its unit tests.

It rebuilds from the CDC versions of engine_status_registry, the maintenance work orders and the
pulses' event times what the dbt models must hold, with plain loops instead of window functions:

status_intervals         the SCD2 history: versions in log order, replays collapsed; an interval
                         opens when (status, status_changed_ts) changes or after a delete; a delete
                         closes the open interval at its commit time; boundary times are made
                         monotonic by a suffix minimum (a change backdated before earlier ones
                         supersedes them) and intervals without duration are dropped
platform_outages         every pause of the fleet's pulses longer than the gap, as [start, end)
failures                 one per run of unavailable intervals that contains a FAILURE, with the
                         time to repair and the operating time and cycles since the start of the
                         operating run before it (a return to service or a registration)
lifecycle_status_time    time in every status per engine lifecycle
daily_reliability        availability, failures, MTTR and MTBF per UTC day and engine, plus 'ALL'
failure_mode_pareto      work orders per failure mode, ranked, with cumulative shares

Times are integer microseconds since the epoch (UTC), as toUnixTimestamp64Micro returns them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import accumulate, groupby, pairwise
from typing import Any

AVAILABLE = frozenset({"HEALTHY", "WARNING", "CRITICAL"})
STATUSES = ("HEALTHY", "WARNING", "CRITICAL", "FAILURE", "MAINTENANCE_RESET")
FLEET = "ALL"
DAY_US = 86_400_000_000
EPOCH = date(1970, 1, 1)
Outage = tuple[int, int]


@dataclass(frozen=True)
class RegistryVersion:
    """One CDC version of an engine_status_registry row (the before-image for a delete)."""

    engine_id: str
    cdc_lsn: int
    cdc_op: str
    is_deleted: bool
    operational_status: str
    engine_model: str
    lifecycle_id: int
    cycle: int
    status_changed_us: int
    last_telemetry_us: int | None
    committed_us: int


@dataclass(frozen=True)
class StatusInterval:
    """One row of the SCD2 status history; valid_to_us is None while the interval is open."""

    engine_id: str
    engine_model: str
    operational_status: str
    lifecycle_id: int
    cycle_at_entry: int
    valid_from_us: int
    valid_to_us: int | None
    closed_by_delete: bool
    entry_cdc_op: str
    entry_cdc_lsn: int

    @property
    def is_available(self) -> bool:
        return self.operational_status in AVAILABLE

    def end_us(self, as_of_us: int) -> int:
        """End of the interval, an open one measured up to the history's horizon."""
        return self.valid_to_us if self.valid_to_us is not None else max(self.valid_from_us, as_of_us)


@dataclass(frozen=True)
class Failure:
    """One failure: a run of unavailable intervals that contains a FAILURE interval."""

    engine_id: str
    engine_model: str
    lifecycle_id: int
    failure_cycle: int
    failed_at_us: int
    returned_us: int | None
    operating_since_us: int | None
    operating_cycles: int | None

    @property
    def repair_us(self) -> int | None:
        return None if self.returned_us is None else self.returned_us - self.failed_at_us

    @property
    def operating_us(self) -> int | None:
        return None if self.operating_since_us is None else self.failed_at_us - self.operating_since_us


@dataclass(frozen=True)
class WorkOrder:
    """Current version of a maintenance work order."""

    maintenance_id: int
    engine_id: str
    lifecycle_id: int
    failure_mode: str
    maintenance_status: str
    failure_cycle: int
    failure_us: int
    downtime_seconds: int | None


@dataclass
class _Tally:
    """Accumulated time per status and failure statistics of one report cell."""

    status_us: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    engines: set[str] = field(default_factory=set)
    failures: int = 0
    repairs: list[int] = field(default_factory=list)
    operating: list[int] = field(default_factory=list)
    cycles: list[int] = field(default_factory=list)

    def add_failure(self, failure: Failure) -> None:
        self.failures += 1
        if failure.repair_us is not None:
            self.repairs.append(failure.repair_us)
        if failure.operating_us is not None:
            self.operating.append(failure.operating_us)
        if failure.operating_cycles is not None:
            self.cycles.append(failure.operating_cycles)


def _mean(values: Sequence[int], scale: float = 1.0) -> float | None:
    return sum(values) / len(values) / scale if values else None


def platform_outages(pulse_us: Iterable[int], min_gap_us: int) -> list[Outage]:
    """int_platform_outages: every pause between two pulses of the fleet longer than ``min_gap_us``."""
    return [(start, end) for start, end in pairwise(sorted(set(pulse_us))) if end - start > min_gap_us]


def as_of_us(versions: Iterable[RegistryVersion]) -> int:
    """Horizon of the history: the newest pulse, status change or delete commit the log reports."""
    return max(
        version.committed_us
        if version.is_deleted
        else max(version.status_changed_us, version.last_telemetry_us or version.status_changed_us)
        for version in versions
    )


def _boundaries(versions: Sequence[RegistryVersion]) -> list[RegistryVersion]:
    """Versions that open an interval, and deletes, in log order."""
    boundaries: list[RegistryVersion] = []
    previous: RegistryVersion | None = None
    for version in versions:
        state = (version.operational_status, version.status_changed_us)
        if (
            version.is_deleted
            or previous is None
            or previous.is_deleted
            or state != (previous.operational_status, previous.status_changed_us)
        ):
            boundaries.append(version)
        previous = version
    return boundaries


def _engine_intervals(versions: Sequence[RegistryVersion]) -> list[StatusInterval]:
    boundaries = _boundaries(versions)
    times = [b.committed_us if b.is_deleted else b.status_changed_us for b in boundaries]
    effective = list(accumulate(reversed(times), min))[::-1]
    kept: list[tuple[int, RegistryVersion, int | None]] = []
    for index, boundary in enumerate(boundaries):
        if boundary.is_deleted:
            continue
        valid_to = effective[index + 1] if index + 1 < len(boundaries) else None
        if valid_to is None or valid_to > effective[index]:
            kept.append((index, boundary, valid_to))
    intervals = []
    for number, (index, boundary, valid_to) in enumerate(kept):
        next_index = kept[number + 1][0] if number + 1 < len(kept) else len(boundaries)
        intervals.append(
            StatusInterval(
                engine_id=boundary.engine_id,
                engine_model=boundary.engine_model,
                operational_status=boundary.operational_status,
                lifecycle_id=boundary.lifecycle_id,
                cycle_at_entry=boundary.cycle,
                valid_from_us=effective[index],
                valid_to_us=valid_to,
                closed_by_delete=any(b.is_deleted for b in boundaries[index + 1 : next_index]),
                entry_cdc_op=boundary.cdc_op,
                entry_cdc_lsn=boundary.cdc_lsn,
            )
        )
    return intervals


def status_intervals(versions: Iterable[RegistryVersion]) -> list[StatusInterval]:
    """The SCD2 status history of every engine (see the module doc)."""
    by_engine: dict[str, dict[int, RegistryVersion]] = defaultdict(dict)
    for version in versions:
        by_engine[version.engine_id][version.cdc_lsn] = version  # replays collapse
    intervals = []
    for engine_id in sorted(by_engine):
        log = by_engine[engine_id]
        intervals += _engine_intervals([log[lsn] for lsn in sorted(log)])
    return intervals


def _runs(intervals: Sequence[StatusInterval]) -> list[tuple[list[StatusInterval], bool]]:
    """Runs of one engine's intervals of equal availability, each with whether it directly
    follows the previous run (the history's start and a deregistration break runs)."""
    runs: list[tuple[list[StatusInterval], bool]] = []
    previous: StatusInterval | None = None
    for interval in intervals:
        contiguous = previous is not None and not previous.closed_by_delete
        if contiguous and previous is not None and previous.is_available == interval.is_available:
            runs[-1][0].append(interval)
        else:
            runs.append(([interval], contiguous))
        previous = interval
    return runs


def failures(intervals: Iterable[StatusInterval]) -> list[Failure]:
    """One failure per run of unavailable intervals that contains a FAILURE (see the module doc)."""
    result = []
    ordered = sorted(intervals, key=lambda i: (i.engine_id, i.valid_from_us))
    for _, engine_intervals in groupby(ordered, key=lambda i: i.engine_id):
        runs = _runs(list(engine_intervals))
        for number, (run, follows) in enumerate(runs):
            failed = [i for i in run if i.operational_status == "FAILURE"]
            if run[0].is_available or not failed:
                continue
            last = run[-1]
            returned = None if last.closed_by_delete else last.valid_to_us
            since = cycles = None
            if follows:
                before, before_follows = runs[number - 1]
                start = before[0]
                if before_follows or start.entry_cdc_op == "c":
                    since = start.valid_from_us
                    if (
                        start.lifecycle_id == failed[0].lifecycle_id
                        and failed[0].cycle_at_entry >= start.cycle_at_entry
                    ):
                        cycles = failed[0].cycle_at_entry - start.cycle_at_entry + 1
            result.append(
                Failure(
                    engine_id=failed[0].engine_id,
                    engine_model=run[0].engine_model,
                    lifecycle_id=failed[0].lifecycle_id,
                    failure_cycle=failed[0].cycle_at_entry,
                    failed_at_us=failed[0].valid_from_us,
                    returned_us=returned,
                    operating_since_us=since,
                    operating_cycles=cycles,
                )
            )
    return result


def _status_columns(status_us: dict[str, int]) -> dict[str, float]:
    columns = {f"{status.lower()}_seconds": status_us.get(status, 0) / 1e6 for status in STATUSES}
    available = sum(status_us.get(status, 0) for status in AVAILABLE)
    observed = sum(status_us.values())
    return {
        **columns,
        "available_seconds": available / 1e6,
        "observed_seconds": observed / 1e6,
        "availability": available / observed if observed > 0 else None,
    }


def lifecycle_status_time(
    intervals: Iterable[StatusInterval], failure_list: Iterable[Failure], as_of: int
) -> dict[tuple[str, int], dict[str, Any]]:
    """fct_engine_lifecycle_status_time by (engine_id, lifecycle_id)."""
    grouped: defaultdict[tuple[str, int], list[StatusInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[(interval.engine_id, interval.lifecycle_id)].append(interval)
    failure_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for failure in failure_list:
        failure_counts[(failure.engine_id, failure.lifecycle_id)] += 1
    rows = {}
    for key, members in grouped.items():
        members.sort(key=lambda i: i.valid_from_us)
        status_us: defaultdict[str, int] = defaultdict(int)
        for interval in members:
            status_us[interval.operational_status] += interval.end_us(as_of) - interval.valid_from_us
        columns = _status_columns(status_us)
        rows[key] = {
            "engine_model": members[0].engine_model,
            "observed_from_us": members[0].valid_from_us,
            "observed_to_us": max(interval.end_us(as_of) for interval in members),
            "is_current": int(any(i.valid_to_us is None for i in members)),
            "status_intervals": len(members),
            "failures": failure_counts[key],
            **{name: value for name, value in columns.items() if name != "observed_seconds"},
            "unavailable_seconds": columns["observed_seconds"] - columns["available_seconds"],
            "observed_seconds": columns["observed_seconds"],
        }
    return rows


def daily_reliability(
    intervals: Iterable[StatusInterval], failure_list: Iterable[Failure], as_of: int
) -> dict[tuple[date, str], dict[str, Any]]:
    """rpt_fleet_reliability_daily by (day, engine_id), the fleet under engine_id 'ALL'."""
    cells: defaultdict[tuple[int, str], _Tally] = defaultdict(_Tally)
    for interval in intervals:
        start, end = interval.valid_from_us, interval.end_us(as_of)
        while start < end:
            day = start // DAY_US
            piece_end = min(end, (day + 1) * DAY_US)
            for scope in (interval.engine_id, FLEET):
                cells[(day, scope)].status_us[interval.operational_status] += piece_end - start
                cells[(day, scope)].engines.add(interval.engine_id)
            start = piece_end
    for failure in failure_list:
        for scope in (failure.engine_id, FLEET):
            cells[(failure.failed_at_us // DAY_US, scope)].add_failure(failure)
    rows = {}
    for (day, scope), tally in cells.items():
        columns = _status_columns(tally.status_us)
        rows[(EPOCH + timedelta(days=day), scope)] = {
            "engines": len(tally.engines),
            **columns,
            "failures": tally.failures,
            "repairs": len(tally.repairs),
            "mttr_seconds": _mean(tally.repairs, 1e6),
            "operating_periods": len(tally.operating),
            "mtbf_seconds": _mean(tally.operating, 1e6),
            "mtbf_cycles": _mean(tally.cycles),
        }
    return rows


def failure_mode_pareto(work_orders: Iterable[WorkOrder]) -> dict[tuple[str, str], dict[str, Any]]:
    """rpt_failure_mode_pareto by (engine_id, failure_mode), the fleet under engine_id 'ALL'."""
    scopes: defaultdict[str, defaultdict[str, list[WorkOrder]]] = defaultdict(lambda: defaultdict(list))
    for order in work_orders:
        for scope in (order.engine_id, FLEET):
            scopes[scope][order.failure_mode].append(order)
    rows = {}
    for scope, modes in scopes.items():
        total = sum(len(orders) for orders in modes.values())
        ranked = sorted(modes.items(), key=lambda item: (-len(item[1]), item[0]))
        cumulative = 0
        for rank, (mode, orders) in enumerate(ranked, start=1):
            cumulative += len(orders)
            downtimes = [o.downtime_seconds for o in orders if o.downtime_seconds is not None]
            rows[(scope, mode)] = {
                "pareto_rank": rank,
                "work_orders": len(orders),
                "completed_work_orders": sum(o.maintenance_status == "COMPLETED" for o in orders),
                "share_of_work_orders": len(orders) / total,
                "cumulative_share": cumulative / total,
                "engines_affected": len({o.engine_id for o in orders}),
                "mean_failure_cycle": sum(o.failure_cycle for o in orders) / len(orders),
                "total_downtime_seconds": sum(downtimes) if downtimes else None,
                "mean_downtime_seconds": _mean(downtimes),
                "first_failure_us": min(o.failure_us for o in orders),
                "last_failure_us": max(o.failure_us for o in orders),
            }
    return rows
