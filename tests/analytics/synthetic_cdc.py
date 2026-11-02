"""Deterministic synthetic CDC history of the jet-engine fleet for the analytics checks.

The simulation mirrors the platform's PostgreSQL tables and their Debezium change stream:
1 Hz pulses grouped in 10-pulse cycles, lifecycles that degrade until a FAILURE pulse, a
maintenance work order opened at the failure and completed 20 s later, the lifecycle's
non-HEALTHY pulses linked to the work order by UPDATEs, and the registry row written the way the
simulator writes it (fleet_runtime): on every status change and on the first pulse of every
cycle, FAILURE at the failure pulse, MAINTENANCE_RESET into the next lifecycle when the repair
completes (plus one spare engine that is registered but never reports telemetry). JT9D-001's
registration row is dated after the start of its history, the way the first-boot backfill writes
history behind a registration the CDC connector has already captured.

``build_history`` splits the change stream into two delivery phases around a cut-off:

* ``base``: everything committed before the cut-off, minus the pulses and registry events held
  back below, plus at-least-once replays, one insert+delete pair and the spare engine's
  deregistration (a delete of its registry row).
* ``increment``: everything after the cut-off plus late data: pulses withheld from ``base``
  (one group in a lifecycle's baseline period), GE90-202's whole registry history (telemetry
  landed first), GE90-201's first MAINTENANCE_RESET (out of log order: the engine's later
  versions landed first), replays of already published versions (a registry change and a pulse's
  newest version), the spare engine's re-registration, a late delete and a late sensor
  correction of already published pulses.
  ``late_visible`` pulses must be loaded with an ingested_at older than the dbt watermark.

The manifest key ``reliability`` names these registry scenarios and the failures the fixture
simulated (failure, return to service, operating run): the truth of the reliability checks.

``History.frames`` holds the high-frequency frame features of the same pulses, delivered in the
same two phases (synthetic_hf.py; manifest key ``hf``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from synthetic_fleet import (
    FLEET,
    MAINTENANCE_SECONDS,
    PULSES_PER_CYCLE,
    SIM_START,
    SPARE_ENGINE,
    EnginePlan,
    Lifecycle,
    Pulse,
    simulate_engine,
)
from synthetic_hf import FrameCopy, build_hf_frames

FIRST_LSN = 26_000_000
REGISTERED_AT = SIM_START - timedelta(seconds=1)
# JT9D-001's history is written behind its registration (its first failure is ~400 s in)
BACKFILLED_ENGINE = "JT9D-001"
BACKFILL_REGISTERED_AT = SIM_START + timedelta(seconds=450)
# the spare engine, registered the day before (its first status interval spans a UTC midnight),
# has its registry row deleted before the cut-off and inserted again after it
SPARE_REGISTERED_AT = SIM_START - timedelta(days=1)
DEREGISTERED_BEFORE_CUTOFF = timedelta(seconds=200)
REREGISTERED_AFTER_CUTOFF = timedelta(seconds=150)
REPAIR_ACTIONS = {
    "HPT_BLADE_EROSION": "HPT_BLADE_REPLACEMENT",
    "BEARING_DEGRADATION": "BEARING_REPLACEMENT",
    "OIL_SYSTEM_LEAK": "OIL_SEAL_REPLACEMENT",
    "COMPRESSOR_FOULING": "COMPRESSOR_WASH",
}


@dataclass
class CdcEvent:
    """One Debezium change event; image is the after-image (before-image for deletes)."""

    table: str
    op: str
    commit_ts: datetime
    image: dict[str, Any]
    lsn: int = 0
    phase: str = "base"
    late_visible: bool = False


@dataclass
class History:
    """The change stream, increment-phase RUL predictions, HF frame copies and the facts the checks verify."""

    events: list[CdcEvent]
    cutoff: datetime
    predictions: list[dict[str, Any]]
    manifest: dict[str, Any]
    frames: list[FrameCopy]


def _telemetry_image(pulse: Pulse, maintenance_id: int | None = None) -> dict[str, Any]:
    return {
        "telemetry_id": pulse.telemetry_id,
        "engine_id": pulse.engine_id,
        "lifecycle_id": pulse.lifecycle_id,
        "cycle_number": pulse.cycle_number,
        "event_ts": pulse.event_ts,
        **pulse.readings,
        "wear_factor": pulse.wear,
        "status": pulse.status,
        "maintenance_id": maintenance_id,
        "created_ts": pulse.event_ts + timedelta(milliseconds=3),
    }


class _RegistryState:
    """Mutable registry row of one engine; ``update`` returns a snapshot of it for a CDC event."""

    def __init__(self, plan: EnginePlan, registered: datetime = REGISTERED_AT) -> None:
        self.row: dict[str, Any] = {
            "engine_id": plan.engine_id,
            "engine_model": plan.engine_model,
            "operational_status": "HEALTHY",
            "current_lifecycle_id": 1,
            "current_cycle": 0,
            "wear_factor": 0.0,
            "cycles_since_new": 0,
            "lifecycles_completed": 0,
            "last_maintenance_ts": None,
            "status_changed_ts": registered,
            "last_telemetry_ts": None,
            "updated_ts": registered,
            "registered_ts": registered,
        }

    def update(self, at: datetime, **changes: Any) -> dict[str, Any]:
        status = changes.get("operational_status")
        if status is not None and status != self.row["operational_status"]:
            self.row["status_changed_ts"] = at
        self.row.update(changes, updated_ts=at)
        return dict(self.row)


def _failure_events(lifecycle: Lifecycle, maintenance_id: int, registry: _RegistryState) -> list[CdcEvent]:
    failure = lifecycle.pulses[-1]
    mode = lifecycle.failure_mode or ""
    flagged = [pulse for pulse in lifecycle.pulses if pulse.status != "HEALTHY"]
    at = failure.event_ts + timedelta(milliseconds=5)
    work_order = {
        "maintenance_id": maintenance_id,
        "engine_id": failure.engine_id,
        "lifecycle_id": lifecycle.lifecycle_id,
        "failure_cycle": failure.cycle_number,
        "failure_ts": failure.event_ts,
        "failure_mode": mode,
        "maintenance_status": "OPEN",
        "action_taken": None,
        "opened_ts": at,
        "completed_ts": None,
        "downtime_seconds": None,
        "warning_rows_flagged": None,
        "notes": "synthetic analytics fixture",
    }
    done = at + timedelta(seconds=MAINTENANCE_SECONDS)
    completed = {
        **work_order,
        "maintenance_status": "COMPLETED",
        "action_taken": REPAIR_ACTIONS[mode],
        "completed_ts": done,
        "downtime_seconds": MAINTENANCE_SECONDS,
        "warning_rows_flagged": len(flagged),
    }
    events = [CdcEvent("maintenance_log", "c", at, work_order)]
    events += [
        CdcEvent(
            "engine_telemetry", "u", at + timedelta(milliseconds=1), _telemetry_image(pulse, maintenance_id)
        )
        for pulse in flagged
    ]
    # REGISTRY_FAILURE_SQL in the failure's transaction: the failed cycle counts as flown
    row = registry.row
    flown = row["cycles_since_new"] + max(failure.cycle_number - row["current_cycle"], 0) + 1
    events.append(
        CdcEvent(
            "engine_status_registry",
            "u",
            at + timedelta(milliseconds=2),
            registry.update(
                failure.event_ts,
                operational_status="FAILURE",
                current_cycle=failure.cycle_number,
                wear_factor=1.0,
                cycles_since_new=flown,
                last_telemetry_ts=failure.event_ts,
            ),
        )
    )
    events.append(CdcEvent("maintenance_log", "u", done, completed))
    # REGISTRY_RESET_SQL with the repair; the first pulse of the next lifecycle sets HEALTHY
    events.append(
        CdcEvent(
            "engine_status_registry",
            "u",
            done + timedelta(milliseconds=1),
            registry.update(
                done,
                operational_status="MAINTENANCE_RESET",
                current_lifecycle_id=lifecycle.lifecycle_id + 1,
                current_cycle=0,
                wear_factor=0.0,
                lifecycles_completed=row["lifecycles_completed"] + 1,
                last_maintenance_ts=done,
            ),
        )
    )
    return events


def _lifecycle_events(lifecycle: Lifecycle, registry: _RegistryState) -> list[CdcEvent]:
    """Pulse inserts and the registry writes of fleet_runtime.registry_update: a status change at the
    pulse that changes the status, progress on the first pulse of every new cycle."""
    events: list[CdcEvent] = []
    for pulse in lifecycle.pulses:
        events.append(
            CdcEvent(
                "engine_telemetry", "c", pulse.event_ts + timedelta(milliseconds=3), _telemetry_image(pulse)
            )
        )
        row = registry.row
        new_cycles = max(pulse.cycle_number - row["current_cycle"], 0)
        if pulse.status == "FAILURE" or (pulse.status == row["operational_status"] and not new_cycles):
            continue  # _failure_events writes the FAILURE
        events.append(
            CdcEvent(
                "engine_status_registry",
                "u",
                pulse.event_ts + timedelta(milliseconds=4),
                registry.update(
                    pulse.event_ts,
                    operational_status=pulse.status,
                    current_lifecycle_id=pulse.lifecycle_id,
                    current_cycle=pulse.cycle_number,
                    wear_factor=pulse.wear,
                    cycles_since_new=row["cycles_since_new"] + new_cycles,
                    last_telemetry_ts=pulse.event_ts,
                ),
            )
        )
    return events


def simulate(rng: random.Random) -> tuple[list[Lifecycle], list[CdcEvent], datetime]:
    """Run the fleet simulation; returns lifecycles, the change stream (no LSNs yet) and the cut-off."""
    lead = FLEET[0]
    lead_failure_s = sum(lead.completed_cycles) * PULSES_PER_CYCLE + 2 * (MAINTENANCE_SECONDS + 1)
    cutoff = SIM_START + timedelta(seconds=lead_failure_s - 100)
    end = cutoff + timedelta(seconds=300)
    lifecycles = [lc for plan in FLEET for lc in simulate_engine(plan, end, rng)]
    pulses = sorted((p for lc in lifecycles for p in lc.pulses), key=lambda p: (p.event_ts, p.engine_id))
    for telemetry_id, pulse in enumerate(pulses, start=1):
        pulse.telemetry_id = telemetry_id
    registries = {
        plan.engine_id: _RegistryState(
            plan, BACKFILL_REGISTERED_AT if plan.engine_id == BACKFILLED_ENGINE else REGISTERED_AT
        )
        for plan in FLEET
    }
    events = [
        CdcEvent("engine_status_registry", "c", SIM_START - timedelta(seconds=1), dict(state.row))
        for state in [*registries.values(), _RegistryState(SPARE_ENGINE, SPARE_REGISTERED_AT)]
    ]
    completed = sorted((lc for lc in lifecycles if lc.failure_mode), key=lambda lc: lc.pulses[-1].event_ts)
    work_orders = {id(lc): number for number, lc in enumerate(completed, start=1)}
    for plan in FLEET:
        for lifecycle in (lc for lc in lifecycles if lc.plan is plan):
            events += _lifecycle_events(lifecycle, registries[plan.engine_id])
            if lifecycle.failure_mode:
                events += _failure_events(lifecycle, work_orders[id(lifecycle)], registries[plan.engine_id])
    return lifecycles, events, cutoff


def _assign_lsns(events: list[CdcEvent], rng: random.Random) -> None:
    """Order the stream by commit time and give every change a strictly increasing LSN."""
    events.sort(key=lambda event: event.commit_ts)
    lsn = FIRST_LSN
    for event in events:
        lsn += rng.randrange(80, 400)
        event.lsn = lsn


def _prediction(
    lifecycle: Lifecycle, at: datetime, rul_cycles: float, risk_band: str, pulse: Pulse | None = None
) -> dict[str, Any]:
    pulse = lifecycle.pulses[-1] if pulse is None else pulse
    return {
        "predicted_at": at,
        "engine_id": pulse.engine_id,
        "engine_model": lifecycle.plan.engine_model,
        "lifecycle_id": lifecycle.lifecycle_id,
        "cycle_number": pulse.cycle_number,
        "telemetry_id": pulse.telemetry_id,
        "feature_event_ts": pulse.event_ts,
        "operational_status": pulse.status,
        **pulse.readings,
        "predicted_rul_cycles": rul_cycles,
        "predicted_rul_hours": rul_cycles * 2.0,
        "failure_probability": 0.9 if risk_band == "HIGH" else 0.1,
        "failure_within_horizon": int(rul_cycles <= 25),
        "horizon_cycles": 25,
        "risk_band": risk_band,
        "model_version": "rul-20261025T080000Z",
    }


def _newest_pulse_at(lifecycle: Lifecycle, at: datetime) -> Pulse:
    """The pulse a live scoring at ``at`` sees: the newest one measured by then."""
    return [pulse for pulse in lifecycle.pulses if pulse.event_ts <= at][-1]


def _predictions(lifecycle_of: dict[tuple[str, int], Lifecycle], cutoff: datetime) -> list[dict[str, Any]]:
    """Predictions written during the increment: a stale one for an ended lifecycle, three current.

    CFM56-104's are scored like the live loop scores, from the newest pulse at scoring time, so
    the board must age them; JT9D-001's current one scores its newest pulse (nothing to age).
    """

    def at(seconds: int) -> datetime:
        return cutoff + timedelta(seconds=seconds)

    cfm56 = lifecycle_of[("CFM56-104", 4)]
    return [
        _prediction(lifecycle_of[("JT9D-001", 3)], at(50), 5.0, "HIGH"),
        _prediction(lifecycle_of[("JT9D-001", 4)], at(250), 80.0, "LOW"),
        _prediction(cfm56, at(100), 40.0, "MEDIUM", _newest_pulse_at(cfm56, at(100))),
        _prediction(cfm56, at(200), 30.0, "MEDIUM", _newest_pulse_at(cfm56, at(200))),
    ]


def _aged_risk_band(risk_band: str, rul_cycles: float, horizon_cycles: int) -> str:
    """predict_rul's RUL rule re-applied to an aged RUL; a band never de-escalates."""
    if risk_band == "HIGH" or rul_cycles <= horizon_cycles:
        return "HIGH"
    if risk_band == "MEDIUM" or rul_cycles <= 2 * horizon_cycles:
        return "MEDIUM"
    return "LOW"


def expected_board_predictions(
    predictions: list[dict[str, Any]], registry: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any] | None]:
    """Per engine, the newest prediction of its current lifecycle aged to its registry cycle."""
    board: dict[str, dict[str, Any] | None] = {}
    for engine_id, row in registry.items():
        current = [
            p
            for p in predictions
            if p["engine_id"] == engine_id and p["lifecycle_id"] == row["current_lifecycle_id"]
        ]
        if not current:
            board[engine_id] = None
            continue
        newest = max(current, key=lambda p: p["predicted_at"])
        elapsed = max(row["current_cycle"] - newest["cycle_number"], 0)
        rul = max(newest["predicted_rul_cycles"] - elapsed, 0.0)
        raw_rul = newest["predicted_rul_cycles"]
        board[engine_id] = {
            "prediction_cycle_number": newest["cycle_number"],
            "cycles_since_prediction": elapsed,
            "predicted_rul_cycles_adjusted": rul,
            "predicted_rul_hours_adjusted": newest["predicted_rul_hours"] * rul / raw_rul if raw_rul else 0.0,
            "risk_band_adjusted": _aged_risk_band(newest["risk_band"], rul, newest["horizon_cycles"]),
        }
    return board


def _reregistration(cutoff: datetime) -> list[CdcEvent]:
    """The spare engine is deregistered (its registry row deleted) before the cut-off and registered
    again after it; whole milliseconds, because ClickHouse keeps the commit time of a change
    (cdc_source_ts, the end of the deleted row's interval) to the millisecond."""
    spare = _RegistryState(SPARE_ENGINE, SPARE_REGISTERED_AT)
    again = _RegistryState(SPARE_ENGINE, cutoff + REREGISTERED_AFTER_CUTOFF)
    return [
        CdcEvent("engine_status_registry", "d", cutoff - DEREGISTERED_BEFORE_CUTOFF, dict(spare.row)),
        CdcEvent("engine_status_registry", "c", cutoff + REREGISTERED_AFTER_CUTOFF, dict(again.row)),
    ]


def _registry_version(events: list[CdcEvent], engine_id: str, status: str) -> CdcEvent:
    """The engine's first registry version in ``status`` (log order)."""
    return next(
        event
        for event in events
        if event.table == "engine_status_registry"
        and event.image["engine_id"] == engine_id
        and event.image["operational_status"] == status
    )


def _expected_failures(lifecycle_of: dict[tuple[str, int], Lifecycle]) -> list[dict[str, Any]]:
    """Every failure the fixture simulated, as fct_engine_failures must show it.

    An engine returns to service with the first pulse of its next lifecycle. The operating run
    before a failure starts at the previous return to service, or at the registration for a first
    lifecycle, except for the backfilled engine, whose history starts behind its registration.
    """
    failures = []
    for (engine_id, lifecycle_id), lifecycle in sorted(lifecycle_of.items()):
        if lifecycle.failure_mode is None:
            continue
        failure, returned = lifecycle.pulses[-1], lifecycle_of[(engine_id, lifecycle_id + 1)].pulses[0]
        since: datetime | None = lifecycle.pulses[0].event_ts if lifecycle_id > 1 else REGISTERED_AT
        if lifecycle_id == 1 and engine_id == BACKFILLED_ENGINE:
            since = None
        failures.append(
            {
                "engine_id": engine_id,
                "lifecycle_id": lifecycle_id,
                "failure_cycle": failure.cycle_number,
                "failed_at": failure.event_ts.isoformat(),
                "returned_to_service_at": returned.event_ts.isoformat(),
                "operating_since": None if since is None else since.isoformat(),
                "operating_cycles": None if since is None else failure.cycle_number + 1,
                "failure_mode": lifecycle.failure_mode,
            }
        )
    return failures


def _pulses_near(lifecycle: Lifecycle, at: datetime, count: int) -> list[Pulse]:
    start = next(index for index, pulse in enumerate(lifecycle.pulses) if pulse.event_ts >= at)
    return lifecycle.pulses[start : start + count]


def build_history(seed: int) -> History:
    """Simulate the fleet and assign every change event to its delivery phase (see module doc)."""
    rng = random.Random(seed)
    lifecycles, events, cutoff = simulate(rng)
    lifecycle_of = {(lc.plan.engine_id, lc.lifecycle_id): lc for lc in lifecycles}
    running = {lc.plan.engine_id: lc for lc in lifecycles if lc.failure_mode is None}

    deleted_late = lifecycle_of[("JT9D-001", 2)].pulses[100]
    deleted_base = lifecycle_of[("GE90-201", 1)].pulses[50]
    corrected = running["GE90-202"].pulses[60]  # inside the baseline period of a running lifecycle
    correction = {**_telemetry_image(corrected), "egt_c": round(corrected.readings["egt_c"] + 15.0, 4)}
    events += [
        CdcEvent("engine_telemetry", "d", cutoff + timedelta(seconds=30), _telemetry_image(deleted_late)),
        CdcEvent("engine_telemetry", "u", cutoff + timedelta(seconds=40), correction),
        CdcEvent(
            "engine_telemetry",
            "d",
            deleted_base.event_ts + timedelta(milliseconds=500),
            _telemetry_image(deleted_base),
        ),
        *_reregistration(cutoff),
    ]
    _assign_lsns(events, rng)
    # final registry row per engine (events are in commit order; replays are added below)
    registry = {e.image["engine_id"]: e.image for e in events if e.table == "engine_status_registry"}
    out_of_order = _registry_version(events, "GE90-201", "MAINTENANCE_RESET")
    replayed = _registry_version(events, "CFM56-104", "FAILURE")
    flagged_failure = lifecycle_of[("CFM56-104", 1)].pulses[-1].telemetry_id
    replayed_pulse = next(
        e
        for e in events
        if e.table == "engine_telemetry" and e.op == "u" and e.image["telemetry_id"] == flagged_failure
    )
    first_change = next(
        event
        for event in events
        if event.table == "engine_status_registry"
        and event.image["engine_id"] == BACKFILLED_ENGINE
        and event.image["status_changed_ts"] != BACKFILL_REGISTERED_AT
    )

    baseline_late = lifecycle_of[("CFM56-104", 2)].pulses[30:33]
    window_late = _pulses_near(running["GE90-201"], cutoff - timedelta(seconds=150), 4)
    tail_late = _pulses_near(lifecycle_of[("JT9D-001", 3)], cutoff - timedelta(seconds=120), 2)
    withheld = {pulse.telemetry_id for pulse in [*baseline_late, *window_late, *tail_late]}
    late_visible = {pulse.telemetry_id for pulse in window_late}
    for event in events:
        event.phase = "base" if event.commit_ts < cutoff else "increment"
        if event.table == "engine_telemetry" and event.image["telemetry_id"] in withheld:
            event.phase = "increment"
            event.late_visible = event.image["telemetry_id"] in late_visible
        elif event is out_of_order or (
            event.table == "engine_status_registry" and event.image["engine_id"] == "GE90-202"
        ):
            event.phase = "increment"

    replays = [
        event
        for number, event in enumerate(events)
        if (event.table == "engine_telemetry" and number % 97 == 0)
        or (event.table == "engine_status_registry" and number % 41 == 0)
        or (event.table == "maintenance_log" and event.image["maintenance_id"] == 1)
    ]
    events += [
        CdcEvent(e.table, e.op, e.commit_ts, dict(e.image), e.lsn, e.phase, e.late_visible) for e in replays
    ]
    # already published versions delivered again in a later batch: a registry change and the
    # newest version of a pulse (its back-flag UPDATE)
    events += [
        CdcEvent(e.table, e.op, e.commit_ts, dict(e.image), e.lsn, "increment")
        for e in (replayed, replayed_pulse)
    ]

    predictions = _predictions(lifecycle_of, cutoff)
    frames, hf_manifest = build_hf_frames(lifecycles, cutoff, seed)
    manifest = {
        "cutoff": cutoff.isoformat(),
        "engines": [plan.engine_id for plan in (*FLEET, SPARE_ENGINE)],
        "silent_engine": SPARE_ENGINE.engine_id,
        "completed_lifecycles": [
            [lc.plan.engine_id, lc.lifecycle_id] for lc in lifecycles if lc.failure_mode
        ],
        "running_lifecycles": [[lc.plan.engine_id, lc.lifecycle_id] for lc in running.values()],
        "withheld_ids": sorted(withheld),
        "late_visible_ids": sorted(late_visible),
        "deleted_ids": [deleted_late.telemetry_id, deleted_base.telemetry_id],
        "corrected_id": corrected.telemetry_id,
        "corrected_egt_c": correction["egt_c"],
        "lagging_registry_engine": "GE90-202",
        "replayed_events": len(replays) + 2,
        # newest prediction of each engine's current lifecycle, as fct_engine_health_status must show it
        "expected_predicted_rul": {"JT9D-001": 80.0, "CFM56-104": 30.0, "GE90-201": None, "GE90-202": None},
        "expected_board_predictions": expected_board_predictions(predictions, registry),
        "hf": hf_manifest,
        "reliability": {
            "backfilled_engine": {
                "engine_id": BACKFILLED_ENGINE,
                "registered_at": BACKFILL_REGISTERED_AT.isoformat(),
                "first_change_at": first_change.image["status_changed_ts"].isoformat(),
            },
            "deregistered_engine": {
                "engine_id": SPARE_ENGINE.engine_id,
                "registered_at": SPARE_REGISTERED_AT.isoformat(),
                "deregistered_at": (cutoff - DEREGISTERED_BEFORE_CUTOFF).isoformat(),
                "registered_again_at": (cutoff + REREGISTERED_AFTER_CUTOFF).isoformat(),
            },
            "out_of_order": {
                "engine_id": out_of_order.image["engine_id"],
                "cdc_lsn": out_of_order.lsn,
                "operational_status": out_of_order.image["operational_status"],
                "lifecycle_id": out_of_order.image["current_lifecycle_id"],
            },
            "replayed": {"engine_id": replayed.image["engine_id"], "cdc_lsn": replayed.lsn},
            "failures": _expected_failures(lifecycle_of),
        },
    }
    return History(events, cutoff, predictions, manifest, frames)
