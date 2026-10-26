"""Deterministic synthetic CDC history of the jet-engine fleet for the analytics checks.

The simulation mirrors the platform's PostgreSQL tables and their Debezium change stream:
1 Hz pulses grouped in 10-pulse cycles, lifecycles that degrade until a FAILURE pulse, a
maintenance work order opened at the failure and completed 20 s later, the lifecycle's
non-HEALTHY pulses linked to the work order by UPDATEs, and a registry row updated every cycle.

``build_history`` splits the change stream into two delivery phases around a cut-off:

* ``base``: everything committed before the cut-off, minus the pulses and registry events held
  back below, plus at-least-once replays and one insert+delete pair.
* ``increment``: everything after the cut-off plus late data: pulses withheld from ``base``
  (one group in a lifecycle's baseline period), GE90-202's whole registry history (telemetry
  landed first), a late delete and a late sensor correction of already published pulses.
  ``late_visible`` pulses must be loaded with an ingested_at older than the dbt watermark.
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
    EnginePlan,
    Lifecycle,
    Pulse,
    simulate_engine,
)

FIRST_LSN = 26_000_000
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
    """The change stream, increment-phase RUL predictions and the facts the checks verify."""

    events: list[CdcEvent]
    cutoff: datetime
    predictions: list[dict[str, Any]]
    manifest: dict[str, Any]


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
    """Mutable registry row of one engine; ``image`` snapshots it for a CDC event."""

    def __init__(self, plan: EnginePlan) -> None:
        registered = SIM_START - timedelta(seconds=1)
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
    events.append(
        CdcEvent(
            "engine_status_registry",
            "u",
            at + timedelta(milliseconds=2),
            registry.update(at, operational_status="FAILURE", wear_factor=1.0),
        )
    )
    events.append(
        CdcEvent(
            "engine_status_registry",
            "u",
            at + timedelta(seconds=1),
            registry.update(at + timedelta(seconds=1), operational_status="MAINTENANCE_RESET"),
        )
    )
    events.append(CdcEvent("maintenance_log", "u", done, completed))
    completed_count = registry.row["lifecycles_completed"] + 1
    events.append(
        CdcEvent(
            "engine_status_registry",
            "u",
            done + timedelta(milliseconds=1),
            registry.update(
                done,
                operational_status="HEALTHY",
                current_lifecycle_id=lifecycle.lifecycle_id + 1,
                current_cycle=0,
                wear_factor=0.0,
                lifecycles_completed=completed_count,
                last_maintenance_ts=done,
            ),
        )
    )
    return events


def _lifecycle_events(lifecycle: Lifecycle, registry: _RegistryState) -> list[CdcEvent]:
    events: list[CdcEvent] = []
    for pulse in lifecycle.pulses:
        events.append(
            CdcEvent(
                "engine_telemetry", "c", pulse.event_ts + timedelta(milliseconds=3), _telemetry_image(pulse)
            )
        )
        if pulse.index % PULSES_PER_CYCLE == PULSES_PER_CYCLE - 1 and pulse.status != "FAILURE":
            events.append(
                CdcEvent(
                    "engine_status_registry",
                    "u",
                    pulse.event_ts + timedelta(milliseconds=4),
                    registry.update(
                        pulse.event_ts,
                        operational_status=pulse.status,
                        current_cycle=pulse.cycle_number,
                        wear_factor=pulse.wear,
                        cycles_since_new=registry.row["cycles_since_new"] + 1,
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
    registries = {plan.engine_id: _RegistryState(plan) for plan in FLEET}
    events = [
        CdcEvent("engine_status_registry", "c", SIM_START - timedelta(seconds=1), dict(state.row))
        for state in registries.values()
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


def _prediction(lifecycle: Lifecycle, at: datetime, rul_cycles: float, risk_band: str) -> dict[str, Any]:
    pulse = lifecycle.pulses[-1]
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


def _predictions(lifecycle_of: dict[tuple[str, int], Lifecycle], cutoff: datetime) -> list[dict[str, Any]]:
    """Predictions written during the increment: a stale one for an ended lifecycle, two for CFM56-104."""

    def at(seconds: int) -> datetime:
        return cutoff + timedelta(seconds=seconds)

    return [
        _prediction(lifecycle_of[("JT9D-001", 3)], at(50), 5.0, "HIGH"),
        _prediction(lifecycle_of[("JT9D-001", 4)], at(250), 80.0, "LOW"),
        _prediction(lifecycle_of[("CFM56-104", 4)], at(100), 30.0, "MEDIUM"),
        _prediction(lifecycle_of[("CFM56-104", 4)], at(200), 25.0, "HIGH"),
    ]


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
    ]
    _assign_lsns(events, rng)

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
        elif event.table == "engine_status_registry" and event.image["engine_id"] == "GE90-202":
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

    manifest = {
        "cutoff": cutoff.isoformat(),
        "engines": [plan.engine_id for plan in FLEET],
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
        "replayed_events": len(replays),
        # newest prediction of each engine's current lifecycle, as fct_engine_health_status must show it
        "expected_predicted_rul": {"JT9D-001": 80.0, "CFM56-104": 25.0, "GE90-201": None, "GE90-202": None},
    }
    return History(events, cutoff, _predictions(lifecycle_of, cutoff), manifest)
