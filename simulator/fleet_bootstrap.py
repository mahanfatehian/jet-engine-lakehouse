"""Simulator start-up: engine registration, resume after a restart, history backfill.

Resume re-derives every in-progress lifecycle under the generation settings stored with it
(simulator_lifecycle_params) and checks that the rebuilt trajectory reproduces the stored telemetry. A
lifecycle keeps the settings it started with when SIM_SEED, SIM_MEAN_LIFE_CYCLES or SIM_PULSES_PER_CYCLE
change; the new values apply from its engine's next lifecycle. A lifecycle written before the settings
were recorded is adopted when the current settings reproduce it, and refused otherwise
(ResumeMismatchError, exit code 2) instead of continuing a corrupted curve.

Backfill works per engine: every configured engine without any telemetry gets its history in ONE
transaction, so an interrupted first boot completes on the next start and an engine added to
SIM_ENGINE_IDS later gets history too; an engine with telemetry is never backfilled again.

Like fleet_runtime, this module never imports engine_generator at runtime: engines are built by the
factory that ``engine_generator.main`` hands in and are persisted through the fleet_db store.
"""

from __future__ import annotations

import logging
import threading
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from fleet_runtime import (
    WriteOp,
    completion_at,
    failure_event,
    lifecycle_settings,
    progress_update,
    registry_update,
)

if TYPE_CHECKING:
    from engine_generator import EngineSimulator, Pulse
    from fleet_config import SimulatorConfig
    from fleet_db import EngineSnapshot, FleetDatabase

logger = logging.getLogger("fleet_bootstrap")

EngineFactory = Callable[..., "EngineSimulator"]
MAX_PARTIAL_LIFE_FRACTION = 0.7
WEAR_TOLERANCE = 1e-6  # stored wear is rounded to 6 decimals, so a faithful rebuild matches exactly


class ResumeMismatchError(RuntimeError):
    """Stored telemetry of an in-progress lifecycle cannot be reproduced by the settings at hand."""


def restore_engine(snapshot: EngineSnapshot, factory: EngineFactory) -> EngineSimulator:
    """Rebuild an engine twin from its registry row, current-lifecycle telemetry and work order.

    A lifecycle with telemetry is rebuilt under its stored generation settings (the current SIM_*
    values when none are stored, which ``resume_fleet`` then verifies); a lifecycle without telemetry
    starts under the current SIM_* values.
    """
    work_order = snapshot.work_order
    if work_order is not None and work_order.status == "COMPLETED":
        # Repair committed while the registry still names the failed lifecycle (only after manual
        # edits): move on to the next lifecycle so its work-order key stays unique.
        logger.warning(
            "%s: lifecycle %d is already repaired; continuing in lifecycle %d",
            snapshot.engine_id,
            snapshot.lifecycle_id,
            snapshot.lifecycle_id + 1,
        )
        return factory(
            snapshot.engine_id,
            lifecycle_id=snapshot.lifecycle_id + 1,
            status=snapshot.operational_status,
            cycles_since_new=snapshot.cycles_since_new,
            last_event_ts=snapshot.last_event_ts,
        )
    # Registry progress may lag the telemetry by up to one cycle; telemetry is authoritative.
    current_cycle = max(snapshot.current_cycle, snapshot.last_cycle_number or 0)
    engine = factory(
        snapshot.engine_id,
        lifecycle_id=snapshot.lifecycle_id,
        pulse_index=snapshot.pulses_in_lifecycle,
        status=snapshot.operational_status,
        current_cycle=current_cycle,
        cycles_since_new=snapshot.cycles_since_new + current_cycle - snapshot.current_cycle,
        wear_factor=snapshot.last_wear_factor or 0.0,
        last_event_ts=snapshot.last_event_ts,
        settings=snapshot.settings if snapshot.pulses_in_lifecycle else None,
    )
    if work_order is not None:
        # FAILURE committed before the restart: the repair completes once it is due (at once if overdue).
        engine.resume_maintenance(work_order.failure_ts, work_order.failure_mode)
    return engine


def resume_mismatch(engine: EngineSimulator, snapshot: EngineSnapshot) -> str | None:
    """How the twin's rebuilt trajectory contradicts the stored telemetry of its lifecycle (None: agrees).

    The stored last pulse (pulse count, highest cycle, highest wear: w never decreases within a
    lifecycle) must be the rebuilt pulse at that index: same cycle, wear within WEAR_TOLERANCE, and
    not yet the failure.
    """
    pulses, path = snapshot.pulses_in_lifecycle, engine.wear_path
    stored = f"cycle {snapshot.last_cycle_number}, w={snapshot.last_wear_factor:.6f}"
    if pulses >= len(path):
        return (
            f"{pulses} pulses stored without a failure (last at {stored}), "
            f"but the rebuilt lifecycle fails at pulse {len(path)}"
        )
    cycle, wear = (pulses - 1) // engine.settings.pulses_per_cycle, float(path[pulses - 1])
    if cycle != snapshot.last_cycle_number or abs(wear - snapshot.last_wear_factor) > WEAR_TOLERANCE:
        return f"stored pulse {pulses} is at {stored}, the rebuilt one at cycle {cycle}, w={wear:.6f}"
    return None


def _mismatch_error(mismatches: Sequence[tuple[EngineSimulator, EngineSnapshot, str]]) -> ResumeMismatchError:
    details = " | ".join(
        f"{engine.engine_id} lifecycle {engine.lifecycle_id} under its "
        f"{'recorded' if snapshot.settings is not None else 'current'} settings "
        f"({engine.settings.describe()}): {mismatch}"
        for engine, snapshot, mismatch in mismatches
    )
    if any(snapshot.settings is None for _, snapshot, _ in mismatches):
        remedy = (
            "Restore the SIM_SEED, SIM_MEAN_LIFE_CYCLES and SIM_PULSES_PER_CYCLE values these lifecycles "
            "were generated with (they predate the per-lifecycle record of those settings), or run "
        )
    else:
        remedy = "Their telemetry was changed outside the simulator: run "
    return ResumeMismatchError(
        f"cannot resume without corrupting the stored lifecycles: {details}. "
        f"{remedy}`make clean` to start over from an empty database"
    )


def resume_fleet(
    store: FleetDatabase, config: SimulatorConfig, factory: EngineFactory
) -> list[tuple[EngineSimulator, EngineSnapshot]]:
    """Rebuild every configured engine and check, adopt or announce the settings of in-progress lifecycles.

    Raises ResumeMismatchError, before writing anything, when the stored telemetry of any in-progress
    lifecycle cannot be reproduced.
    """
    restored = []
    for engine_id in config.engine_ids:
        snapshot = store.load_snapshot(engine_id)
        restored.append((restore_engine(snapshot, factory), snapshot))
    # Lifecycles without telemetry, or already ended by a failure, generate nothing more under old settings.
    in_progress = [(e, s) for e, s in restored if s.pulses_in_lifecycle and s.work_order is None]
    mismatches = [(e, s, m) for e, s in in_progress if (m := resume_mismatch(e, s)) is not None]
    if mismatches:
        raise _mismatch_error(mismatches)
    adopted = [engine for engine, snapshot in in_progress if snapshot.settings is None]
    if adopted:
        store.execute_with_retry([("record_settings", lifecycle_settings(engine)) for engine in adopted])
        logger.info(
            "Recorded the settings (%s) of lifecycles started before they were stored: %s",
            config.generation.describe(),
            ", ".join(f"{engine.engine_id} L{engine.lifecycle_id}" for engine in adopted),
        )
    for engine, snapshot in in_progress:
        if snapshot.settings is not None and snapshot.settings != config.generation:
            logger.warning(
                "%s lifecycle %d keeps the settings it started with; the change %s applies from lifecycle %d",
                engine.engine_id,
                engine.lifecycle_id,
                snapshot.settings.changes_to(config.generation),
                engine.lifecycle_id + 1,
            )
    return restored


def lifecycle_history(
    engine: EngineSimulator,
    start_ts: datetime,
    tick: timedelta,
    repair: timedelta,
    pulse_limit: int | None = None,
) -> tuple[list[WriteOp], datetime]:
    """Simulate one backdated lifecycle from ``start_ts``; return its writes and the next lifecycle start.

    Without ``pulse_limit`` the lifecycle runs to failure and includes the work order plus its completion
    ``repair`` later; with it, the lifecycle stops early and remains the engine's current one.
    """
    ops: list[WriteOp] = []
    batch: list[Pulse] = []
    ts = start_ts
    while pulse_limit is None or engine.pulse_index < pulse_limit:
        if engine.pulse_index == 0:
            ops.append(("record_settings", lifecycle_settings(engine)))
        pulse = engine.next_pulse(ts)
        if pulse.is_failure:
            if batch:
                ops.append(("insert_pulses", batch))
            ops.append(("record_failure", failure_event(engine, pulse, "backfill")))
            engine.advance(pulse)
            completion = completion_at(engine, pulse.event_ts + repair)
            ops.append(("complete_maintenance", completion))
            engine.start_next_lifecycle(completion)
            return ops, completion.completed_ts + tick
        batch.append(pulse)
        update = registry_update(engine, pulse, include_progress=False)
        if update is not None:
            ops += [("insert_pulses", batch), ("update_registry", update)]
            batch = []
        engine.advance(pulse)
        ts += tick
    if batch:
        ops.append(("insert_pulses", batch))
    if engine.pulse_index:
        ops.append(("update_registry", progress_update(engine)))
    return ops, ts


def engine_history(engine: EngineSimulator, config: SimulatorConfig) -> list[tuple[int, list[WriteOp]]]:
    """Backfill writes of one engine per lifecycle: SIM_BACKFILL_LIFECYCLES failed-and-repaired ones
    plus a partial current one, backdated so the last event lies one tick before now."""
    tick = timedelta(seconds=config.tick_seconds)
    repair = timedelta(seconds=config.maintenance_seconds)
    count = config.backfill_lifecycles
    lengths = engine.lifecycle_lengths(count + 1)
    rng = np.random.default_rng(zlib.crc32(f"{config.seed}|{engine.engine_id}|backfill".encode()))
    partial = int(rng.uniform(0.0, MAX_PARTIAL_LIFE_FRACTION) * lengths[-1])
    # Event-time span from the first backfilled pulse to the last backfilled event (pulse or repair).
    span = sum(((n - 1) * tick + repair + tick for n in lengths[:-1]), timedelta()) + (partial - 1) * tick
    start = datetime.now(UTC) - tick - span
    history = []
    for index in range(count + 1):
        lifecycle_id = engine.lifecycle_id
        ops, start = lifecycle_history(engine, start, tick, repair, partial if index == count else None)
        if ops:
            history.append((lifecycle_id, ops))
    return history


def _log_backfilled(engine_id: str, lifecycle_id: int, ops: Sequence[WriteOp]) -> None:
    rows = sum(len(payload) for name, payload in ops if name == "insert_pulses")
    failure = next((payload for name, payload in ops if name == "record_failure"), None)
    if failure is None:
        logger.info("Backfilled %s lifecycle %d: %d pulses, still in service", engine_id, lifecycle_id, rows)
        return
    logger.info(
        "Backfilled %s lifecycle %d: %d pulses, %s at cycle %d, repaired",
        engine_id,
        lifecycle_id,
        rows + 1,
        failure.failure_mode,
        failure.pulse.cycle_number,
    )


def backfill_fleet(
    engines: Sequence[EngineSimulator],
    store: FleetDatabase,
    config: SimulatorConfig,
    stop_event: threading.Event,
) -> None:
    """Write the history of every engine in ``engines`` (engines without any telemetry).

    History looks exactly like an earlier live run: pulses one tick apart, repairs SIM_MAINTENANCE_SECONDS
    long, the same writes as the live loop. Each engine's history is ONE transaction, so an engine has
    either all of it or no telemetry at all; a shutdown request stops between engines and the next start
    backfills the engines still without telemetry.
    """
    names = [engine.engine_id for engine in engines]
    logger.info(
        "Backfilling %d lifecycles plus a partial one for the engines without telemetry: %s",
        config.backfill_lifecycles,
        ", ".join(names),
    )
    for done, engine in enumerate(engines):
        if stop_event.is_set():
            logger.info(
                "Backfill interrupted by shutdown request; on the next start: %s", ", ".join(names[done:])
            )
            return
        history = engine_history(engine, config)
        store.execute_with_retry([op for _, ops in history for op in ops])
        for lifecycle_id, ops in history:
            _log_backfilled(engine.engine_id, lifecycle_id, ops)
    logger.info("Backfill complete: %s", " | ".join(engine.describe() for engine in engines))


def bootstrap_fleet(
    store: FleetDatabase,
    config: SimulatorConfig,
    factory: EngineFactory,
    engine_models: Mapping[str, str],
    *,
    allow_backfill: bool,
    stop_event: threading.Event,
) -> list[EngineSimulator]:
    """Register the fleet, rebuild every twin from the database and backfill engines without telemetry."""
    store.ensure_schema()
    created = store.register_engines(engine_models)
    logger.info("Engine registry ready: %d engines, %d newly registered", len(engine_models), created)
    restored = resume_fleet(store, config, factory)
    blank = [engine for engine, snapshot in restored if snapshot.last_event_ts is None]
    backfilled = blank if allow_backfill and config.backfill_lifecycles > 0 else []
    resumed = [engine for engine, _ in restored if engine not in backfilled]
    if resumed:
        logger.info("Resuming fleet: %s", " | ".join(engine.describe() for engine in resumed))
    if backfilled:
        backfill_fleet(backfilled, store, config, stop_event)
    return [engine for engine, _ in restored]
