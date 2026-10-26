"""Fleet runtime of the simulator: registry and work-order bookkeeping plus the drift-free live loop.

Every live tick becomes ONE database transaction (executed by fleet_db) that holds, in order: the repairs
that became due, the generation settings of lifecycles whose first pulse this is, one pulse per operating
engine, FAILURE events, then registry updates. Engines are duck-typed ``engine_generator.EngineSimulator``
objects, so this module never imports engine_generator at runtime (that would load the entry-point script
a second time).
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine_generator import EngineSimulator, Pulse
    from fleet_config import GenerationSettings, SimulatorConfig
    from fleet_db import FleetDatabase

logger = logging.getLogger("fleet_runtime")

WriteOp = tuple[str, Any]
"""One database write as (fleet_db writer method, payload); a list of them runs in ONE transaction."""

HEARTBEAT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class LifecycleSettings:
    """Generation settings of one lifecycle, stored with its first pulse (simulator_lifecycle_params)."""

    engine_id: str
    lifecycle_id: int
    settings: GenerationSettings


@dataclass(frozen=True, slots=True)
class RegistryUpdate:
    """engine_status_registry progress after a pulse; ``status`` is None when the status is unchanged."""

    engine_id: str
    lifecycle_id: int
    current_cycle: int
    wear_factor: float
    cycles_since_new: int
    event_ts: datetime
    status: str | None


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """A FAILURE pulse plus the work order it opens."""

    pulse: Pulse
    failure_mode: str
    cycles_since_new: int
    notes: str


@dataclass(frozen=True, slots=True)
class MaintenanceCompletion:
    """Closure of the work order of ``lifecycle_id``; the engine restarts in the next lifecycle."""

    engine_id: str
    lifecycle_id: int
    failure_ts: datetime
    completed_ts: datetime
    action_taken: str

    @property
    def next_lifecycle_id(self) -> int:
        """Lifecycle that starts after this repair."""
        return self.lifecycle_id + 1

    @property
    def downtime_seconds(self) -> int:
        """Whole seconds between failure and repair completion."""
        return max(0, round((self.completed_ts - self.failure_ts).total_seconds()))


def lifecycle_settings(engine: EngineSimulator) -> LifecycleSettings:
    """Settings record of the engine's current lifecycle (written with that lifecycle's first pulse)."""
    return LifecycleSettings(engine.engine_id, engine.lifecycle_id, engine.settings)


def registry_update(
    engine: EngineSimulator, pulse: Pulse, *, include_progress: bool = True
) -> RegistryUpdate | None:
    """Registry write implied by a non-failure ``pulse``: always on a status change, and on the first
    pulse of every new cycle when ``include_progress`` is set."""
    status_changed = pulse.status != engine.status
    if not status_changed and not (include_progress and pulse.cycle_number > engine.current_cycle):
        return None
    return RegistryUpdate(
        engine_id=engine.engine_id,
        lifecycle_id=engine.lifecycle_id,
        current_cycle=pulse.cycle_number,
        wear_factor=pulse.wear_factor,
        cycles_since_new=engine.cycles_since_new_at(pulse),
        event_ts=pulse.event_ts,
        status=pulse.status if status_changed else None,
    )


def progress_update(engine: EngineSimulator) -> RegistryUpdate:
    """Registry progress matching the engine's last committed pulse."""
    if engine.last_event_ts is None:
        raise RuntimeError(f"{engine.engine_id} has not emitted telemetry yet")
    return RegistryUpdate(
        engine_id=engine.engine_id,
        lifecycle_id=engine.lifecycle_id,
        current_cycle=engine.current_cycle,
        wear_factor=engine.wear_factor,
        cycles_since_new=engine.cycles_since_new,
        event_ts=engine.last_event_ts,
        status=None,
    )


def failure_event(engine: EngineSimulator, pulse: Pulse, source: str) -> FailureEvent:
    """Work order of a FAILURE ``pulse``; ``source`` (live or backfill) is recorded in the notes."""
    mode = engine.lifecycle.failure_mode
    notes = (
        f"{source}: {mode} at cycle {pulse.cycle_number}; EGT {pulse.egt_c:.1f} C, "
        f"core {pulse.core_temp_c:.1f} C, vibration {pulse.vibration_mm_s:.2f} mm/s, "
        f"oil {pulse.oil_pressure_psi:.1f} psi"
    )
    # The cycle in which the engine failed counts as flown.
    return FailureEvent(pulse, mode, engine.cycles_since_new_at(pulse) + 1, notes)


def completion_at(engine: EngineSimulator, completed_ts: datetime) -> MaintenanceCompletion:
    """Repair completion of the engine's open work order at ``completed_ts``."""
    if engine.maintenance_since is None:
        raise RuntimeError(f"{engine.engine_id} has no open work order")
    return MaintenanceCompletion(
        engine.engine_id,
        engine.lifecycle_id,
        engine.maintenance_since,
        completed_ts,
        engine.maintenance_action,
    )


def completion_due(
    engine: EngineSimulator, now: datetime, repair_seconds: float
) -> MaintenanceCompletion | None:
    """The repair completion once the engine has been under maintenance for ``repair_seconds``."""
    if engine.maintenance_since is None or (now - engine.maintenance_since).total_seconds() < repair_seconds:
        return None
    return completion_at(engine, now)


def build_tick_ops(
    completions: Sequence[tuple[EngineSimulator, MaintenanceCompletion]],
    pulses: Sequence[tuple[EngineSimulator, Pulse]],
) -> tuple[list[WriteOp], dict[str, int]]:
    """Ordered writes of one live tick and, per failing engine, the index of its record_failure op."""
    ops: list[WriteOp] = [("complete_maintenance", completion) for _, completion in completions]
    ops += [
        ("record_settings", lifecycle_settings(engine)) for engine, pulse in pulses if pulse.pulse_index == 0
    ]
    regular = [pulse for _, pulse in pulses if not pulse.is_failure]
    if regular:
        ops.append(("insert_pulses", regular))
    failure_slots: dict[str, int] = {}
    for engine, pulse in pulses:
        if pulse.is_failure:
            failure_slots[engine.engine_id] = len(ops)
            ops.append(("record_failure", failure_event(engine, pulse, "live")))
        elif (update := registry_update(engine, pulse)) is not None:
            ops.append(("update_registry", update))
    return ops, failure_slots


class FleetRunner:
    """Live loop: every tick, ONE transaction carries each operating engine's pulse and any due repairs."""

    def __init__(
        self,
        engines: Sequence[EngineSimulator],
        store: FleetDatabase,
        config: SimulatorConfig,
        stop_event: threading.Event,
    ) -> None:
        self._engines = list(engines)
        self._store = store
        self._tick_seconds = config.tick_seconds
        self._repair_seconds = config.maintenance_seconds
        self._stop = stop_event
        self.ticks = self.pulses = self.failures = self.dropped_ticks = 0

    def run(self, max_ticks: int | None = None) -> None:
        """Tick against a monotonic deadline (no drift) until stopped or ``max_ticks`` ticks have run."""
        deadline = last_heartbeat = time.monotonic()
        while not self._stop.is_set() and (max_ticks is None or self.ticks < max_ticks):
            self.run_tick(datetime.now(UTC))
            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                self.log_heartbeat()
                last_heartbeat = time.monotonic()
            deadline += self._tick_seconds
            delay = deadline - time.monotonic()
            if delay < -self._tick_seconds:
                # Bursting to catch up would squeeze event time; skip the missed ticks instead.
                logger.warning("Live loop %.1f s behind schedule: skipping missed ticks", -delay)
                deadline, delay = time.monotonic(), 0.0
            if delay > 0:
                self._stop.wait(delay)
        self.log_heartbeat()

    def run_tick(self, now: datetime) -> bool:
        """Persist one tick at wall-clock ``now``; False when a database error dropped it."""
        self.ticks += 1
        completions = [
            (engine, completion)
            for engine in self._engines
            if (completion := completion_due(engine, now, self._repair_seconds)) is not None
        ]
        pulses = [(engine, engine.next_pulse(now)) for engine in self._engines if not engine.in_maintenance]
        ops, failure_slots = build_tick_ops(completions, pulses)
        results = self._store.execute(ops)
        if results is None:
            self.dropped_ticks += 1
            logger.warning("Tick %d dropped (database unavailable); engines continue next tick", self.ticks)
            return False
        for engine, completion in completions:
            engine.start_next_lifecycle(completion)
            logger.info(
                "%s repaired (%s, %d s downtime): lifecycle %d starts in MAINTENANCE_RESET",
                engine.engine_id,
                completion.action_taken,
                completion.downtime_seconds,
                engine.lifecycle_id,
            )
        for engine, pulse in pulses:
            slot = failure_slots.get(engine.engine_id)
            self._apply_pulse(engine, pulse, None if slot is None else results[slot])
        return True

    def _apply_pulse(self, engine: EngineSimulator, pulse: Pulse, maintenance_id: Any) -> None:
        previous = engine.status
        engine.advance(pulse)
        self.pulses += 1
        if pulse.is_failure:
            self.failures += 1
            logger.warning(
                "%s %s -> FAILURE in lifecycle %d at cycle %d (%s): work order %s opened, repair %.0f s",
                engine.engine_id,
                previous,
                pulse.lifecycle_id,
                pulse.cycle_number,
                engine.lifecycle.failure_mode,
                maintenance_id,
                self._repair_seconds,
            )
        elif pulse.status != previous:
            logger.info(
                "%s lifecycle %d cycle %d: %s -> %s (w=%.3f)",
                engine.engine_id,
                pulse.lifecycle_id,
                pulse.cycle_number,
                previous,
                pulse.status,
                pulse.wear_factor,
            )

    def log_heartbeat(self) -> None:
        """Log loop counters and the state of every engine."""
        logger.info(
            "Fleet heartbeat: %d ticks, %d pulses, %d failures, %d dropped ticks | %s",
            self.ticks,
            self.pulses,
            self.failures,
            self.dropped_ticks,
            " | ".join(engine.describe() for engine in self._engines),
        )


def configure_logging(level_name: str) -> None:
    """Configure root logging to stdout; unknown LOG_LEVEL values fall back to INFO."""
    level = logging.getLevelNamesMapping().get(level_name.strip().upper())
    logging.basicConfig(
        level=logging.INFO if level is None else level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if level is None:
        logger.warning("Unknown LOG_LEVEL %r: using INFO", level_name)


def install_signal_handlers(stop_event: threading.Event) -> None:
    """SIGTERM/SIGINT only set ``stop_event``, so the running transaction completes before shutdown."""

    def request_stop(signum: int, _frame: object) -> None:
        logger.info(
            "%s received: finishing the current transaction, then exiting", signal.Signals(signum).name
        )
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
