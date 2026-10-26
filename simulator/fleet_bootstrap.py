"""Simulator start-up: SIM_* configuration, engine registration, resume after a restart, history backfill.

Like fleet_runtime, this module never imports engine_generator at runtime: engines are built by the
factory that ``engine_generator.main`` hands in and are persisted through the fleet_db store.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from fleet_runtime import WriteOp, completion_at, failure_event, progress_update, registry_update

if TYPE_CHECKING:
    from engine_generator import EngineSimulator, Pulse
    from fleet_db import EngineSnapshot, FleetDatabase

logger = logging.getLogger("fleet_bootstrap")

EngineFactory = Callable[..., "EngineSimulator"]
DEFAULT_ENGINE_IDS = ("JT9D-001", "JT9D-002", "CFM56-104", "CFM56-105", "GE90-201", "GE90-202")
MAX_ENGINE_ID_LENGTH = 32  # engine_status_registry.engine_id is VARCHAR(32)
MAX_PARTIAL_LIFE_FRACTION = 0.7


def _env_value(env: Mapping[str, str], name: str, cast: Callable[[str], Any], default: Any) -> Any:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc


@dataclass(frozen=True)
class SimulatorConfig:
    """Simulator settings; the defaults are the contract defaults of the SIM_* variables."""

    engine_ids: tuple[str, ...] = DEFAULT_ENGINE_IDS
    tick_seconds: float = 1.0
    pulses_per_cycle: int = 10
    mean_life_cycles: float = 180.0
    backfill_lifecycles: int = 4
    maintenance_seconds: float = 20.0
    seed: int = 42
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        checks = (
            (bool(self.engine_ids), "SIM_ENGINE_IDS lists no engine"),
            (len(set(self.engine_ids)) == len(self.engine_ids), "SIM_ENGINE_IDS contains duplicates"),
            (
                all(len(engine_id) <= MAX_ENGINE_ID_LENGTH for engine_id in self.engine_ids),
                f"engine ids are limited to {MAX_ENGINE_ID_LENGTH} characters",
            ),
            (0 < self.tick_seconds < math.inf, "SIM_TICK_SECONDS must be > 0"),
            (self.pulses_per_cycle >= 1, "SIM_PULSES_PER_CYCLE must be >= 1"),
            (2 <= self.mean_life_cycles < math.inf, "SIM_MEAN_LIFE_CYCLES must be >= 2"),
            (self.backfill_lifecycles >= 0, "SIM_BACKFILL_LIFECYCLES must be >= 0"),
            (0 <= self.maintenance_seconds < math.inf, "SIM_MAINTENANCE_SECONDS must be >= 0"),
        )
        problems = [message for valid, message in checks if not valid]
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SimulatorConfig:
        """Read SIM_* and LOG_LEVEL; unset or blank variables keep the contract defaults."""
        env = os.environ if env is None else env
        raw_ids = env.get("SIM_ENGINE_IDS", "").strip()
        engine_ids = tuple(part.strip() for part in raw_ids.split(",") if part.strip())
        return cls(
            engine_ids=engine_ids if raw_ids else DEFAULT_ENGINE_IDS,
            tick_seconds=_env_value(env, "SIM_TICK_SECONDS", float, cls.tick_seconds),
            pulses_per_cycle=_env_value(env, "SIM_PULSES_PER_CYCLE", int, cls.pulses_per_cycle),
            mean_life_cycles=_env_value(env, "SIM_MEAN_LIFE_CYCLES", float, cls.mean_life_cycles),
            backfill_lifecycles=_env_value(env, "SIM_BACKFILL_LIFECYCLES", int, cls.backfill_lifecycles),
            maintenance_seconds=_env_value(env, "SIM_MAINTENANCE_SECONDS", float, cls.maintenance_seconds),
            seed=_env_value(env, "SIM_SEED", int, cls.seed),
            log_level=env.get("LOG_LEVEL", "").strip() or cls.log_level,
        )

    def summary(self) -> str:
        """One-line description for the start-up log."""
        return (
            f"{len(self.engine_ids)} engines ({', '.join(self.engine_ids)}), tick {self.tick_seconds:g} s, "
            f"{self.pulses_per_cycle} pulses/cycle, mean life {self.mean_life_cycles:g} cycles, "
            f"repair {self.maintenance_seconds:g} s, backfill {self.backfill_lifecycles} lifecycles, "
            f"seed {self.seed}"
        )


def restore_engine(snapshot: EngineSnapshot, factory: EngineFactory) -> EngineSimulator:
    """Rebuild an engine twin from its registry row, current-lifecycle telemetry and work order."""
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
        wear_floor=snapshot.last_wear_factor or 0.0,
        last_event_ts=snapshot.last_event_ts,
    )
    if work_order is not None:
        # FAILURE committed before the restart: the repair completes once it is due (at once if overdue).
        engine.resume_maintenance(work_order.failure_ts, work_order.failure_mode)
    return engine


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
    """Write SIM_BACKFILL_LIFECYCLES failed-and-repaired lifecycles plus a partial current one per engine.

    History is backdated so each engine's last event lies one tick before now, with pulses one tick apart
    and repairs SIM_MAINTENANCE_SECONDS long: it looks exactly like an earlier live run. Every lifecycle
    is one transaction issuing the same writes as the live loop.
    """
    tick = timedelta(seconds=config.tick_seconds)
    repair = timedelta(seconds=config.maintenance_seconds)
    count = config.backfill_lifecycles
    logger.info("Empty telemetry table: backfilling %d lifecycles plus a partial one per engine", count)
    for engine in engines:
        lengths = engine.lifecycle_lengths(count + 1)
        rng = np.random.default_rng(zlib.crc32(f"{config.seed}|{engine.engine_id}|backfill".encode()))
        partial = int(rng.uniform(0.0, MAX_PARTIAL_LIFE_FRACTION) * lengths[-1])
        # Event-time span from the first backfilled pulse to the last backfilled event (pulse or repair).
        span = sum(((n - 1) * tick + repair + tick for n in lengths[:-1]), timedelta()) + (partial - 1) * tick
        start = datetime.now(UTC) - tick - span
        for index in range(count + 1):
            if stop_event.is_set():
                logger.info("Backfill interrupted by shutdown request")
                return
            lifecycle_id = engine.lifecycle_id
            ops, start = lifecycle_history(engine, start, tick, repair, partial if index == count else None)
            if ops:
                store.execute_with_retry(ops)
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
    """Register the fleet, rebuild every twin from the database and backfill history into an empty one."""
    created = store.register_engines(engine_models)
    logger.info("Engine registry ready: %d engines, %d newly registered", len(engine_models), created)
    backfill = allow_backfill and config.backfill_lifecycles > 0 and store.telemetry_is_empty()
    engines = [restore_engine(store.load_snapshot(engine_id), factory) for engine_id in config.engine_ids]
    if backfill:
        backfill_fleet(engines, store, config, stop_event)
    else:
        logger.info("Resuming fleet: %s", " | ".join(engine.describe() for engine in engines))
    return engines
