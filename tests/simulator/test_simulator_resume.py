"""Restart semantics of the simulator against an in-memory database: stored per-lifecycle settings,
adoption or refusal of lifecycles written before they were stored, and per-engine backfill."""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import engine_generator as eg
import pytest
from fleet_bootstrap import ResumeMismatchError, bootstrap_fleet
from fleet_config import GenerationSettings, SimulatorConfig
from fleet_runtime import FleetRunner

OLD = SimulatorConfig(
    engine_ids=("JT9D-001", "CFM56-104", "GE90-201"),
    tick_seconds=1.0,
    pulses_per_cycle=3,
    mean_life_cycles=8,
    backfill_lifecycles=0,
    maintenance_seconds=2.0,
)
NEW = replace(OLD, mean_life_cycles=4, pulses_per_cycle=2, seed=7)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class MemoryFleetDB:
    """Applies the simulator's write ops to in-memory tables; serves snapshots like fleet_db.SNAPSHOT_SQL."""

    def __init__(self) -> None:
        self.registry: dict[str, dict[str, Any]] = {}
        self.telemetry: list[eg.Pulse] = []
        self.work_orders: dict[tuple[str, int], dict[str, Any]] = {}
        self.settings: dict[tuple[str, int], GenerationSettings] = {}
        self.transactions = 0
        self.stop_after: tuple[int, threading.Event] | None = None

    def ensure_schema(self) -> None:
        """The simulator-owned table always exists in memory."""

    def register_engines(self, engine_models: dict[str, str]) -> int:
        fresh = [engine_id for engine_id in engine_models if engine_id not in self.registry]
        for engine_id in fresh:
            self.registry[engine_id] = {"status": "HEALTHY", "lifecycle": 1, "cycle": 0, "csn": 0, "done": 0}
        return len(fresh)

    def execute(self, ops: list[tuple[str, Any]]) -> list[Any]:
        self.transactions += 1
        results = [getattr(self, f"_{name}")(payload) for name, payload in ops]
        if self.stop_after is not None and self.transactions >= self.stop_after[0]:
            self.stop_after[1].set()  # SIGTERM arrives while this transaction commits
        return results

    execute_with_retry = execute

    def _record_settings(self, record: Any) -> None:
        self.settings[(record.engine_id, record.lifecycle_id)] = record.settings

    def _insert_pulses(self, pulses: list[eg.Pulse]) -> None:
        self.telemetry.extend(pulses)

    def _record_failure(self, event: Any) -> int:
        pulse = event.pulse
        self.telemetry.append(pulse)
        key = (pulse.engine_id, pulse.lifecycle_id)
        self.work_orders[key] = {
            "status": "OPEN",
            "failure_ts": pulse.event_ts,
            "failure_mode": event.failure_mode,
        }
        self.registry[pulse.engine_id] |= {"status": "FAILURE", "cycle": pulse.cycle_number}
        self.registry[pulse.engine_id]["csn"] = event.cycles_since_new
        return len(self.work_orders)

    def _update_registry(self, update: Any) -> None:
        row = self.registry[update.engine_id]
        row |= {
            "lifecycle": update.lifecycle_id,
            "cycle": update.current_cycle,
            "csn": update.cycles_since_new,
        }
        row["status"] = update.status or row["status"]

    def _complete_maintenance(self, completion: Any) -> None:
        self.work_orders[(completion.engine_id, completion.lifecycle_id)]["status"] = "COMPLETED"
        row = self.registry[completion.engine_id]
        row |= {"status": "MAINTENANCE_RESET", "lifecycle": completion.next_lifecycle_id, "cycle": 0}
        row["done"] += 1

    def rows(self, engine_id: str, lifecycle_id: int | None = None) -> list[eg.Pulse]:
        return sorted(
            (
                p
                for p in self.telemetry
                if p.engine_id == engine_id and lifecycle_id in (None, p.lifecycle_id)
            ),
            key=lambda p: p.event_ts,
        )

    def load_snapshot(self, engine_id: str) -> SimpleNamespace:
        row = self.registry[engine_id]
        current = self.rows(engine_id, row["lifecycle"])
        everything = self.rows(engine_id)
        order = self.work_orders.get((engine_id, row["lifecycle"]))
        return SimpleNamespace(
            engine_id=engine_id,
            operational_status=row["status"],
            lifecycle_id=row["lifecycle"],
            current_cycle=row["cycle"],
            cycles_since_new=row["csn"],
            pulses_in_lifecycle=len(current),
            last_cycle_number=max((p.cycle_number for p in current), default=None),
            last_wear_factor=max((p.wear_factor for p in current), default=None),
            last_event_ts=everything[-1].event_ts if everything else None,
            settings=self.settings.get((engine_id, row["lifecycle"])),
            work_order=SimpleNamespace(**order) if order else None,
        )


def start(db: MemoryFleetDB, config: SimulatorConfig, stop: threading.Event | None = None) -> FleetRunner:
    """One simulator start: bootstrap (with backfill) against ``db``, returning the live runner."""
    stop = stop or threading.Event()
    factory = functools.partial(eg.EngineSimulator, config=config)
    models = {engine_id: eg.build_profile(engine_id).engine_model for engine_id in config.engine_ids}
    engines = bootstrap_fleet(db, config, factory, models, allow_backfill=True, stop_event=stop)
    return FleetRunner(engines, db, config, stop)


def run(runner: FleetRunner, first_second: int, ticks: int, base: datetime = T0) -> int:
    """Run ``ticks`` live ticks one second apart from ``base``; returns the next free second.

    Backfilled history ends just before the wall-clock now, so tests with backfill pass a base taken
    right before the simulator starts.
    """
    for second in range(first_second, first_second + ticks):
        assert runner.run_tick(base + timedelta(seconds=second))
    return first_second + ticks


def assert_every_lifecycle_follows_its_recorded_settings(db: MemoryFleetDB) -> None:
    """Each stored lifecycle is exactly the trajectory of its recorded settings: wear, cycle, failure."""
    for engine_id in db.registry:
        for lifecycle_id in sorted({p.lifecycle_id for p in db.rows(engine_id)}):
            settings = db.settings[(engine_id, lifecycle_id)]
            twin = eg.EngineSimulator(engine_id, OLD, lifecycle_id=lifecycle_id, settings=settings)
            rows = db.rows(engine_id, lifecycle_id)
            assert len(rows) <= len(twin.wear_path)
            for index, pulse in enumerate(rows):
                assert pulse.wear_factor == twin.wear_path[index], (engine_id, lifecycle_id, index)
                assert pulse.cycle_number == index // settings.pulses_per_cycle, (engine_id, lifecycle_id)
            assert [p.is_failure for p in rows].count(True) == (len(rows) == len(twin.wear_path))


def test_restart_with_the_same_settings_continues_every_lifecycle_exactly() -> None:
    db = MemoryFleetDB()
    second = run(start(db, OLD), 0, 45)
    second = run(start(db, OLD), second + 30, 60)
    assert_every_lifecycle_follows_its_recorded_settings(db)
    assert set(db.settings.values()) == {OLD.generation}
    assert all(row["done"] >= 1 for row in db.registry.values())


def test_changed_settings_apply_from_the_next_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    db = MemoryFleetDB()
    second = run(start(db, OLD), 0, 40)
    in_progress = {
        e: r["lifecycle"] for e, r in db.registry.items() if db.load_snapshot(e).pulses_in_lifecycle
    }
    in_progress = {e: lc for e, lc in in_progress.items() if db.load_snapshot(e).work_order is None}
    assert in_progress, "the scenario needs engines in the middle of a lifecycle"
    with caplog.at_level(logging.WARNING, logger="fleet_bootstrap"):
        runner = start(db, NEW)
    for engine_id, lifecycle_id in in_progress.items():
        assert f"{engine_id} lifecycle {lifecycle_id} keeps the settings it started with" in caplog.text
    assert "SIM_SEED 42 -> 7, SIM_MEAN_LIFE_CYCLES 8 -> 4, SIM_PULSES_PER_CYCLE 3 -> 2" in caplog.text
    run(runner, second + 30, 120)
    assert_every_lifecycle_follows_its_recorded_settings(db)
    for engine_id, lifecycle_id in in_progress.items():
        assert db.settings[(engine_id, lifecycle_id)] == OLD.generation
        assert db.work_orders[(engine_id, lifecycle_id)]["status"] == "COMPLETED", "finished on its old path"
        assert db.settings[(engine_id, lifecycle_id + 1)] == NEW.generation


def test_lifecycle_without_recorded_settings_is_adopted_when_the_settings_reproduce_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = MemoryFleetDB()
    second = run(start(db, OLD), 0, 40)
    db.settings.clear()  # a database written before the simulator recorded its settings
    with caplog.at_level(logging.INFO, logger="fleet_bootstrap"):
        runner = start(db, OLD)
    assert (
        "Recorded the settings (SIM_SEED=42, SIM_MEAN_LIFE_CYCLES=8, SIM_PULSES_PER_CYCLE=3)" in caplog.text
    )
    snapshots = [db.load_snapshot(engine_id) for engine_id in db.registry]
    current = {(s.engine_id, s.lifecycle_id) for s in snapshots if s.pulses_in_lifecycle and not s.work_order}
    assert current and current == set(db.settings), "exactly the in-progress lifecycles are recorded"
    run(runner, second + 30, 40)
    for engine_id in db.registry:  # lifecycles finished before the upgrade stay without a record
        lifecycles = {p.lifecycle_id for p in db.rows(engine_id)}
        db.settings |= {(engine_id, lc): OLD.generation for lc in lifecycles}
    assert_every_lifecycle_follows_its_recorded_settings(db)


def test_lifecycle_without_recorded_settings_is_refused_under_other_settings() -> None:
    db = MemoryFleetDB()
    run(start(db, OLD), 0, 40)
    db.settings.clear()
    transactions, rows = db.transactions, len(db.telemetry)
    with pytest.raises(ResumeMismatchError) as refused:
        start(db, NEW)
    message = str(refused.value)
    assert (
        "under its current settings (SIM_SEED=7, SIM_MEAN_LIFE_CYCLES=4, SIM_PULSES_PER_CYCLE=2)" in message
    )
    assert "Restore the SIM_SEED, SIM_MEAN_LIFE_CYCLES and SIM_PULSES_PER_CYCLE values" in message
    assert "`make clean`" in message
    assert (db.transactions, len(db.telemetry), db.settings) == (transactions, rows, {}), "nothing written"


def test_lifecycle_that_contradicts_its_recorded_settings_is_refused() -> None:
    db = MemoryFleetDB()
    run(start(db, OLD), 0, 40)
    engine_id = next(e for e in db.registry if db.load_snapshot(e).pulses_in_lifecycle > 2)
    # A second writer (before the single-writer lock existed) wrote some pulse indexes twice.
    db.telemetry += db.rows(engine_id, db.registry[engine_id]["lifecycle"])[-2:]
    with pytest.raises(
        ResumeMismatchError, match=f"{engine_id} lifecycle .* under its recorded settings"
    ) as err:
        start(db, OLD)
    assert "changed outside the simulator" in str(err.value)


def test_interrupted_first_boot_backfill_completes_on_the_next_start() -> None:
    config = replace(OLD, backfill_lifecycles=3)
    db, stop, base = MemoryFleetDB(), threading.Event(), datetime.now(UTC)
    db.stop_after = (1, stop)  # SIGTERM while the first engine's history commits
    start(db, config, stop)
    assert {p.engine_id for p in db.telemetry} == {"JT9D-001"} and db.registry["JT9D-001"]["done"] == 3
    first_engine_rows = db.rows("JT9D-001")
    db.stop_after = None
    run(start(db, config), 0, 5, base)
    assert all(row["done"] >= 3 for row in db.registry.values()), "every engine got its history"
    assert db.rows("JT9D-001")[: len(first_engine_rows)] == first_engine_rows, "not backfilled twice"
    assert len(db.rows("JT9D-001")) == len(first_engine_rows) + 5
    assert_every_lifecycle_follows_its_recorded_settings(db)


def test_engine_added_to_the_fleet_later_gets_history_too() -> None:
    config = replace(OLD, backfill_lifecycles=2)
    db, base = MemoryFleetDB(), datetime.now(UTC)
    second = run(start(db, config), 0, 10, base)
    rows_before = {engine_id: len(db.rows(engine_id)) for engine_id in config.engine_ids}
    grown = replace(config, engine_ids=(*config.engine_ids, "GE90-202"))
    run(start(db, grown), second, 1, base)
    assert db.registry["GE90-202"]["done"] == 2
    *history, first_live = db.rows("GE90-202")
    assert history and all(p.event_ts < first_live.event_ts for p in history)
    assert first_live.event_ts == base + timedelta(seconds=second), "history ends before its live pulses"
    for engine_id, count in rows_before.items():
        assert len(db.rows(engine_id)) == count + 1, f"{engine_id} only got its live pulse"
