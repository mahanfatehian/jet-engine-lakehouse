"""Lifecycle state machine, live tick transactions, backfill and resume of the simulator (no database)."""

from __future__ import annotations

import itertools
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import engine_generator as eg
import pytest
from fleet_bootstrap import backfill_fleet, bootstrap_fleet, restore_engine
from fleet_config import GenerationSettings, SimulatorConfig
from fleet_runtime import FleetRunner, LifecycleSettings, build_tick_ops, completion_at, registry_update

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SMALL = SimulatorConfig(
    engine_ids=("JT9D-001", "CFM56-104", "GE90-201"),
    tick_seconds=1.0,
    pulses_per_cycle=2,
    mean_life_cycles=6,
    backfill_lifecycles=2,
    maintenance_seconds=3.0,
)
STATUS_ORDER = {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2, "FAILURE": 3}


class FakeStore:
    """In-memory stand-in for fleet_db.FleetDatabase that records every transaction.

    ``with_telemetry`` names the engines whose snapshot shows earlier telemetry (all others are fresh).
    """

    def __init__(self, *, with_telemetry: tuple[str, ...] = ()) -> None:
        self.transactions: list[list[tuple[str, Any]]] = []
        self.drop_next = False
        self.with_telemetry = with_telemetry
        self.registered: dict[str, str] = {}
        self.schema_checks = 0
        self._work_orders = 0

    def execute(self, ops: list[tuple[str, Any]]) -> list[Any] | None:
        if self.drop_next:
            self.drop_next = False
            return None
        self.transactions.append(list(ops))
        results: list[Any] = []
        for name, _ in ops:
            if name == "record_failure":
                self._work_orders += 1
                results.append(self._work_orders)
            else:
                results.append(None)
        return results

    def execute_with_retry(self, ops: list[tuple[str, Any]]) -> list[Any]:
        results = self.execute(ops)
        assert results is not None
        return results

    def ensure_schema(self) -> None:
        self.schema_checks += 1

    def register_engines(self, engine_models: dict[str, str]) -> int:
        self.registered = dict(engine_models)
        return len(engine_models)

    def load_snapshot(self, engine_id: str) -> SimpleNamespace:
        if engine_id in self.with_telemetry:
            # Previous lifecycle failed and was repaired; the current one has no pulse yet.
            return snapshot(
                engine_id, lifecycle_id=2, operational_status="MAINTENANCE_RESET", last_event_ts=T0
            )
        return snapshot(engine_id)

    def ops(self, name: str) -> list[Any]:
        return [payload for ops in self.transactions for op, payload in ops if op == name]

    def pulses(self) -> list[eg.Pulse]:
        rows = [pulse for batch in self.ops("insert_pulses") for pulse in batch]
        return rows + [event.pulse for event in self.ops("record_failure")]


def snapshot(engine_id: str, **overrides: Any) -> SimpleNamespace:
    """A freshly registered engine as fleet_db.FleetDatabase.load_snapshot would return it."""
    fields = {
        "engine_id": engine_id,
        "operational_status": "HEALTHY",
        "lifecycle_id": 1,
        "current_cycle": 0,
        "cycles_since_new": 0,
        "pulses_in_lifecycle": 0,
        "last_cycle_number": None,
        "last_wear_factor": None,
        "last_event_ts": None,
        "settings": None,
        "work_order": None,
    }
    return SimpleNamespace(**(fields | overrides))


def factory(engine_id: str, **kwargs: Any) -> eg.EngineSimulator:
    return eg.EngineSimulator(engine_id, SMALL, **kwargs)


def run_to_failure(engine: eg.EngineSimulator, start: datetime = T0) -> list[eg.Pulse]:
    pulses, ts = [], start
    while not pulses or not pulses[-1].is_failure:
        pulses.append(engine.next_pulse(ts))
        engine.advance(pulses[-1])
        ts += timedelta(seconds=1)
    return pulses


def advance_to_last_pulse(engine: eg.EngineSimulator) -> datetime:
    """Commit every pulse before the FAILURE one; returns the time of the pending failure pulse."""
    ts = T0
    while engine.pulse_index < len(engine.wear_path) - 1:
        engine.advance(engine.next_pulse(ts))
        ts += timedelta(seconds=1)
    return ts


def test_every_lifecycle_walks_healthy_warning_critical_failure() -> None:
    # Needs enough pulses per lifecycle: with only a dozen, one wear step can jump the 0.2-wide CRITICAL band.
    config = SimulatorConfig(pulses_per_cycle=4, mean_life_cycles=30)
    for engine_id in ("JT9D-001", "CFM56-104", "GE90-201"):
        engine = eg.EngineSimulator(engine_id, config)
        for _ in range(5):
            pulses = run_to_failure(engine)
            ranks = [STATUS_ORDER[p.status] for p in pulses]
            assert ranks == sorted(ranks) and ranks.count(3) == 1 and set(ranks) == {0, 1, 2, 3}
            cycles = [p.cycle_number for p in pulses]
            assert cycles == sorted(cycles) and cycles[0] == 0
            assert all(cycles.count(c) == config.pulses_per_cycle for c in range(cycles[-1]))
            engine.start_next_lifecycle(completion_at(engine, pulses[-1].event_ts))


def test_failure_takes_the_engine_into_maintenance_until_repaired() -> None:
    engine = factory("JT9D-001")
    pulses = run_to_failure(engine)
    cycles = [p.cycle_number for p in pulses]
    assert pulses[-1].wear_factor == 1.0 and pulses[0].wear_factor == 0.0
    assert engine.in_maintenance and engine.cycles_since_new == cycles[-1] + 1
    with pytest.raises(RuntimeError):
        engine.next_pulse(T0 + timedelta(days=1))
    completion = completion_at(engine, pulses[-1].event_ts + timedelta(seconds=3))
    assert completion.downtime_seconds == 3
    assert completion.action_taken == eg.FAILURE_MODES[engine.lifecycle.failure_mode].maintenance_action
    engine.start_next_lifecycle(completion)
    assert (engine.lifecycle_id, engine.status, engine.pulse_index) == (2, "MAINTENANCE_RESET", 0)
    first = engine.next_pulse(T0 + timedelta(days=1))
    assert (first.lifecycle_id, first.cycle_number, first.wear_factor, first.status) == (2, 0, 0.0, "HEALTHY")
    assert registry_update(engine, first) is not None, "MAINTENANCE_RESET -> HEALTHY must reach the registry"


def test_registry_updates_once_per_cycle_and_on_every_status_change() -> None:
    engine = factory("CFM56-104")
    updates = []
    for index in range(200):
        pulse = engine.next_pulse(T0 + timedelta(seconds=index))
        if pulse.is_failure:
            break
        updates.append((pulse, registry_update(engine, pulse)))
        engine.advance(pulse)
    for pulse, update in updates:
        starts_cycle = pulse.pulse_index > 0 and pulse.pulse_index % SMALL.pulses_per_cycle == 0
        status_change = update is not None and update.status is not None
        assert (update is not None) == (starts_cycle or status_change)
        if update is not None:
            assert update.cycles_since_new == pulse.cycle_number


def test_tick_ops_put_repairs_first_and_record_failures_in_the_same_transaction() -> None:
    failing, healthy, repaired = factory("JT9D-001"), factory("CFM56-104"), factory("GE90-201")
    failure_pulse = failing.next_pulse(advance_to_last_pulse(failing))
    assert failure_pulse.is_failure
    run_to_failure(repaired)
    completion = completion_at(repaired, T0 + timedelta(hours=1))
    pulses = [(failing, failure_pulse), (healthy, healthy.next_pulse(T0))]
    ops, slots = build_tick_ops([(repaired, completion)], pulses)
    names = ["complete_maintenance", "record_settings", "insert_pulses", "record_failure"]
    assert [name for name, _ in ops] == names
    # Only the healthy engine emits the first pulse of its lifecycle, which carries the lifecycle's settings.
    assert ops[1][1] == LifecycleSettings("CFM56-104", 1, SMALL.generation)
    assert ops[2][1] == [pulses[1][1]] and slots == {"JT9D-001": 3}
    assert ops[3][1].pulse == failure_pulse and "live" in ops[3][1].notes


def test_live_loop_fails_repairs_and_restarts_every_engine() -> None:
    store = FakeStore()
    engines = [factory(engine_id) for engine_id in SMALL.engine_ids]
    runner = FleetRunner(engines, store, SMALL, threading.Event())
    for second in range(120):
        assert runner.run_tick(T0 + timedelta(seconds=second))
    failures, repairs = store.ops("record_failure"), store.ops("complete_maintenance")
    assert runner.failures == len(failures) >= 2 * len(engines)
    assert len(repairs) >= len(failures) - len(engines)
    for engine in engines:
        mine = [event.pulse for event in failures if event.pulse.engine_id == engine.engine_id]
        fixes = [c for c in repairs if c.engine_id == engine.engine_id]
        pulses = [p for p in store.pulses() if p.engine_id == engine.engine_id]
        for event_pulse, fix in zip(mine, fixes, strict=False):
            assert fix.lifecycle_id == event_pulse.lifecycle_id
            assert fix.downtime_seconds >= SMALL.maintenance_seconds
            assert not [p for p in pulses if event_pulse.event_ts < p.event_ts <= fix.completed_ts]
        lifecycles = [p.lifecycle_id for p in sorted(pulses, key=lambda p: p.event_ts)]
        assert lifecycles == sorted(lifecycles)
        # Settings are stored once per lifecycle, in the transaction of its first pulse.
        records = [r for r in store.ops("record_settings") if r.engine_id == engine.engine_id]
        assert [r.lifecycle_id for r in records] == sorted(set(lifecycles))
        assert all(r.settings == SMALL.generation for r in records)
    for ops in store.transactions:
        engine_ids = [p.engine_id for name, batch in ops if name == "insert_pulses" for p in batch]
        engine_ids += [event.pulse.engine_id for name, event in ops if name == "record_failure"]
        assert len(engine_ids) == len(set(engine_ids)), "one pulse per engine per tick"


def test_dropped_tick_leaves_the_twins_untouched() -> None:
    store = FakeStore()
    engines = [factory(engine_id) for engine_id in SMALL.engine_ids]
    runner = FleetRunner(engines, store, SMALL, threading.Event())
    runner.run_tick(T0)
    before = [(e.pulse_index, e.status, e.current_cycle) for e in engines]
    store.drop_next = True
    assert not runner.run_tick(T0 + timedelta(seconds=1))
    assert [(e.pulse_index, e.status, e.current_cycle) for e in engines] == before
    assert runner.dropped_ticks == 1 and runner.run_tick(T0 + timedelta(seconds=2))
    assert all(e.pulse_index == 2 for e in engines)


def test_runner_honours_max_ticks_and_the_stop_event() -> None:
    config = SimulatorConfig(
        engine_ids=("JT9D-001",), tick_seconds=0.01, pulses_per_cycle=2, mean_life_cycles=6
    )
    runner = FleetRunner([eg.EngineSimulator("JT9D-001", config)], FakeStore(), config, threading.Event())
    runner.run(max_ticks=5)
    assert runner.ticks == 5
    stopped = threading.Event()
    stopped.set()
    idle = FleetRunner([eg.EngineSimulator("JT9D-001", config)], FakeStore(), config, stopped)
    idle.run()
    assert idle.ticks == 0


def test_backfill_writes_backdated_lifecycles_ending_before_now() -> None:
    store = FakeStore()
    engines = [factory(engine_id) for engine_id in SMALL.engine_ids]
    started = datetime.now(UTC)
    backfill_fleet(engines, store, SMALL, threading.Event())
    tick, repair = timedelta(seconds=SMALL.tick_seconds), timedelta(seconds=SMALL.maintenance_seconds)
    for engine in engines:
        pulses = sorted(
            (p for p in store.pulses() if p.engine_id == engine.engine_id), key=lambda p: p.event_ts
        )
        stamps = [p.event_ts for p in pulses]
        assert len(stamps) == len(set(stamps))
        assert stamps[-1] <= started - tick + timedelta(seconds=1)
        failures = [p for p in pulses if p.is_failure]
        assert [p.lifecycle_id for p in failures] == [1, 2]
        for failure in failures:
            nxt = next((p for p in pulses if p.lifecycle_id == failure.lifecycle_id + 1), None)
            assert nxt is None or nxt.event_ts - failure.event_ts == repair + tick
            same = [p.event_ts for p in pulses if p.lifecycle_id == failure.lifecycle_id]
            assert all(b - a == tick for a, b in itertools.pairwise(same))
        assert engine.lifecycle_id == 3 and not engine.in_maintenance
        records = [r for r in store.ops("record_settings") if r.engine_id == engine.engine_id]
        assert [r.lifecycle_id for r in records] == sorted({p.lifecycle_id for p in pulses})
    repairs = store.ops("complete_maintenance")
    assert len(repairs) == 2 * len(engines) and all(r.downtime_seconds == 3 for r in repairs)
    assert {u.status for u in store.ops("update_registry")} >= {"HEALTHY", "WARNING", "CRITICAL"}
    assert all("backfill" in event.notes for event in store.ops("record_failure"))
    # One transaction per engine: an interrupted backfill never leaves an engine with partial history.
    assert len(store.transactions) == len(engines)
    for ops, engine in zip(store.transactions, engines, strict=True):
        written = {p.engine_id for name, batch in ops if name == "insert_pulses" for p in batch}
        assert written == {engine.engine_id}


class StopDuringWriteStore(FakeStore):
    """FakeStore whose first retried write coincides with a shutdown request (SIGTERM)."""

    def __init__(self, stop: threading.Event) -> None:
        super().__init__()
        self.stop = stop

    def execute_with_retry(self, ops: list[tuple[str, Any]]) -> list[Any]:
        self.stop.set()
        return super().execute_with_retry(ops)


def test_backfill_stops_between_engines_on_a_shutdown_request() -> None:
    stop = threading.Event()
    store = StopDuringWriteStore(stop)
    engines = [factory(engine_id) for engine_id in SMALL.engine_ids]
    backfill_fleet(engines, store, SMALL, stop)
    assert {p.engine_id for p in store.pulses()} == {"JT9D-001"}, "the running transaction completes"
    assert [engine.lifecycle_id for engine in engines] == [3, 1, 1]


def test_bootstrap_backfills_exactly_the_engines_without_telemetry() -> None:
    models = {engine_id: eg.build_profile(engine_id).engine_model for engine_id in SMALL.engine_ids}
    fresh = FakeStore()
    engines = bootstrap_fleet(
        fresh, SMALL, factory, models, allow_backfill=True, stop_event=threading.Event()
    )
    assert fresh.registered == models and fresh.schema_checks == 1 and fresh.ops("record_failure")
    assert all(engine.lifecycle_id == 3 for engine in engines)
    partly = FakeStore(with_telemetry=("JT9D-001",))
    engines = bootstrap_fleet(
        partly, SMALL, factory, models, allow_backfill=True, stop_event=threading.Event()
    )
    assert {p.engine_id for p in partly.pulses()} == {"CFM56-104", "GE90-201"}
    assert (engines[0].lifecycle_id, engines[0].pulse_index) == (2, 0), "resumed, never backfilled again"
    for allowed, count in ((False, 2), (True, 0)):
        store = FakeStore()
        config = replace(SMALL, backfill_lifecycles=count)
        bootstrap_fleet(store, config, factory, models, allow_backfill=allowed, stop_event=threading.Event())
        assert store.transactions == []


def test_resume_continues_the_same_wear_trajectory() -> None:
    original = factory("GE90-201")
    for second in range(7):
        original.advance(original.next_pulse(T0 + timedelta(seconds=second)))
    restored = restore_engine(
        snapshot(
            "GE90-201",
            current_cycle=original.current_cycle - 1,
            cycles_since_new=original.cycles_since_new - 1,
            pulses_in_lifecycle=original.pulse_index,
            last_cycle_number=original.current_cycle,
            last_wear_factor=original.wear_factor,
            last_event_ts=original.last_event_ts,
            operational_status=original.status,
        ),
        factory,
    )
    assert restored.cycles_since_new == original.cycles_since_new
    ts = T0 + timedelta(minutes=5)
    expected, actual = original.next_pulse(ts), restored.next_pulse(ts)
    assert (actual.cycle_number, actual.wear_factor, actual.pulse_index) == (
        expected.cycle_number,
        expected.wear_factor,
        expected.pulse_index,
    )
    assert restored.next_pulse(original.last_event_ts).event_ts > original.last_event_ts


def test_resume_completes_an_open_work_order_and_skips_a_closed_one() -> None:
    failed_at = T0 + timedelta(hours=1)
    order = SimpleNamespace(status="OPEN", failure_ts=failed_at, failure_mode="OIL_SYSTEM_LEAK")
    engine = restore_engine(snapshot("JT9D-001", operational_status="FAILURE", work_order=order), factory)
    assert engine.in_maintenance and engine.maintenance_action == "OIL_PUMP_AND_SEAL_OVERHAUL"
    store = FakeStore()
    FleetRunner([engine], store, SMALL, threading.Event()).run_tick(failed_at + timedelta(seconds=10))
    (repair,) = store.ops("complete_maintenance")
    assert (repair.lifecycle_id, repair.downtime_seconds) == (1, 10) and engine.lifecycle_id == 2
    closed = SimpleNamespace(status="COMPLETED", failure_ts=failed_at, failure_mode="OIL_SYSTEM_LEAK")
    moved_on = restore_engine(snapshot("JT9D-002", lifecycle_id=4, work_order=closed), factory)
    assert (moved_on.lifecycle_id, moved_on.pulse_index, moved_on.in_maintenance) == (5, 0, False)


def test_config_reads_contract_variables_and_rejects_bad_values() -> None:
    assert SimulatorConfig.from_env({}) == SimulatorConfig()
    assert SimulatorConfig().engine_ids == (
        "JT9D-001",
        "JT9D-002",
        "CFM56-104",
        "CFM56-105",
        "GE90-201",
        "GE90-202",
    )
    env = {
        "SIM_ENGINE_IDS": " JT9D-001 , GE90-201 ",
        "SIM_TICK_SECONDS": "0.2",
        "SIM_PULSES_PER_CYCLE": "2",
        "SIM_MEAN_LIFE_CYCLES": "12",
        "SIM_BACKFILL_LIFECYCLES": "0",
        "SIM_MAINTENANCE_SECONDS": "3",
        "SIM_SEED": "7",
        "LOG_LEVEL": "DEBUG",
    }
    config = SimulatorConfig.from_env(env)
    assert config.engine_ids == ("JT9D-001", "GE90-201") and config.tick_seconds == 0.2
    assert (config.pulses_per_cycle, config.mean_life_cycles, config.backfill_lifecycles) == (2, 12.0, 0)
    assert (config.maintenance_seconds, config.seed, config.log_level) == (3.0, 7, "DEBUG")
    assert config.generation == GenerationSettings(seed=7, mean_life_cycles=12.0, pulses_per_cycle=2)
    for name, value in (
        ("SIM_TICK_SECONDS", "0"),
        ("SIM_PULSES_PER_CYCLE", "1.5"),
        ("SIM_ENGINE_IDS", " , "),
        ("SIM_SEED", str(2**63)),  # stored per lifecycle in a BIGINT column
    ):
        with pytest.raises(ValueError, match="SIM_"):
            SimulatorConfig.from_env({name: value})
    with pytest.raises(ValueError, match="duplicates"):
        SimulatorConfig.from_env({"SIM_ENGINE_IDS": "JT9D-001,JT9D-001"})
    assert SimulatorConfig.from_env({"SIM_SEED": str(-(2**63))}).seed == -(2**63)


def test_generation_settings_describe_their_values_and_changes_exactly() -> None:
    old, new = GenerationSettings(2**62, 180.0, 10), GenerationSettings(7, 60.5, 10)
    assert old.describe() == f"SIM_SEED={2**62}, SIM_MEAN_LIFE_CYCLES=180, SIM_PULSES_PER_CYCLE=10"
    assert old.changes_to(new) == f"SIM_SEED {2**62} -> 7, SIM_MEAN_LIFE_CYCLES 180 -> 60.5"
    assert old.changes_to(old) == ""
