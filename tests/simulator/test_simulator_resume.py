"""Restart semantics of the simulator against an in-memory database: stored per-lifecycle settings,
adoption or refusal of lifecycles written before they were stored, per-engine backfill, the HF frame
identity (frame_seq, telemetry_id, frame_start_ts) across restarts, and the resumable HF backfill."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import engine_generator as eg
import pytest
from fleet_bootstrap import ResumeMismatchError, bootstrap_fleet
from fleet_config import GenerationSettings, SimulatorConfig
from fleet_runtime import FleetRunner, HfBackfillMarker, RecordedFailure
from hf_config import HfConfig
from hf_frames import FrameRequest, FrameTap
from hf_runtime import HfPath, open_hf_path

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
STORED_COLUMNS = (  # fleet_db.StoredPulse after telemetry_id: no theta, no failure mode
    "lifecycle_id",
    "cycle_number",
    "event_ts",
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "wear_factor",
    "status",
)


class MemoryFleetDB:
    """Applies the simulator's write ops to in-memory tables; serves snapshots like fleet_db.SNAPSHOT_SQL.

    Telemetry rows get BIGSERIAL-like ids (``ids`` maps (engine_id, event_ts) to them), returned like
    fleet_writer returns them: a list per insert_pulses, a RecordedFailure per record_failure.
    """

    def __init__(self) -> None:
        self.registry: dict[str, dict[str, Any]] = {}
        self.telemetry: list[eg.Pulse] = []
        self.ids: dict[tuple[str, datetime], int] = {}
        self.work_orders: dict[tuple[str, int], dict[str, Any]] = {}
        self.settings: dict[tuple[str, int], GenerationSettings] = {}
        self.hf_markers: dict[str, HfBackfillMarker] = {}
        self.marker_reads = 0
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

    def _insert_pulses(self, pulses: list[eg.Pulse]) -> list[int]:
        self.telemetry.extend(pulses)
        for pulse in pulses:
            self.ids[(pulse.engine_id, pulse.event_ts)] = len(self.ids) + 1
        return [self.ids[(pulse.engine_id, pulse.event_ts)] for pulse in pulses]

    def _record_failure(self, event: Any) -> RecordedFailure:
        pulse = event.pulse
        (telemetry_id,) = self._insert_pulses([pulse])
        key = (pulse.engine_id, pulse.lifecycle_id)
        self.work_orders[key] = {
            "status": "OPEN",
            "failure_ts": pulse.event_ts,
            "failure_mode": event.failure_mode,
        }
        self.registry[pulse.engine_id] |= {"status": "FAILURE", "cycle": pulse.cycle_number}
        self.registry[pulse.engine_id]["csn"] = event.cycles_since_new
        return RecordedFailure(len(self.work_orders), telemetry_id)

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

    def _mark_hf_backfill(self, marker: HfBackfillMarker) -> None:
        self.hf_markers[marker.engine_id] = marker

    def _clear_hf_backfill(self, engine_ids: list[str]) -> None:
        for engine_id in engine_ids:
            self.hf_markers.pop(engine_id, None)

    def hf_backfill_markers(self) -> list[HfBackfillMarker]:
        self.marker_reads += 1
        return [self.hf_markers[engine_id] for engine_id in sorted(self.hf_markers)]

    def backfilled_history(self, marker: HfBackfillMarker) -> SimpleNamespace:
        """Only what PostgreSQL stores: the telemetry columns and the recorded settings."""
        rows = [
            SimpleNamespace(
                telemetry_id=self.ids[(p.engine_id, p.event_ts)], **{c: getattr(p, c) for c in STORED_COLUMNS}
            )
            for p in self.rows(marker.engine_id)
            if p.event_ts <= marker.last_event_ts
        ]
        settings = {lc: s for (engine_id, lc), s in self.settings.items() if engine_id == marker.engine_id}
        return SimpleNamespace(rows=rows, settings=settings)

    def clear_hf_backfill(self, engine_ids: list[str]) -> bool:
        self.execute([("clear_hf_backfill", list(engine_ids))])
        return True

    def telemetry_counts(self, engine_ids: list[str]) -> dict[str, int]:
        return {engine_id: len(self.rows(engine_id)) for engine_id in engine_ids}

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


class RecordingPublisher:
    """hf_producer.FramePublisher stand-in that keeps every frame request it is handed."""

    def __init__(self) -> None:
        self.frames: list[FrameRequest] = []

    def submit(self, requests: list[FrameRequest]) -> None:
        self.frames.extend(requests)

    submit_backfill = submit


def start_with_frames(
    db: MemoryFleetDB, config: SimulatorConfig, publisher: RecordingPublisher
) -> FleetRunner:
    """One simulator start with the HF tap wired in like engine_generator.main does (counts read first)."""
    stop = threading.Event()
    factory = functools.partial(eg.EngineSimulator, config=config)
    models = {engine_id: eg.build_profile(engine_id).engine_model for engine_id in config.engine_ids}
    tap = FrameTap(publisher, models, db.telemetry_counts(list(config.engine_ids)))
    engines = bootstrap_fleet(
        db, config, factory, models, allow_backfill=True, stop_event=stop, on_commit=tap.backfill
    )
    return FleetRunner(engines, db, config, stop, tap.live)


def test_frames_carry_the_pulse_ordinal_row_id_and_event_time_across_restarts() -> None:
    config = replace(OLD, backfill_lifecycles=1)
    db, publisher, base = MemoryFleetDB(), RecordingPublisher(), datetime.now(UTC)
    second = run(start_with_frames(db, config, publisher), 0, 25, base)
    # Restart with one more engine: the others resume their sequence, the newcomer is backfilled from 1.
    grown = replace(config, engine_ids=(*config.engine_ids, "GE90-202"))
    run(start_with_frames(db, grown, publisher), second + 10, 25, base)
    for engine_id in grown.engine_ids:
        rows = db.rows(engine_id)
        frames = [frame for frame in publisher.frames if frame.engine_id == engine_id]
        assert len(rows) > 25 and len(frames) == len(rows), "one frame per committed pulse"
        assert [frame.frame_seq for frame in frames] == list(range(1, len(rows) + 1))
        assert [frame.frame_start for frame in frames] == [pulse.event_ts for pulse in rows]
        assert [frame.telemetry_id for frame in frames] == [db.ids[(engine_id, p.event_ts)] for p in rows]
        assert [(f.lifecycle_id, f.cycle_number) for f in frames] == [
            (p.lifecycle_id, p.cycle_number) for p in rows
        ]
        assert {frame.engine_model for frame in frames} == {eg.build_profile(engine_id).engine_model}


class KafkaStandIn:
    """Kafka for the real HF publisher: ready unless ``ready`` is False; acknowledges the frames ``acks``
    accepts (all by default) and holds the others; keeps every record built and every one acknowledged."""

    topic = "engine_hf.sensor_frames"
    fatal_error = None

    def __init__(
        self, *, ready: bool = True, acks: Callable[[dict[str, Any]], bool] = lambda record: True
    ) -> None:
        self.ready = ready
        self.acks = acks
        self.schema_id: int | None = None
        self.built: list[dict[str, Any]] = []
        self.acknowledged: list[dict[str, Any]] = []
        self.pending: list[tuple[dict[str, Any], Any]] = []

    def check_ready(self, timeout: float) -> str | None:
        self.schema_id = 1
        return None if self.ready else "topic engine_hf.sensor_frames does not exist yet"

    def serialize(self, record: dict[str, Any]) -> bytes:
        self.built.append(record)
        return f"{record['engine_id']}#{record['frame_seq']}".encode()

    def produce(self, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.pending.append((self.built[-1], on_delivery))

    def poll(self, timeout: float) -> int:
        acks = self.acks  # read once: a test may swap it while the publisher's worker polls
        served = [(record, callback) for record, callback in self.pending if acks(record)]
        self.pending = [(record, callback) for record, callback in self.pending if not acks(record)]
        for record, callback in served:
            callback(None, None)
            self.acknowledged.append(record)
        return len(served)

    def flush(self, timeout: float) -> int:
        self.poll(0)
        return len(self.pending)

    def close(self) -> None:
        pass


def start_hf(
    db: MemoryFleetDB, config: SimulatorConfig, sink_factory: Callable[[], Any], *, enabled: bool = True
) -> tuple[HfPath, FleetRunner]:
    """One simulator start wired like engine_generator.main: the HF path (real publisher, the given Kafka),
    bootstrap with the backfill markers, then the resume of the frames earlier backfills still owe."""
    stop = threading.Event()
    factory = functools.partial(eg.EngineSimulator, config=config)
    models = {engine_id: eg.build_profile(engine_id).engine_model for engine_id in config.engine_ids}
    hf = open_hf_path(HfConfig(enabled=enabled), config.seed, db, models, sink_factory=sink_factory)
    engines = bootstrap_fleet(
        db,
        config,
        factory,
        models,
        allow_backfill=True,
        stop_event=stop,
        on_commit=hf.backfill,
        mark_hf_backfill=hf.marks_backfill,
    )
    hf.resume_backfill(factory, stop)
    return hf, FleetRunner(engines, db, config, stop, hf.live)


def wait_for(condition: Callable[[], bool], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def key(record: dict[str, Any]) -> tuple[str, int]:
    return record["engine_id"], record["telemetry_id"]


def test_backfill_frames_a_stop_left_unacknowledged_are_rebuilt_byte_identical_later() -> None:
    config = replace(OLD, backfill_lifecycles=2)
    db, base = MemoryFleetDB(), datetime.now(UTC)
    silent = KafkaStandIn(acks=lambda record: False)  # takes every frame, acknowledges none
    hf, runner = start_hf(db, config, lambda: silent)
    history = {engine_id: db.rows(engine_id) for engine_id in config.engine_ids}
    owed = {(e, db.ids[(e, p.event_ts)]) for e, rows in history.items() for p in rows}
    wait_for(lambda: len(silent.built) == len(owed))
    second = run(runner, 0, 3, base)
    hf.close()  # make down while Kafka still owes every acknowledgement
    written = {e: HfBackfillMarker(e, len(rows), rows[-1].event_ts) for e, rows in history.items()}
    assert db.hf_markers == written, "written with the history, kept until acknowledged"
    never_up = KafkaStandIn(ready=False)
    hf, runner = start_hf(db, config, lambda: never_up)  # the resumed backfill waits for Kafka ...
    second = run(runner, second, 1, base)
    hf.close()  # ... and the stack is stopped again before Kafka ever came up
    assert never_up.built == [] and db.hf_markers == written
    healthy = KafkaStandIn()
    hf, runner = start_hf(db, config, lambda: healthy)
    wait_for(lambda: owed <= {key(record) for record in healthy.acknowledged})
    run(runner, second, 1, base)  # the tick after the acknowledgements clears the markers
    hf.close()
    assert db.hf_markers == {}
    originals = {key(record): record for record in silent.built}
    resumed = [record for record in healthy.acknowledged if key(record) in owed]
    assert len(resumed) == len(owed) and all(record == originals[key(record)] for record in resumed)
    for engine_id, rows in history.items():
        seqs = [r["frame_seq"] for r in resumed if r["engine_id"] == engine_id]
        assert seqs == list(range(1, len(rows) + 1)), "the original frame_seq: the pulse ordinal"
        (live,) = [r for r in healthy.acknowledged if r["engine_id"] == engine_id and key(r) not in owed]
        assert live["frame_seq"] == len(db.rows(engine_id)), "live frames continue after every stored row"
    again = KafkaStandIn()
    hf, _ = start_hf(db, config, lambda: again)
    hf.close()
    assert again.built == [] and db.hf_markers == {}, "a paid backfill is never sent again"


def test_a_restart_mid_drain_resends_only_the_unacknowledged_engines_and_a_newcomer_once() -> None:
    config = replace(OLD, backfill_lifecycles=1)
    db, base = MemoryFleetDB(), datetime.now(UTC)
    kafka = KafkaStandIn(acks=lambda record: False)
    hf, runner = start_hf(db, config, lambda: kafka)
    history = {engine_id: len(db.rows(engine_id)) for engine_id in config.engine_ids}
    # Kafka acknowledges JT9D-001's history and every live frame, then the stack is stopped.
    kafka.acks = lambda r: r["engine_id"] == "JT9D-001" or r["frame_seq"] > history[r["engine_id"]]
    wait_for(lambda: len(kafka.acknowledged) == history["JT9D-001"])
    second = run(runner, 0, 2, base)
    hf.close()
    assert set(db.hf_markers) == {"CFM56-104", "GE90-201"}
    grown = replace(config, engine_ids=(*config.engine_ids, "GE90-202"))
    healthy = KafkaStandIn()
    hf, runner = start_hf(db, grown, lambda: healthy)
    owed = history["CFM56-104"] + history["GE90-201"] + len(db.rows("GE90-202"))
    wait_for(lambda: len(healthy.acknowledged) == owed)
    run(runner, second, 1, base)
    hf.close()
    assert db.hf_markers == {}
    delivered = kafka.acknowledged + healthy.acknowledged
    for engine_id in grown.engine_ids:  # every pulse's frame exactly once, in its ordinal: no gap, no repeat
        rows = db.rows(engine_id)
        frames = sorted((r for r in delivered if r["engine_id"] == engine_id), key=lambda r: r["frame_seq"])
        assert [r["frame_seq"] for r in frames] == list(range(1, len(rows) + 1)), engine_id
        assert [r["telemetry_id"] for r in frames] == [db.ids[(engine_id, p.event_ts)] for p in rows]


def test_with_hf_disabled_no_marker_is_written_and_earlier_ones_wait_untouched() -> None:
    config = replace(OLD, backfill_lifecycles=1, engine_ids=("JT9D-001", "GE90-201"))
    db, base = MemoryFleetDB(), datetime.now(UTC)
    hf, _ = start_hf(db, config, lambda: KafkaStandIn(ready=False))  # Kafka never came up
    hf.close()
    markers, reads = dict(db.hf_markers), db.marker_reads
    assert set(markers) == {"JT9D-001", "GE90-201"}

    def no_kafka() -> Any:
        raise AssertionError("SIM_HF_ENABLED=false must not create a Kafka client")

    grown = replace(config, engine_ids=(*config.engine_ids, "CFM56-104"))
    hf, runner = start_hf(db, grown, no_kafka, enabled=False)
    run(runner, 0, 2, base)
    hf.close()
    assert db.registry["CFM56-104"]["done"] == 1, "the newcomer got its history ..."
    assert db.hf_markers == markers, (
        "... but owes no frames, and the earlier markers wait for an enabled start"
    )
    assert db.marker_reads == reads
