"""Resumable first-boot HF backfill: the per-engine acknowledgement ledger, the rebuild of a marked history's
frame requests from its stored rows, the resume at start-up, clearing paid markers, and the PostgreSQL side
(simulator_hf_backfill DDL, reads, writes, a clear that never reconnects)."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import itertools
import logging
import sys
import threading
import types
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import engine_generator as eg
import pytest
from fleet_bootstrap import engine_history, hf_backfill_marker
from fleet_config import GenerationSettings, SimulatorConfig
from fleet_runtime import HfBackfillMarker, RecordedFailure, committed_pulses
from fleet_writer import CLEAR_HF_BACKFILL_SQL, MARK_HF_BACKFILL_SQL, FleetWriter
from hf_backfill import (
    BackfillLedger,
    BackfillMismatchError,
    clear_paid,
    rebuild_requests,
    resume_backfill,
)
from hf_config import HfConfig
from hf_frames import FrameBuilder, FrameRequest, FrameTap

CONFIG = SimulatorConfig(
    engine_ids=("JT9D-001", "CFM56-104", "GE90-201"),
    pulses_per_cycle=3,
    mean_life_cycles=8,
    backfill_lifecycles=2,
    maintenance_seconds=2.0,
)
EVENT_TS = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


@pytest.fixture
def fleet_db(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """fleet_db imported afresh; where the image has no psycopg2 (data-platform), against a stand-in driver
    that offers only what fleet_db touches at import time."""
    if importlib.util.find_spec("psycopg2") is None:
        driver = types.ModuleType("psycopg2")
        driver.OperationalError = type("OperationalError", (Exception,), {})  # type: ignore[attr-defined]
        driver.InterfaceError = type("InterfaceError", (Exception,), {})  # type: ignore[attr-defined]
        extras = types.ModuleType("psycopg2.extras")
        extras.execute_values = None  # type: ignore[attr-defined]
        driver.extras = extras  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "psycopg2", driver)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    monkeypatch.setitem(sys.modules, "fleet_db", None)  # whatever was there is restored afterwards
    del sys.modules["fleet_db"]
    return importlib.import_module("fleet_db")


class Recorder:
    """FramePublisher stand-in: a real ledger, plus every backfill request it was handed."""

    def __init__(self) -> None:
        self.ledger = BackfillLedger()
        self.backfill: list[FrameRequest] = []

    def submit(self, requests: list[FrameRequest]) -> None:
        raise AssertionError("no live frames here")

    def submit_backfill(self, requests: list[FrameRequest]) -> None:
        self.ledger.expect(request.engine_id for request in requests)
        self.backfill.extend(requests)


@dataclass
class Backfilled:
    """One engine's first-boot backfill: its marker, what PostgreSQL returns for it, what it queued."""

    marker: HfBackfillMarker
    history: Any
    requests: list[FrameRequest]


def backfilled(engine_id: str, fleet_db: types.ModuleType, first_id: int = 1000) -> Backfilled:
    """The history fleet_bootstrap writes for ``engine_id``, with BIGSERIAL-like telemetry ids, and the
    frame requests FrameTap queued for it."""
    writes = [op for _, ops in engine_history(eg.EngineSimulator(engine_id, CONFIG), CONFIG) for op in ops]
    ids = itertools.count(first_id)
    results: list[Any] = []
    for name, payload in writes:
        if name == "insert_pulses":
            results.append([next(ids) for _ in payload])
        else:
            results.append(RecordedFailure(1, next(ids)) if name == "record_failure" else None)
    publisher = Recorder()
    FrameTap(publisher, {engine_id: eg.build_profile(engine_id).engine_model}, {}).backfill(writes, results)
    columns = fleet_db.StoredPulse._fields[1:]
    rows = [
        fleet_db.StoredPulse(telemetry_id, *(getattr(pulse, name) for name in columns))
        for pulse, telemetry_id in committed_pulses(writes, results)
    ]
    settings = {op.lifecycle_id: op.settings for name, op in writes if name == "record_settings"}
    marker = hf_backfill_marker(engine_id, writes)
    assert marker is not None
    return Backfilled(marker, fleet_db.BackfilledHistory(rows, settings), publisher.backfill)


def factory_under(config: SimulatorConfig) -> functools.partial[eg.EngineSimulator]:
    return functools.partial(eg.EngineSimulator, config=config)


def test_the_ledger_pays_an_engine_once_every_backfill_frame_of_it_is_acknowledged() -> None:
    ledger = BackfillLedger()
    ledger.expect(["JT9D-001", "GE90-201", "JT9D-001"])
    ledger.acknowledged("CFM56-104")  # a frame it never expected (a live one) changes nothing
    ledger.acknowledged("JT9D-001")
    assert ledger.take_paid() == [] and ledger.owed() == {"GE90-201": 1, "JT9D-001": 1}
    ledger.acknowledged("GE90-201")
    ledger.acknowledged("JT9D-001")
    ledger.acknowledged("JT9D-001")  # a duplicate report after the engine was paid
    assert ledger.take_paid() == ["GE90-201", "JT9D-001"] and ledger.take_paid() == []
    ledger.return_paid(["GE90-201"])
    assert ledger.take_paid() == ["GE90-201"] and ledger.owed() == {}
    assert ledger.queued("JT9D-001") and not ledger.queued("CFM56-104")


def test_the_rebuilt_requests_are_the_original_ones_and_build_the_same_bytes(
    fleet_db: types.ModuleType,
) -> None:
    done = backfilled("GE90-201", fleet_db)
    assert done.marker.frames == len(done.requests) == len(done.history.rows) > 4 * CONFIG.pulses_per_cycle
    assert done.marker.last_event_ts == done.requests[-1].frame_start
    # SIM_* changed since the backfill: every lifecycle is re-derived under the settings recorded for it.
    changed = factory_under(replace(CONFIG, seed=7, mean_life_cycles=4, pulses_per_cycle=2))
    rebuilt = rebuild_requests(done.marker, done.history, changed)
    assert rebuilt == done.requests, "telemetry_id, frame_seq, event time, theta, failure mode and readings"
    assert [r.frame_seq for r in rebuilt] == list(range(1, done.marker.frames + 1))
    assert {r.lifecycle_id for r in rebuilt} >= {1, 2} and len({r.failure_mode for r in rebuilt}) >= 1
    original, again = FrameBuilder(HfConfig(), CONFIG.seed), FrameBuilder(HfConfig(), CONFIG.seed)
    unrelated = replace(done.requests[0], engine_id="JT9D-002", telemetry_id=1)
    built = []
    for request in done.requests:  # the original worker built live frames in between
        original.build(unrelated)
        built.append(original.build(request))
    assert [again.build(request) for request in rebuilt] == built


def test_a_history_its_recorded_settings_do_not_generate_is_refused(fleet_db: types.ModuleType) -> None:
    done = backfilled("JT9D-001", fleet_db)
    factory = factory_under(CONFIG)
    rows, settings = done.history.rows, done.history.settings

    def refusal(history_rows: list[Any], recorded: dict[int, GenerationSettings], frames: int) -> str:
        marker = replace(done.marker, frames=frames)
        with pytest.raises(BackfillMismatchError) as refused:
            rebuild_requests(marker, fleet_db.BackfilledHistory(history_rows, recorded), factory)
        return str(refused.value)

    edited = [*rows[:5], rows[5]._replace(egt_c=rows[5].egt_c + 0.5), *rows[6:]]
    assert refusal(edited, settings, len(rows)).endswith(
        "is not what its recorded settings generate: egt_c differ"
    )
    without = {lifecycle: s for lifecycle, s in settings.items() if lifecycle != 2}
    assert refusal(rows, without, len(rows)) == "lifecycle 2 has no recorded generation settings"
    assert refusal(rows[:-1], settings, len(rows)).startswith(f"{len(rows) - 1} telemetry rows up to ")
    assert refusal([], settings, len(rows)).endswith(f"instead of {len(rows)}")
    assert refusal([], settings, len(rows)).startswith("no telemetry rows up to ")
    failure = next(index for index, row in enumerate(rows) if row.status == "FAILURE")
    late = rows[failure]._replace(telemetry_id=1, event_ts=rows[failure].event_ts + timedelta(milliseconds=1))
    after_failure = [*rows[: failure + 1], late, *rows[failure + 1 : -1]]
    assert "is under maintenance" in refusal(after_failure, settings, len(rows))


class MarkerStore:
    """simulator_hf_backfill in memory; ``connected=False`` is a database the clear cannot reach right now."""

    def __init__(self, histories: dict[str, Backfilled]) -> None:
        self.markers = {engine_id: done.marker for engine_id, done in histories.items()}
        self.histories = histories
        self.cleared: list[list[str]] = []
        self.connected = True
        self.broken = False

    def hf_backfill_markers(self) -> list[HfBackfillMarker]:
        return [self.markers[engine_id] for engine_id in sorted(self.markers)]

    def backfilled_history(self, marker: HfBackfillMarker) -> Any:
        return self.histories[marker.engine_id].history

    def clear_hf_backfill(self, engine_ids: list[str]) -> bool:
        if self.broken:
            raise RuntimeError("relation simulator_hf_backfill does not exist")
        if not self.connected:
            return False
        self.cleared.append(list(engine_ids))
        for engine_id in engine_ids:
            self.markers.pop(engine_id)
        return True


def test_resume_requeues_marked_engines_skips_its_own_and_drops_markers_without_rows(
    fleet_db: types.ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    owed, own = backfilled("JT9D-001", fleet_db), backfilled("GE90-201", fleet_db, first_id=5000)
    gone = Backfilled(HfBackfillMarker("CFM56-104", 40, EVENT_TS), fleet_db.BackfilledHistory([], {}), [])
    removed = backfilled("CFM56-999", fleet_db, first_id=9000)  # no longer in SIM_ENGINE_IDS
    store = MarkerStore({"JT9D-001": owed, "GE90-201": own, "CFM56-104": gone, "CFM56-999": removed})
    publisher = Recorder()
    publisher.submit_backfill(own.requests)  # backfilled by this very start, before the resume
    with caplog.at_level(logging.INFO, logger="hf_backfill"):
        queued = resume_backfill(store, factory_under(CONFIG), publisher, threading.Event())
    assert queued == len(owed.requests) + len(removed.requests)
    assert publisher.backfill == own.requests + removed.requests + owed.requests, "own frames queued once"
    assert store.cleared == [["CFM56-104"]] and set(store.markers) == {"CFM56-999", "GE90-201", "JT9D-001"}
    assert set(publisher.ledger.owed()) == {"CFM56-999", "GE90-201", "JT9D-001"}, "kept until acknowledged"
    assert (
        "Cannot rebuild the 40 HF backfill frames CFM56-104 owes (no telemetry rows up to "
        "2026-11-02T09:00:00+00:00 instead of 40): its marker is dropped" in caplog.text
    )
    frames = len(owed.requests)
    assert f"Resumed the HF backfill of JT9D-001: {frames} frames queued again (frame_seq 1..{frames}" in (
        caplog.text
    )


def test_a_shutdown_request_leaves_the_resume_to_the_next_start(
    fleet_db: types.ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    store = MarkerStore({"JT9D-001": backfilled("JT9D-001", fleet_db)})
    publisher, stop = Recorder(), threading.Event()
    stop.set()
    with caplog.at_level(logging.INFO, logger="hf_backfill"):
        assert resume_backfill(store, factory_under(CONFIG), publisher, stop) == 0
    assert publisher.backfill == [] and set(store.markers) == {"JT9D-001"}
    assert "HF backfill resume interrupted by shutdown request; on the next start: JT9D-001" in caplog.text


def test_paid_markers_are_cleared_when_the_database_allows_it_and_kept_otherwise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MarkerStore({})
    store.markers = {e: HfBackfillMarker(e, 1, EVENT_TS) for e in ("JT9D-001", "GE90-201", "CFM56-104")}
    ledger = BackfillLedger()
    ledger.expect(["JT9D-001", "GE90-201", "CFM56-104"])
    clear_paid(ledger, store)
    assert store.cleared == [], "nothing paid: no statement at all"
    ledger.acknowledged("JT9D-001")
    store.connected = False  # no open connection (the tick that would reconnect comes next)
    clear_paid(ledger, store)
    ledger.acknowledged("GE90-201")
    store.connected = True
    with caplog.at_level(logging.INFO, logger="hf_backfill"):
        clear_paid(ledger, store)
    assert store.cleared == [["JT9D-001", "GE90-201"]] and set(store.markers) == {"CFM56-104"}
    assert "HF backfill acknowledged by Kafka, marker cleared: JT9D-001, GE90-201" in caplog.text
    ledger.acknowledged("CFM56-104")
    store.broken = True
    clear_paid(ledger, store)  # never raises into the tick loop; the next start re-sends those frames
    assert "HF backfill markers of ['CFM56-104'] not cleared" in caplog.text and ledger.take_paid() == []


class RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append((sql, params))


def test_the_writer_records_and_deletes_markers() -> None:
    cursor = RecordingCursor()
    ops = [
        ("mark_hf_backfill", HfBackfillMarker("GE90-201", 4321, EVENT_TS)),
        ("clear_hf_backfill", ("JT9D-001", "GE90-201")),
    ]
    FleetWriter(cursor).apply(ops)  # type: ignore[arg-type]
    assert cursor.statements == [
        (MARK_HF_BACKFILL_SQL, {"engine_id": "GE90-201", "frames": 4321, "last_event_ts": EVENT_TS}),
        (CLEAR_HF_BACKFILL_SQL, (["JT9D-001", "GE90-201"],)),
    ]
    assert "ON CONFLICT (engine_id) DO UPDATE" in MARK_HF_BACKFILL_SQL


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn
        self._sql = ""

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        if self._conn.error is not None:
            raise self._conn.error
        self._conn.statements.append((sql, params))
        self._sql = sql

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._conn.answers.get(self._sql, [])


class FakeConnection:
    """Duck-typed psycopg2 connection: records statements, answers fetchall from ``answers``."""

    def __init__(self, answers: dict[str, list[tuple[Any, ...]]] | None = None) -> None:
        self.answers = answers or {}
        self.statements: list[tuple[str, Any]] = []
        self.closed = 0
        self.commits = 0
        self.error: Exception | None = None

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.commits += exc_type is None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = 1


def database(fleet_db: types.ModuleType, conn: FakeConnection | None) -> Any:
    store = fleet_db.FleetDatabase(fleet_db.DatabaseSettings(), threading.Event())
    store._conn = conn
    return store


def test_the_schema_check_creates_the_marker_table_and_the_reads_parse_it(fleet_db: types.ModuleType) -> None:
    row = (1001, 2, 7, EVENT_TS, 9300.5, 600.25, 680.0, 1.25, 4200.0, 60.5, 0.4321, "HEALTHY")
    conn = FakeConnection(
        {
            fleet_db.HF_BACKFILL_MARKERS_SQL: [("GE90-201", 4321, EVENT_TS)],
            fleet_db.BACKFILLED_ROWS_SQL: [row],
            fleet_db.ENGINE_SETTINGS_SQL: [(1, 42, 8, 3), (2, -(2**63), 180.25, 10)],
        }
    )
    store = database(fleet_db, conn)
    store.ensure_schema()
    assert [sql for sql, _ in conn.statements] == [fleet_db.LIFECYCLE_PARAMS_DDL, fleet_db.HF_BACKFILL_DDL]
    assert "CREATE TABLE IF NOT EXISTS simulator_hf_backfill" in fleet_db.HF_BACKFILL_DDL
    (marker,) = store.hf_backfill_markers()
    assert marker == HfBackfillMarker("GE90-201", 4321, EVENT_TS)
    history = store.backfilled_history(marker)
    assert conn.statements[-2:] == [
        (fleet_db.BACKFILLED_ROWS_SQL, ("GE90-201", EVENT_TS)),
        (fleet_db.ENGINE_SETTINGS_SQL, ("GE90-201",)),
    ]
    assert "WHERE engine_id = %s AND event_ts <= %s" in fleet_db.BACKFILLED_ROWS_SQL
    assert "ORDER BY event_ts" in fleet_db.BACKFILLED_ROWS_SQL
    assert history.rows == [fleet_db.StoredPulse(*row)] and history.rows[0].wear_factor == 0.4321
    assert history.settings == {
        1: GenerationSettings(42, 8.0, 3),
        2: GenerationSettings(-(2**63), 180.25, 10),
    }


def test_clearing_markers_never_reconnects(
    fleet_db: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_connect(**_: Any) -> None:
        raise AssertionError("the clear must not open a connection")

    monkeypatch.setattr(fleet_db.psycopg2, "connect", no_connect, raising=False)
    assert database(fleet_db, None).clear_hf_backfill(["JT9D-001"]) is False
    closed = FakeConnection()
    closed.closed = 1
    assert database(fleet_db, closed).clear_hf_backfill(["JT9D-001"]) is False and closed.statements == []
    conn = FakeConnection()
    store = database(fleet_db, conn)
    assert store.clear_hf_backfill(("JT9D-001", "GE90-201")) is True
    assert conn.statements == [(CLEAR_HF_BACKFILL_SQL, (["JT9D-001", "GE90-201"],))] and conn.commits == 1
    conn.error = fleet_db.psycopg2.OperationalError("server closed the connection unexpectedly")
    assert store.clear_hf_backfill(["JT9D-001"]) is False, "a transient error: tried again after a tick"
    assert conn.closed and store.clear_hf_backfill(["JT9D-001"]) is False
