"""Resumable first-boot HF backfill: the frames a backfilled history still owes, and how they get paid.

A backfilled history's frames are queued in memory, and on a fresh stack they wait minutes for Kafka, the
topic and the registry, which start after PostgreSQL. So that a stop in that window (make down, a crash,
Kafka never coming up) loses nothing for good, every engine's backfill transaction also writes a marker,
``simulator_hf_backfill`` (engine_id, frames, last_event_ts): the engine's telemetry rows up to
last_event_ts, its first ``frames`` rows (frame_seq 1..frames), owe one frame each.

    paid       the publisher counts the delivery reports of every engine's backfill frames (BackfillLedger);
               once Kafka has acknowledged all of them, the main thread deletes the engine's marker after
               its next tick or at shutdown. A frame that failed, was discarded with a replaced client or
               was still queued or unacknowledged at shutdown leaves the marker in place.
    resumed    on start, every marked engine's frame requests are rebuilt from PostgreSQL and queued as
               backfill again, behind the same BACKFILL_SHARE rule as a first boot (hf_queue), so live
               frames keep flowing. telemetry_id and event time come from the stored rows and frame_seq is
               the row's ordinal. theta and the failure mode are not stored: each lifecycle is re-simulated
               from pulse 0 under the settings recorded for it (simulator_lifecycle_params), which is how
               the backfill generated it, and every stored reading, wear value, cycle and status must come
               out exactly the same. A history that does not is never sent with invented physics: it is
               logged and its marker dropped.

The rebuilt requests equal the original ones, and FrameBuilder draws each frame's noise from a stream keyed
by SIM_SEED and the telemetry_id, so under an unchanged SIM_SEED a rebuilt frame is byte-identical to the
original. After a SIM_SEED change the pulses keep their physics (state, readings, failure mode), while
sensor biases and noise come from the new seed, as for every live frame since the change. A frame that
reaches Kafka twice collapses in raw_engine_hf_features (ReplacingMergeTree on engine_id, telemetry_id).

Only an enabled HF path (SIM_HF_ENABLED) writes markers; only a running publisher reads and clears them. A
disabled path never touches them and needs no Kafka: markers of an earlier run wait for the next enabled
start, and nothing else reads them. A marker describes rows in PostgreSQL, not SIM_ENGINE_IDS: an engine
removed from the fleet still gets the frames its history owes, and a marker whose rows are gone is dropped.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from hf_frames import FrameRequest

if TYPE_CHECKING:
    from engine_generator import EngineSimulator, Pulse
    from fleet_bootstrap import EngineFactory
    from fleet_db import BackfilledHistory, StoredPulse
    from fleet_runtime import HfBackfillMarker

logger = logging.getLogger("hf_backfill")

EXACT_FIELDS = ("lifecycle_id", "cycle_number", "status")
READING_FIELDS = (
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "wear_factor",
)
READING_TOLERANCE = 1e-9  # relative: DOUBLE PRECISION round-trips exactly; this only absorbs formatting


class BackfillStore(Protocol):
    """The part of fleet_db.FleetDatabase the resumable backfill needs."""

    def hf_backfill_markers(self) -> list[HfBackfillMarker]:
        """Every engine whose backfilled history still owes HF frames."""

    def backfilled_history(self, marker: HfBackfillMarker) -> BackfilledHistory:
        """The rows the marker covers, in event order, and the engine's recorded lifecycle settings."""

    def clear_hf_backfill(self, engine_ids: Sequence[str]) -> bool:
        """Delete markers on the open connection; False when that is not possible right now."""


class BackfillPublisher(Protocol):
    """The part of hf_producer.FramePublisher the resumable backfill needs."""

    ledger: BackfillLedger

    def submit_backfill(self, requests: Sequence[FrameRequest]) -> None:
        """Queue backfill frame requests (each expects a delivery report in the ledger)."""


class BackfillMismatchError(ValueError):
    """The stored rows of a marked history cannot be rebuilt into the frames they owe."""


class BackfillLedger:
    """Per engine, the backfill frames Kafka has still to acknowledge (thread-safe: the publisher's worker
    reports deliveries, the main thread collects the engines whose frames are all acknowledged)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owed: dict[str, int] = {}
        self._paid: list[str] = []
        self._queued: set[str] = set()

    def expect(self, engine_ids: Iterable[str]) -> None:
        """One owed frame per entry: the engine id of every backfill frame about to be queued."""
        with self._lock:
            for engine_id, frames in Counter(engine_ids).items():
                self._owed[engine_id] = self._owed.get(engine_id, 0) + frames
                self._queued.add(engine_id)

    def acknowledged(self, engine_id: str) -> None:
        """Kafka acknowledged one backfill frame of ``engine_id``."""
        with self._lock:
            left = self._owed.get(engine_id)
            if left is None:
                return
            if left > 1:
                self._owed[engine_id] = left - 1
                return
            del self._owed[engine_id]
            self._paid.append(engine_id)

    def take_paid(self) -> list[str]:
        """Engines whose backfill frames Kafka has all acknowledged since the last call."""
        with self._lock:
            paid, self._paid = self._paid, []
            return paid

    def return_paid(self, engine_ids: Iterable[str]) -> None:
        """Hand back engines whose marker could not be cleared yet (the next take_paid returns them)."""
        with self._lock:
            self._paid.extend(engine_ids)

    def queued(self, engine_id: str) -> bool:
        """True once this process has queued backfill frames of ``engine_id``."""
        with self._lock:
            return engine_id in self._queued

    def owed(self) -> dict[str, int]:
        """Backfill frames still waiting for an acknowledgement, per engine."""
        with self._lock:
            return dict(sorted(self._owed.items()))


def _same(name: str, rebuilt: object, stored: object) -> bool:
    if name in EXACT_FIELDS:
        return rebuilt == stored
    return math.isclose(float(rebuilt), float(stored), rel_tol=READING_TOLERANCE, abs_tol=READING_TOLERANCE)


def _replay(engine: EngineSimulator, row: StoredPulse) -> Pulse:
    """The engine's next pulse at the row's event time, after checking it is the stored one."""
    try:
        pulse = engine.next_pulse(row.event_ts)
    except RuntimeError as exc:  # stored rows go on after the lifecycle's FAILURE pulse
        raise BackfillMismatchError(f"telemetry_id {row.telemetry_id}: {exc}") from exc
    fields = EXACT_FIELDS + READING_FIELDS
    differs = [name for name in fields if not _same(name, getattr(pulse, name), getattr(row, name))]
    if differs:
        raise BackfillMismatchError(
            f"telemetry_id {row.telemetry_id} (lifecycle {row.lifecycle_id}) is not what its recorded "
            f"settings generate: {', '.join(differs)} differ"
        )
    engine.advance(pulse)
    return pulse


def rebuild_requests(
    marker: HfBackfillMarker, history: BackfilledHistory, factory: EngineFactory
) -> list[FrameRequest]:
    """The frame requests the marked history's backfill queued, rebuilt from its stored rows (module doc).

    Raises BackfillMismatchError when the rows are not the ``marker.frames`` pulses their recorded settings
    generate.
    """
    rows = history.rows
    if len(rows) != marker.frames:
        found = f"{len(rows)} telemetry rows" if rows else "no telemetry rows"
        raise BackfillMismatchError(
            f"{found} up to {marker.last_event_ts.isoformat()} instead of {marker.frames}"
        )
    requests: list[FrameRequest] = []
    engine: EngineSimulator | None = None
    for frame_seq, row in enumerate(rows, start=1):
        if engine is None or row.lifecycle_id != engine.lifecycle_id:
            settings = history.settings.get(row.lifecycle_id)
            if settings is None:
                raise BackfillMismatchError(
                    f"lifecycle {row.lifecycle_id} has no recorded generation settings"
                )
            # Every backfilled lifecycle started at pulse 0, which also seeds its noise stream.
            engine = factory(marker.engine_id, lifecycle_id=row.lifecycle_id, settings=settings)
        pulse = _replay(engine, row)
        requests.append(
            FrameRequest.from_pulse(pulse, engine.profile.engine_model, row.telemetry_id, frame_seq)
        )
    return requests


def resume_backfill(
    store: BackfillStore,
    factory: EngineFactory,
    publisher: BackfillPublisher,
    stop_event: threading.Event,
) -> int:
    """Queue again the frames every marked engine still owes; returns how many (start-up, main thread).

    Engines whose backfill this process has queued already (backfilled by this very start) are skipped; a
    shutdown request stops between engines, leaving the remaining markers to the next start.
    """
    markers = [m for m in store.hf_backfill_markers() if not publisher.ledger.queued(m.engine_id)]
    queued = 0
    for done, marker in enumerate(markers):
        if stop_event.is_set():
            names = ", ".join(m.engine_id for m in markers[done:])
            logger.info("HF backfill resume interrupted by shutdown request; on the next start: %s", names)
            break
        try:
            requests = rebuild_requests(marker, store.backfilled_history(marker), factory)
        except BackfillMismatchError as exc:
            logger.error(
                "Cannot rebuild the %d HF backfill frames %s owes (%s): its marker is dropped and those "
                "pulses keep no HF features",
                marker.frames,
                marker.engine_id,
                exc,
            )
            store.clear_hf_backfill([marker.engine_id])
            continue
        publisher.submit_backfill(requests)
        queued += len(requests)
        logger.info(
            "Resumed the HF backfill of %s: %d frames queued again (frame_seq 1..%d, history up to %s)",
            marker.engine_id,
            len(requests),
            len(requests),
            marker.last_event_ts.isoformat(),
        )
    return queued


def clear_paid(ledger: BackfillLedger, store: BackfillStore) -> None:
    """Delete the markers of engines whose backfill frames Kafka has all acknowledged (main thread).

    Never raises: an engine whose marker cannot be cleared now is handed back for the next call, and one
    whose marker survives a database error only has its frames sent again by the next start.
    """
    paid = ledger.take_paid()
    if not paid:
        return
    try:
        cleared = store.clear_hf_backfill(paid)
    except Exception:
        logger.exception("HF backfill markers of %s not cleared; the next start sends them again", paid)
        return
    if not cleared:
        ledger.return_paid(paid)
        return
    logger.info("HF backfill acknowledged by Kafka, marker cleared: %s", ", ".join(paid))
