"""Unit tests of the HF fixture (synthetic_hf.py) the dbt integration harness seeds.

Its rows must have exactly the columns the streaming job writes into raw_engine_hf_features,
and it must hold every case the integration checks rely on, delivered the way the single
Spark writer delivers copies (a later copy lands in a later insert, with a later ingested_at),
because the raw table keeps the copy with the newest ingested_at when it merges.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hf_reference import rolling_spectral_features, staged_frames
from synthetic_cdc import History, build_history
from synthetic_hf import HF_COLUMNS, KAFKA_PARTITIONS

REPO = Path(__file__).resolve().parents[2]
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MICROSECOND = timedelta(microseconds=1)
CASES = (
    "flat_line",
    "stopped_spool",
    "missing_egt_b",
    "missing_oil_qty_a",
    "dead_oil_dp_a",
    "cold_oil",
    "infinite_crest",
    "negative_bpfo",
    "same_batch_copies",
    "replayed",
    "relabelled",
    "late_visible",
    "lagging",
)


@pytest.fixture(scope="module")
def history() -> History:
    return build_history(42)


def _pulses(history: History) -> dict[int, dict[str, Any]]:
    """First image of every telemetry pulse, by telemetry id."""
    return {
        event.image["telemetry_id"]: event.image
        for event in reversed(history.events)
        if event.table == "engine_telemetry" and event.op == "c"
    }


def _table_columns(ddl: str, table: str) -> list[str]:
    body = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)", ddl, re.DOTALL)
    assert body, f"{table} not found in init-db.sql"
    return [line.split()[0] for line in body.group(1).splitlines() if line.strip()]


def test_rows_have_the_columns_of_the_clickhouse_table() -> None:
    ddl = (REPO / "infrastructure" / "clickhouse" / "init-db.sql").read_text(encoding="utf-8")
    assert [*HF_COLUMNS, "ingested_at"] == _table_columns(ddl, "raw_engine_hf_features")


def test_rows_have_the_columns_the_streaming_job_writes() -> None:
    pytest.importorskip("pandas")
    hf_features = pytest.importorskip("hf_features")
    assert HF_COLUMNS == hf_features.FEATURE_COLUMN_NAMES


def test_the_fixture_is_deterministic(history: History) -> None:
    again = build_history(42)
    # repr: NaN values never compare equal
    assert repr([(copy.row, copy.phase) for copy in again.frames]) == repr(
        [(copy.row, copy.phase) for copy in history.frames]
    )
    assert again.manifest["hf"] == history.manifest["hf"]


def test_every_frame_is_keyed_and_timed_like_its_pulse(history: History) -> None:
    pulses, relabelled = _pulses(history), history.manifest["hf"]["relabelled"]
    for copy in history.frames:
        row, pulse = copy.row, pulses[copy.row["telemetry_id"]]
        assert (row["engine_id"], row["cycle_number"], row["frame_start_ts"]) == (
            pulse["engine_id"],
            pulse["cycle_number"],
            pulse["event_ts"],
        )
        if row["lifecycle_id"] != pulse["lifecycle_id"]:
            assert (row["telemetry_id"], row["lifecycle_id"], copy.phase) == (
                relabelled["telemetry_id"],
                relabelled["from_lifecycle"],
                "base",
            )


def test_every_case_of_the_integration_checks_is_present(history: History) -> None:
    hf = history.manifest["hf"]
    assert [name for name in CASES if not hf["cases"].get(name)] == []
    framed = {copy.row["telemetry_id"] for copy in history.frames}
    assert framed.isdisjoint(hf["pulses_without_frame"])
    assert hf["frames"] == len(framed)
    silent = tuple(hf["silent_lifecycle"])
    assert not [
        copy for copy in history.frames if (copy.row["engine_id"], copy.row["lifecycle_id"]) == silent
    ]


def test_every_failure_pulse_has_a_frame(history: History) -> None:
    # the calibration guard judges the bearing flag at the FAILURE pulses
    failures = {tid for tid, pulse in _pulses(history).items() if pulse["status"] == "FAILURE"}
    assert failures and failures <= {copy.row["telemetry_id"] for copy in history.frames}


def test_late_visible_frames_have_no_copy_in_another_insert(history: History) -> None:
    deliveries: defaultdict[int, set[tuple[str, bool]]] = defaultdict(set)
    for copy in history.frames:
        deliveries[copy.row["telemetry_id"]].add((copy.phase, copy.late_visible))
    late = [tid for tid, kinds in deliveries.items() if ("increment", True) in kinds]
    assert late and all(deliveries[tid] == {("increment", True)} for tid in late)


def test_copies_are_delivered_in_order_with_increasing_kafka_offsets(history: History) -> None:
    ranks = [
        (copy.phase == "increment", copy.phase == "increment" and not copy.late_visible)
        for copy in history.frames
    ]
    assert ranks == sorted(ranks)  # base, then the late-visible insert, then the increment
    offsets: defaultdict[int, int] = defaultdict(lambda: -1)
    for copy in history.frames:
        partition = copy.row["kafka_partition"]
        assert partition == zlib.crc32(copy.row["engine_id"].encode()) % KAFKA_PARTITIONS
        assert copy.row["kafka_offset"] > offsets[partition]
        offsets[partition] = copy.row["kafka_offset"]


def test_bearing_failures_stand_out_of_healthy_pulses(history: History) -> None:
    # the calibration guard judges the mart (completed lifecycles): its default BPFO threshold
    # flags every bearing failure and no healthy pulse there; the spall of GE90-202's running
    # lifecycle is the board's case
    copies = [
        {
            **copy.row,
            "frame_micros": (copy.row["frame_start_ts"] - EPOCH) // MICROSECOND,
            "ingested_at": number,
        }
        for number, copy in enumerate(history.frames)  # a later delivery is a later ingestion
    ]
    reference = rolling_spectral_features(staged_frames(copies).values())
    pulses = _pulses(history)
    modes = {
        (event.image["engine_id"], event.image["lifecycle_id"]): event.image["failure_mode"]
        for event in history.events
        if event.table == "maintenance_log"
    }
    at_bearing_failures = [
        row["fan_amp_bpfo_mean_5m"]
        for (engine_id, lifecycle_id, tid), row in reference.items()
        if pulses[tid]["status"] == "FAILURE" and modes[(engine_id, lifecycle_id)] == "BEARING_DEGRADATION"
    ]
    at_healthy = [
        row["fan_amp_bpfo_mean_5m"]
        for (engine_id, lifecycle_id, tid), row in reference.items()
        if (engine_id, lifecycle_id) in modes
        and pulses[tid]["status"] == "HEALTHY"
        and row["fan_amp_bpfo_mean_5m"] is not None
    ]
    assert len(at_bearing_failures) == 3 and min(at_bearing_failures) > 0.5
    assert at_healthy and max(at_healthy) < 0.5
