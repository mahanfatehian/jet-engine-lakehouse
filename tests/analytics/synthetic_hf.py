"""Synthetic high-frequency (HF) frame features of the analytics fixture: raw_engine_hf_features rows.

The Spark job reduces every HF sensor frame to one raw_engine_hf_features row (CONTRACT_V2.md
sections 1, 6 and 7). Here every pulse of synthetic_fleet gets such a row, keyed like its pulse
(same telemetry_id, frame_start_ts == event_ts), whose features follow the pulse's wear w and its
lifecycle's failure-mode signature: the fan bearing outer-race tone (BPFO) grows ~w^3.5 and the
impulsiveness (kurtosis) ~w^4, strongest in BEARING_DEGRADATION; oil temperature, quantity and
filter dP and the EGT A/B split follow the oil and hot-section signatures. Values are rounded to
Float32 like the column stores them, so a reference computed from them is exact.

``build_hf_frames`` also injects the cases the dbt models must handle:

* missing frames: single dropped frames, a 40 s HF outage and a lifecycle without any frame (HF
  never delivered after JT9D-001's repair): the spectral features of those pulses are NULL;
* unusable values: a flat-line fan channel (NaN crest factor, kurtosis, band share and centroid,
  zero amplitudes), a stopped fan spool (NaN shaft and BPFO tones), a missing EGT_B probe (NaN
  egt_ab_split), an infinite crest factor and a negative BPFO amplitude; a negative oil
  temperature (cold soak) is a valid reading;
* absent process keys: OIL_QTY_A missing in single frames, OIL_DP_A missing for 330 s (5-minute
  means without any value); ClickHouse would read 0 for a missing key;
* at-least-once delivery: two copies in one insert (the Kafka position decides), replays with the
  increment (the newer copy wins) and a copy that corrects the lifecycle of a frame (the frame
  moves to its lifecycle). Every later copy is inserted later, as the single Spark writer does,
  with a later ingested_at: raw_engine_hf_features (ReplacingMergeTree(ingested_at)) and
  stg_engine_hf_features both keep the copy with the newest ingested_at;
* HF lag: frames of the last 20 s before the cut-off and a late-visible block (ingested_at
  behind the dbt watermark, frames without another copy) land with the increment;
* a bearing spall in GE90-202's running lifecycle, which the health board must flag.
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple

from synthetic_fleet import Lifecycle, Pulse

# raw_engine_hf_features in ClickHouse column order (hf_features.FEATURE_COLUMN_NAMES of the
# streaming job); the server fills ingested_at.
CHANNEL_FEATURES = (
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "amp_1x_n1",
    "amp_1x_n2",
    "amp_bpfo",
    "band_energy_hf",
    "spectral_centroid_hz",
)
HF_COLUMNS: tuple[str, ...] = (
    "engine_id",
    "engine_model",
    "lifecycle_id",
    "cycle_number",
    "telemetry_id",
    "frame_seq",
    "frame_start_ts",
    "frame_duration_ms",
    "n1_rpm",
    "n2_rpm",
    "bpfo_order",
    "vibration_sample_rate_hz",
    "vibration_channels",
    "process_channels",
    *(f"{channel}_{feature}" for channel in ("fan", "core") for feature in CHANNEL_FEATURES),
    "egt_ab_split",
    "process_means",
    "frame_bytes",
    "kafka_partition",
    "kafka_offset",
)
KAFKA_PARTITIONS = 6  # HF_TOPIC_PARTITIONS; the message key is the engine id
HF_LAG = timedelta(seconds=20)
SPALL_SECONDS = 200
NAN = float("nan")

N1_NOMINAL_RPM = {"JT9D-7R4": 3650.0, "CFM56-7B": 5175.0, "GE90-115B": 2355.0}
BPFO_ORDER = {"JT9D-7R4": 4.62, "CFM56-7B": 4.81, "GE90-115B": 5.07}
OIL_TANK_L = {"JT9D-7R4": 26.0, "CFM56-7B": 20.0, "GE90-115B": 34.0}
# process parameter -> (offset of sensor B, noise sigma of both sensors)
SENSOR_PAIRS = {
    "EGT": (0.0, 0.3),  # sensor B also reads the hot-section split
    "N1": (1.5, 0.5),
    "N2": (-2.0, 0.5),
    "OIL_DP": (0.1, 0.05),
    "OIL_P": (0.2, 0.05),
    "OIL_QTY": (0.05, 0.02),
    "OIL_T": (0.3, 0.1),
}


class Signature(NamedTuple):
    """How a failure mode shows in the HF frames (CONTRACT_V2.md section 3)."""

    bearing: float  # fan bearing defect: BPFO tone ~w^3.5 and impacts ~w^4
    hot_section: float  # core unbalance (1xN2) and the EGT A/B split ~w^2
    oil_leak: float  # share of the oil tank lost at w = 1
    oil_heat: float  # oil temperature rise factor
    oil_debris: float  # oil filter dP rise at w = 1 (psid)


SIGNATURES = {
    "BEARING_DEGRADATION": Signature(1.0, 0.2, 0.0, 1.5, 20.0),
    "OIL_SYSTEM_LEAK": Signature(0.35, 0.2, 0.45, 1.5, 5.0),
    "HPT_BLADE_EROSION": Signature(0.05, 1.0, 0.0, 1.0, 1.5),
    "COMPRESSOR_FOULING": Signature(0.05, 0.4, 0.0, 1.0, 1.5),
}


@dataclass
class FrameCopy:
    """One delivered copy of a frame: its raw_engine_hf_features row and its delivery phase."""

    row: dict[str, Any]
    phase: str
    late_visible: bool = False


@dataclass(frozen=True)
class Scenario:
    """Where the injected cases sit: (engine_id, lifecycle_id[, pulse indices or pulse index]).

    ``picks`` collects the telemetry ids each case was applied to (for the manifest).
    """

    cutoff: datetime
    silent_lifecycle: tuple[str, int] = ("JT9D-001", 4)
    outage: tuple[str, int, range] = ("GE90-201", 4, range(150, 190))
    late_visible_block: tuple[str, int, range] = ("GE90-201", 4, range(22, 52))
    dead_oil_dp: tuple[str, int, range] = ("CFM56-104", 4, range(20, 350))
    cold_oil: tuple[str, int, range] = ("GE90-202", 1, range(0, 5))
    spall_lifecycle: tuple[str, int] = ("GE90-202", 4)
    relabelled: tuple[str, int, int] = ("GE90-201", 2, 100)  # pulse index; first copy names lifecycle 1
    infinite_crest: tuple[str, int, int] = ("GE90-202", 2, 77)
    negative_bpfo: tuple[str, int, int] = ("CFM56-104", 1, 123)
    picks: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def dropped(self, lifecycle: Lifecycle, pulse: Pulse) -> bool:
        """No frame for this pulse: the silent lifecycle, the outage, single drops (never a FAILURE)."""
        key = (lifecycle.plan.engine_id, lifecycle.lifecycle_id)
        if key == self.silent_lifecycle or _in_range(self.outage, lifecycle, pulse):
            return True
        return pulse.index % 31 == 17 and pulse.status != "FAILURE"


def _in_range(spec: tuple[str, int, range], lifecycle: Lifecycle, pulse: Pulse) -> bool:
    engine_id, lifecycle_id, indices = spec
    return (lifecycle.plan.engine_id, lifecycle.lifecycle_id) == (engine_id, lifecycle_id) and (
        pulse.index in indices
    )


def _is_pulse(spec: tuple[str, int, int], lifecycle: Lifecycle, pulse: Pulse) -> bool:
    return (lifecycle.plan.engine_id, lifecycle.lifecycle_id, pulse.index) == spec


def f32(value: float) -> float:
    """``value`` as a Float32 column stores it (round to nearest float32)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def signature_of(lifecycle: Lifecycle) -> Signature:
    """A running lifecycle wears like the engine's last planned failure mode (synthetic_fleet)."""
    return SIGNATURES[lifecycle.failure_mode or lifecycle.plan.failure_modes[-1]]


def vibration_features(
    pulse: Pulse, signature: Signature, spall: bool, rng: random.Random
) -> dict[str, float]:
    """The 18 per-channel features of the fan (fan_frame_vel) and core (core_trf_vel) channels."""
    w = pulse.wear
    defect = signature.bearing * w**3.5 + (0.3 if spall else 0.0)
    impacts = signature.bearing * w**4 + (0.5 if spall else 0.0)
    fan_rms = max(pulse.readings["vibration_mm_s"], 0.05)
    kurtosis = 1.95 + 4.5 * impacts + rng.gauss(0.0, 0.02)
    crest = 1.45 + 0.35 * (kurtosis - 1.95) + rng.gauss(0.0, 0.01)
    fan_bpfo = 0.03 + 4.0 * defect + abs(rng.gauss(0.0, 0.005))
    core_rms = 0.6 * fan_rms
    channels = {  # values in CHANNEL_FEATURES order
        "fan": (
            fan_rms,
            fan_rms * crest,
            crest,
            kurtosis,
            0.55 * fan_rms + abs(rng.gauss(0.0, 0.005)),
            0.08 + abs(rng.gauss(0.0, 0.003)),
            fan_bpfo,
            min(0.04 + 0.5 * defect, 0.95),
            150.0 + 400.0 * defect,
        ),
        "core": (
            core_rms,
            1.5 * core_rms,
            1.5,
            2.0 + rng.gauss(0.0, 0.02),
            0.05,
            0.3 * core_rms * (1.0 + 2.0 * signature.hot_section * w * w),
            0.3 * fan_bpfo,
            0.03,
            300.0,
        ),
    }
    return {
        f"{channel}_{name}": f32(value)
        for channel, values in channels.items()
        for name, value in zip(CHANNEL_FEATURES, values, strict=True)
    }


def process_means(
    lifecycle: Lifecycle, pulse: Pulse, n1_rpm: float, signature: Signature, rng: random.Random
) -> dict[str, float]:
    """Frame means of the redundant A/B process channels, in name order like the Spark job."""
    w, readings = pulse.wear, pulse.readings
    sensor_a = {
        "EGT": readings["egt_c"],
        "N1": n1_rpm,
        "N2": readings["rpm"],
        "OIL_DP": 6.0 + signature.oil_debris * w**3,
        "OIL_P": readings["oil_pressure_psi"],
        "OIL_QTY": OIL_TANK_L[lifecycle.plan.engine_model] * (0.85 - 0.04 * w - signature.oil_leak * w * w),
        "OIL_T": 92.0 + 22.0 * signature.oil_heat * w * w,
    }
    egt_split = 0.4 + 12.0 * signature.hot_section * w * w
    means: dict[str, float] = {}
    for name, value in sensor_a.items():
        offset, noise = SENSOR_PAIRS[name]
        means[f"{name}_A"] = f32(value + rng.gauss(0.0, noise))
        means[f"{name}_B"] = f32(value + (egt_split if name == "EGT" else offset) + rng.gauss(0.0, noise))
    return dict(sorted(means.items()))


def frame_row(
    lifecycle: Lifecycle, pulse: Pulse, frame_seq: int, scenario: Scenario, rng: random.Random
) -> dict[str, Any]:
    """The raw_engine_hf_features row of one pulse's frame, scenario injections applied."""
    model = lifecycle.plan.engine_model
    signature = signature_of(lifecycle)
    spall = (lifecycle.plan.engine_id, lifecycle.lifecycle_id) == scenario.spall_lifecycle and (
        pulse.event_ts > lifecycle.pulses[-1].event_ts - timedelta(seconds=SPALL_SECONDS)
    )
    n1_rpm = N1_NOMINAL_RPM[model] * (1.0 + 0.006 * pulse.wear) + rng.gauss(0.0, 2.0)
    means = process_means(lifecycle, pulse, n1_rpm, signature, rng)
    row: dict[str, Any] = {
        "engine_id": lifecycle.plan.engine_id,
        "engine_model": model,
        "lifecycle_id": lifecycle.lifecycle_id,
        "cycle_number": pulse.cycle_number,
        "telemetry_id": pulse.telemetry_id,
        "frame_seq": frame_seq,
        "frame_start_ts": pulse.event_ts,
        "frame_duration_ms": 1000,
        "n1_rpm": f32(n1_rpm),
        "n2_rpm": f32(pulse.readings["rpm"]),
        "bpfo_order": f32(BPFO_ORDER[model]),
        "vibration_sample_rate_hz": 2048,
        "vibration_channels": 2,
        **vibration_features(pulse, signature, spall, rng),
        "process_means": means,
    }
    _inject_faults(row, lifecycle, pulse, scenario)
    means = row["process_means"]
    split = abs(means["EGT_A"] - means["EGT_B"]) if {"EGT_A", "EGT_B"} <= means.keys() else NAN
    row.update(
        egt_ab_split=f32(split),
        process_channels=len(means),
        frame_bytes=2 * 2048 * 4 + len(means) * 8 * 4 + 180,
        kafka_partition=0,
        kafka_offset=0,
    )
    return row


def _inject_faults(row: dict[str, Any], lifecycle: Lifecycle, pulse: Pulse, scenario: Scenario) -> None:
    """Unusable values and absent process keys the streaming job emits (see the module doc)."""
    tid, means = pulse.telemetry_id, row["process_means"]
    faults: list[tuple[str, bool, Callable[[], object]]] = [
        ("flat_line", tid % 97 == 13, lambda: _flat_line(row, "fan")),
        ("stopped_spool", tid % 89 == 5, lambda: _stopped_fan_spool(row)),
        ("missing_egt_b", tid % 83 == 11, lambda: means.pop("EGT_B")),
        ("missing_oil_qty_a", tid % 79 == 3, lambda: means.pop("OIL_QTY_A")),
        ("dead_oil_dp_a", _in_range(scenario.dead_oil_dp, lifecycle, pulse), lambda: means.pop("OIL_DP_A")),
        ("cold_oil", _in_range(scenario.cold_oil, lifecycle, pulse), lambda: _cold_soak(means, pulse.index)),
        (
            "infinite_crest",
            _is_pulse(scenario.infinite_crest, lifecycle, pulse),
            lambda: row.update(fan_crest_factor=math.inf),
        ),
        (
            "negative_bpfo",
            _is_pulse(scenario.negative_bpfo, lifecycle, pulse),
            lambda: row.update(fan_amp_bpfo=f32(-0.4)),
        ),
    ]
    for name, applies, inject in faults:
        if applies:
            inject()
            scenario.picks[name].append(tid)


def _flat_line(row: dict[str, Any], channel: str) -> None:
    # a constant waveform: no variance, so no crest factor, kurtosis, band share or centroid
    for feature in ("rms", "peak", "amp_1x_n1", "amp_1x_n2", "amp_bpfo"):
        row[f"{channel}_{feature}"] = 0.0
    for feature in ("crest_factor", "kurtosis", "band_energy_hf", "spectral_centroid_hz"):
        row[f"{channel}_{feature}"] = NAN


def _stopped_fan_spool(row: dict[str, Any]) -> None:
    # N1 <= 0: no fan shaft tone and no BPFO (a multiple of the N1 frequency) on either channel
    row["n1_rpm"] = 0.0
    for channel in ("fan", "core"):
        row[f"{channel}_amp_1x_n1"] = NAN
        row[f"{channel}_amp_bpfo"] = NAN


def _cold_soak(means: dict[str, float], index: int) -> None:
    # oil below 0 degC after a cold night, a valid reading: -6.5 .. -0.5 degC over pulses 0..4
    means["OIL_T_A"] = f32(-6.5 + 1.5 * index)
    means["OIL_T_B"] = f32(-6.2 + 1.5 * index)


def frame_sequences(lifecycles: list[Lifecycle]) -> dict[int, int]:
    """Per-engine 1-based pulse ordinal of every telemetry id (the frame_seq of its frame)."""
    by_engine: defaultdict[str, list[Pulse]] = defaultdict(list)
    for lifecycle in lifecycles:
        by_engine[lifecycle.plan.engine_id].extend(lifecycle.pulses)
    return {
        pulse.telemetry_id: number
        for pulses in by_engine.values()
        for number, pulse in enumerate(sorted(pulses, key=lambda p: p.event_ts), start=1)
    }


def deliveries(
    row: dict[str, Any], lifecycle: Lifecycle, pulse: Pulse, scenario: Scenario
) -> list[FrameCopy]:
    """The copies through which one frame lands, in delivery order within their phase."""
    tid = pulse.telemetry_id
    late_block = _in_range(scenario.late_visible_block, lifecycle, pulse)
    phase = "base" if pulse.event_ts < scenario.cutoff - HF_LAG and not late_block else "increment"
    if _is_pulse(scenario.relabelled, lifecycle, pulse):
        # the first copy names the previous lifecycle; the increment brings the corrected one
        scenario.picks["relabelled"].append(tid)
        return [
            FrameCopy({**row, "lifecycle_id": row["lifecycle_id"] - 1}, phase),
            FrameCopy(row, "increment"),
        ]
    copies = [FrameCopy(row, phase, late_visible=late_block)]
    if tid % 41 == 0:  # redelivered within the same micro-batch
        copies.append(FrameCopy(dict(row), phase, late_visible=late_block))
        scenario.picks["same_batch_copies"].append(tid)
    if phase == "base" and tid % 53 == 7:  # replayed after a restart, with the increment
        copies.append(FrameCopy(dict(row), "increment"))
        scenario.picks["replayed"].append(tid)
    if late_block:
        scenario.picks["late_visible"].append(tid)
    elif phase == "increment" and pulse.event_ts < scenario.cutoff:
        scenario.picks["lagging"].append(tid)
    return copies


def delivery_order(copies: list[FrameCopy]) -> list[FrameCopy]:
    """Base first; in the increment the late-visible copies first (they are inserted first)."""
    rank = {("base", False): 0, ("increment", True): 1, ("increment", False): 2}
    return sorted(
        copies, key=lambda copy: (rank[(copy.phase, copy.late_visible)], copy.row["frame_start_ts"])
    )


def assign_kafka_positions(copies: list[FrameCopy]) -> None:
    """Partition by engine id (the message key), offsets in delivery order per partition."""
    offsets: defaultdict[int, int] = defaultdict(int)
    for copy in copies:
        partition = zlib.crc32(copy.row["engine_id"].encode()) % KAFKA_PARTITIONS
        copy.row = {**copy.row, "kafka_partition": partition, "kafka_offset": offsets[partition]}
        offsets[partition] += 1


def build_hf_frames(lifecycles: list[Lifecycle], cutoff: datetime, seed: int) -> tuple[list[FrameCopy], dict]:
    """Every delivered copy, in delivery order, and the HF part of the scenario manifest."""
    rng = random.Random(seed * 7919 + 17)  # independent of the CDC simulation's generator
    scenario = Scenario(cutoff)
    sequences = frame_sequences(lifecycles)
    copies: list[FrameCopy] = []
    without_frame: list[int] = []
    for lifecycle in lifecycles:
        for pulse in lifecycle.pulses:
            if scenario.dropped(lifecycle, pulse):
                without_frame.append(pulse.telemetry_id)
                continue
            row = frame_row(lifecycle, pulse, sequences[pulse.telemetry_id], scenario, rng)
            copies += deliveries(row, lifecycle, pulse, scenario)
    ordered = delivery_order(copies)
    assign_kafka_positions(ordered)
    engine, lifecycle_id, _ = scenario.relabelled
    manifest = {
        "frames": len({copy.row["telemetry_id"] for copy in ordered}),
        "copies": len(ordered),
        "pulses_without_frame": sorted(without_frame),
        "silent_lifecycle": list(scenario.silent_lifecycle),
        "bearing_alarm_engine": scenario.spall_lifecycle[0],
        "dead_oil_dp_lifecycle": list(scenario.dead_oil_dp[:2]),
        "relabelled": {
            "engine_id": engine,
            "telemetry_id": scenario.picks["relabelled"][0],
            "from_lifecycle": lifecycle_id - 1,
            "to_lifecycle": lifecycle_id,
        },
        # telemetry ids per injected case (see the module doc)
        "cases": {name: sorted(ids) for name, ids in scenario.picks.items()},
    }
    return ordered, manifest
