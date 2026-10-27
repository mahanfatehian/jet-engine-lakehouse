"""Independent pure-Python reference of the spectral feature models (CONTRACT_V2.md section 8).

verify_spectral.py compares the dbt models against this, computed from the raw frame copies:

stg_engine_hf_features   one row per (engine_id, telemetry_id): the copy with the newest
                         (ingested_at, kafka_partition, kafka_offset). A value that is NaN,
                         infinite, or negative where the quantity cannot be is missing, and so is
                         every feature of a channel whose kurtosis is undefined (a flat line: a
                         dead or stuck sensor, whose zero amplitudes are no measurement). The oil
                         parameters are the channel-A means of process_means, missing when the
                         frame does not carry the key (ClickHouse would read 0).
int_vibration_features   per frame, the mean of each spectral source over the frames of its
                         (engine_id, lifecycle_id) within [t - 300 s, t] (RANGE over epoch
                         seconds, timestamp peers included), missing values skipped, None when
                         the window holds no value; frames_5m counts the frames in the window.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

CHANNELS = ("fan", "core")
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
# staging column -> (process_means key, non-negative); a temperature in degC may be negative
OIL_PARAMETERS = {
    "oil_temp_c": ("OIL_T_A", False),
    "oil_quantity": ("OIL_QTY_A", True),
    "oil_filter_dp": ("OIL_DP_A", True),
}
VIBRATION_FEATURES = tuple(f"{channel}_{feature}" for channel in CHANNELS for feature in CHANNEL_FEATURES)
MEASUREMENTS = (*VIBRATION_FEATURES, "egt_ab_split", *OIL_PARAMETERS)
# SPECTRAL_FEATURE_COLUMNS of the ML contract, each with its per-frame staging column
SPECTRAL_SOURCES = (
    ("fan_amp_bpfo_mean_5m", "fan_amp_bpfo"),
    ("fan_kurtosis_mean_5m", "fan_kurtosis"),
    ("fan_crest_factor_mean_5m", "fan_crest_factor"),
    ("fan_amp_1x_n1_mean_5m", "fan_amp_1x_n1"),
    ("core_amp_1x_n2_mean_5m", "core_amp_1x_n2"),
    ("fan_band_energy_hf_mean_5m", "fan_band_energy_hf"),
    ("oil_temp_c_mean_5m", "oil_temp_c"),
    ("oil_quantity_mean_5m", "oil_quantity"),
    ("oil_filter_dp_mean_5m", "oil_filter_dp"),
    ("egt_ab_split_mean_5m", "egt_ab_split"),
)
SPECTRAL_FEATURES = tuple(name for name, _ in SPECTRAL_SOURCES)
WINDOW_SECONDS = 300
FrameKey = tuple[str, int]  # (engine_id, telemetry_id)
SpectralKey = tuple[str, int, int]  # (engine_id, lifecycle_id, telemetry_id)


def usable(value: float | None, non_negative: bool) -> float | None:
    """The value when a model can use it, else None (missing)."""
    if value is None or not math.isfinite(value) or (non_negative and value < 0):
        return None
    return float(value)


def staged_measurements(copy: Mapping[str, Any]) -> dict[str, float | None]:
    """The measurement columns of stg_engine_hf_features for one raw copy (see the module doc)."""
    staged: dict[str, float | None] = {}
    for channel in CHANNELS:
        kurtosis = copy[f"{channel}_kurtosis"]
        live = kurtosis is not None and math.isfinite(kurtosis)
        for feature in CHANNEL_FEATURES:
            column = f"{channel}_{feature}"
            staged[column] = usable(copy[column], non_negative=True) if live else None
    staged["egt_ab_split"] = usable(copy["egt_ab_split"], non_negative=True)
    means = copy["process_means"]
    for column, (parameter, non_negative) in OIL_PARAMETERS.items():
        staged[column] = usable(means[parameter], non_negative) if parameter in means else None
    return staged


def newest_copies(copies: Iterable[Mapping[str, Any]]) -> dict[FrameKey, Mapping[str, Any]]:
    """The winning copy of every frame: newest ingested_at, then Kafka partition and offset."""
    newest: dict[FrameKey, Mapping[str, Any]] = {}
    for copy in copies:
        key = (copy["engine_id"], copy["telemetry_id"])
        position = (copy["ingested_at"], copy["kafka_partition"], copy["kafka_offset"])
        current = newest.get(key)
        if current is None or position > (
            current["ingested_at"],
            current["kafka_partition"],
            current["kafka_offset"],
        ):
            newest[key] = copy
    return newest


@dataclass(frozen=True)
class StagedFrame:
    """One frame as stg_engine_hf_features FINAL must hold it."""

    engine_id: str
    lifecycle_id: int
    telemetry_id: int
    frame_micros: int
    measurements: dict[str, float | None]

    @property
    def epoch(self) -> float:
        # the same Float64 the SQL model orders its RANGE frame by
        return self.frame_micros / 1_000_000.0


def staged_frames(copies: Iterable[Mapping[str, Any]]) -> dict[FrameKey, StagedFrame]:
    """The staged frame of every winning copy (``copies`` carry frame_micros, not frame_start_ts)."""
    return {
        key: StagedFrame(
            copy["engine_id"],
            copy["lifecycle_id"],
            copy["telemetry_id"],
            copy["frame_micros"],
            staged_measurements(copy),
        )
        for key, copy in newest_copies(copies).items()
    }


def _mean_of_present(window: list[StagedFrame], source: str) -> float | None:
    values = [value for frame in window if (value := frame.measurements[source]) is not None]
    return math.fsum(values) / len(values) if values else None


def rolling_spectral_features(frames: Iterable[StagedFrame]) -> dict[SpectralKey, dict[str, Any]]:
    """Expected int_vibration_features row (10 means + frames_5m) of every frame."""
    lifecycles: defaultdict[tuple[str, int], list[StagedFrame]] = defaultdict(list)
    for frame in frames:
        lifecycles[(frame.engine_id, frame.lifecycle_id)].append(frame)
    expected: dict[SpectralKey, dict[str, Any]] = {}
    for (engine_id, lifecycle_id), members in lifecycles.items():
        ordered = sorted(members, key=lambda frame: (frame.epoch, frame.telemetry_id))
        epochs = [frame.epoch for frame in ordered]
        for frame in ordered:
            start = bisect.bisect_left(epochs, frame.epoch - WINDOW_SECONDS)
            window = ordered[start : bisect.bisect_right(epochs, frame.epoch)]
            row: dict[str, Any] = {
                name: _mean_of_present(window, source) for name, source in SPECTRAL_SOURCES
            }
            row["frames_5m"] = len(window)
            expected[(engine_id, lifecycle_id, frame.telemetry_id)] = row
    return expected


def pulse_spectral_features(
    keys: Iterable[SpectralKey], reference: Mapping[SpectralKey, Mapping[str, Any]]
) -> dict[SpectralKey, dict[str, float | None]]:
    """int_ml_features' spectral columns per pulse: its own frame's row, all None without one."""
    return {
        key: {name: reference[key][name] if key in reference else None for name in SPECTRAL_FEATURES}
        for key in keys
    }


def bearing_defect_flag(
    amp_bpfo: float | None, kurtosis: float | None, bpfo_limit: float, kurtosis_limit: float
) -> int:
    """is_bearing_defect_anomaly: either KPI above its threshold; an unknown KPI never raises it."""
    bpfo_high = amp_bpfo is not None and amp_bpfo > bpfo_limit
    kurtosis_high = kurtosis is not None and kurtosis > kurtosis_limit
    return int(bpfo_high or kurtosis_high)


def board_spectral_kpis(
    lifecycle_frames: Iterable[StagedFrame],
    reference: Mapping[SpectralKey, Mapping[str, Any]],
    kpis: tuple[str, ...],
    limits: tuple[float, float],
) -> dict[str, Any]:
    """Board columns of an engine: the newest frame of its current lifecycle (none: all unknown).

    ``kpis`` starts with the BPFO and kurtosis means, which ``limits`` apply to.
    """
    newest = max(lifecycle_frames, key=lambda frame: (frame.frame_micros, frame.telemetry_id), default=None)
    values = reference[(newest.engine_id, newest.lifecycle_id, newest.telemetry_id)] if newest else {}
    return {
        "frame_micros": newest.frame_micros if newest else None,
        **{kpi: values.get(kpi) for kpi in kpis},
        "flag": bearing_defect_flag(values.get(kpis[0]), values.get(kpis[1]), *limits),
    }
