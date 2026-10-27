"""Unit tests of hf_reference.py, the pure-Python reference of the spectral feature models.

The integration harness trusts this reference to judge the dbt models, so its semantics are
pinned here: the staging rules (unusable values, flat-line channels, absent process keys,
newest copy), the RANGE window of int_vibration_features and the board's bearing flag.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hf_reference import (
    CHANNEL_FEATURES,
    MEASUREMENTS,
    SPECTRAL_FEATURES,
    StagedFrame,
    bearing_defect_flag,
    board_spectral_kpis,
    newest_copies,
    pulse_spectral_features,
    rolling_spectral_features,
    staged_measurements,
    usable,
)

NAN = float("nan")
T0 = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)
ENGINE = "GE90-201"
BOARD_KPIS = ("fan_amp_bpfo_mean_5m", "fan_kurtosis_mean_5m")
LIMITS = (0.5, 3.5)


def raw_copy(**overrides: Any) -> dict[str, Any]:
    """A raw_engine_hf_features copy with plain values (1.0); ``overrides`` replace columns."""
    copy: dict[str, Any] = {
        f"{channel}_{feature}": 1.0 for channel in ("fan", "core") for feature in CHANNEL_FEATURES
    }
    copy.update(
        engine_id=ENGINE,
        lifecycle_id=1,
        telemetry_id=1,
        frame_micros=0,
        egt_ab_split=0.5,
        process_means={"OIL_DP_A": 6.0, "OIL_QTY_A": 20.0, "OIL_T_A": 92.0},
        ingested_at=T0,
        kafka_partition=0,
        kafka_offset=0,
    )
    copy.update(overrides)
    return copy


def frame(telemetry_id: int, seconds: float, lifecycle_id: int = 1, **values: float | None) -> StagedFrame:
    """A staged frame at ``seconds`` with every measurement 1.0 except ``values``."""
    measurements: dict[str, float | None] = dict.fromkeys(MEASUREMENTS, 1.0)
    measurements.update(values)
    return StagedFrame(ENGINE, lifecycle_id, telemetry_id, round(seconds * 1_000_000), measurements)


@pytest.mark.parametrize(
    ("value", "non_negative", "expected"),
    [
        (1.5, True, 1.5),
        (0.0, True, 0.0),
        (-0.1, True, None),
        (-6.5, False, -6.5),
        (NAN, True, None),
        (math.inf, False, None),
        (-math.inf, False, None),
        (None, True, None),
    ],
)
def test_usable_keeps_only_values_a_model_can_use(
    value: float | None, non_negative: bool, expected: float | None
) -> None:
    assert usable(value, non_negative) == expected


def test_a_flat_line_channel_yields_no_feature_not_even_its_zero_amplitudes() -> None:
    flat = {f"fan_{feature}": 0.0 for feature in ("rms", "peak", "amp_1x_n1", "amp_1x_n2", "amp_bpfo")}
    flat |= {
        f"fan_{feature}": NAN
        for feature in ("crest_factor", "kurtosis", "band_energy_hf", "spectral_centroid_hz")
    }
    staged = staged_measurements(raw_copy(**flat))
    assert {staged[f"fan_{feature}"] for feature in CHANNEL_FEATURES} == {None}
    assert {staged[f"core_{feature}"] for feature in CHANNEL_FEATURES} == {1.0}


def test_an_undefined_tone_is_missing_on_its_own() -> None:
    # a stopped spool: no shaft tone and no BPFO, the rest of the channel is a measurement
    staged = staged_measurements(raw_copy(fan_amp_1x_n1=NAN, fan_amp_bpfo=NAN, egt_ab_split=NAN))
    assert (
        staged["fan_amp_1x_n1"] is None and staged["fan_amp_bpfo"] is None and staged["egt_ab_split"] is None
    )
    assert staged["fan_kurtosis"] == 1.0 and staged["fan_rms"] == 1.0


def test_an_absent_process_key_is_missing_never_zero() -> None:
    staged = staged_measurements(
        raw_copy(process_means={"OIL_T_A": -6.5, "OIL_QTY_A": -1.0, "OIL_DP_B": 6.1})
    )
    assert staged["oil_filter_dp"] is None  # OIL_DP_A absent (channel B does not stand in)
    assert staged["oil_temp_c"] == -6.5  # a temperature in degC may be negative
    assert staged["oil_quantity"] is None  # a quantity may not


def test_the_newest_ingestion_wins_then_the_kafka_position() -> None:
    old = raw_copy(ingested_at=T0, kafka_offset=9, egt_ab_split=1.0)
    new = raw_copy(ingested_at=T0 + timedelta(seconds=1), kafka_offset=3, egt_ab_split=2.0)
    tie = raw_copy(ingested_at=T0 + timedelta(seconds=1), kafka_offset=4, egt_ab_split=3.0)
    other = raw_copy(telemetry_id=2, egt_ab_split=4.0)
    assert newest_copies([new, old])[(ENGINE, 1)]["egt_ab_split"] == 2.0
    winners = newest_copies([tie, other, new, old])
    assert {key: copy["egt_ab_split"] for key, copy in winners.items()} == {
        (ENGINE, 1): 3.0,
        (ENGINE, 2): 4.0,
    }


def test_the_window_is_the_300_seconds_up_to_the_frame_with_its_peers() -> None:
    frames = [
        frame(1, 0.0, fan_amp_bpfo=1.0),
        frame(2, 100.0, fan_amp_bpfo=2.0),
        frame(3, 400.0, fan_amp_bpfo=4.0),
        frame(4, 400.0, fan_amp_bpfo=6.0),
        frame(5, 700.0, fan_amp_bpfo=8.0),
    ]
    rows = rolling_spectral_features(frames)
    means = {
        tid: (rows[(ENGINE, 1, tid)]["fan_amp_bpfo_mean_5m"], rows[(ENGINE, 1, tid)]["frames_5m"])
        for tid in range(1, 6)
    }
    # frame 2 sits exactly 300 s before frames 3 and 4 (included), which are each other's peers
    assert means == {1: (1.0, 1), 2: (1.5, 2), 3: (4.0, 3), 4: (4.0, 3), 5: (6.0, 3)}


def test_missing_values_are_skipped_and_a_window_without_any_is_missing() -> None:
    rows = rolling_spectral_features(
        [
            frame(1, 0.0, oil_filter_dp=None),
            frame(2, 10.0, oil_filter_dp=4.0),
            frame(3, 400.0, oil_filter_dp=None),
        ]
    )
    assert rows[(ENGINE, 1, 1)]["oil_filter_dp_mean_5m"] is None
    assert rows[(ENGINE, 1, 2)]["oil_filter_dp_mean_5m"] == 4.0
    assert rows[(ENGINE, 1, 3)]["oil_filter_dp_mean_5m"] is None  # frame 2 is 390 s back
    assert rows[(ENGINE, 1, 3)]["frames_5m"] == 1


def test_windows_stay_inside_their_lifecycle() -> None:
    rows = rolling_spectral_features(
        [frame(1, 0.0, lifecycle_id=1, egt_ab_split=10.0), frame(2, 5.0, lifecycle_id=2)]
    )
    assert (rows[(ENGINE, 2, 2)]["egt_ab_split_mean_5m"], rows[(ENGINE, 2, 2)]["frames_5m"]) == (1.0, 1)


def test_a_pulse_without_its_own_frame_has_no_spectral_feature() -> None:
    reference = rolling_spectral_features([frame(1, 0.0)])
    pulses = pulse_spectral_features([(ENGINE, 1, 1), (ENGINE, 1, 2), (ENGINE, 2, 1)], reference)
    assert list(pulses[(ENGINE, 1, 1)]) == list(SPECTRAL_FEATURES)
    assert set(pulses[(ENGINE, 1, 1)].values()) == {1.0}
    assert set(pulses[(ENGINE, 1, 2)].values()) == {None}
    assert set(pulses[(ENGINE, 2, 1)].values()) == {None}  # the frame belongs to another lifecycle


@pytest.mark.parametrize(
    ("amp_bpfo", "kurtosis", "flag"),
    [
        (0.6, None, 1),
        (None, 3.6, 1),
        (0.6, 3.0, 1),
        (0.4, 3.4, 0),
        (0.5, 3.5, 0),
        (0.4, None, 0),
        (None, None, 0),
    ],
)
def test_the_bearing_flag_needs_a_known_kpi_above_its_threshold(
    amp_bpfo: float | None, kurtosis: float | None, flag: int
) -> None:
    assert bearing_defect_flag(amp_bpfo, kurtosis, *LIMITS) == flag


def test_the_board_reads_the_newest_frame_of_the_lifecycle() -> None:
    frames = [
        frame(1, 0.0, fan_amp_bpfo=0.2),
        frame(3, 50.0, fan_amp_bpfo=2.0),
        frame(2, 50.0, fan_amp_bpfo=0.1),
    ]
    board = board_spectral_kpis(frames, rolling_spectral_features(frames), BOARD_KPIS, LIMITS)
    # peers at 50 s: the higher telemetry id is the newer pulse; its window holds all three
    assert board["frame_micros"] == 50_000_000
    assert board["fan_amp_bpfo_mean_5m"] == pytest.approx(2.3 / 3)
    assert (board["fan_kurtosis_mean_5m"], board["flag"]) == (1.0, 1)


def test_the_board_reads_unknown_without_a_frame_of_the_lifecycle() -> None:
    assert board_spectral_kpis([], {}, BOARD_KPIS, LIMITS) == {
        "frame_micros": None,
        "fan_amp_bpfo_mean_5m": None,
        "fan_kurtosis_mean_5m": None,
        "flag": 0,
    }
