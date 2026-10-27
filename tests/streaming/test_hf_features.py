"""Unit tests of the HF feature math on synthetic signals (numpy and pandas only, no Spark).

Signals are built with known tones, impulses and noise at the default edge rate (2048 Hz,
1 s frames, so 1 Hz bins), and the features are checked against their closed forms.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hf_features import (
    FEATURE_COLUMN_NAMES,
    FEATURE_INPUT_COLUMNS,
    Spectrum,
    channel_features,
    decode_f32le,
    egt_ab_split,
    feature_batches,
    features_frame,
    process_channel_means,
    waveform_statistics,
)

FS = 2048
N1_RPM, N2_RPM, BPFO_ORDER = 3650.0, 7800.0, 4.62  # JT9D: f_N1 60.83 Hz, f_N2 130 Hz, BPFO 281 Hz
F_N1, F_N2 = N1_RPM / 60, N2_RPM / 60
F_BPFO = BPFO_ORDER * F_N1


def tone(frequency_hz: float, amplitude: float, phase: float = 0.3, samples: int = FS) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency_hz * np.arange(samples) / FS + phase)


def impulse_train(rate_hz: float, ring_hz: float = 600.0, decay_s: float = 0.0015) -> np.ndarray:
    """Exponentially decaying ring-downs (bearing impacts) every 1 / rate_hz seconds."""
    t = np.arange(FS) / FS
    since_impact = t % (1.0 / rate_hz)
    return np.exp(-since_impact / decay_s) * np.sin(2 * np.pi * ring_hz * since_impact)


def f32(*values: float) -> bytes:
    return np.array(values, dtype="<f4").tobytes()


def test_decode_f32le_reads_packed_little_endian_float32() -> None:
    raw = f32(1.5, -2.25, 3.0e5)
    decoded = decode_f32le(raw)
    assert decoded.dtype == np.float64
    assert decoded.tolist() == [1.5, -2.25, 300000.0]
    assert decode_f32le(raw + b"\x01\x02").tolist() == [1.5, -2.25, 300000.0]  # torn tail ignored
    assert decode_f32le(np.array([1.5], dtype=">f4").tobytes()).tolist() != [1.5]  # byte order matters
    assert decode_f32le(b"").size == 0
    assert decode_f32le(None).size == 0


def test_statistics_of_a_sine_follow_their_closed_forms() -> None:
    sine = tone(64.0, 2.0, phase=0.0)  # 32 samples per period: the crests are sampled exactly
    stats = waveform_statistics(sine)
    assert stats["rms"] == pytest.approx(2.0 / math.sqrt(2), rel=1e-9)
    assert stats["peak"] == pytest.approx(2.0, rel=1e-12)
    assert stats["crest_factor"] == pytest.approx(math.sqrt(2), rel=1e-9)
    assert stats["kurtosis"] == pytest.approx(1.5, rel=1e-9)


def test_gaussian_noise_has_a_kurtosis_of_three() -> None:
    noise = np.random.default_rng(7).normal(0.0, 1.3, 2**15)
    assert waveform_statistics(noise)["kurtosis"] == pytest.approx(3.0, abs=0.15)


def test_an_impulse_train_is_impulsive() -> None:
    impacts = impulse_train(rate_hz=40.0) + np.random.default_rng(3).normal(0.0, 0.01, FS)
    stats = waveform_statistics(impacts)
    assert stats["kurtosis"] > 5.0
    assert stats["crest_factor"] > 4.0


@pytest.mark.parametrize("frequency_hz", [39.0, 61.0, 130.0, 281.0, 415.0, 1000.0])
def test_on_bin_tones_read_their_exact_amplitude(frequency_hz: float) -> None:
    spectrum = Spectrum.of(tone(frequency_hz, 2.5), FS)
    assert spectrum.resolution_hz == 1.0
    assert spectrum.enbw_bins == pytest.approx(1.5)
    assert spectrum.tone_amplitude(frequency_hz, 2.0) == pytest.approx(2.5, rel=1e-6)


@pytest.mark.parametrize("bin_offset", [0.1, 0.25, 0.5, 0.75])
def test_tones_between_bins_read_their_amplitude_within_two_percent(bin_offset: float) -> None:
    frequency_hz = 60.0 + bin_offset
    spectrum = Spectrum.of(tone(frequency_hz, 3.0), FS)
    assert spectrum.tone_amplitude(frequency_hz, 2.0) == pytest.approx(3.0, rel=0.002)  # 2 % required
    if bin_offset == 0.5:  # what the highest bin alone would report: Hann scalloping loss
        highest = spectrum.amplitudes[np.abs(spectrum.frequencies_hz - frequency_hz) <= 2.0].max()
        assert highest == pytest.approx(3.0 * 0.8488, rel=0.01)


def test_tones_are_scaled_for_any_sample_rate() -> None:
    samples = 4096  # 25.6 kHz for 160 ms: 6.25 Hz bins, as at full fidelity
    signal = 1.7 * np.sin(2 * np.pi * 812.5 * np.arange(samples) / 25600.0)
    spectrum = Spectrum.of(signal, 25600.0)
    assert spectrum.resolution_hz == 6.25
    assert spectrum.tone_amplitude(812.5, 2 * 6.25) == pytest.approx(1.7, rel=1e-6)


def test_shaft_and_bearing_tones_are_found_from_the_spool_speeds() -> None:
    slipped_bpfo = F_BPFO * 1.02  # rolling elements slip: the defect tone sits 2 % high
    noise = np.random.default_rng(11).normal(0.0, 0.02, FS)
    signal = tone(F_N1, 1.0) + tone(2 * F_N1, 0.4) + tone(F_N2, 0.7) + tone(slipped_bpfo, 0.3) + noise
    features = channel_features(signal, FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert features["amp_1x_n1"] == pytest.approx(1.0, rel=0.02)
    assert features["amp_1x_n2"] == pytest.approx(0.7, rel=0.02)
    assert features["amp_bpfo"] == pytest.approx(0.3, rel=0.02)


def test_bearing_tone_beyond_three_percent_is_not_counted() -> None:
    features = channel_features(tone(F_BPFO * 1.05, 0.3), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert features["amp_bpfo"] < 0.01


@pytest.mark.parametrize("offset_bins", [-2.0, -1.4, 1.5, 2.0])
def test_shaft_tone_is_found_up_to_two_bins_off_the_spool_speed(offset_bins: float) -> None:
    signal = tone(F_N1 + offset_bins, 1.0)  # F_N1 = 60.83 Hz lies between two bins
    assert channel_features(signal, FS, N1_RPM, N2_RPM, BPFO_ORDER)["amp_1x_n1"] == pytest.approx(
        1.0, rel=0.02
    )


@pytest.mark.parametrize("offset_bins", [-4.3, 4.3, 6.0])
def test_shaft_tone_farther_away_is_not_reported(offset_bins: float) -> None:
    # Only the skirt of the tone reaches the band: its peak lies outside.
    signal = tone(F_N1 + offset_bins, 1.0)
    assert channel_features(signal, FS, N1_RPM, N2_RPM, BPFO_ORDER)["amp_1x_n1"] < 0.05


def test_band_energy_share_and_centroid() -> None:
    low = channel_features(tone(100.0, 1.0), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert low["band_energy_hf"] < 1e-6
    assert low["spectral_centroid_hz"] == pytest.approx(100.0, abs=0.5)
    high = channel_features(tone(700.0, 1.0), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert high["band_energy_hf"] == pytest.approx(1.0, abs=1e-6)
    assert high["spectral_centroid_hz"] == pytest.approx(700.0, abs=0.5)
    pair = channel_features(tone(200.0, 1.0) + tone(800.0, 1.0), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert pair["band_energy_hf"] == pytest.approx(0.5, abs=1e-6)
    assert pair["spectral_centroid_hz"] == pytest.approx(500.0, abs=0.5)
    white = np.random.default_rng(5).normal(0.0, 1.0, 16 * FS)  # 16 s: the share is ~0.5 +- 0.01
    noise = channel_features(white, FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert noise["band_energy_hf"] == pytest.approx(0.5, abs=0.04)
    assert noise["spectral_centroid_hz"] == pytest.approx(FS / 4, rel=0.05)


def test_impacts_shift_energy_to_high_frequencies() -> None:
    healthy = channel_features(tone(F_N1, 1.0) + tone(F_N2, 0.5), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    damaged = channel_features(
        tone(F_N1, 1.0) + tone(F_N2, 0.5) + 2.0 * impulse_train(F_BPFO / 4, ring_hz=700.0),
        FS,
        N1_RPM,
        N2_RPM,
        BPFO_ORDER,
    )
    assert damaged["band_energy_hf"] > healthy["band_energy_hf"] + 0.1
    assert damaged["spectral_centroid_hz"] > 1.5 * healthy["spectral_centroid_hz"]
    assert damaged["kurtosis"] > healthy["kurtosis"]
    assert damaged["crest_factor"] > healthy["crest_factor"]


def test_undefined_features_are_nan_not_errors() -> None:
    flat = channel_features(np.zeros(FS), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert (flat["rms"], flat["peak"], flat["amp_1x_n1"]) == (0.0, 0.0, 0.0)
    for name in ("crest_factor", "kurtosis", "band_energy_hf", "spectral_centroid_hz"):
        assert math.isnan(flat[name]), name
    stopped = channel_features(tone(F_N2, 1.0), FS, 0.0, N2_RPM, BPFO_ORDER)
    assert math.isnan(stopped["amp_1x_n1"]) and math.isnan(stopped["amp_bpfo"])
    assert stopped["amp_1x_n2"] == pytest.approx(1.0, rel=0.02)
    beyond_nyquist = channel_features(tone(F_N2, 1.0), FS, N1_RPM, 70_000.0, BPFO_ORDER)
    assert math.isnan(beyond_nyquist["amp_1x_n2"])
    assert all(math.isnan(value) for value in channel_features(np.empty(0), FS, N1_RPM, N2_RPM, 4.6).values())
    short = channel_features(np.array([1.0, -1.0, 1.0]), FS, N1_RPM, N2_RPM, BPFO_ORDER)
    assert short["rms"] == 1.0 and math.isnan(short["amp_1x_n1"])


def test_process_means_and_egt_split() -> None:
    parameters = np.array(["EGT_A", "EGT_B", "OIL_T_A", "EGT_A", "OIL_QTY_A"], dtype=object)
    samples = [f32(610.0, 612.0), f32(613.0, 615.0), f32(80.5), f32(999.0), b""]
    means = process_channel_means(parameters, samples)
    assert means == {"EGT_A": 611.0, "EGT_B": 614.0, "OIL_T_A": 80.5}  # first EGT_A wins, empty left out
    assert means["OIL_T_A"] == float(np.float32(80.5))
    assert list(process_channel_means(parameters[::-1], samples[::-1])) == ["EGT_A", "EGT_B", "OIL_T_A"]
    assert egt_ab_split(means) == 3.0
    assert math.isnan(egt_ab_split({"EGT_A": 611.0}))
    assert process_channel_means(None, None) == {}
    assert process_channel_means(np.array([], dtype=object), np.array([], dtype=object)) == {}
    assert process_channel_means(["A", "B"], [None, f32(1.0, 3.0)]) == {"B": 2.0}


def frame_input(telemetry_id: int, start: datetime, **overrides: Any) -> dict[str, Any]:
    """One row in the FEATURE_INPUT_COLUMNS layout (what hf_decode.feature_input ships)."""
    row = {
        "engine_id": "JT9D-001",
        "engine_model": "JT9D-7R4",
        "lifecycle_id": 3,
        "cycle_number": 17,
        "telemetry_id": telemetry_id,
        "frame_seq": telemetry_id - 4000,
        "frame_start_us": (start - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1),
        "frame_duration_ms": 1000,
        "n1_rpm": np.float32(N1_RPM),
        "n2_rpm": np.float32(N2_RPM),
        "bpfo_order": np.float32(BPFO_ORDER),
        "vibration_sample_rate_hz": FS,
        "vibration_channels": 3,
        "process_channels": 2,
        "fan_samples": (tone(F_N1, 2.0) + tone(F_BPFO, 0.25)).astype("<f4").tobytes(),
        "core_samples": tone(F_N2, 0.8).astype("<f4").tobytes(),
        "process_parameters": np.array(["EGT_A", "EGT_B"], dtype=object),
        "process_samples": np.array([f32(610.0, 612.0), f32(609.0, 609.0)], dtype=object),
        "frame_bytes": 16574,
        "kafka_partition": 2,
        "kafka_offset": 7,
        "decode_error": None,
    }
    row.update(overrides)
    return row


def test_features_frame_has_the_raw_engine_hf_features_layout() -> None:
    starts = [
        datetime(2026, 10, 26, 9, 0, 1, 123456, tzinfo=UTC),
        datetime(2290, 1, 1, 0, 0, 0, 1, tzinfo=UTC),
    ]
    frames = pd.DataFrame(
        [frame_input(4242, starts[0]), frame_input(4243, starts[1], fan_samples=f32(*([0.0] * 64)))],
        columns=list(FEATURE_INPUT_COLUMNS),
    )
    table = features_frame(frames)
    assert tuple(table.columns) == FEATURE_COLUMN_NAMES
    assert (table["lifecycle_id"].dtype, table["telemetry_id"].dtype) == (np.int32, np.int64)
    assert (table["fan_amp_bpfo"].dtype, table["n1_rpm"].dtype) == (np.float32, np.float32)
    assert str(table["frame_start_ts"].dtype) == "datetime64[us, UTC]"
    # Exact to the microsecond, also beyond the year 2262 where nanosecond timestamps end.
    assert [ts.to_pydatetime() for ts in table["frame_start_ts"]] == starts
    first = table.iloc[0]
    assert (first["engine_id"], first["telemetry_id"], first["frame_seq"], first["kafka_offset"]) == (
        "JT9D-001",
        4242,
        242,
        7,
    )
    assert first["fan_amp_1x_n1"] == pytest.approx(2.0, rel=0.02)
    assert first["fan_amp_bpfo"] == pytest.approx(0.25, rel=0.02)
    assert first["core_amp_1x_n2"] == pytest.approx(0.8, rel=0.02)
    # Within 0.5 %: 1 s holds a non-integer number of periods of each tone.
    assert first["fan_rms"] == pytest.approx(math.sqrt((2.0**2 + 0.25**2) / 2), rel=5e-3)
    assert first["process_means"] == {"EGT_A": 611.0, "EGT_B": 609.0}
    assert first["egt_ab_split"] == pytest.approx(2.0)
    assert math.isnan(table.iloc[1]["fan_crest_factor"])  # a flat-line fan channel


def test_feature_batches_yields_one_table_per_arrow_batch() -> None:
    start = datetime(2026, 10, 26, 9, 0, 1, tzinfo=UTC)
    rejected = frame_input(3, start, decode_error="null value", fan_samples=None, process_samples=None)
    batches = [
        pd.DataFrame(
            [frame_input(1, start), rejected, frame_input(2, start)], columns=list(FEATURE_INPUT_COLUMNS)
        ),
        pd.DataFrame(columns=list(FEATURE_INPUT_COLUMNS)),
    ]
    tables = list(feature_batches(iter(batches)))
    assert [len(table) for table in tables] == [2, 0]  # the rejected record is dropped
    assert tables[0]["telemetry_id"].tolist() == [1, 2]
    assert tuple(tables[1].columns) == FEATURE_COLUMN_NAMES
    assert tables[1]["telemetry_id"].dtype == np.int64
