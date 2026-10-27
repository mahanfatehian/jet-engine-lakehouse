"""Vibration synthesis of the HF frames: tones on the spool and bearing frequencies, failure-mode signatures
that grow with wear, and waveform levels that agree with the tick's 1 Hz summary."""

from __future__ import annotations

from types import SimpleNamespace

import hf_signals as hs
import numpy as np
import pytest

FS = 2048
# N2 (the summary rpm) of each engine type, from engine_generator.ENGINE_TYPES.
N2_RPM = {"JT9D-7R4": 7800.0, "CFM56-7B": 14400.0, "GE90-115B": 9300.0}


def summary(
    *,
    wear: float = 0.0,
    mode: str = "HPT_BLADE_EROSION",
    rpm: float = 14400.0,
    vibration: float = 1.1,
    theta=1.0,
) -> SimpleNamespace:
    """A pulse's 1 Hz state (the hf_signals.PulseSummary protocol)."""
    return SimpleNamespace(
        rpm=rpm,
        core_temp_c=575.0,
        egt_c=650.0,
        vibration_mm_s=vibration,
        fuel_flow_kg_h=1250.0,
        oil_pressure_psi=45.0,
        wear_factor=wear,
        theta=theta,
        failure_mode=mode,
    )


def amplitude_spectrum(samples: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and 0-peak amplitudes of a Hann-windowed waveform."""
    window = np.hanning(samples.size)
    amplitudes = np.abs(np.fft.rfft((samples - samples.mean()) * window)) * 2.0 / window.sum()
    return np.fft.rfftfreq(samples.size, 1.0 / fs), amplitudes


def kurtosis(samples: np.ndarray) -> float:
    deviation = samples.astype(np.float64) - samples.mean()
    return float(np.mean(deviation**4) / np.mean(deviation**2) ** 2)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def frame(model: str, state: SimpleNamespace, seed: int, fs: int = FS, channels: int = 2) -> tuple:
    rng = np.random.default_rng(seed)
    frame_state = hs.frame_state(state, model, rng)
    return frame_state, hs.synthesize_vibration(frame_state, fs, fs, channels, rng)


@pytest.mark.parametrize("model", sorted(N2_RPM))
def test_fan_and_core_channels_peak_at_the_n1_and_n2_bins(model: str) -> None:
    state, vibration = frame(model, summary(rpm=N2_RPM[model]), seed=3)
    freqs, fan = amplitude_spectrum(vibration.waveforms[hs.FAN_CHANNEL], FS)
    _, core = amplitude_spectrum(vibration.waveforms[hs.CORE_CHANNEL], FS)
    f_n1, f_n2 = state.n1_rpm / 60.0, state.n2_rpm / 60.0
    assert freqs[1] == 1.0, "a 1 s frame resolves 1 Hz"
    assert int(np.argmax(fan)) == round(f_n1), "fan unbalance (1xN1) dominates the fan frame channel"
    assert int(np.argmax(core)) == round(f_n2), "core unbalance (1xN2) dominates the turbine rear frame"
    assert fan[round(2 * f_n1)] > 10 * np.median(fan), "2xN1 is a visible line as well"


def test_spool_speeds_and_bpfo_follow_the_engine_type_theta_and_wear() -> None:
    rng = np.random.default_rng(0)
    for model, spool in hs.SPOOL_TYPES.items():
        state = hs.frame_state(summary(rpm=N2_RPM[model], theta=1.02, wear=1.0), model, rng)
        expected = spool.n1_nominal_rpm * 1.02 * (1 + hs.SPOOL_DRIFT * 0.8)
        assert state.n1_rpm == pytest.approx(expected, rel=0.003)
        assert state.n2_rpm == N2_RPM[model] and state.bpfo_order == spool.bpfo_order
        assert state.f_bpfo == pytest.approx(spool.bpfo_order * state.n1_rpm / 60.0)
    assert hs.spool_type("XYZ-1") == hs.SPOOL_TYPES["CFM56-7B"], (
        "unknown models fall back like engine_generator"
    )
    fouled = hs.frame_state(
        summary(wear=1.0, mode="COMPRESSOR_FOULING"), "CFM56-7B", np.random.default_rng(0)
    )
    healthy = hs.frame_state(
        summary(wear=0.0, mode="COMPRESSOR_FOULING"), "CFM56-7B", np.random.default_rng(0)
    )
    drift = hs.SPOOL_TYPES["CFM56-7B"].n1_nominal_rpm * hs.SPOOL_DRIFT * 1.5  # same noise draw in both frames
    assert fouled.n1_rpm - healthy.n1_rpm == pytest.approx(drift, rel=1e-9), (
        "fouling (m = 1.5) drifts N1 most"
    )


def test_fan_rms_is_the_summary_vibration_and_every_channel_a_fixed_fraction_of_it() -> None:
    rng = np.random.default_rng(11)
    for index in range(40):
        target = float(rng.uniform(0.5, 9.0))
        mode = sorted(hs.MODE_SIGNATURES)[index % 4]
        state = summary(wear=float(rng.uniform()), mode=mode, vibration=target)
        _, vibration = frame("GE90-115B", state, seed=index, channels=16)
        levels = [rms(waveform) for waveform in vibration.waveforms]
        assert levels[hs.FAN_CHANNEL] == pytest.approx(target, rel=0.01)
        assert levels[hs.CORE_CHANNEL] == pytest.approx(0.6 * target, rel=0.01)
        fractions = [channel.rms_fraction for channel in hs.VIBRATION_CHANNELS]
        assert levels == pytest.approx([target * fraction for fraction in fractions], rel=0.01)
    assert vibration.waveforms.dtype == np.float32 and vibration.waveforms.shape == (16, FS)


def bearing_signature(mode: str, wear: float, frames: int = 6) -> tuple[float, float]:
    """Mean BPFO amplitude (+-3 % band, like the streaming job) and fan kurtosis at the same summary level."""
    amplitudes, kurtoses = [], []
    for seed in range(frames):
        state, vibration = frame("CFM56-7B", summary(wear=wear, mode=mode, vibration=3.0), seed=seed)
        freqs, fan = amplitude_spectrum(vibration.waveforms[hs.FAN_CHANNEL], FS)
        band = np.abs(freqs - state.f_bpfo) <= 0.03 * state.f_bpfo + 0.5
        amplitudes.append(float(fan[band].max()))
        kurtoses.append(kurtosis(vibration.waveforms[hs.FAN_CHANNEL]))
    return float(np.mean(amplitudes)), float(np.mean(kurtoses))


def test_bearing_defect_lines_and_impulsiveness_grow_with_wear_only_in_bearing_modes() -> None:
    wears = (0.1, 0.4, 0.7, 1.0)
    bearing = [bearing_signature("BEARING_DEGRADATION", wear) for wear in wears]
    amplitudes, kurtoses = zip(*bearing, strict=True)
    assert list(amplitudes) == sorted(amplitudes) and amplitudes[-1] > 20 * amplitudes[0]
    assert kurtoses[-1] > 4.0 and kurtoses[-1] > kurtoses[0] + 2.0, "spall impacts make the signal impulsive"
    for mode in ("HPT_BLADE_EROSION", "COMPRESSOR_FOULING"):
        amplitude, peakedness = bearing_signature(mode, 1.0)
        assert amplitude < 0.2 * amplitudes[-1] and peakedness < 3.0, mode
    leak, _ = bearing_signature("OIL_SYSTEM_LEAK", 1.0)
    assert 0.2 * amplitudes[-1] < leak < amplitudes[-1], "oil starvation damages the bearing moderately"


def test_impacts_raise_the_band_above_a_quarter_of_the_sample_rate() -> None:
    def high_band_share(mode: str) -> float:
        _, vibration = frame("JT9D-7R4", summary(wear=1.0, mode=mode, rpm=7800.0, vibration=3.0), seed=5)
        freqs, fan = amplitude_spectrum(vibration.waveforms[hs.FAN_CHANNEL], FS)
        power = fan**2
        return float(power[freqs > FS / 4].sum() / power.sum())

    assert high_band_share("BEARING_DEGRADATION") > 0.3 > 0.1 > high_band_share("HPT_BLADE_EROSION")


def test_tones_above_the_anti_aliasing_edge_are_not_synthesized() -> None:
    rng = np.random.default_rng(2)
    f_n1 = 0.3 * FS  # 2xN1 (0.6 fs) and every BPFO line lie beyond the 0.45 fs filter edge
    state = hs.FrameState(
        summary(rpm=600.0), "CFM56-7B", f_n1 * 60.0, 4.81, hs.MODE_SIGNATURES["HPT_BLADE_EROSION"]
    )
    vibration = hs.synthesize_vibration(state, FS, FS, 2, rng)
    _, fan = amplitude_spectrum(vibration.waveforms[hs.FAN_CHANNEL], FS)
    noise = float(np.median(fan))
    assert fan[round(f_n1)] > 50 * noise, "1xN1 is below the edge"
    assert fan[round(FS - 2 * f_n1)] < 5 * noise, "no alias of 2xN1 folds back into the band"


def test_impact_train_rms_tracks_its_level_and_sharpens_at_higher_sample_rates() -> None:
    rng = np.random.default_rng(4)
    trains = {fs: hs.impact_train(fs, fs, 415.0, 2.0, 1.0, rng) for fs in (2048, 25_600)}
    for fs, train in trains.items():
        assert train.shape == (fs,)
        # Log-normal strengths (sigma 0.5) carry an RMS of exp(0.25) = 1.28 times the level.
        assert rms(train) / 2.0 == pytest.approx(np.exp(0.25), rel=0.2), fs
    assert kurtosis(trains[25_600]) > 1.5 * kurtosis(trains[2048]), "sharper impulses at a higher rate"
    assert not hs.impact_train(100, FS, 415.0, 0.0, 1.0, rng).any()
    assert not hs.impact_train(100, FS, 0.0, 1.0, 1.0, rng).any(), "a stopped spool strikes nothing"


def test_tracked_one_per_rev_levels_are_the_fan_and_core_tone_rms() -> None:
    state, vibration = frame("CFM56-7B", summary(wear=0.5, mode="HPT_BLADE_EROSION", vibration=2.0), seed=9)
    _, fan = amplitude_spectrum(vibration.waveforms[hs.FAN_CHANNEL], FS)
    _, core = amplitude_spectrum(vibration.waveforms[hs.CORE_CHANNEL], FS)
    # Hann scalloping reads an off-bin tone up to 15 % low; the tracked values are exact RMS levels.
    assert fan[round(state.n1_rpm / 60)] == pytest.approx(np.sqrt(2) * vibration.tracked_n1_mm_s, rel=0.16)
    assert core[round(state.n2_rpm / 60)] == pytest.approx(np.sqrt(2) * vibration.tracked_n2_mm_s, rel=0.16)


def test_the_catalog_starts_with_the_fan_and_core_channels_and_synthesis_is_seeded() -> None:
    names = [channel.name for channel in hs.VIBRATION_CHANNELS]
    assert names[:2] == ["fan_frame_vel", "core_trf_vel"] and len(set(names)) == len(names) == 16
    assert all(channel.rms_fraction <= 1.0 and len(channel.weights) == 6 for channel in hs.VIBRATION_CHANNELS)
    state = summary(wear=0.8, mode="BEARING_DEGRADATION", vibration=4.0)
    first, second = frame("JT9D-7R4", state, seed=21)[1], frame("JT9D-7R4", state, seed=21)[1]
    assert np.array_equal(first.waveforms, second.waveforms)
