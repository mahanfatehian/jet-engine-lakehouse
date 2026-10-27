"""High-frequency vibration synthesis: kHz velocity waveforms that agree with the tick's 1 Hz summary.

Each frame mixes the same physical sources into every accelerometer channel (vectorised numpy, float32):

    source         frequency                        level before scaling              physical cause
    1xN1 tone      f_N1 = N1 / 60                   1 + 0.15 w                        fan unbalance
    2xN1 tone      2 f_N1                           0.25 + 0.1 w                      shaft misalignment
    1xN2 tone      f_N2 = N2 / 60                   1 + 2.5 c w^2                     core unbalance
    BPFO tones     k bpfo_order f_N1, k = 1, 2, 3   b (0.01 + 1.6 w^3) (1, .5, .25)   No.1 bearing outer race
    impact train   ring-downs at the BPFO rate      b 1.6 w^3.5 (RMS)                 spall impacts
    noise          broadband                        0.12 + 0.25 w                     flow, rub, sensor noise

b and c come from the lifecycle's failure mode (MODE_SIGNATURES): BEARING_DEGRADATION b = 1, OIL_SYSTEM_LEAK
b = 0.35 (oil-starved bearings), the others b = 0.05; HPT_BLADE_EROSION and COMPRESSOR_FOULING unbalance the
core. Impacts strike a structural resonance at 0.35 fs (above fs/4, so they raise the high band), decay within
15 % of the impact interval, jitter by 1 % (rolling-element slip), vary in strength (log-normal) and become
regular as the spall grows (an impact on 40 % of the ball passes for a new defect, all of them at w = 1).
Tones at or above 0.45 fs are not synthesized: the anti-aliasing filter of a real DAQ removes them.
Measured with the streaming job's feature code at the default 2048 Hz, a bearing failure takes the fan
channel's kurtosis from ~1.9 (tones) to 5..9 and its BPFO line to 10..14x the level of the HPT and
compressor modes (about twice the oil-leak level); at 25.6 kHz the sharper impulses push the kurtosis
past 10.

Each channel weights the sources by its mounting position (VIBRATION_CHANNELS) and is finally scaled so
that the fan channel's RMS equals the tick's vibration_mm_s and every other channel's RMS is a fixed
fraction of it: the HF data add spectral shape and impulsiveness (the failure-mode signatures), never a
level that contradicts the 1 Hz summary.

Spools: N2 is the tick's rpm; N1 = N1_nom theta (1 + 0.006 m w) + noise with the per-type N1_nom and m the
rpm sensitivity of the lifecycle (1.5 when compressor fouling dominates, else 0.8, as in engine_generator).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

VIBRATION_UNIT = "mm/s"
TONE_CUTOFF = 0.45  # fraction of fs: anti-aliasing filter edge
RESONANCE = 0.35  # fraction of fs: structural resonance excited by the bearing impacts
BPFO_HARMONICS = (1.0, 0.5, 0.25)
IMPACT_DECAY = 0.15  # ring-down time constant as a fraction of the impact interval
MIN_DECAY_SAMPLES = 1.5
IMPACT_JITTER = 0.01
IMPACT_SPREAD = 0.5  # sigma of the log-normal impact strength
N1_NOISE = 0.0005  # fraction of N1_nom
SPOOL_DRIFT = 0.006


class SpoolType(NamedTuple):
    """Fan speed at theta = 1 and the fan's No.1 bearing outer-race defect order of one engine type."""

    n1_nominal_rpm: float
    bpfo_order: float


SPOOL_TYPES: dict[str, SpoolType] = {
    "JT9D-7R4": SpoolType(3650.0, 4.62),
    "CFM56-7B": SpoolType(5175.0, 4.81),
    "GE90-115B": SpoolType(2355.0, 5.07),
}
FALLBACK_MODEL = "CFM56-7B"  # engine_generator's fallback type for unknown engine-id prefixes


class ModeSignature(NamedTuple):
    """How a failure mode shows in the vibration: bearing defect scale b, core unbalance growth c, and the rpm
    sensitivity m that also drifts N1."""

    bearing: float
    core_unbalance: float
    spool_sensitivity: float


MODE_SIGNATURES: dict[str, ModeSignature] = {
    "BEARING_DEGRADATION": ModeSignature(1.0, 0.15, 0.8),
    "OIL_SYSTEM_LEAK": ModeSignature(0.35, 0.15, 0.8),
    "HPT_BLADE_EROSION": ModeSignature(0.05, 1.0, 0.8),
    "COMPRESSOR_FOULING": ModeSignature(0.05, 0.7, 1.5),
}
NEUTRAL_SIGNATURE = ModeSignature(0.05, 0.15, 0.8)


class VibrationChannel(NamedTuple):
    """One accelerometer: RMS relative to the fan channel and the weights of the sources it picks up, in the
    order 1xN1, 2xN1, 1xN2, BPFO tones, impacts, noise."""

    name: str
    description: str
    rms_fraction: float
    weights: tuple[float, float, float, float, float, float]


VIBRATION_CHANNELS: tuple[VibrationChannel, ...] = (
    VibrationChannel(
        "fan_frame_vel", "fan frame (No.1 bearing), radial", 1.00, (1.0, 1.0, 0.3, 1.0, 1.0, 1.0)
    ),
    VibrationChannel("core_trf_vel", "turbine rear frame, radial", 0.60, (0.3, 0.3, 1.0, 0.3, 0.25, 1.0)),
    VibrationChannel("fan_frame_vel_b", "fan frame, redundant sensor", 1.00, (1.0, 1.0, 0.3, 1.0, 1.0, 1.0)),
    VibrationChannel("no1_bearing_vel", "No.1 bearing housing", 0.80, (0.9, 0.8, 0.2, 1.6, 1.8, 0.8)),
    VibrationChannel("no2_bearing_vel", "No.2 bearing housing", 0.75, (0.8, 0.7, 0.3, 1.2, 1.3, 0.8)),
    VibrationChannel("no3_bearing_vel", "No.3 (HP thrust) bearing", 0.70, (0.3, 0.3, 1.2, 0.3, 0.3, 0.9)),
    VibrationChannel("no4_bearing_vel", "No.4 (HP roller) bearing", 0.65, (0.25, 0.3, 1.3, 0.25, 0.25, 0.9)),
    VibrationChannel("fan_case_axial_vel", "fan case, axial", 0.55, (0.7, 1.2, 0.2, 0.6, 0.7, 1.0)),
    VibrationChannel("hpc_case_vel", "HP compressor case", 0.50, (0.3, 0.2, 1.1, 0.2, 0.2, 1.2)),
    VibrationChannel("hpt_case_vel", "HP turbine case", 0.55, (0.2, 0.2, 1.2, 0.2, 0.2, 1.3)),
    VibrationChannel("lpt_case_vel", "LP turbine case", 0.60, (0.8, 0.5, 0.5, 0.3, 0.3, 1.1)),
    VibrationChannel("agb_vel", "accessory gearbox (N2 driven)", 0.45, (0.2, 0.2, 1.0, 0.1, 0.1, 1.4)),
    VibrationChannel("tgb_vel", "transfer gearbox", 0.45, (0.3, 0.2, 0.9, 0.2, 0.2, 1.3)),
    VibrationChannel("inlet_cowl_vel", "inlet cowl", 0.40, (1.0, 0.6, 0.1, 0.3, 0.2, 1.0)),
    VibrationChannel("pylon_fwd_vel", "forward engine mount", 0.35, (0.9, 0.5, 0.4, 0.2, 0.2, 0.9)),
    VibrationChannel(
        "core_trf_vel_b", "turbine rear frame, redundant sensor", 0.60, (0.3, 0.3, 1.0, 0.3, 0.25, 1.0)
    ),
)
FAN_CHANNEL, CORE_CHANNEL = 0, 1


class PulseSummary(Protocol):
    """The 1 Hz state an HF frame is synthesized around (an engine_generator.Pulse or a frame request)."""

    rpm: float
    core_temp_c: float
    egt_c: float
    vibration_mm_s: float
    fuel_flow_kg_h: float
    oil_pressure_psi: float
    wear_factor: float
    theta: float
    failure_mode: str


@dataclass(frozen=True, slots=True)
class FrameState:
    """Physical state of one frame: the tick's summary plus the spool speeds derived from it."""

    summary: PulseSummary
    engine_model: str
    n1_rpm: float
    bpfo_order: float
    signature: ModeSignature

    @property
    def wear(self) -> float:
        """Ground-truth wear w of the tick (drives the defect sources, never sent)."""
        return self.summary.wear_factor

    @property
    def n2_rpm(self) -> float:
        """Core spool speed: the tick's rpm."""
        return self.summary.rpm

    @property
    def f_bpfo(self) -> float:
        """Bearing outer-race defect frequency in Hz."""
        return self.bpfo_order * self.n1_rpm / 60.0


def spool_type(engine_model: str) -> SpoolType:
    """Spool constants of ``engine_model`` (CFM56-7B for an unknown model, like engine_generator)."""
    return SPOOL_TYPES.get(engine_model, SPOOL_TYPES[FALLBACK_MODEL])


def frame_state(summary: PulseSummary, engine_model: str, rng: np.random.Generator) -> FrameState:
    """Frame state of ``summary``: N1 from theta and wear (see module doc), N2 = rpm."""
    spool = spool_type(engine_model)
    signature = MODE_SIGNATURES.get(summary.failure_mode, NEUTRAL_SIGNATURE)
    drift = 1.0 + SPOOL_DRIFT * signature.spool_sensitivity * summary.wear_factor
    noise = N1_NOISE * spool.n1_nominal_rpm * float(rng.standard_normal())
    n1 = spool.n1_nominal_rpm * summary.theta * drift + noise
    return FrameState(summary, engine_model, n1, spool.bpfo_order, signature)


class SourceLevels(NamedTuple):
    """Amplitudes of the vibration sources before the channels are scaled to the summary level."""

    n1: float
    n1x2: float
    n2: float
    bpfo: float
    impacts: float
    noise: float


def source_levels(wear: float, signature: ModeSignature) -> SourceLevels:
    """Source amplitudes for wear ``wear`` under the failure mode ``signature`` (table in the module doc)."""
    w, b = wear, signature.bearing
    return SourceLevels(
        n1=1.0 + 0.15 * w,
        n1x2=0.25 + 0.1 * w,
        n2=1.0 + 2.5 * signature.core_unbalance * w**2,
        bpfo=b * (0.01 + 1.6 * w**3),
        impacts=b * 1.6 * w**3.5,
        noise=0.12 + 0.25 * w,
    )


def _tone(t: np.ndarray, freq: float, amplitude: float, fs: int, rng: np.random.Generator) -> np.ndarray:
    if freq >= TONE_CUTOFF * fs or amplitude == 0.0:
        return np.zeros_like(t)
    return amplitude * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))


def impact_train(
    n: int, fs: int, rate_hz: float, level: float, regularity: float, rng: np.random.Generator
) -> np.ndarray:
    """``n`` samples of decaying ring-downs at ``rate_hz`` impacts per second (bearing spall strikes).

    Each impact excites the structural resonance at RESONANCE * fs; an impact occurs on a fraction
    ``regularity`` of the passes with a log-normal strength. The ring-down is normalised so that a regular
    train of unit strength has an RMS of ``level`` at every sample rate: a higher rate resolves sharper
    impulses (higher kurtosis), not more energy.
    """
    if level == 0.0 or not rate_hz > 0.0:
        return np.zeros(n)
    interval = fs / rate_hz
    count = math.ceil(n / interval) + 1
    positions = rng.uniform(0.0, interval) + interval * np.arange(count)
    positions += rng.normal(0.0, IMPACT_JITTER * interval, count)
    struck = (rng.random(count) < regularity) & (positions >= 0.0) & (positions < n)
    starts = np.rint(positions[struck]).astype(np.int64)
    strengths = level * rng.lognormal(0.0, IMPACT_SPREAD, starts.size)
    decay = max(MIN_DECAY_SAMPLES, IMPACT_DECAY * interval)
    k = np.arange(math.ceil(6.0 * decay))
    kernel = np.exp(-k / decay) * np.cos(2.0 * np.pi * RESONANCE * k)
    kernel *= math.sqrt(interval / float(np.sum(kernel**2)))
    index = (starts[:, None] + k[None, :]).ravel()
    weights = (strengths[:, None] * kernel[None, :]).ravel()
    inside = index < n
    return np.bincount(index[inside], weights=weights[inside], minlength=n)


@functools.cache
def channel_matrix(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Source weights (count x 6) and RMS fractions (count) of the first ``count`` catalog channels
    (cached: callers must not modify them)."""
    channels = VIBRATION_CHANNELS[:count]
    weights = np.array([channel.weights for channel in channels], dtype=np.float32)
    return weights, np.array([channel.rms_fraction for channel in channels])


@dataclass(frozen=True, slots=True)
class VibrationFrame:
    """Waveforms (channels x samples, float32, mm/s) plus the tracked 1/rev levels of the process channels."""

    waveforms: np.ndarray
    tracked_n1_mm_s: float
    tracked_n2_mm_s: float


def synthesize_vibration(
    state: FrameState, fs: int, samples: int, channels: int, rng: np.random.Generator
) -> VibrationFrame:
    """Velocity waveforms of the first ``channels`` catalog channels for one frame (see module doc)."""
    levels = source_levels(state.wear, state.signature)
    t = np.arange(samples) / fs
    f_n1, f_n2, f_bpfo = state.n1_rpm / 60.0, state.n2_rpm / 60.0, state.f_bpfo
    bpfo = sum(
        _tone(t, order * f_bpfo, levels.bpfo * share, fs, rng)
        for order, share in enumerate(BPFO_HARMONICS, start=1)
    )
    regularity = 0.4 + 0.6 * state.wear
    sources = np.stack(
        [
            _tone(t, f_n1, levels.n1, fs, rng),
            _tone(t, 2.0 * f_n1, levels.n1x2, fs, rng),
            _tone(t, f_n2, levels.n2, fs, rng),
            bpfo,
            impact_train(samples, fs, f_bpfo, levels.impacts, regularity, rng),
        ]
    ).astype(np.float32)
    weights, fractions = channel_matrix(channels)
    waveforms = weights[:, :5] @ sources
    noise = rng.standard_normal((channels, samples), dtype=np.float32)
    noise *= weights[:, 5:] * np.float32(levels.noise)
    waveforms += noise
    rms = np.sqrt(np.einsum("ij,ij->i", waveforms, waveforms, dtype=np.float64) / samples)
    scale = state.summary.vibration_mm_s * fractions / rms
    waveforms *= scale[:, None].astype(np.float32)
    tone_rms = math.sqrt(0.5)
    return VibrationFrame(
        waveforms=waveforms,
        tracked_n1_mm_s=float(scale[FAN_CHANNEL] * weights[FAN_CHANNEL, 0] * levels.n1 * tone_rms),
        tracked_n2_mm_s=float(scale[CORE_CHANNEL] * weights[CORE_CHANNEL, 2] * levels.n2 * tone_rms),
    )
