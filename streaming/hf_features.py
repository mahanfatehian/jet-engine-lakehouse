"""Per-frame features of the high-frequency (HF) sensor frames, in numpy and pandas only.

A frame is one engine over one frame period (SIM_HF_FRAME_MS, 1 s by default): kHz vibration
velocity waveforms in mm/s plus about 80 redundant process channels, each a packed
little-endian float32 array. :func:`feature_batches`, the ``mapInPandas`` function applied to
the rows of ``hf_decode.feature_input``, reduces every frame to one ``raw_engine_hf_features``
row (``FEATURE_COLUMNS``: ClickHouse column order and Spark types). Only two vibration
channels are reduced, ``fan_frame_vel`` (prefix ``fan``) and ``core_trf_vel`` (prefix
``core``); any other channel is kept in the bronze lake only. Per prefix ``c``:

    c_rms                   sqrt(mean(x^2))
    c_peak                  max |x|
    c_crest_factor          peak / rms: sqrt(2) for a sine, large for impacts
    c_kurtosis              Pearson kurtosis m4 / m2^2: 3 for Gaussian noise, > 3 for impacts
    c_amp_1x_n1             0-peak amplitude of the fan shaft tone, within +-2 bins of N1/60
    c_amp_1x_n2             0-peak amplitude of the core shaft tone, within +-2 bins of N2/60
    c_amp_bpfo              0-peak amplitude of the bearing outer-race defect tone, within
                            +-3 % of bpfo_order * N1/60 (rolling-element slip shifts it a little)
    c_band_energy_hf        share of the spectral energy above fs/4
    c_spectral_centroid_hz  power-weighted mean frequency

Spectrum: the mean-removed waveform times a periodic Hann window, single-sided and divided by
the window's coherent gain, so that an on-bin sine of amplitude A reads A. Between two bins the
highest bin reads up to 15 % low (Hann scalloping loss), so a tone's amplitude is measured from
the energy of the highest bin of its search band and the two bins on either side,
sqrt(sum(a_k^2) / ENBW) with the window's equivalent noise bandwidth (1.5 bins for Hann): exact
on a bin and within 0.1 % anywhere between bins (see ``Spectrum.tone_amplitude``).

Process channels: ``process_means`` maps every parameter, in name order, to its mean over the
frame (the first of duplicated names wins; a channel without samples is left out, so readers
test ``mapContains`` rather than rely on ClickHouse's 0 for a missing key); ``egt_ab_split`` is
|mean(EGT_A) - mean(EGT_B)|, the disagreement of the redundant EGT probes.

An undefined value is NaN, never an exception (a failing batch would stop the query): a
flat-line waveform has no crest factor, kurtosis, band share or centroid, a non-positive spool
speed no shaft tone, a frame without EGT_A or EGT_B no split. ``hf_decode`` has already
rejected frames without a fan or core waveform and channels whose byte length is not a
multiple of four.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FAN_CHANNEL = "fan_frame_vel"
CORE_CHANNEL = "core_trf_vel"
# Feature prefix -> the vibration channel it is computed from, and the input column of its samples.
VIBRATION_PREFIXES: dict[str, str] = {"fan": FAN_CHANNEL, "core": CORE_CHANNEL}
SAMPLE_COLUMNS: dict[str, str] = {prefix: f"{prefix}_samples" for prefix in VIBRATION_PREFIXES}
CHANNEL_FEATURES: tuple[str, ...] = (
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
EGT_PARAMETERS: tuple[str, str] = ("EGT_A", "EGT_B")

TONE_SEARCH_BINS = 2  # shaft tones: highest bin within +-2 bins of the spool frequency
BPFO_SEARCH_FRACTION = 0.03  # bearing tone: highest bin within +-3 % of its nominal frequency
TONE_LOBE_BINS = 2  # bins on either side of that bin whose energy belongs to the tone
HF_BAND_START_FRACTION = 0.25  # band_energy_hf: the share above fs / 4
MIN_SPECTRUM_SAMPLES = 4
SECONDS_PER_MINUTE = 60.0
_NAN = float("nan")

# Columns of the frames handed to feature_batches; hf_decode.feature_input builds them.
FEATURE_INPUT_COLUMNS: tuple[str, ...] = (
    "engine_id",
    "engine_model",
    "lifecycle_id",
    "cycle_number",
    "telemetry_id",
    "frame_seq",
    "frame_start_us",
    "frame_duration_ms",
    "n1_rpm",
    "n2_rpm",
    "bpfo_order",
    "vibration_sample_rate_hz",
    "vibration_channels",
    "process_channels",
    *SAMPLE_COLUMNS.values(),
    "process_parameters",
    "process_samples",
    "frame_bytes",
    "kafka_partition",
    "kafka_offset",
    "decode_error",
)

# raw_engine_hf_features in ClickHouse column order (the server fills ingested_at), with the
# Spark SQL type of every column: the output schema of the mapInPandas stage.
FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("engine_id", "string"),
    ("engine_model", "string"),
    ("lifecycle_id", "int"),
    ("cycle_number", "int"),
    ("telemetry_id", "bigint"),
    ("frame_seq", "bigint"),
    ("frame_start_ts", "timestamp"),
    ("frame_duration_ms", "int"),
    ("n1_rpm", "float"),
    ("n2_rpm", "float"),
    ("bpfo_order", "float"),
    ("vibration_sample_rate_hz", "int"),
    ("vibration_channels", "int"),
    ("process_channels", "int"),
    *((f"{prefix}_{feature}", "float") for prefix in VIBRATION_PREFIXES for feature in CHANNEL_FEATURES),
    ("egt_ab_split", "float"),
    ("process_means", "map<string,float>"),
    ("frame_bytes", "int"),
    ("kafka_partition", "int"),
    ("kafka_offset", "bigint"),
)
FEATURE_COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURE_COLUMNS)
FEATURE_SCHEMA_DDL = ", ".join(f"{name} {spark_type}" for name, spark_type in FEATURE_COLUMNS)

_PASSTHROUGH_COLUMNS = tuple(name for name in FEATURE_COLUMN_NAMES if name in FEATURE_INPUT_COLUMNS)
_NUMPY_DTYPES = {"int": "int32", "bigint": "int64", "float": "float32"}
_OUTPUT_DTYPES = {name: _NUMPY_DTYPES[kind] for name, kind in FEATURE_COLUMNS if kind in _NUMPY_DTYPES}


def decode_f32le(raw: bytes | None) -> np.ndarray:
    """Samples of a packed little-endian float32 array as float64 (whole values only)."""
    if not raw:
        return np.empty(0)
    return np.frombuffer(raw, dtype="<f4", count=len(raw) // 4).astype(np.float64)


def waveform_statistics(samples: np.ndarray) -> dict[str, float]:
    """``rms``, ``peak``, ``crest_factor`` and Pearson ``kurtosis`` of one waveform."""
    if samples.size == 0:
        return dict.fromkeys(("rms", "peak", "crest_factor", "kurtosis"), _NAN)
    rms = math.sqrt(float(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    deviation = samples - np.mean(samples)
    m2 = float(np.mean(np.square(deviation)))
    m4 = float(np.mean(np.square(np.square(deviation))))
    return {"rms": rms, "peak": peak, "crest_factor": _ratio(peak, rms), "kurtosis": _ratio(m4, m2 * m2)}


@functools.lru_cache(maxsize=8)
def hann_window(size: int) -> np.ndarray:
    """Periodic (DFT-even) Hann window, read-only and cached per frame length."""
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)
    window.setflags(write=False)
    return window


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Single-sided spectrum of a mean-removed, Hann-windowed waveform (see the module doc).

    ``amplitudes`` are 0-peak amplitudes per bin; ``power`` is the one-sided power per bin on
    an arbitrary scale, for ratios.
    """

    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    power: np.ndarray
    resolution_hz: float
    enbw_bins: float

    @classmethod
    def of(cls, samples: np.ndarray, sample_rate_hz: float) -> Spectrum:
        """Spectrum of at least two ``samples`` taken at ``sample_rate_hz``."""
        size = samples.size
        window = hann_window(size)
        magnitude = np.abs(np.fft.rfft((samples - np.mean(samples)) * window))
        one_sided = np.full(magnitude.size, 2.0)
        one_sided[0] = 1.0
        if size % 2 == 0:
            one_sided[-1] = 1.0  # the Nyquist bin has no mirror image either
        gain = float(np.sum(window))
        return cls(
            frequencies_hz=np.fft.rfftfreq(size, d=1.0 / sample_rate_hz),
            amplitudes=magnitude * one_sided / gain,
            power=np.square(magnitude) * one_sided,
            resolution_hz=sample_rate_hz / size,
            enbw_bins=size * float(np.sum(np.square(window))) / gain**2,
        )

    def tone_amplitude(self, frequency_hz: float, half_width_hz: float) -> float:
        """0-peak amplitude of the strongest tone within ``frequency_hz`` +- ``half_width_hz``.

        The band holds the bins up to half a bin beyond the half width, so that the nearest
        bin of any tone within the half width is in it; the tone sits at the band's highest
        bin. When that bin is a local maximum of the spectrum, the amplitude comes from the
        energy of the bin and its two neighbours on either side (free of scalloping loss);
        otherwise the band only holds the skirt of a tone outside it, and the bin's own
        amplitude is the answer. NaN when the frequency is not positive or no bin lies in the
        band (above Nyquist).
        """
        if not (math.isfinite(frequency_hz) and frequency_hz > 0):
            return _NAN
        reach_hz = half_width_hz + self.resolution_hz / 2
        band = np.flatnonzero(np.abs(self.frequencies_hz - frequency_hz) <= reach_hz)
        if band.size == 0:
            return _NAN
        peak = int(band[np.argmax(self.amplitudes[band])])
        if self.amplitudes[peak] < np.max(self.amplitudes[max(peak - 1, 0) : peak + 2]):
            return float(self.amplitudes[peak])
        lobe = self.amplitudes[max(peak - TONE_LOBE_BINS, 0) : peak + TONE_LOBE_BINS + 1]
        return math.sqrt(float(np.sum(np.square(lobe))) / self.enbw_bins)

    def energy_share_above(self, cutoff_hz: float) -> float:
        """Share of the spectral energy above ``cutoff_hz`` (NaN for a flat-line waveform)."""
        above = float(np.sum(self.power[self.frequencies_hz > cutoff_hz]))
        return _ratio(above, float(np.sum(self.power)))

    def centroid_hz(self) -> float:
        """Power-weighted mean frequency (NaN for a flat-line waveform)."""
        return _ratio(float(np.sum(self.frequencies_hz * self.power)), float(np.sum(self.power)))


def channel_features(
    samples: np.ndarray, sample_rate_hz: float, n1_rpm: float, n2_rpm: float, bpfo_order: float
) -> dict[str, float]:
    """The ``CHANNEL_FEATURES`` of one vibration waveform, in that order."""
    features = dict.fromkeys(CHANNEL_FEATURES, _NAN)
    features.update(waveform_statistics(samples))
    if samples.size < MIN_SPECTRUM_SAMPLES or not sample_rate_hz > 0:
        return features
    spectrum = Spectrum.of(samples, float(sample_rate_hz))
    shaft_band_hz = TONE_SEARCH_BINS * spectrum.resolution_hz
    f_n1 = float(n1_rpm) / SECONDS_PER_MINUTE
    f_bpfo = float(bpfo_order) * f_n1
    features.update(
        amp_1x_n1=spectrum.tone_amplitude(f_n1, shaft_band_hz),
        amp_1x_n2=spectrum.tone_amplitude(float(n2_rpm) / SECONDS_PER_MINUTE, shaft_band_hz),
        amp_bpfo=spectrum.tone_amplitude(f_bpfo, max(BPFO_SEARCH_FRACTION * f_bpfo, spectrum.resolution_hz)),
        band_energy_hf=spectrum.energy_share_above(HF_BAND_START_FRACTION * sample_rate_hz),
        spectral_centroid_hz=spectrum.centroid_hz(),
    )
    return features


def process_channel_means(
    parameters: Sequence[str] | None, samples: Sequence[bytes] | None
) -> dict[str, float]:
    """Mean of every process channel over the frame, keyed (and ordered) by parameter name.

    All channels are decoded in one pass (a frame has about 80 of them).
    """
    if parameters is None or samples is None or len(samples) == 0:
        return {}
    chunks = [raw or b"" for raw in samples]
    counts = np.array([len(raw) // 4 for raw in chunks], dtype=np.int64)
    values = decode_f32le(b"".join(raw[: count * 4] for raw, count in zip(chunks, counts, strict=True)))
    sums = np.bincount(np.repeat(np.arange(counts.size), counts), weights=values, minlength=counts.size)
    means: dict[str, float] = {}
    for name, total, count in zip(parameters, sums.tolist(), counts.tolist(), strict=True):
        if count and name not in means:
            means[str(name)] = total / count
    return dict(sorted(means.items()))


def egt_ab_split(process_means: Mapping[str, float]) -> float:
    """|mean(EGT_A) - mean(EGT_B)| (NaN when either probe is missing)."""
    probe_a, probe_b = (process_means.get(name, _NAN) for name in EGT_PARAMETERS)
    return abs(probe_a - probe_b)


def feature_row(frame: Any) -> dict[str, Any]:
    """One ``raw_engine_hf_features`` row of a frame carrying ``FEATURE_INPUT_COLUMNS`` attributes.

    ``frame_start_ts`` is still epoch microseconds here; :func:`features_frame` converts it.
    """
    row: dict[str, Any] = {name: getattr(frame, name) for name in _PASSTHROUGH_COLUMNS}
    row["frame_start_ts"] = frame.frame_start_us
    for prefix, column in SAMPLE_COLUMNS.items():
        features = channel_features(
            decode_f32le(getattr(frame, column)),
            frame.vibration_sample_rate_hz,
            frame.n1_rpm,
            frame.n2_rpm,
            frame.bpfo_order,
        )
        row.update((f"{prefix}_{name}", value) for name, value in features.items())
    means = process_channel_means(frame.process_parameters, frame.process_samples)
    row["egt_ab_split"] = egt_ab_split(means)
    row["process_means"] = means
    return row


def utc_timestamps(micros: pd.Series) -> pd.Series:
    """Epoch microseconds as timezone-aware UTC timestamps of microsecond resolution.

    Microsecond resolution keeps the conversion exact over the whole DateTime64 range, which
    nanosecond timestamps (ending in 2262) do not cover.
    """
    instants = micros.to_numpy(dtype=np.int64).astype("datetime64[us]")
    return pd.Series(instants, index=micros.index).dt.tz_localize("UTC")


def features_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """Feature rows, in ``FEATURE_COLUMNS`` order and types, of a batch of frames."""
    records = [feature_row(frame) for frame in frames.itertuples(index=False)]
    table = pd.DataFrame.from_records(records, columns=list(FEATURE_COLUMN_NAMES))
    table["frame_start_ts"] = utc_timestamps(table["frame_start_ts"])
    return table.astype(_OUTPUT_DTYPES)


def feature_batches(batches: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """The ``mapInPandas`` function: one feature table per Arrow batch of frames.

    Records that ``hf_decode`` rejected (``decode_error`` set) are dropped here; they are
    dead-lettered by the ClickHouse query.
    """
    for frames in batches:
        yield features_frame(frames[frames["decode_error"].isna()])


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, NaN unless the denominator is positive."""
    return numerator / denominator if denominator > 0 else _NAN
