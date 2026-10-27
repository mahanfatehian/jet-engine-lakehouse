"""HF sensor frames: one Avro EngineSensorFrame record per committed telemetry pulse.

Flow: a committed transaction (a live tick, or one engine's backfilled history) reaches FrameTap, which
turns every written pulse into a lightweight FrameRequest: the pulse's 1 Hz state, its telemetry_id (from
INSERT ... RETURNING) and its frame_seq. The publisher's worker thread later builds the record with
FrameBuilder (vibration waveforms from hf_signals, A/B process channels from hf_channels), serializes and
produces it, so the tick never pays for synthesis or Kafka.

Record fields follow infrastructure/schemas/engine_sensor_frame.avsc: frame_start_ts is the pulse's
event_ts in exact integer microseconds (the join key with engine_telemetry, together with telemetry_id),
samples are packed little-endian float32 bytes, channels come in catalog order (fan_frame_vel and
core_trf_vel first, then the A and B sensor of every process parameter).

frame_seq is the 1-based ordinal of the pulse among its engine's telemetry rows: FrameSequencer starts at
the engine's row count at start-up and advances for every committed pulse, also when the frame is later
dropped, so consumers see exactly the lost frames as gaps, across restarts too.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from fleet_runtime import committed_pulses
from hf_channels import PROCESS_CHANNELS, ProcessSensors, egt_split, parameter_means
from hf_signals import VIBRATION_CHANNELS, VIBRATION_UNIT, frame_state, synthesize_vibration

if TYPE_CHECKING:
    from engine_generator import Pulse
    from fleet_runtime import WriteOp
    from hf_config import HfConfig

logger = logging.getLogger("hf_frames")

SCHEMA_VERSION = 1
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
SAMPLE_DTYPE = "<f4"
TOPIC_MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # max.message.bytes of the HF topics (infrastructure/kafka)
FRAME_SIZE_HEADROOM = 64 * 1024  # record-batch and wire-format framing around one frame
HEADER_BYTES = 128  # upper bound of the fixed fields: ids, timestamps, rates, spool speeds


def epoch_micros(ts: datetime) -> int:
    """Exact integer microseconds since the epoch of an aware datetime (Avro timestamp-micros)."""
    if ts.tzinfo is None:
        raise ValueError(f"naive timestamp {ts.isoformat()}: frames need an aware UTC event time")
    return (ts - EPOCH) // timedelta(microseconds=1)


def pack_samples(samples: np.ndarray) -> bytes:
    """Little-endian float32 bytes of ``samples`` (the samples_f32le encoding)."""
    return np.asarray(samples, dtype=SAMPLE_DTYPE).tobytes()


@dataclass(frozen=True, slots=True)
class FrameRequest:
    """Everything needed to build one frame later, on the publisher's thread: the pulse's 1 Hz state plus
    its identity in the frame stream."""

    engine_id: str
    engine_model: str
    lifecycle_id: int
    cycle_number: int
    telemetry_id: int
    frame_seq: int
    frame_start: datetime
    rpm: float
    core_temp_c: float
    egt_c: float
    vibration_mm_s: float
    fuel_flow_kg_h: float
    oil_pressure_psi: float
    wear_factor: float
    theta: float
    failure_mode: str

    @classmethod
    def from_pulse(cls, pulse: Pulse, engine_model: str, telemetry_id: int, frame_seq: int) -> FrameRequest:
        """Request for the frame of a committed ``pulse``."""
        return cls(
            engine_id=pulse.engine_id,
            engine_model=engine_model,
            lifecycle_id=pulse.lifecycle_id,
            cycle_number=pulse.cycle_number,
            telemetry_id=telemetry_id,
            frame_seq=frame_seq,
            frame_start=pulse.event_ts,
            rpm=pulse.rpm,
            core_temp_c=pulse.core_temp_c,
            egt_c=pulse.egt_c,
            vibration_mm_s=pulse.vibration_mm_s,
            fuel_flow_kg_h=pulse.fuel_flow_kg_h,
            oil_pressure_psi=pulse.oil_pressure_psi,
            wear_factor=pulse.wear_factor,
            theta=pulse.theta,
            failure_mode=pulse.failure_mode,
        )


def _varint_bytes(value: int) -> int:
    """Length of an Avro zig-zag varint."""
    return max(1, ((2 * value).bit_length() + 6) // 7)


def _channel_bytes(name: str, unit: str, samples: int) -> int:
    payload = 4 * samples
    strings = sum(_varint_bytes(len(text)) + len(text.encode()) for text in (name, unit))
    return strings + _varint_bytes(payload) + payload


def estimated_frame_bytes(config: HfConfig) -> int:
    """Serialized size of one frame under ``config`` (Confluent framing included; within a few bytes)."""
    vibration = sum(
        _channel_bytes(channel.name, VIBRATION_UNIT, config.vibration_samples)
        for channel in VIBRATION_CHANNELS[: config.vibration_channels]
    )
    process = sum(_channel_bytes(name, unit, config.process_samples) for name, unit in PROCESS_CHANNELS)
    return HEADER_BYTES + vibration + process


def check_frame_size(config: HfConfig) -> int:
    """Estimated frame size; ValueError when a frame would not fit the topic's 8 MB message limit."""
    size = estimated_frame_bytes(config)
    if size > TOPIC_MAX_MESSAGE_BYTES - FRAME_SIZE_HEADROOM:
        raise ValueError(
            f"one HF frame would be {size / 1e6:.1f} MB, above the topic limit of "
            f"{TOPIC_MAX_MESSAGE_BYTES / 1e6:.1f} MB: lower SIM_HF_SAMPLE_RATE_HZ, SIM_HF_VIB_CHANNELS or "
            "SIM_HF_FRAME_MS"
        )
    return size


class FrameBuilder:
    """Builds the Avro record of a frame request. Not thread-safe: one builder per producing thread."""

    def __init__(self, config: HfConfig, seed: int) -> None:
        self.config = config
        self.frame_bytes = check_frame_size(config)
        self._rng = np.random.default_rng([seed % 2**64, zlib.crc32(b"hf-frames")])
        self._sensors = ProcessSensors(seed)
        self._vibration = [channel.name for channel in VIBRATION_CHANNELS[: config.vibration_channels]]

    def build(self, request: FrameRequest) -> dict[str, Any]:
        """The EngineSensorFrame record (a dict in schema field order) of ``request``."""
        config = self.config
        state = frame_state(request, request.engine_model, self._rng)
        vibration = synthesize_vibration(
            state, config.sample_rate_hz, config.vibration_samples, config.vibration_channels, self._rng
        )
        process = self._sensors.sample(
            request.engine_id,
            parameter_means(state, vibration),
            egt_split(state),
            config.process_samples,
            self._rng,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_id": request.engine_id,
            "engine_model": request.engine_model,
            "lifecycle_id": request.lifecycle_id,
            "cycle_number": request.cycle_number,
            "telemetry_id": request.telemetry_id,
            "frame_seq": request.frame_seq,
            "frame_start_ts": epoch_micros(request.frame_start),
            "frame_duration_ms": config.frame_ms,
            "n1_rpm": state.n1_rpm,
            "n2_rpm": state.n2_rpm,
            "bpfo_order": state.bpfo_order,
            "vibration_sample_rate_hz": config.sample_rate_hz,
            "vibration": [
                {"channel": name, "unit": VIBRATION_UNIT, "samples_f32le": pack_samples(samples)}
                for name, samples in zip(self._vibration, vibration.waveforms, strict=True)
            ],
            "process_sample_rate_hz": config.process_rate_hz,
            "process": [
                {"parameter": name, "unit": unit, "samples_f32le": pack_samples(samples)}
                for (name, unit), samples in zip(PROCESS_CHANNELS, process, strict=True)
            ],
        }


class FrameSequencer:
    """Per-engine frame_seq: the 1-based ordinal of each committed pulse among the engine's telemetry rows."""

    def __init__(self, telemetry_counts: Mapping[str, int]) -> None:
        self._last = dict(telemetry_counts)

    def next(self, engine_id: str) -> int:
        """frame_seq of the engine's next committed pulse."""
        self._last[engine_id] = self._last.get(engine_id, 0) + 1
        return self._last[engine_id]


class FrameSubmitter(Protocol):
    """Where FrameTap hands its requests (hf_producer.FramePublisher)."""

    def submit(self, requests: Sequence[FrameRequest]) -> None:
        """Queue live frames (best effort)."""

    def submit_backfill(self, requests: Sequence[FrameRequest]) -> None:
        """Queue backfilled history (never dropped for a full buffer)."""


class FrameTap:
    """Commit hooks of the fleet: every pulse a committed transaction wrote becomes a frame request.

    Errors are logged and never reach the fleet: the HF path must not stop the lossless 1 Hz path.
    """

    def __init__(
        self,
        publisher: FrameSubmitter,
        engine_models: Mapping[str, str],
        telemetry_counts: Mapping[str, int],
    ) -> None:
        self._publisher = publisher
        self._models = dict(engine_models)
        self._sequencer = FrameSequencer(telemetry_counts)
        self.errors = 0

    def requests(self, ops: Sequence[WriteOp], results: Sequence[Any]) -> list[FrameRequest]:
        """Frame requests of a committed transaction, in event order (advances the frame sequence)."""
        return [
            FrameRequest.from_pulse(
                pulse, self._models[pulse.engine_id], telemetry_id, self._sequencer.next(pulse.engine_id)
            )
            for pulse, telemetry_id in committed_pulses(ops, results)
        ]

    def live(self, ops: Sequence[WriteOp], results: Sequence[Any]) -> None:
        """Commit hook of the live loop."""
        try:
            self._publisher.submit(self.requests(ops, results))
        except Exception:
            self.errors += 1
            logger.exception("HF frames of a live tick were lost")

    def backfill(self, ops: Sequence[WriteOp], results: Sequence[Any]) -> None:
        """Commit hook of an engine's backfilled history."""
        try:
            requests = self.requests(ops, results)
            self._publisher.submit_backfill(requests)
        except Exception:
            self.errors += 1
            logger.exception("HF frames of a backfilled history were lost")
            return
        engine_id = requests[0].engine_id if requests else "?"
        logger.info("Queued %d backfill HF frames of %s", len(requests), engine_id)
