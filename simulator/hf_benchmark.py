"""HF load test (``engine_generator.py --benchmark``): what one producer process sustains, and what that
means for the "30 TB per 30 minutes" scale of full-fidelity engine data.

Synthetic engines BENCH-0001.. (no PostgreSQL, no pacing) cycle through the engine types, failure modes and
a spread of wear levels; their frames go to HF_BENCHMARK_TOPIC (engine_hf.benchmark: ten minutes of
retention, no consumer) with the live producer's settings. Measured over --benchmark-seconds (MB = 10^6 B):

    frames/s, samples/s       frames built, serialized and handed to the Kafka client per second
    serialized MB/s           Confluent wire-format bytes per second
    delivered MB/s            bytes Kafka acknowledged (delivery reports) per second, the final flush included
    wire MB/s                 bytes the client sent to the brokers after compression (client statistics)
    build+serialize p50/p99   per-frame waveform synthesis plus Avro serialization time
    producer errors           failed produce calls and deliveries, frames unacknowledged after the flush

Projection: one engine at full fidelity (25.6 kHz x 16 vibration channels plus 1000 process channels at
16 Hz, float32) streams 1.70 MB/s of samples, and 30 TB per 30 min is 16.7 GB/s, about 9,800 such engines.
Engines per producer process = delivered MB/s / 1.70 MB/s, producer processes = 16.7 GB/s / delivered
MB/s, brokers at an assumed 100 MB/s ingest each = 167 for the raw stream (fewer after compression). The
producer's cost grows with the samples, so measure at full fidelity for the closest projection:
``--benchmark-sample-rate 25600 --benchmark-vib-channels 16``.

Output: a human table and one JSON line on stdout. Exit code 0 when every frame was delivered; 1 when frames
failed or Kafka, the topic or the registry stayed unavailable for BENCHMARK_READY_SECONDS.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from hf_channels import PROCESS_CHANNELS
from hf_frames import FrameBuilder, FrameRequest
from hf_kafka import open_kafka_sink
from hf_producer import READY_TIMEOUT_SECONDS, FrameSink, backoff_delay, first_line
from hf_signals import MODE_SIGNATURES

if TYPE_CHECKING:
    from hf_config import HfConfig

logger = logging.getLogger("hf_benchmark")

SAMPLE_BYTES = 4  # float32
TARGET_BYTES, TARGET_SECONDS = 30e12, 30 * 60.0  # "30 TB per 30 minutes"
BROKER_INGEST_BYTES_PER_S = 100e6  # assumed sustained ingest of one broker's partition set
BENCHMARK_READY_SECONDS = 60.0
FLUSH_SECONDS = 30.0
STATISTICS_SETTLE_SECONDS = 1.5
BUFFER_RETRY_SECONDS = 0.01
CLIENT_ID = "engine-simulator-hf-benchmark"


@dataclass(frozen=True)
class Fidelity:
    """Channel layout of one engine's HF stream."""

    vibration_rate_hz: int
    vibration_channels: int
    process_rate_hz: int
    process_channels: int

    @property
    def samples_per_second(self) -> int:
        """Samples one engine produces per second."""
        return self.vibration_rate_hz * self.vibration_channels + self.process_rate_hz * self.process_channels

    @property
    def bytes_per_second(self) -> int:
        """Sample payload of one engine per second (float32)."""
        return SAMPLE_BYTES * self.samples_per_second


FULL_FIDELITY = Fidelity(
    vibration_rate_hz=25_600, vibration_channels=16, process_rate_hz=16, process_channels=1000
)


class BenchmarkSink(FrameSink, Protocol):
    """A frame sink that also reports the bytes it sent to the brokers (None without statistics)."""

    @property
    def wire_bytes(self) -> int | None:
        """Bytes sent to the brokers so far."""


class SyntheticFleet:
    """Frame requests of the engines BENCH-0001.. in round-robin order (one frame period apart per engine)."""

    def __init__(
        self, engines: int, nominal: Mapping[str, Mapping[str, float]], frame_ms: int, start: datetime
    ) -> None:
        models, modes = sorted(nominal), sorted(MODE_SIGNATURES)
        self._templates = [
            _template(index, engines, models[index % len(models)], modes[index % len(modes)], nominal, start)
            for index in range(engines)
        ]
        self._period = timedelta(milliseconds=frame_ms)
        self._sequence = [0] * engines
        self._telemetry_id = 0
        self._next = 0

    def next(self) -> FrameRequest:
        """The next engine's next frame request."""
        index = self._next
        self._next = (index + 1) % len(self._templates)
        self._sequence[index] += 1
        self._telemetry_id += 1
        template = self._templates[index]
        return replace(
            template,
            telemetry_id=self._telemetry_id,
            frame_seq=self._sequence[index],
            frame_start=template.frame_start + self._sequence[index] * self._period,
        )


def _template(
    index: int,
    engines: int,
    model: str,
    mode: str,
    nominal: Mapping[str, Mapping[str, float]],
    start: datetime,
) -> FrameRequest:
    """Engine ``index``: its type's nominal state at wear (index + 0.5) / engines (vibration on the fleet's
    mean wear curve), so the load covers healthy and failing engines alike."""
    wear = (index + 0.5) / engines
    base = nominal[model]
    return FrameRequest(
        engine_id=f"BENCH-{index + 1:04d}",
        engine_model=model,
        lifecycle_id=1,
        cycle_number=0,
        telemetry_id=0,
        frame_seq=0,
        frame_start=start,
        rpm=base["rpm"],
        core_temp_c=base["core_temp_c"],
        egt_c=base["egt_c"],
        vibration_mm_s=base["vibration_mm_s"] * (1.0 + 3.5 * wear**2.5),
        fuel_flow_kg_h=base["fuel_flow_kg_h"],
        oil_pressure_psi=base["oil_pressure_psi"],
        wear_factor=wear,
        theta=1.0,
        failure_mode=mode,
    )


@dataclass
class Measurement:
    """Raw counts and timings of one load test."""

    frames: int = 0
    serialized_bytes: int = 0
    delivered_frames: int = 0
    delivered_bytes: int = 0
    errors: int = 0
    produce_seconds: float = 0.0
    total_seconds: float = 0.0
    wire_bytes: int | None = None
    latencies_s: list[float] = field(default_factory=list)

    def delivered(self, size: int, error: Any, _message: Any) -> None:
        """Delivery report of one frame of ``size`` bytes."""
        if error is None:
            self.delivered_frames += 1
            self.delivered_bytes += size
        else:
            self.errors += 1


def wait_ready(
    sink: FrameSink,
    stop: threading.Event,
    limit_s: float = BENCHMARK_READY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Retry the sink's readiness check with backoff for up to ``limit_s`` seconds."""
    deadline, attempt = clock() + limit_s, 0
    while not stop.is_set():
        try:
            reason = sink.check_ready(READY_TIMEOUT_SECONDS)
        except Exception as exc:
            reason = first_line(exc)
        if reason is None:
            return True
        attempt += 1
        remaining = deadline - clock()
        if remaining <= 0:
            logger.error("Kafka not ready after %g s, giving up: %s", limit_s, reason)
            return False
        logger.warning("Waiting for Kafka (attempt %d): %s", attempt, reason)
        stop.wait(min(backoff_delay(attempt), remaining))
    return False


def measure(
    sink: BenchmarkSink,
    builder: FrameBuilder,
    fleet: SyntheticFleet,
    seconds: float,
    stop: threading.Event,
    clock: Callable[[], float] = time.perf_counter,
) -> Measurement:
    """Build, serialize and produce frames as fast as possible for ``seconds``, then flush."""
    result = Measurement()
    start = clock()
    while not stop.is_set() and (began := clock()) - start < seconds:
        request = fleet.next()
        value = sink.serialize(builder.build(request))
        result.latencies_s.append(clock() - began)
        result.frames += 1
        result.serialized_bytes += len(value)
        _produce(
            sink, request.engine_id.encode(), value, functools.partial(result.delivered, len(value)), result
        )
        sink.poll(0)
    result.produce_seconds = clock() - start
    result.errors += sink.flush(FLUSH_SECONDS)
    result.total_seconds = clock() - start
    settle = clock() + STATISTICS_SETTLE_SECONDS  # one more statistics event covers the flush
    while clock() < settle:
        sink.poll(0.1)
    result.wire_bytes = sink.wire_bytes
    return result


def _produce(sink: FrameSink, key: bytes, value: bytes, callback: Any, result: Measurement) -> None:
    while True:
        try:
            sink.produce(key, value, callback)
            return
        except BufferError:
            sink.poll(BUFFER_RETRY_SECONDS)  # the client's queue is full: the load test runs at its limit
        except Exception as exc:
            result.errors += 1
            logger.warning("produce failed: %s", first_line(exc))
            return


def projection(delivered_bytes_per_s: float, compression_ratio: float | None) -> dict[str, Any]:
    """Full-fidelity scale-out implied by one producer's delivered throughput (see the module doc)."""
    per_engine = FULL_FIDELITY.bytes_per_second
    target = TARGET_BYTES / TARGET_SECONDS
    producers = math.ceil(target / delivered_bytes_per_s) if delivered_bytes_per_s > 0 else None
    compressed = (
        math.ceil(target / compression_ratio / BROKER_INGEST_BYTES_PER_S) if compression_ratio else None
    )
    return {
        "full_fidelity_mb_s_per_engine": round(per_engine / 1e6, 4),
        "engines_per_producer": round(delivered_bytes_per_s / per_engine, 2),
        "target_gb_s": round(target / 1e9, 2),
        "engines_at_target": round(target / per_engine),
        "producers_needed": producers,
        "broker_ingest_mb_s": BROKER_INGEST_BYTES_PER_S / 1e6,
        "brokers_needed_raw": math.ceil(target / BROKER_INGEST_BYTES_PER_S),
        "brokers_needed_compressed": compressed,
    }


def summarize(result: Measurement, config: HfConfig, topic: str, engines: int) -> dict[str, Any]:
    """Every reported number (the JSON line), derived from the raw measurement."""
    samples = (
        config.vibration_channels * config.vibration_samples + len(PROCESS_CHANNELS) * config.process_samples
    )
    produce_s, total_s = max(result.produce_seconds, 1e-9), max(result.total_seconds, 1e-9)
    delivered_rate = result.delivered_bytes / total_s
    ratio = (
        result.delivered_bytes / result.wire_bytes if result.wire_bytes and result.delivered_bytes else None
    )
    p50, p99 = np.percentile(result.latencies_s, [50, 99]) * 1e3 if result.latencies_s else (0.0, 0.0)
    return {
        "topic": topic,
        "engines": engines,
        "compression": config.compression,
        "vibration_sample_rate_hz": config.sample_rate_hz,
        "vibration_channels": config.vibration_channels,
        "process_sample_rate_hz": config.process_rate_hz,
        "process_channels": len(PROCESS_CHANNELS),
        "frame_ms": config.frame_ms,
        "seconds": round(result.produce_seconds, 3),
        "flush_seconds": round(result.total_seconds - result.produce_seconds, 3),
        "frames": result.frames,
        "delivered_frames": result.delivered_frames,
        "errors": result.errors,
        "frames_per_s": round(result.frames / produce_s, 2),
        "samples_per_s": round(result.frames * samples / produce_s),
        "serialized_mb_s": round(result.serialized_bytes / produce_s / 1e6, 3),
        "delivered_mb_s": round(delivered_rate / 1e6, 3),
        "wire_mb_s": round(result.wire_bytes / total_s / 1e6, 3) if result.wire_bytes else None,
        "compression_ratio": round(ratio, 3) if ratio else None,
        "avg_frame_kb": round(result.serialized_bytes / result.frames / 1e3, 2) if result.frames else 0.0,
        "build_serialize_ms_p50": round(float(p50), 3),
        "build_serialize_ms_p99": round(float(p99), 3),
        "projection": projection(delivered_rate, ratio),
    }


def format_report(summary: Mapping[str, Any]) -> str:
    """The human-readable table of a summary: the measurement, then the full-fidelity projection."""
    p = summary["projection"]
    wire = "n/a"
    if summary["wire_mb_s"] is not None:
        wire = f"{summary['wire_mb_s']:.2f} (compression {summary['compression_ratio']}x)"
    latency = f"{summary['build_serialize_ms_p50']:.2f} / {summary['build_serialize_ms_p99']:.2f} ms"
    measured = [
        ("duration", f"{summary['seconds']:.1f} s producing + {summary['flush_seconds']:.1f} s flush"),
        ("frames produced / delivered", f"{summary['frames']:,} / {summary['delivered_frames']:,}"),
        ("frames/s", f"{summary['frames_per_s']:,.1f}"),
        ("samples/s", f"{summary['samples_per_s'] / 1e6:,.2f} M"),
        ("average frame", f"{summary['avg_frame_kb']:.1f} KB"),
        ("serialized MB/s", f"{summary['serialized_mb_s']:.2f}"),
        ("delivered MB/s", f"{summary['delivered_mb_s']:.2f}"),
        (f"wire MB/s ({summary['compression']})", wire),
        ("build+serialize p50 / p99", latency),
        ("producer errors", str(summary["errors"])),
    ]
    brokers = f"{p['brokers_needed_raw']} raw"
    if p["brokers_needed_compressed"] is not None:
        brokers += f", {p['brokers_needed_compressed']} after compression"
    producers = "n/a" if p["producers_needed"] is None else f"{p['producers_needed']:,}"
    projected = [
        ("full fidelity per engine", f"{p['full_fidelity_mb_s_per_engine']:.2f} MB/s"),
        ("engines per producer process", f"{p['engines_per_producer']:,.1f}"),
        (
            "30 TB per 30 min",
            f"{p['target_gb_s']:.1f} GB/s = {p['engines_at_target']:,} full-fidelity engines",
        ),
        ("producer processes needed", producers),
        (f"brokers at {p['broker_ingest_mb_s']:.0f} MB/s each", brokers),
    ]
    width = max(len(name) for name, _ in measured + projected)
    full = FULL_FIDELITY
    return "\n".join(
        [
            f"HF load test: {summary['engines']} engines, {summary['vibration_channels']} x "
            f"{summary['vibration_sample_rate_hz']} Hz vibration + {summary['process_channels']} x "
            f"{summary['process_sample_rate_hz']} Hz process channels, {summary['frame_ms']} ms frames "
            f"-> {summary['topic']}",
            *(f"  {name:<{width}}  {value}" for name, value in measured),
            f"Projection to full fidelity ({full.vibration_rate_hz / 1000:g} kHz x {full.vibration_channels} "
            f"vibration + {full.process_channels} process channels at {full.process_rate_hz} Hz), "
            "from the delivered MB/s:",
            *(f"  {name:<{width}}  {value}" for name, value in projected),
        ]
    )


def run_benchmark(
    config: HfConfig,
    options: Any,
    seed: int,
    nominal: Mapping[str, Mapping[str, float]],
    stop: threading.Event,
    sink_factory: Callable[[str], BenchmarkSink] | None = None,
) -> int:
    """The load test of the module doc; returns the process exit code."""
    topic = options.benchmark_topic or config.benchmark_topic
    factory = sink_factory or functools.partial(_kafka_sink, config)
    sink = factory(topic)
    try:
        logger.info(
            "HF load test: %d engines for %.0f s -> %s",
            options.benchmark_engines,
            options.benchmark_seconds,
            topic,
        )
        if not wait_ready(sink, stop, BENCHMARK_READY_SECONDS):
            return 1
        fleet = SyntheticFleet(options.benchmark_engines, nominal, config.frame_ms, datetime.now(UTC))
        result = measure(sink, FrameBuilder(config, seed), fleet, options.benchmark_seconds, stop)
    finally:
        sink.close()
    summary = summarize(result, config, topic, options.benchmark_engines)
    print(format_report(summary), flush=True)
    print(json.dumps({"hf_benchmark": summary}, separators=(",", ":")), flush=True)
    return 0 if result.errors == 0 and result.delivered_frames == result.frames else 1


def _kafka_sink(config: HfConfig, topic: str) -> BenchmarkSink:
    return open_kafka_sink(config, topic, client_id=CLIENT_ID, statistics=True)
