"""Settings of the high-frequency (HF) sensor-frame path: frame shape, Kafka, Schema Registry, buffer.

Read from SIM_HF_* plus the Kafka and Schema Registry variables of CONTRACT_V2 section 4. The host defaults
point at the published ports of the local stack and at this repository's schema file, so the simulator also
runs outside compose. A leaf module (standard library and fleet_config only): reading the settings never
imports numpy or the Kafka client.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from fleet_config import env_value

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
)
COMPRESSION_TYPES = ("none", "gzip", "snappy", "lz4", "zstd")
SAMPLE_RATE_RANGE_HZ = (1024, 102_400)  # Nyquist must clear the BPFO harmonics; 100 kHz is lab-grade DAQ
PROCESS_RATE_RANGE_HZ = (1, 1000)
FRAME_MS_RANGE = (100, 10_000)
VIBRATION_CHANNEL_RANGE = (2, 16)  # fan_frame_vel and core_trf_vel always, then the hf_signals catalog
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def boolean(raw: str) -> bool:
    """true/false, 1/0, yes/no or on/off in any case."""
    word = raw.strip().lower()
    if word in TRUE_WORDS | FALSE_WORDS:
        return word in TRUE_WORDS
    raise ValueError(f"{raw!r} is not a boolean")


def _in_range(name: str, value: int, bounds: tuple[int, int]) -> tuple[bool, str]:
    return bounds[0] <= value <= bounds[1], f"{name} must be within {bounds[0]}..{bounds[1]}"


@dataclass(frozen=True)
class HfConfig:
    """HF path settings; the defaults are the contract defaults (laptop scale)."""

    enabled: bool = True
    bootstrap_servers: str = "localhost:9092"
    schema_registry_url: str = "http://localhost:8081"
    topic: str = "engine_hf.sensor_frames"
    benchmark_topic: str = "engine_hf.benchmark"
    schema_path: Path = DEFAULT_SCHEMA_PATH
    sample_rate_hz: int = 2048
    vibration_channels: int = 2
    process_rate_hz: int = 8
    frame_ms: int = 1000
    compression: str = "zstd"
    queue_max_mb: int = 256

    def __post_init__(self) -> None:
        checks = (
            (bool(self.bootstrap_servers.strip()), "KAFKA_BOOTSTRAP_SERVERS is empty"),
            (bool(self.schema_registry_url.strip()), "SCHEMA_REGISTRY_URL is empty"),
            (bool(self.topic.strip() and self.benchmark_topic.strip()), "HF topic names must not be empty"),
            _in_range("SIM_HF_SAMPLE_RATE_HZ", self.sample_rate_hz, SAMPLE_RATE_RANGE_HZ),
            _in_range("SIM_HF_VIB_CHANNELS", self.vibration_channels, VIBRATION_CHANNEL_RANGE),
            _in_range("SIM_HF_PROCESS_RATE_HZ", self.process_rate_hz, PROCESS_RATE_RANGE_HZ),
            _in_range("SIM_HF_FRAME_MS", self.frame_ms, FRAME_MS_RANGE),
            (
                self.sample_rate_hz * self.frame_ms % 1000 == 0
                and self.process_rate_hz * self.frame_ms % 1000 == 0,
                "SIM_HF_FRAME_MS must hold a whole number of vibration and process samples",
            ),
            (self.compression in COMPRESSION_TYPES, f"SIM_HF_COMPRESSION must be one of {COMPRESSION_TYPES}"),
            (self.queue_max_mb >= 1, "SIM_HF_QUEUE_MAX_MB must be >= 1"),
        )
        problems = [message for valid, message in checks if not valid]
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HfConfig:
        """Read the HF variables; unset or blank ones keep the defaults."""
        env = os.environ if env is None else env
        return cls(
            enabled=env_value(env, "SIM_HF_ENABLED", boolean, cls.enabled),
            bootstrap_servers=env.get("KAFKA_BOOTSTRAP_SERVERS", "").strip() or cls.bootstrap_servers,
            schema_registry_url=env.get("SCHEMA_REGISTRY_URL", "").strip() or cls.schema_registry_url,
            topic=env.get("HF_TOPIC", "").strip() or cls.topic,
            benchmark_topic=env.get("HF_BENCHMARK_TOPIC", "").strip() or cls.benchmark_topic,
            schema_path=Path(env.get("HF_SCHEMA_PATH", "").strip() or cls.schema_path),
            sample_rate_hz=env_value(env, "SIM_HF_SAMPLE_RATE_HZ", int, cls.sample_rate_hz),
            vibration_channels=env_value(env, "SIM_HF_VIB_CHANNELS", int, cls.vibration_channels),
            process_rate_hz=env_value(env, "SIM_HF_PROCESS_RATE_HZ", int, cls.process_rate_hz),
            frame_ms=env_value(env, "SIM_HF_FRAME_MS", int, cls.frame_ms),
            compression=env.get("SIM_HF_COMPRESSION", "").strip().lower() or cls.compression,
            queue_max_mb=env_value(env, "SIM_HF_QUEUE_MAX_MB", int, cls.queue_max_mb),
        )

    def for_benchmark(self, sample_rate_hz: int | None, vibration_channels: int | None) -> HfConfig:
        """These settings with the load test's optional overrides (validated like the variables)."""
        return replace(
            self,
            sample_rate_hz=sample_rate_hz or self.sample_rate_hz,
            vibration_channels=vibration_channels or self.vibration_channels,
        )

    @property
    def vibration_samples(self) -> int:
        """Samples per vibration channel and frame."""
        return self.sample_rate_hz * self.frame_ms // 1000

    @property
    def process_samples(self) -> int:
        """Samples per process channel and frame."""
        return self.process_rate_hz * self.frame_ms // 1000

    @property
    def queue_max_bytes(self) -> int:
        """SIM_HF_QUEUE_MAX_MB in bytes."""
        return self.queue_max_mb * 1024 * 1024

    def summary(self) -> str:
        """One-line description for the start-up log."""
        return (
            f"{self.vibration_channels} vibration channels at {self.sample_rate_hz} Hz, process channels at "
            f"{self.process_rate_hz} Hz, {self.frame_ms} ms frames -> {self.topic} on "
            f"{self.bootstrap_servers} "
            f"({self.compression}, buffer {self.queue_max_mb} MiB), schema {self.schema_path} via "
            f"{self.schema_registry_url}"
        )
