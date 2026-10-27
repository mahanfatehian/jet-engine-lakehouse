"""HF load test: the synthetic fleet, the measurement and its derived rates, the report table with its JSON
line, the 30 TB per 30 min projection, the exit codes and the command line (fake Kafka sink, no network)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import engine_generator as eg
import hf_benchmark as hb
import pytest
from fleet_config import parse_args
from hf_config import HfConfig

NOMINAL = {model: dict(zip(eg.CHANNELS, values, strict=True)) for model, values in eg.ENGINE_TYPES.values()}
START = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


class FakeBenchmarkSink:
    """Acknowledges every frame on poll; the wire carries 80 % of the serialized bytes (zstd stand-in)."""

    topic = "engine_hf.benchmark"
    fatal_error = None

    def __init__(self, *, ready: bool = True, failing_engine: str | None = None) -> None:
        self.ready = ready
        self.failing_engine = failing_engine
        self.schema_id: int | None = None
        self.pending: list[tuple[bytes, bytes, Any]] = []
        self.keys: list[bytes] = []
        self.wire = 0
        self.closed = False

    def check_ready(self, timeout: float) -> str | None:
        return None if self.ready else "topic engine_hf.benchmark does not exist yet"

    def serialize(self, record: dict[str, Any]) -> bytes:
        identity = {name: record[name] for name in ("engine_id", "frame_seq", "telemetry_id")}
        return json.dumps(identity).encode().ljust(20_000, b" ")

    def produce(self, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.pending.append((key, value, on_delivery))

    def poll(self, timeout: float) -> int:
        served, self.pending = self.pending, []
        for key, value, callback in served:
            self.keys.append(key)
            self.wire += int(0.8 * len(value))
            failed = key.decode() == self.failing_engine
            callback("KafkaError{code=_MSG_TIMED_OUT}" if failed else None, None)
        return len(served)

    def flush(self, timeout: float) -> int:
        self.poll(0)
        return 0

    @property
    def wire_bytes(self) -> int:
        return self.wire

    def close(self) -> None:
        self.closed = True


def options(**overrides: Any) -> SimpleNamespace:
    values = {"benchmark_topic": None, "benchmark_engines": 4, "benchmark_seconds": 0.3}
    return SimpleNamespace(**(values | overrides))


def test_full_fidelity_and_the_thirty_terabyte_projection() -> None:
    full = hb.FULL_FIDELITY
    assert (full.vibration_rate_hz, full.vibration_channels, full.process_rate_hz, full.process_channels) == (
        25_600,
        16,
        16,
        1000,
    )
    assert full.samples_per_second == 425_600 and full.bytes_per_second == 1_702_400
    projected = hb.projection(50e6, 1.25)
    assert projected == {
        "full_fidelity_mb_s_per_engine": 1.7024,
        "engines_per_producer": 29.37,  # 50 MB/s / 1.7024 MB/s
        "target_gb_s": 16.67,  # 30e12 B / 1800 s
        "engines_at_target": 9790,
        "producers_needed": 334,  # ceil(16.67 GB/s / 50 MB/s)
        "broker_ingest_mb_s": 100.0,
        "brokers_needed_raw": 167,
        "brokers_needed_compressed": 134,  # ceil(16.67 GB/s / 1.25 / 100 MB/s)
    }
    nothing = hb.projection(0.0, None)
    assert (
        nothing["engines_per_producer"],
        nothing["producers_needed"],
        nothing["brokers_needed_compressed"],
    ) == (
        0.0,
        None,
        None,
    )


def test_summary_derives_rates_sizes_and_latency_percentiles() -> None:
    measured = hb.Measurement(
        frames=1000,
        serialized_bytes=20_000_000,
        delivered_frames=1000,
        delivered_bytes=20_000_000,
        produce_seconds=10.0,
        total_seconds=12.5,
        wire_bytes=16_000_000,
        latencies_s=[index / 1000 for index in range(1, 101)],
    )
    summary = hb.summarize(measured, HfConfig(), "engine_hf.benchmark", 24)
    assert summary["frames_per_s"] == 100.0 and summary["samples_per_s"] == 100 * (2 * 2048 + 80 * 8)
    assert (summary["serialized_mb_s"], summary["delivered_mb_s"], summary["wire_mb_s"]) == (2.0, 1.6, 1.28)
    assert (summary["compression_ratio"], summary["avg_frame_kb"]) == (1.25, 20.0)
    assert (summary["build_serialize_ms_p50"], summary["build_serialize_ms_p99"]) == (50.5, 99.01)
    assert (summary["seconds"], summary["flush_seconds"], summary["errors"]) == (10.0, 2.5, 0)
    assert summary["projection"] == hb.projection(1.6e6, 1.25)
    assert (summary["process_channels"], summary["vibration_channels"], summary["compression"]) == (
        80,
        2,
        "zstd",
    )


def test_run_prints_a_table_and_one_json_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(hb, "STATISTICS_SETTLE_SECONDS", 0.0)
    sink = FakeBenchmarkSink()
    code = hb.run_benchmark(HfConfig(), options(), 42, NOMINAL, threading.Event(), lambda topic: sink)
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and sink.closed
    assert lines[0] == (
        "HF load test: 4 engines, 2 x 2048 Hz vibration + 80 x 8 Hz process channels, 1000 ms frames "
        "-> engine_hf.benchmark"
    )
    text = "\n".join(lines)
    for label in ("frames/s", "samples/s", "serialized MB/s", "delivered MB/s", "wire MB/s (zstd)"):
        assert f"  {label}" in text
    assert "build+serialize p50 / p99" in text and "producer errors" in text
    assert "Projection to full fidelity (25.6 kHz x 16 vibration + 1000 process channels at 16 Hz)" in text
    assert "30 TB per 30 min" in text and "16.7 GB/s = 9,790 full-fidelity engines" in text
    assert "brokers at 100 MB/s each" in text and "167 raw" in text
    result = json.loads(lines[-1])["hf_benchmark"]
    assert result["frames"] == result["delivered_frames"] > 4 and result["errors"] == 0
    assert result["compression_ratio"] == 1.25 and result["topic"] == "engine_hf.benchmark"
    assert {key.decode() for key in sink.keys} == {"BENCH-0001", "BENCH-0002", "BENCH-0003", "BENCH-0004"}


def test_lost_frames_or_an_unavailable_kafka_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(hb, "STATISTICS_SETTLE_SECONDS", 0.0)
    lossy = FakeBenchmarkSink(failing_engine="BENCH-0002")
    assert hb.run_benchmark(HfConfig(), options(), 42, NOMINAL, threading.Event(), lambda topic: lossy) == 1
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["hf_benchmark"]
    assert result["errors"] > 0 and result["delivered_frames"] < result["frames"]
    monkeypatch.setattr(hb, "BENCHMARK_READY_SECONDS", 0.05)
    monkeypatch.setattr(hb, "backoff_delay", lambda attempt: 0.01)
    absent = FakeBenchmarkSink(ready=False)
    with caplog.at_level(logging.WARNING, logger="hf_benchmark"):
        code = hb.run_benchmark(HfConfig(), options(), 42, NOMINAL, threading.Event(), lambda topic: absent)
    assert code == 1 and absent.closed and capsys.readouterr().out == ""
    assert (
        "Kafka not ready after 0.05 s, giving up: topic engine_hf.benchmark does not exist yet" in caplog.text
    )


def test_a_stop_request_ends_the_measurement_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hb, "STATISTICS_SETTLE_SECONDS", 0.0)
    stop = threading.Event()
    stop.set()
    sink = FakeBenchmarkSink()
    fleet = hb.SyntheticFleet(2, NOMINAL, 1000, START)
    result = hb.measure(sink, hb.FrameBuilder(HfConfig(), 1), fleet, 60.0, stop)
    assert result.frames == 0 and result.produce_seconds < 1.0


def test_synthetic_fleet_cycles_engines_types_modes_and_wear() -> None:
    fleet = hb.SyntheticFleet(8, NOMINAL, 500, START)
    first, second = [fleet.next() for _ in range(8)], [fleet.next() for _ in range(8)]
    assert [request.engine_id for request in first] == [f"BENCH-{n:04d}" for n in range(1, 9)]
    assert [request.engine_model for request in first[:3]] == ["CFM56-7B", "GE90-115B", "JT9D-7R4"]
    assert [request.failure_mode for request in first[:4]] == sorted(hb.MODE_SIGNATURES)
    assert [request.wear_factor for request in first] == [(n + 0.5) / 8 for n in range(8)]
    assert [r.frame_seq for r in first] == [1] * 8 and [r.frame_seq for r in second] == [2] * 8
    assert [r.telemetry_id for r in first + second] == list(range(1, 17))
    assert second[0].frame_start - first[0].frame_start == timedelta(milliseconds=500)
    jt9d = first[2]
    assert (
        jt9d.rpm == NOMINAL["JT9D-7R4"]["rpm"] and jt9d.vibration_mm_s > NOMINAL["JT9D-7R4"]["vibration_mm_s"]
    )


def test_benchmark_options_require_benchmark_and_have_defaults() -> None:
    plain = parse_args([])
    assert not plain.benchmark and (plain.benchmark_seconds, plain.benchmark_engines) == (60.0, 24)
    full = parse_args(
        [
            "--benchmark",
            "--benchmark-seconds",
            "20",
            "--benchmark-engines",
            "48",
            "--benchmark-topic",
            "load.test",
            "--benchmark-sample-rate",
            "25600",
            "--benchmark-vib-channels",
            "16",
        ]
    )
    assert (full.benchmark_seconds, full.benchmark_engines, full.benchmark_topic) == (20.0, 48, "load.test")
    assert (full.benchmark_sample_rate, full.benchmark_vib_channels) == (25_600, 16)
    for argv in (
        ["--benchmark-seconds", "5"],
        ["--benchmark", "--max-ticks", "3"],
        ["--benchmark", "--no-backfill"],
        ["--benchmark", "--benchmark-seconds", "0"],
        ["--benchmark", "--benchmark-engines", "0"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            parse_args(argv)
        assert exit_info.value.code == 2, argv
