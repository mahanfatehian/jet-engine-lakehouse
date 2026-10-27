"""HF path settings and wiring: SIM_HF_* and Kafka variables, the producer configuration, the schema file, the
client callbacks, and how the simulator process opens the path (or leaves it out entirely when disabled)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import engine_generator as eg
import hf_kafka
import pytest
from fleet_config import SimulatorConfig
from hf_config import DEFAULT_SCHEMA_PATH, HfConfig, boolean
from hf_runtime import HfPath, load_hf_config, open_hf_path

REPO = Path(__file__).resolve().parents[2]
COMPOSE_ENV = {
    "KAFKA_BOOTSTRAP_SERVERS": "kafka:29092",
    "SCHEMA_REGISTRY_URL": "http://schema-registry:8081",
    "HF_TOPIC": "engine_hf.sensor_frames",
    "HF_BENCHMARK_TOPIC": "engine_hf.benchmark",
    "HF_SCHEMA_PATH": "/schemas/engine_sensor_frame.avsc",
    "SIM_HF_ENABLED": "true",
    "SIM_HF_SAMPLE_RATE_HZ": "4096",
    "SIM_HF_VIB_CHANNELS": "4",
    "SIM_HF_PROCESS_RATE_HZ": "16",
    "SIM_HF_FRAME_MS": "500",
    "SIM_HF_COMPRESSION": "LZ4",
    "SIM_HF_QUEUE_MAX_MB": "64",
}


def test_defaults_are_the_contract_defaults_for_a_host_run() -> None:
    config = HfConfig.from_env({})
    assert config == HfConfig()
    assert (config.enabled, config.bootstrap_servers, config.schema_registry_url) == (
        True,
        "localhost:9092",
        "http://localhost:8081",
    )
    assert (config.topic, config.benchmark_topic) == ("engine_hf.sensor_frames", "engine_hf.benchmark")
    assert (
        config.schema_path
        == DEFAULT_SCHEMA_PATH
        == REPO / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
    )
    assert config.schema_path.is_file()
    assert (config.sample_rate_hz, config.vibration_channels, config.process_rate_hz, config.frame_ms) == (
        2048,
        2,
        8,
        1000,
    )
    assert (config.compression, config.queue_max_mb, config.queue_max_bytes) == ("zstd", 256, 256 * 2**20)
    assert (config.vibration_samples, config.process_samples) == (2048, 8)


def test_every_variable_is_read_like_compose_passes_it() -> None:
    config = HfConfig.from_env(COMPOSE_ENV)
    assert (config.bootstrap_servers, config.schema_registry_url) == (
        "kafka:29092",
        "http://schema-registry:8081",
    )
    assert config.schema_path == Path("/schemas/engine_sensor_frame.avsc")
    assert (config.sample_rate_hz, config.vibration_channels, config.process_rate_hz, config.frame_ms) == (
        4096,
        4,
        16,
        500,
    )
    assert (config.vibration_samples, config.process_samples, config.compression) == (2048, 8, "lz4")
    assert config.queue_max_mb == 64 and "4 vibration channels at 4096 Hz" in config.summary()
    for word, value in (
        ("false", False),
        ("0", False),
        ("No", False),
        ("off", False),
        ("TRUE", True),
        ("on", True),
    ):
        assert boolean(word) is value
        assert HfConfig.from_env({"SIM_HF_ENABLED": word}).enabled is value


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SIM_HF_ENABLED", "maybe", "SIM_HF_ENABLED='maybe' is not a valid boolean"),
        ("SIM_HF_SAMPLE_RATE_HZ", "100", "SIM_HF_SAMPLE_RATE_HZ must be within 1024..102400"),
        ("SIM_HF_SAMPLE_RATE_HZ", "2k", "SIM_HF_SAMPLE_RATE_HZ='2k' is not a valid int"),
        ("SIM_HF_VIB_CHANNELS", "17", "SIM_HF_VIB_CHANNELS must be within 2..16"),
        ("SIM_HF_PROCESS_RATE_HZ", "0", "SIM_HF_PROCESS_RATE_HZ must be within 1..1000"),
        ("SIM_HF_FRAME_MS", "333", "whole number of vibration and process samples"),
        ("SIM_HF_COMPRESSION", "brotli", "SIM_HF_COMPRESSION must be one of"),
        ("SIM_HF_QUEUE_MAX_MB", "0", "SIM_HF_QUEUE_MAX_MB must be >= 1"),
    ],
)
def test_invalid_values_are_rejected_naming_the_variable(name: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        HfConfig.from_env({name: value})


def test_load_hf_config_applies_the_load_test_overrides_and_checks_the_message_limit() -> None:
    fleet = SimpleNamespace(benchmark=False, benchmark_sample_rate=None, benchmark_vib_channels=None)
    assert load_hf_config(fleet, {}) == HfConfig()
    bench = SimpleNamespace(benchmark=True, benchmark_sample_rate=25_600, benchmark_vib_channels=16)
    full = load_hf_config(bench, {})
    assert (full.sample_rate_hz, full.vibration_channels) == (25_600, 16)
    huge = {"SIM_HF_SAMPLE_RATE_HZ": "102400", "SIM_HF_VIB_CHANNELS": "16", "SIM_HF_FRAME_MS": "2000"}
    with pytest.raises(ValueError, match="above the topic limit"):
        load_hf_config(fleet, huge)
    assert not load_hf_config(fleet, huge | {"SIM_HF_ENABLED": "false"}).enabled, "no frames, no size limit"
    with pytest.raises(ValueError, match="SIM_HF_VIB_CHANNELS"):
        load_hf_config(
            SimpleNamespace(benchmark=True, benchmark_sample_rate=None, benchmark_vib_channels=99), {}
        )


def test_producer_is_idempotent_compressed_batched_and_never_creates_topics() -> None:
    settings = hf_kafka.producer_config(HfConfig.from_env(COMPOSE_ENV), "engine-simulator-hf")
    assert settings["bootstrap.servers"] == "kafka:29092" and settings["client.id"] == "engine-simulator-hf"
    assert (settings["acks"], settings["enable.idempotence"], settings["compression.type"]) == (
        "all",
        True,
        "lz4",
    )
    assert settings["max.in.flight.requests.per.connection"] <= 5, "idempotence keeps order up to 5 in flight"
    assert (settings["linger.ms"], settings["batch.size"]) == (50, 2**20)
    assert settings["message.max.bytes"] == 8 * 2**20 == 8_388_608, "the HF topics' max.message.bytes"
    assert settings["allow.auto.create.topics"] is False and settings["partitioner"] == "murmur2_random"
    assert settings["queue.buffering.max.kbytes"] == 64 * 1024 + 8 * 1024
    assert settings["statistics.interval.ms"] == 0
    assert hf_kafka.producer_config(HfConfig(), "x", statistics=True)["statistics.interval.ms"] == 1000


def test_the_schema_file_is_read_and_bad_files_are_named(tmp_path: Path) -> None:
    text = hf_kafka.read_schema(DEFAULT_SCHEMA_PATH)
    assert json.loads(text)["name"] == "EngineSensorFrame"
    with pytest.raises(ValueError, match=r"missing\.avsc"):
        hf_kafka.read_schema(tmp_path / "missing.avsc")
    (tmp_path / "array.avsc").write_text('{"type": "array", "items": "float"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an Avro record schema"):
        hf_kafka.read_schema(tmp_path / "array.avsc")


class FakeKafkaError:
    def __init__(self, code: int, text: str, fatal: bool = False) -> None:
        self._code, self._text, self._fatal = code, text, fatal

    def code(self) -> int:
        return self._code

    def str(self) -> str:
        return self._text

    def fatal(self) -> bool:
        return self._fatal


def test_client_errors_are_logged_once_a_minute_and_statistics_kept(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    events = hf_kafka.ClientEvents(clock=lambda: now[0])
    assert set(events.callbacks(statistics=True)) == {"error_cb", "stats_cb"}
    assert set(events.callbacks(statistics=False)) == {"error_cb"}
    down = FakeKafkaError(-195, "kafka:29092/bootstrap: Connect to ipv4#172.18.0.3:29092 failed")
    with caplog.at_level(logging.WARNING, logger="hf_kafka"):
        for second in (0.0, 10.0, 59.0, 60.0):
            now[0] = second
            events.on_error(down)
        events.on_error(FakeKafkaError(-187, "1/1 brokers are down"))
        events.on_error(FakeKafkaError(-150, "Producer fenced", fatal=True))
    assert events.errors == 6 and events.fatal == "Producer fenced"
    assert caplog.text.count("Connect to ipv4") == 2 and "1/1 brokers are down" in caplog.text
    assert "Kafka producer fatal error: Producer fenced" in caplog.text
    events.on_stats(json.dumps({"tx_bytes": 123_456, "txmsgs": 9}))
    events.on_stats("not json")
    assert events.tx_bytes == 123_456


def test_librdkafka_log_lines_pass_once_a_minute_per_facility() -> None:
    now = [0.0]
    repeat = hf_kafka.RepeatFilter(interval=60.0, clock=lambda: now[0])

    def record(facility: str, message: str) -> logging.LogRecord:
        args = (facility, "engine-simulator-hf#producer-1", message)
        return logging.LogRecord("hf_kafka.librdkafka", logging.ERROR, __file__, 1, "%s [%s] %s", args, None)

    passed = []
    for second, facility in ((0, "FAIL"), (4, "FAIL"), (8, "CONFWARN"), (59, "FAIL"), (60, "FAIL")):
        now[0] = second
        passed.append(repeat.filter(record(facility, "kafka:29092/bootstrap: Connect failed")))
    assert passed == [True, False, True, False, True]
    plain = logging.LogRecord("x", logging.INFO, __file__, 1, "no arguments", None, None)
    assert repeat.filter(plain) and not repeat.filter(plain)
    first, again = hf_kafka.client_logger(), hf_kafka.client_logger()
    assert first is again and first.name == "hf_kafka.librdkafka"
    assert sum(isinstance(item, hf_kafka.RepeatFilter) for item in first.filters) == 1


class FakeProducer:
    def __init__(self, topics: dict[str, Any] | Exception) -> None:
        self.topics, self.produced, self.polls = topics, [], []

    def list_topics(self, topic: str, timeout: float) -> SimpleNamespace:
        if isinstance(self.topics, Exception):
            raise self.topics
        return SimpleNamespace(topics=self.topics)

    def produce(self, topic: str, value: bytes, key: bytes, on_delivery: Any) -> None:
        self.produced.append((topic, key, value, on_delivery))

    def poll(self, timeout: float) -> int:
        self.polls.append(timeout)
        return 0

    def flush(self, timeout: float) -> int:
        return 3


class FakeRegistry:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.registered, self.closed = error, [], False

    def register_schema(self, subject: str, schema: Any) -> int:
        if self.error is not None:
            raise self.error
        self.registered.append((subject, schema))
        return 12

    def close(self) -> None:
        self.closed = True


def kafka_sink(producer: FakeProducer, registry: FakeRegistry) -> hf_kafka.KafkaSink:
    return hf_kafka.KafkaSink(
        "engine_hf.sensor_frames",
        producer=producer,
        registry=registry,
        schema="<schema>",
        serializer=lambda record, context: f"{context}:{record['engine_id']}".encode(),
        context="value-context",
        subject="engine_hf.sensor_frames-value",
        events=hf_kafka.ClientEvents(),
        endpoints=("kafka:29092", "http://schema-registry:8081"),
    )


def test_kafka_sink_readiness_names_the_missing_piece_and_delegates_to_the_client() -> None:
    present = {"engine_hf.sensor_frames": SimpleNamespace(error=None, partitions={0: None, 1: None})}
    absent = {
        "engine_hf.sensor_frames": SimpleNamespace(
            error=FakeKafkaError(3, "Broker: Unknown topic"), partitions={}
        )
    }
    down = RuntimeError(
        "KafkaError{code=_TRANSPORT,str=Failed to get metadata: Local: Broker transport failure}"
    )
    unresolved = OSError("[Errno -5] No address associated with hostname")
    assert kafka_sink(FakeProducer(down), FakeRegistry()).check_ready(5) == (
        "Kafka kafka:29092: KafkaError{code=_TRANSPORT,"
        "str=Failed to get metadata: Local: Broker transport failure}"
    )
    assert kafka_sink(FakeProducer(absent), FakeRegistry()).check_ready(5) == (
        "topic engine_hf.sensor_frames does not exist yet (Broker: Unknown topic)"
    )
    assert kafka_sink(FakeProducer(present), FakeRegistry(unresolved)).check_ready(5) == (
        "Schema Registry http://schema-registry:8081 (subject engine_hf.sensor_frames-value): "
        "[Errno -5] No address associated with hostname"
    )
    producer, registry = FakeProducer(present), FakeRegistry()
    sink = kafka_sink(producer, registry)
    assert sink.check_ready(5) is None and sink.schema_id == 12
    assert registry.registered == [("engine_hf.sensor_frames-value", "<schema>")]
    assert sink.serialize({"engine_id": "GE90-201"}) == b"value-context:GE90-201"
    sink.produce(b"GE90-201", b"frame", print)
    assert producer.produced == [("engine_hf.sensor_frames", b"GE90-201", b"frame", print)]
    assert (sink.poll(0.5), sink.flush(1.0), producer.polls) == (0, 3, [0.5])
    assert (sink.fatal_error, sink.wire_bytes) == (None, None)
    sink.close()
    assert registry.closed


def test_a_disabled_path_loads_neither_the_publisher_nor_the_kafka_client() -> None:
    script = (
        "import sys\n"
        "import engine_generator, hf_runtime\n"
        "from hf_config import HfConfig\n"
        "path = hf_runtime.open_hf_path(HfConfig(enabled=False), 42, None, {'JT9D-001': 'JT9D-7R4'})\n"
        "assert path.live is None and path.backfill is None\n"
        "path.close()\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "print(sorted(loaded & {'confluent_kafka', 'fastavro'}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO / "simulator",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


class CountingStore:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls: list[list[str]] = []

    def telemetry_counts(self, engine_ids: list[str]) -> dict[str, int]:
        self.calls.append(list(engine_ids))
        return {engine_id: self.counts.get(engine_id, 0) for engine_id in engine_ids}


class InstantSink:
    """Kafka stand-in that is ready at once and acknowledges every frame on poll."""

    topic = "engine_hf.sensor_frames"
    fatal_error = None

    def __init__(self) -> None:
        self.schema_id: int | None = None
        self.pending: list[tuple[bytes, bytes, Any]] = []
        self.delivered: list[tuple[bytes, bytes]] = []

    def check_ready(self, timeout: float) -> None:
        self.schema_id = 1

    def serialize(self, record: dict[str, Any]) -> bytes:
        return f"{record['engine_id']}|{record['frame_seq']}|{record['telemetry_id']}".encode()

    def produce(self, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.pending.append((key, value, on_delivery))

    def poll(self, timeout: float) -> int:
        served, self.pending = self.pending, []
        for key, value, callback in served:
            self.delivered.append((key, value))
            callback(None, None)
        return len(served)

    def flush(self, timeout: float) -> int:
        self.poll(0)
        return 0

    def close(self) -> None:
        pass


def test_an_enabled_path_resumes_the_sequence_and_publishes_committed_pulses() -> None:
    store, sink = CountingStore({"JT9D-001": 41}), InstantSink()
    models = {"JT9D-001": "JT9D-7R4", "GE90-201": "GE90-115B"}
    path = open_hf_path(HfConfig(), 42, store, models, sink_factory=lambda: sink)
    assert store.calls == [["JT9D-001", "GE90-201"]], "counts are read once, before any backfill"
    event_ts = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
    pulse = eg.EngineSimulator("JT9D-001", SimulatorConfig()).next_pulse(event_ts)
    other = eg.EngineSimulator("GE90-201", SimulatorConfig()).next_pulse(event_ts)
    assert path.live is not None and path.backfill is not None
    path.live([("insert_pulses", [pulse, other])], [[501, 502]])
    deadline = time.monotonic() + 5
    while len(sink.delivered) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    path.close()
    assert sink.delivered == [(b"JT9D-001", b"JT9D-001|42|501"), (b"GE90-201", b"GE90-201|1|502")]


def test_a_kafka_client_that_cannot_be_set_up_disables_only_the_hf_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken() -> Any:
        raise ValueError("cannot read the HF frame schema /schemas/engine_sensor_frame.avsc: No such file")

    with caplog.at_level(logging.ERROR, logger="hf_runtime"):
        path = open_hf_path(HfConfig(), 42, CountingStore({}), {"JT9D-001": "JT9D-7R4"}, sink_factory=broken)
    assert path == HfPath() and path.live is None
    assert (
        "HF sensor frames disabled, the Kafka client cannot be set up: cannot read the HF frame schema"
        in caplog.text
    )
