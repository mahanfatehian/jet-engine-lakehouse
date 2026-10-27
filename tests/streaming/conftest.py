"""Shared fixtures of the streaming tests: Debezium messages, HF frames, the ClickHouse DDL, Spark.

``fixtures/debezium_sample_messages.txt`` was captured from the real stack (Debezium 2.5
on the contract's PostgreSQL schema) and holds c/u/d events of all three tables. HF sensor
frames are synthesised (``make_frame``) and encoded like the edge gateway does (``avro_frames``).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_DIR = REPO_ROOT / "streaming"
SAMPLE_FILE = Path(__file__).parent / "fixtures" / "debezium_sample_messages.txt"
DDL_FILE = REPO_ROOT / "infrastructure" / "clickhouse" / "init-db.sql"
FRAME_SCHEMA_FILE = REPO_ROOT / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
KAFKA_RECORD_DDL = (
    "kafka_key STRING, kafka_value STRING, kafka_topic STRING, "
    "kafka_partition INT, kafka_offset BIGINT, kafka_timestamp TIMESTAMP"
)
HF_TOPIC = "engine_hf.sensor_frames"
HF_SAMPLE_RATE_HZ = 2048
HF_FRAME_START_MICROS = 1_793_005_201_123_456  # 2026-10-26T09:00:01.123456Z
HF_RECORD_TIMESTAMP = datetime(2026, 10, 26, 9, 0, 2, 500000, tzinfo=UTC)
HF_RECORD_DDL = (
    "kafka_key STRING, kafka_value BINARY, kafka_topic STRING, "
    "kafka_partition INT, kafka_offset BIGINT, kafka_timestamp TIMESTAMP"
)
# The .avsc record as a Spark struct (frame_start_ts as the long it is on the wire).
FRAME_STRUCT_DDL = (
    "schema_version INT, engine_id STRING, engine_model STRING, lifecycle_id INT, cycle_number INT, "
    "telemetry_id BIGINT, frame_seq BIGINT, frame_start_ts BIGINT, frame_duration_ms INT, n1_rpm FLOAT, "
    "n2_rpm FLOAT, bpfo_order FLOAT, vibration_sample_rate_hz INT, "
    "vibration ARRAY<STRUCT<channel: STRING, unit: STRING, samples_f32le: BINARY>>, "
    "process_sample_rate_hz INT, "
    "process ARRAY<STRUCT<parameter: STRING, unit: STRING, samples_f32le: BINARY>>"
)
_TABLE_BLOCK = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\)", re.DOTALL)
_COLUMN_LINE = re.compile(r"^\s+(\w+)\s+(.+?),?$")


class KafkaMessage(NamedTuple):
    """One Kafka record in the column order of ``cdc_transforms.KAFKA_COLUMNS``."""

    key: str | None
    value: str | None
    topic: str
    partition: int
    offset: int
    timestamp: datetime


def _parse_capture(path: Path) -> list[KafkaMessage]:
    # Capture format: "##### <topic>" headers, then "Partition:N || Offset:N || <key> || <value>".
    messages: list[KafkaMessage] = []
    topic = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#####"):
            topic = line.removeprefix("#####").strip()
        elif line.strip():
            partition, offset, key, value = line.split(" || ", 3)
            messages.append(
                KafkaMessage(
                    key=key,
                    value=value,
                    topic=topic,
                    partition=int(partition.removeprefix("Partition:")),
                    offset=int(offset.removeprefix("Offset:")),
                    timestamp=datetime(2026, 10, 26, 9, 6, 6, 300000, tzinfo=UTC),
                )
            )
    return messages


def _ddl_columns(path: Path) -> dict[str, dict[str, str]]:
    """Map table -> ordered {column: type} parsed from the DDL file (indexes skipped)."""
    tables: dict[str, dict[str, str]] = {}
    for name, body in _TABLE_BLOCK.findall(path.read_text(encoding="utf-8")):
        columns: dict[str, str] = {}
        for line in body.splitlines():
            match = _COLUMN_LINE.match(line)
            if match and match.group(1) != "INDEX":
                columns[match.group(1)] = match.group(2)
        tables[name] = columns
    return tables


@pytest.fixture(scope="session")
def sample_messages() -> list[KafkaMessage]:
    """The real captured Debezium messages, in capture order."""
    return _parse_capture(SAMPLE_FILE)


@pytest.fixture(scope="session")
def ddl() -> dict[str, dict[str, str]]:
    """Columns and types of every table in infrastructure/clickhouse/init-db.sql."""
    if not DDL_FILE.exists():
        pytest.skip(f"{DDL_FILE} not available")
    return _ddl_columns(DDL_FILE)


@pytest.fixture
def non_utc_process_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the test with the Python process in UTC+03:30 (Iran, no DST)."""
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "IRST-03:30")
        time.tzset()
    if not datetime(2026, 10, 26, 9).astimezone().utcoffset():
        pytest.skip("cannot switch this process to a non-UTC local timezone")
    yield
    monkeypatch.undo()
    if hasattr(time, "tzset"):
        time.tzset()


@pytest.fixture(scope="session")
def spark() -> Iterator[Any]:
    """Local SparkSession; skips the requesting test when PySpark or a JVM is unavailable.

    The JVM gets the job's Avro decoder limits, and the Python workers import the job's
    modules (``hf_features`` in mapInPandas) from the repository rather than from an image.
    """
    pytest.importorskip("pyspark")
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        pytest.skip("no Java runtime: Spark tests need a JVM")
    from pyspark.sql import SparkSession
    from streaming_config import AVRO_DECODER_LIMITS

    # createDataFrame() ships local rows through a Python worker, which must be this very
    # interpreter (PySpark would otherwise look for `python3`, absent on Windows).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(STREAMING_DIR), os.environ.get("PYTHONPATH")))
    )
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("tests-streaming")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.extraJavaOptions", " ".join(("-Duser.timezone=UTC", *AVRO_DECODER_LIMITS)))
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:  # a broken local JVM is an environment problem, not a code defect
        pytest.skip(f"cannot start a local SparkSession: {exc}")
    yield session
    session.stop()


@pytest.fixture
def records_frame(spark: Any) -> Callable[[Sequence[KafkaMessage]], Any]:
    """Factory turning messages into a DataFrame shaped like ``cdc_transforms.kafka_records``."""

    def build(messages: Sequence[KafkaMessage]) -> Any:
        return spark.createDataFrame([tuple(message) for message in messages], schema=KAFKA_RECORD_DDL)

    return build


def _f32le(samples: Sequence[float] | np.ndarray) -> bytes:
    """Packed little-endian float32 samples, as the edge gateway sends them."""
    return np.asarray(samples, dtype="<f4").tobytes()


def _tone(frequency_hz: float, amplitude: float) -> np.ndarray:
    """One second of a sine at the default vibration sample rate."""
    return amplitude * np.sin(
        2 * np.pi * frequency_hz * np.arange(HF_SAMPLE_RATE_HZ) / HF_SAMPLE_RATE_HZ + 0.3
    )


def _sensor_frame(
    telemetry_id: int = 4242,
    *,
    fan: bytes | None = None,
    core: bytes | None = None,
    extra_vibration: Sequence[tuple[str, str, bytes]] = (),
    process: Sequence[tuple[str, str, bytes]] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """One EngineSensorFrame as a dict in .avsc field order (``frame_start_ts`` in epoch micros).

    Default waveforms at JT9D speeds: fan 2.0 mm/s at 1xN1 plus 0.25 mm/s at BPFO, core 0.8 mm/s
    at 1xN2; default process channels EGT_A/EGT_B (610 / 612 degC) and OIL_T_A (85 degC).
    """
    n1_rpm, n2_rpm, bpfo_order = 3650.0, 7800.0, 4.62
    fan = fan if fan is not None else _f32le(_tone(n1_rpm / 60, 2.0) + _tone(bpfo_order * n1_rpm / 60, 0.25))
    core = core if core is not None else _f32le(_tone(n2_rpm / 60, 0.8))
    frame = {
        "schema_version": 1,
        "engine_id": "JT9D-001",
        "engine_model": "JT9D-7R4",
        "lifecycle_id": 3,
        "cycle_number": 17,
        "telemetry_id": telemetry_id,
        "frame_seq": telemetry_id - 4000,
        "frame_start_ts": HF_FRAME_START_MICROS + (telemetry_id - 4242) * 1_000_000,
        "frame_duration_ms": 1000,
        "n1_rpm": n1_rpm,
        "n2_rpm": n2_rpm,
        "bpfo_order": bpfo_order,
        "vibration_sample_rate_hz": HF_SAMPLE_RATE_HZ,
        "vibration": [("fan_frame_vel", "mm/s", fan), ("core_trf_vel", "mm/s", core), *extra_vibration],
        "process_sample_rate_hz": 8,
        "process": list(
            process
            if process is not None
            else [
                ("EGT_A", "degC", _f32le([610.0] * 8)),
                ("EGT_B", "degC", _f32le([612.0] * 8)),
                ("OIL_T_A", "degC", _f32le([85.0] * 8)),
            ]
        ),
    }
    unknown = set(fields) - set(frame)
    assert not unknown, f"not frame fields: {unknown}"
    frame.update(fields)
    return frame


@pytest.fixture
def make_frame() -> Callable[..., dict[str, Any]]:
    """Factory of frame dicts (see ``_sensor_frame``)."""
    return _sensor_frame


@pytest.fixture
def avro_frames(spark: Any) -> Callable[..., list[bytes]]:
    """Factory encoding frame dicts as Confluent-framed Avro values (magic byte, schema id, body).

    The Avro body comes from Spark's ``to_avro`` with the repository's .avsc, so these tests
    need no Avro library of their own; the end-to-end check runs the real producer stack
    (confluent-kafka with fastavro) against the job.
    """
    from pyspark.sql import functions as sf
    from pyspark.sql.avro.functions import to_avro

    schema_json = FRAME_SCHEMA_FILE.read_text(encoding="utf-8")

    def encode(frames: Sequence[dict[str, Any]], schema_id: int = 1) -> list[bytes]:
        table = spark.createDataFrame([tuple(frame.values()) for frame in frames], FRAME_STRUCT_DDL)
        header = b"\x00" + schema_id.to_bytes(4, "big")
        values = table.select(sf.concat(sf.lit(header), to_avro(sf.struct(*table.columns), schema_json)))
        return [bytes(row[0]) for row in values.collect()]

    return encode


@pytest.fixture
def hf_records(spark: Any) -> Callable[..., Any]:
    """Factory of a DataFrame in the ``hf_decode.HF_KAFKA_COLUMNS`` layout, one record per value.

    Record ``i`` gets offset ``first_offset + i`` in partition 0 of the HF topic.
    """

    def build(values: Sequence[bytes | None], key: str | None = "JT9D-001", first_offset: int = 0) -> Any:
        rows = [
            (key, value, HF_TOPIC, 0, first_offset + index, HF_RECORD_TIMESTAMP)
            for index, value in enumerate(values)
        ]
        return spark.createDataFrame(rows, HF_RECORD_DDL)

    return build
