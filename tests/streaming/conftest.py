"""Shared fixtures of the streaming tests: real Debezium messages and a local SparkSession.

``fixtures/debezium_sample_messages.txt`` was captured from the real stack (Debezium 2.5
on the contract's PostgreSQL schema) and holds c/u/d events of all three tables.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pytest

SAMPLE_FILE = Path(__file__).parent / "fixtures" / "debezium_sample_messages.txt"
KAFKA_RECORD_DDL = (
    "kafka_key STRING, kafka_value STRING, kafka_topic STRING, "
    "kafka_partition INT, kafka_offset BIGINT, kafka_timestamp TIMESTAMP"
)


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


@pytest.fixture(scope="session")
def sample_messages() -> list[KafkaMessage]:
    """The real captured Debezium messages, in capture order."""
    return _parse_capture(SAMPLE_FILE)


@pytest.fixture(scope="session")
def spark() -> Iterator[Any]:
    """Local SparkSession; skips the requesting test when PySpark or a JVM is unavailable."""
    pytest.importorskip("pyspark")
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        pytest.skip("no Java runtime: Spark tests need a JVM")
    from pyspark.sql import SparkSession

    # createDataFrame() ships local rows through a Python worker, which must be this very
    # interpreter (PySpark would otherwise look for `python3`, absent on Windows).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("tests-streaming")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
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
