"""Unit tests of the query wiring: dependency groups, recovery texts and Kafka source options.

A fake SparkSession records the reader options and listeners; no JVM is started.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pyspark")

import streaming_queries
from streaming_config import load_settings

LAKE_QUERIES = [
    "bronze_engine_telemetry",
    "silver_engine_telemetry",
    "bronze_engine_status_registry",
    "silver_engine_status_registry",
    "bronze_maintenance_log",
    "silver_maintenance_log",
]


class FakeReader:
    """``spark.readStream`` that records the source format and options."""

    def __init__(self) -> None:
        self.source = ""
        self.options: dict[str, str] = {}

    def format(self, source: str) -> FakeReader:
        self.source = source
        return self

    def option(self, key: str, value: str) -> FakeReader:
        self.options[key] = value
        return self

    def load(self) -> str:
        return "kafka-source-frame"


class FakeSpark:
    def __init__(self) -> None:
        self.readStream = FakeReader()
        self.listeners: list[Any] = []
        self.streams = SimpleNamespace(addListener=self.listeners.append)


class FakeClickHouse:
    def __init__(self) -> None:
        self.checks = 0
        self.writer = object()

    def check_ready(self) -> None:
        self.checks += 1


@pytest.mark.parametrize(("env", "expected"), [({}, "true"), ({"KAFKA_FAIL_ON_DATA_LOSS": "false"}, "false")])
def test_kafka_source_fails_on_data_loss_unless_disabled(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    monkeypatch.setattr(streaming_queries, "kafka_records", lambda source: source)
    spark = FakeSpark()
    frame = streaming_queries.read_kafka(spark, load_settings(env), ["t1", "t2"], "bronze_x")
    assert frame == "kafka-source-frame"
    assert spark.readStream.source == "kafka"
    assert spark.readStream.options["failOnDataLoss"] == expected
    assert spark.readStream.options["subscribe"] == "t1,t2"
    assert spark.readStream.options["groupIdPrefix"] == "iot-engine-bronze_x"


def test_queries_are_grouped_by_the_dependencies_they_need() -> None:
    spark = FakeSpark()
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        spark, load_settings({}), FakeClickHouse()
    )
    assert (clickhouse_group.name, lake_group.name) == ("clickhouse", "lake")
    assert [query.name for query in clickhouse_group.queries] == ["clickhouse_raw"]
    assert [query.name for query in lake_group.queries] == LAKE_QUERIES
    (listener,) = spark.listeners
    assert isinstance(listener, streaming_queries.ProgressLogger)


def test_each_group_checks_only_its_own_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[str] = []
    monkeypatch.setattr(streaming_queries, "check_lake_dependencies", lambda _s: probed.append("kafka+minio"))
    monkeypatch.setattr(
        streaming_queries, "check_clickhouse_dependencies", lambda _s: probed.append("kafka+clickhouse")
    )
    clickhouse = FakeClickHouse()
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        FakeSpark(), load_settings({}), clickhouse
    )
    lake_group.check_ready()
    assert (probed, clickhouse.checks) == (["kafka+minio"], 0)
    clickhouse_group.check_ready()
    assert (probed, clickhouse.checks) == (["kafka+minio", "kafka+clickhouse"], 1)


def test_data_loss_recovery_steps_fit_each_sink() -> None:
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        FakeSpark(), load_settings({}), FakeClickHouse()
    )
    (clickhouse_query,) = clickhouse_group.queries
    assert "delete /opt/spark-checkpoints/clickhouse_raw" in clickhouse_query.data_loss_recovery
    assert "KAFKA_FAIL_ON_DATA_LOSS=false" in clickhouse_query.data_loss_recovery
    silver_maintenance = lake_group.queries[-1]
    assert "KAFKA_FAIL_ON_DATA_LOSS=false" in silver_maintenance.data_loss_recovery
    assert (
        "Do not just delete its checkpoint /opt/spark-checkpoints/silver_maintenance_log"
        in silver_maintenance.data_loss_recovery
    )
    assert "s3a://lakehouse/silver/maintenance_log/_spark_metadata" in silver_maintenance.data_loss_recovery
