"""Unit tests of the query wiring: dependency groups, recovery texts and Kafka source options.

A fake SparkSession records the reader options and listeners; no JVM is started. Covers the
CDC queries and the high-frequency (HF) queries registered in the same supervisor groups.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pyspark")

import hf_queries
import streaming_queries
from clickhouse_sink import MissingTablesError
from hf_decode import FrameSchemaError
from streaming_config import load_settings

REPO_SCHEMA = Path(__file__).resolve().parents[2] / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
CDC_LAKE_QUERIES = [
    "bronze_engine_telemetry",
    "silver_engine_telemetry",
    "bronze_engine_status_registry",
    "silver_engine_status_registry",
    "bronze_maintenance_log",
    "silver_maintenance_log",
]
HF_LAKE_QUERIES = ["bronze_engine_sensor_frames", "silver_engine_hf_features"]


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
    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.checks = 0
        self.checked_tables: list[tuple[str, ...]] = []
        self.writer = object()
        self._missing = missing

    def check_ready(self) -> None:
        self.checks += 1

    def check_tables(self, tables: Any) -> None:
        self.checked_tables.append(tuple(tables))
        if self._missing:
            raise MissingTablesError(f"lacks table(s) {', '.join(self._missing)}")


def query(group: Any, name: str) -> Any:
    return next(spec for spec in group.queries if spec.name == name)


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


@pytest.mark.parametrize(("env", "expected"), [({}, "false"), ({"HF_FAIL_ON_DATA_LOSS": "true"}, "true")])
def test_hf_kafka_source_is_best_effort_unless_configured(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    monkeypatch.setattr(hf_queries, "hf_kafka_records", lambda source: source)
    spark = FakeSpark()
    frame = hf_queries.read_hf_kafka(
        spark, load_settings({"HF_MAX_OFFSETS_PER_TRIGGER": "500", **env}), "hf_x"
    )
    assert frame == "kafka-source-frame"
    assert spark.readStream.options == {
        "kafka.bootstrap.servers": "localhost:9092",
        "subscribe": "engine_hf.sensor_frames",
        "startingOffsets": "earliest",
        "failOnDataLoss": expected,
        "maxOffsetsPerTrigger": "500",
        "groupIdPrefix": "iot-engine-hf_x",
        "kafka.max.partition.fetch.bytes": "8388608",  # the HF topic's max.message.bytes
    }


def test_queries_are_grouped_by_the_dependencies_they_need() -> None:
    spark = FakeSpark()
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        spark, load_settings({}), FakeClickHouse()
    )
    assert (clickhouse_group.name, lake_group.name) == ("clickhouse", "lake")
    # The CDC queries come first in each group, so they are started first.
    assert [spec.name for spec in clickhouse_group.queries] == ["clickhouse_raw", "clickhouse_hf_features"]
    assert [spec.name for spec in lake_group.queries] == CDC_LAKE_QUERIES + HF_LAKE_QUERIES
    (listener,) = spark.listeners
    assert isinstance(listener, streaming_queries.ProgressLogger)


@pytest.mark.parametrize("name", ["silver_engine_hf_features", "bronze_maintenance_log"])
def test_progress_of_every_lake_query_is_logged(name: str, caplog: pytest.LogCaptureFixture) -> None:
    spark = FakeSpark()
    streaming_queries.build_query_groups(spark, load_settings({}), FakeClickHouse())
    (listener,) = spark.listeners
    progress = SimpleNamespace(name=name, batchId=4, numInputRows=360, batchDuration=850)
    with caplog.at_level(logging.INFO):
        listener.onQueryProgress(SimpleNamespace(progress=progress))
    assert f"query {name} committed batch 4: 360 records in 850 ms" in caplog.text


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
    assert clickhouse.checked_tables == []  # the HF tables never gate the CDC query


def test_hf_feature_query_waits_for_its_own_tables_without_touching_kafka() -> None:
    spark, clickhouse = FakeSpark(), FakeClickHouse(missing=("hf_ingest_batches",))
    settings = load_settings({"HF_SCHEMA_PATH": str(REPO_SCHEMA)})
    clickhouse_group, _lake_group = streaming_queries.build_query_groups(spark, settings, clickhouse)
    with pytest.raises(MissingTablesError, match="hf_ingest_batches"):
        query(clickhouse_group, "clickhouse_hf_features").start()
    assert clickhouse.checked_tables == [("raw_engine_hf_features", "hf_ingest_batches")]
    assert spark.readStream.options == {}


@pytest.mark.parametrize(
    ("group_index", "name"), [(0, "clickhouse_hf_features"), (1, "silver_engine_hf_features")]
)
def test_hf_decoding_queries_need_the_schema_file(tmp_path: Path, group_index: int, name: str) -> None:
    spark = FakeSpark()
    settings = load_settings({"HF_SCHEMA_PATH": str(tmp_path / "missing.avsc")})
    groups = streaming_queries.build_query_groups(spark, settings, FakeClickHouse())
    with pytest.raises(FrameSchemaError, match=r"missing\.avsc"):
        query(groups[group_index], name).start()
    assert spark.readStream.options == {}


def test_data_loss_recovery_steps_fit_each_sink() -> None:
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        FakeSpark(), load_settings({}), FakeClickHouse()
    )
    clickhouse_raw = query(clickhouse_group, "clickhouse_raw")
    assert "delete /opt/spark-checkpoints/clickhouse_raw" in clickhouse_raw.data_loss_recovery
    assert "KAFKA_FAIL_ON_DATA_LOSS=false" in clickhouse_raw.data_loss_recovery
    silver_maintenance = query(lake_group, "silver_maintenance_log")
    assert "KAFKA_FAIL_ON_DATA_LOSS=false" in silver_maintenance.data_loss_recovery
    assert (
        "Do not just delete its checkpoint /opt/spark-checkpoints/silver_maintenance_log"
        in silver_maintenance.data_loss_recovery
    )
    assert "s3a://lakehouse/silver/maintenance_log/_spark_metadata" in silver_maintenance.data_loss_recovery


def test_hf_recovery_steps_point_at_the_best_effort_switch() -> None:
    clickhouse_group, lake_group = streaming_queries.build_query_groups(
        FakeSpark(), load_settings({}), FakeClickHouse()
    )
    hf_features = query(clickhouse_group, "clickhouse_hf_features")
    assert "HF_FAIL_ON_DATA_LOSS=false" in hf_features.data_loss_recovery
    assert "delete /opt/spark-checkpoints/clickhouse_hf_features" in hf_features.data_loss_recovery
    bronze = query(lake_group, "bronze_engine_sensor_frames")
    assert "HF_FAIL_ON_DATA_LOSS=false" in bronze.data_loss_recovery
    assert "s3a://lakehouse/bronze/engine_sensor_frames/_spark_metadata" in bronze.data_loss_recovery


@pytest.mark.parametrize(("days", "expected"), [(1, "23h"), (7, "167h")])
def test_bronze_log_forgets_raw_frames_before_the_bucket_expires_them(days: int, expected: str) -> None:
    assert hf_queries.bronze_log_retention(days) == expected
