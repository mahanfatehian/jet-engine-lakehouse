"""Construction of the ten streaming queries, grouped by the dependencies they need.

Path A, group ``lake`` (Kafka + MinIO) - parquet file sinks, one Kafka source and checkpoint
per query:
    bronze_<table>  raw records  -> s3a://<bucket>/bronze/<table>/ingest_date=YYYY-MM-DD/
    silver_<table>  typed images -> s3a://<bucket>/silver/<table>/event_date=YYYY-MM-DD/
Path B, group ``clickhouse`` (Kafka + ClickHouse) - one query over the three topics,
at-least-once into ReplacingMergeTree tables:
    clickhouse_raw  -> raw_engine_telemetry / raw_engine_status_registry /
                       raw_maintenance_log / raw_cdc_dead_letter (foreachBatch)
The high-frequency sensor frames add bronze_engine_sensor_frames and
silver_engine_hf_features to the lake group and clickhouse_hf_features to the ClickHouse
group (see ``hf_queries``); the CDC queries of a group are started first.

The lake is exactly-once only for readers that honour ``<table path>/_spark_metadata``, that
is Spark reading the table root. A micro-batch first writes its part files into the partition
folders; they join the table when the batch is committed to that log. A batch interrupted in
between (a killed driver, an S3 error during the commit) is re-run under new file names and
leaves its first files behind as orphans. Readers that list the files themselves (ClickHouse
``s3()``, DuckDB, pyarrow, Spark on a glob or on one partition folder) must deduplicate:
bronze by (kafka_topic, kafka_partition, kafka_offset); silver by primary key and
``cdc_lsn``, the current row of a key being the one with the highest ``cdc_lsn`` (argMax or
row_number, which also drops exact copies).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Protocol

from cdc_schemas import DEAD_LETTER_COLUMNS, DEAD_LETTER_TABLE, TABLES, TableSpec
from cdc_transforms import (
    BRONZE_PARTITION_COLUMN,
    SILVER_PARTITION_COLUMN,
    bronze_records,
    clickhouse_projection,
    kafka_records,
    parse_cdc_records,
    silver_records,
)
from clickhouse_rows import RowWriter, TableBatch, split_collected
from hf_queries import build_hf_query_specs
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.streaming import StreamingQuery, StreamingQueryListener
from pyspark.sql.streaming.listener import QueryProgressEvent, QueryStartedEvent, QueryTerminatedEvent
from streaming_config import StreamingSettings, trigger_interval
from streaming_readiness import check_clickhouse_dependencies, check_lake_dependencies
from streaming_supervisor import QueryGroup, QuerySpec

logger = logging.getLogger(__name__)

CLICKHOUSE_QUERY_NAME = "clickhouse_raw"
CLICKHOUSE_GROUP = "clickhouse"
LAKE_GROUP = "lake"
_ERROR_POSITION = DEAD_LETTER_COLUMNS.index("error")


class ClickHouseDestination(Protocol):
    """Readiness checks and writer of Path B (``clickhouse_sink.ClickHouseTarget`` in production)."""

    def check_ready(self) -> None:
        """Raise unless ClickHouse is connected and has every raw table."""

    def check_tables(self, tables: Iterable[str]) -> None:
        """Raise unless ``tables`` exist as well (the HF feature query checks its own tables)."""

    @property
    def writer(self) -> RowWriter:
        """Writer bound to the connected client."""


def bronze_query_name(table: TableSpec) -> str:
    """Name (and checkpoint folder) of the bronze query of ``table``."""
    return f"bronze_{table.name}"


def silver_query_name(table: TableSpec) -> str:
    """Name (and checkpoint folder) of the silver query of ``table``."""
    return f"silver_{table.name}"


def read_kafka(
    spark: SparkSession, settings: StreamingSettings, topics: Sequence[str], query_name: str
) -> DataFrame:
    """Streaming DataFrame of ``KAFKA_COLUMNS`` over ``topics`` (each query owns its source).

    ``failOnDataLoss`` follows KAFKA_FAIL_ON_DATA_LOSS (default true): when offsets the query
    has not read yet were deleted by the topic retention, the query fails, and the supervisor
    logs how to recover, instead of silently skipping the gap.
    """
    source = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", ",".join(topics))
        .option("startingOffsets", settings.kafka_starting_offsets)
        .option("failOnDataLoss", str(settings.kafka_fail_on_data_loss).lower())
        .option("maxOffsetsPerTrigger", str(settings.max_offsets_per_trigger))
        .option("groupIdPrefix", f"iot-engine-{query_name}")
        .load()
    )
    return kafka_records(source)


class ClickHouseBatchHandler:
    """``foreachBatch`` callback that routes one micro-batch to the ClickHouse raw tables."""

    def __init__(self, writer: RowWriter, topics: Mapping[str, TableSpec]) -> None:
        self._writer = writer
        self._topics = dict(topics)

    def __call__(self, batch: DataFrame, batch_id: int) -> None:
        """Parse, convert, deduplicate and insert one micro-batch; raises if an insert fails."""
        started = time.monotonic()
        batch.persist()  # one collect per table: read the micro-batch from Kafka only once
        try:
            prepared = [self._collect(batch, topic, table) for topic, table in self._topics.items()]
        finally:
            batch.unpersist()
        for table_batch in prepared:
            table = table_batch.table
            self._writer.insert_rows(table.clickhouse_table, table.clickhouse_columns, table_batch.rows)
        dead_letters = [row for table_batch in prepared for row in table_batch.dead_letters]
        self._writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, dead_letters)
        _log_batch(batch_id, prepared, dead_letters, time.monotonic() - started)

    @staticmethod
    def _collect(batch: DataFrame, topic: str, table: TableSpec) -> TableBatch:
        parsed = parse_cdc_records(batch.where(sf.col("kafka_topic") == topic), table)
        return split_collected(table, clickhouse_projection(parsed, table).collect())


def _log_batch(
    batch_id: int, prepared: Sequence[TableBatch], dead_letters: Sequence[Sequence[Any]], elapsed_s: float
) -> None:
    counts = " ".join(f"{table_batch.table.name}={len(table_batch.rows)}" for table_batch in prepared)
    logger.info(
        "clickhouse batch %d: %s dead_letters=%d duplicates_dropped=%d duration_ms=%d",
        batch_id,
        counts,
        len(dead_letters),
        sum(table_batch.duplicates for table_batch in prepared),
        round(elapsed_s * 1000),
    )
    if dead_letters:
        reasons = sorted({str(row[_ERROR_POSITION]) for row in dead_letters})
        logger.warning("clickhouse batch %d dead-lettered: %s", batch_id, "; ".join(reasons[:5]))


class ProgressLogger(StreamingQueryListener):
    """Logs non-empty micro-batches of the lake queries (Path B logs its own batches)."""

    def __init__(self, query_names: Iterable[str]) -> None:
        super().__init__()
        self._query_names = frozenset(query_names)

    def onQueryStarted(self, event: QueryStartedEvent) -> None:  # noqa: N802 - PySpark listener API
        """Log the start of every query."""
        logger.info("query %s started (id=%s)", event.name, event.id)

    def onQueryProgress(self, event: QueryProgressEvent) -> None:  # noqa: N802 - PySpark listener API
        """Log committed lake micro-batches that contained records."""
        progress = event.progress
        if progress.name in self._query_names and progress.numInputRows > 0:
            logger.info(
                "query %s committed batch %d: %d records in %d ms",
                progress.name,
                progress.batchId,
                progress.numInputRows,
                progress.batchDuration,
            )

    def onQueryTerminated(self, event: QueryTerminatedEvent) -> None:  # noqa: N802 - PySpark listener API
        """Terminations are reported, with the query name, by ``streaming_supervisor``."""


def build_query_groups(
    spark: SparkSession, settings: StreamingSettings, clickhouse: ClickHouseDestination
) -> tuple[QueryGroup, QueryGroup]:
    """Path B (checked and started first: latency-sensitive) and Path A, supervised independently.

    Each query is described by a factory, so that the supervisor can restart it from its
    checkpoint without touching the others.
    """
    hf = build_hf_query_specs(spark, settings, clickhouse)
    lake_queries = tuple(spec for table in TABLES for spec in _lake_query_specs(spark, settings, table))
    lake_queries += hf.lake
    spark.streams.addListener(ProgressLogger(spec.name for spec in lake_queries))
    clickhouse_query = QuerySpec(
        CLICKHOUSE_QUERY_NAME,
        partial(_start_clickhouse_query, spark, settings, clickhouse),
        _clickhouse_recovery(settings),
    )
    return (
        QueryGroup(
            CLICKHOUSE_GROUP,
            partial(_check_clickhouse_ready, settings, clickhouse),
            (clickhouse_query, *hf.clickhouse),
        ),
        QueryGroup(LAKE_GROUP, partial(check_lake_dependencies, settings), lake_queries),
    )


def _check_clickhouse_ready(settings: StreamingSettings, clickhouse: ClickHouseDestination) -> None:
    check_clickhouse_dependencies(settings)
    clickhouse.check_ready()


def _lake_query_specs(
    spark: SparkSession, settings: StreamingSettings, table: TableSpec
) -> tuple[QuerySpec, QuerySpec]:
    bronze, silver = bronze_query_name(table), silver_query_name(table)
    return (
        QuerySpec(
            bronze,
            partial(_start_bronze_query, spark, settings, table),
            _lake_recovery(settings, bronze, settings.lake_path("bronze", table.name)),
        ),
        QuerySpec(
            silver,
            partial(_start_silver_query, spark, settings, table),
            _lake_recovery(settings, silver, settings.lake_path("silver", table.name)),
        ),
    )


def _start_bronze_query(spark: SparkSession, settings: StreamingSettings, table: TableSpec) -> StreamingQuery:
    name = bronze_query_name(table)
    records = bronze_records(read_kafka(spark, settings, [settings.topic(table.name)], name))
    path = settings.lake_path("bronze", table.name)
    return _start_parquet_sink(records, settings, name, path, BRONZE_PARTITION_COLUMN)


def _start_silver_query(spark: SparkSession, settings: StreamingSettings, table: TableSpec) -> StreamingQuery:
    name = silver_query_name(table)
    parsed = parse_cdc_records(read_kafka(spark, settings, [settings.topic(table.name)], name), table)
    path = settings.lake_path("silver", table.name)
    return _start_parquet_sink(silver_records(parsed, table), settings, name, path, SILVER_PARTITION_COLUMN)


def _start_clickhouse_query(
    spark: SparkSession, settings: StreamingSettings, clickhouse: ClickHouseDestination
) -> StreamingQuery:
    topics = {settings.topic(table.name): table for table in TABLES}
    return (
        read_kafka(spark, settings, list(topics), CLICKHOUSE_QUERY_NAME)
        .writeStream.queryName(CLICKHOUSE_QUERY_NAME)
        .foreachBatch(ClickHouseBatchHandler(clickhouse.writer, topics))
        .option("checkpointLocation", settings.checkpoint_path(CLICKHOUSE_QUERY_NAME))
        .trigger(processingTime=trigger_interval(settings.clickhouse_trigger_seconds))
        .start()
    )


def _start_parquet_sink(
    frame: DataFrame, settings: StreamingSettings, query_name: str, path: str, partition_column: str
) -> StreamingQuery:
    # Exactly-once for readers of <path>/_spark_metadata only (see the module docstring).
    return (
        # One file per partition directory and micro-batch instead of one per Kafka partition.
        frame.coalesce(1)
        .writeStream.queryName(query_name)
        .format("parquet")
        .outputMode("append")
        .partitionBy(partition_column)
        .option("path", path)
        .option("checkpointLocation", settings.checkpoint_path(query_name))
        .trigger(processingTime=trigger_interval(settings.lake_trigger_seconds))
        .start()
    )


def _clickhouse_recovery(settings: StreamingSettings) -> str:
    return (
        f"Recovery: with the job stopped, delete {settings.checkpoint_path(CLICKHOUSE_QUERY_NAME)} so that "
        "the query re-reads everything Kafka still retains (KAFKA_STARTING_OFFSETS=earliest; the raw_* "
        "tables absorb the replay), or run the job with KAFKA_FAIL_ON_DATA_LOSS=false until the query "
        "has committed a batch past the gap, then set it back to true."
    )


def _lake_recovery(settings: StreamingSettings, query_name: str, path: str) -> str:
    return (
        "Recovery: run the job with KAFKA_FAIL_ON_DATA_LOSS=false until this query has committed a "
        "batch past the gap, then set it back to true (a Debezium snapshot can backfill the current "
        f"rows). Do not just delete its checkpoint {settings.checkpoint_path(query_name)}: a new run "
        f"starts again at batch 0 and the file sink skips every batch id already committed in "
        f"{path}/_spark_metadata."
    )
