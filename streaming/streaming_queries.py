"""Construction, supervision and shutdown of the seven streaming queries.

Path A - exactly-once parquet file sinks, one Kafka source and checkpoint per query:
    bronze_<table>  raw records  -> s3a://<bucket>/bronze/<table>/ingest_date=YYYY-MM-DD/
    silver_<table>  typed images -> s3a://<bucket>/silver/<table>/event_date=YYYY-MM-DD/
Path B - one query over the three topics, at-least-once into ReplacingMergeTree tables:
    clickhouse_raw  -> raw_engine_telemetry / raw_engine_status_registry /
                       raw_maintenance_log / raw_cdc_dead_letter (foreachBatch)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
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
from clickhouse_rows import TableBatch, split_collected
from pyspark.errors import StreamingQueryException
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.streaming import StreamingQuery, StreamingQueryListener
from pyspark.sql.streaming.listener import QueryProgressEvent, QueryStartedEvent, QueryTerminatedEvent
from streaming_config import StreamingSettings

logger = logging.getLogger(__name__)

CLICKHOUSE_QUERY_NAME = "clickhouse_raw"
EXIT_OK = 0
EXIT_QUERY_FAILED = 1
SUPERVISOR_POLL_SECONDS = 2.0
# docker compose grants 30 s (stop_grace_period) between SIGTERM and SIGKILL.
DRAIN_TIMEOUT_SECONDS = 20.0
_DRAIN_POLL_SECONDS = 0.5
_ERROR_POSITION = DEAD_LETTER_COLUMNS.index("error")


class RowWriter(Protocol):
    """Destination of insert-ready rows (``clickhouse_sink.ClickHouseWriter`` in production)."""

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Insert ``rows`` into ``table`` using exactly ``columns``."""


def bronze_query_name(table: TableSpec) -> str:
    """Name (and checkpoint folder) of the bronze query of ``table``."""
    return f"bronze_{table.name}"


def silver_query_name(table: TableSpec) -> str:
    """Name (and checkpoint folder) of the silver query of ``table``."""
    return f"silver_{table.name}"


def read_kafka(
    spark: SparkSession, settings: StreamingSettings, topics: Sequence[str], query_name: str
) -> DataFrame:
    """Streaming DataFrame of ``KAFKA_COLUMNS`` over ``topics`` (each query owns its source)."""
    source = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", ",".join(topics))
        .option("startingOffsets", settings.kafka_starting_offsets)
        .option("failOnDataLoss", "false")
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
        """Terminations are reported, with the query name, by :func:`supervise`."""


def start_queries(
    spark: SparkSession, settings: StreamingSettings, writer: RowWriter
) -> list[StreamingQuery]:
    """Start Path B first (latency-sensitive), then the bronze and silver query of each table."""
    lake_names = [name for table in TABLES for name in (bronze_query_name(table), silver_query_name(table))]
    spark.streams.addListener(ProgressLogger(lake_names))
    queries = [_start_clickhouse_query(spark, settings, writer)]
    for table in TABLES:
        topic = settings.topic(table.name)
        bronze_name, silver_name = bronze_query_name(table), silver_query_name(table)
        bronze = bronze_records(read_kafka(spark, settings, [topic], bronze_name))
        queries.append(
            _start_parquet_sink(
                bronze,
                settings,
                bronze_name,
                settings.lake_path("bronze", table.name),
                BRONZE_PARTITION_COLUMN,
            )
        )
        silver = silver_records(
            parse_cdc_records(read_kafka(spark, settings, [topic], silver_name), table), table
        )
        queries.append(
            _start_parquet_sink(
                silver,
                settings,
                silver_name,
                settings.lake_path("silver", table.name),
                SILVER_PARTITION_COLUMN,
            )
        )
    return queries


def _start_clickhouse_query(
    spark: SparkSession, settings: StreamingSettings, writer: RowWriter
) -> StreamingQuery:
    topics = {settings.topic(table.name): table for table in TABLES}
    return (
        read_kafka(spark, settings, list(topics), CLICKHOUSE_QUERY_NAME)
        .writeStream.queryName(CLICKHOUSE_QUERY_NAME)
        .foreachBatch(ClickHouseBatchHandler(writer, topics))
        .option("checkpointLocation", settings.checkpoint_path(CLICKHOUSE_QUERY_NAME))
        .trigger(processingTime=_interval(settings.clickhouse_trigger_seconds))
        .start()
    )


def _start_parquet_sink(
    frame: DataFrame, settings: StreamingSettings, query_name: str, path: str, partition_column: str
) -> StreamingQuery:
    # Exactly-once: files of a micro-batch become visible through <path>/_spark_metadata on commit.
    return (
        # One file per partition directory and micro-batch instead of one per Kafka partition.
        frame.coalesce(1)
        .writeStream.queryName(query_name)
        .format("parquet")
        .outputMode("append")
        .partitionBy(partition_column)
        .option("path", path)
        .option("checkpointLocation", settings.checkpoint_path(query_name))
        .trigger(processingTime=_interval(settings.lake_trigger_seconds))
        .start()
    )


def _interval(seconds: float) -> str:
    return f"{max(1, round(seconds * 1000))} milliseconds"


def supervise(spark: SparkSession, queries: Sequence[StreamingQuery], stop_event: threading.Event) -> int:
    """Block until shutdown is requested or any query terminates, then stop all queries.

    Returns ``EXIT_OK`` for a requested shutdown and ``EXIT_QUERY_FAILED`` when a query
    died on its own, so that Docker's restart policy restarts the container.
    """
    while not stop_event.is_set():
        try:
            terminated = spark.streams.awaitAnyTermination(SUPERVISOR_POLL_SECONDS)
        except StreamingQueryException:
            terminated = True  # the failing query is identified and logged below
        if terminated:
            _log_terminated(queries, during_shutdown=stop_event.is_set())
            stop_queries(queries)
            return EXIT_OK if stop_event.is_set() else EXIT_QUERY_FAILED
    logger.info("shutdown requested: draining and stopping %d queries", len(queries))
    stop_queries(queries)
    return EXIT_OK


def _log_terminated(queries: Sequence[StreamingQuery], during_shutdown: bool) -> None:
    level = logging.WARNING if during_shutdown else logging.ERROR
    for query in queries:
        if query.isActive:
            continue
        error = query.exception()
        if error is None:
            logger.log(level, "query %s stopped unexpectedly", query.name)
        else:
            logger.log(level, "query %s failed: %s", query.name, error)


def stop_queries(queries: Sequence[StreamingQuery], drain_timeout_s: float = DRAIN_TIMEOUT_SECONDS) -> None:
    """Stop every query, first letting in-flight micro-batches finish within a shared deadline.

    An interrupted lake batch is simply re-run after a restart, but the files it already
    wrote stay behind as orphans that non-Spark parquet readers would see.
    """
    deadline = time.monotonic() + drain_timeout_s
    for query in queries:
        while query.isActive and query.status["isTriggerActive"] and time.monotonic() < deadline:
            time.sleep(_DRAIN_POLL_SECONDS)
        try:
            query.stop()
        except Exception:  # one failing stop must not leave the remaining queries running
            logger.exception("failed to stop query %s", query.name)
