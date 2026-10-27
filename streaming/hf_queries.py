"""The three high-frequency (HF) streaming queries and the micro-batch handler of the feature query.

Group ``lake`` (Kafka + MinIO), parquet file sinks, each with its own Kafka source and checkpoint:
    bronze_engine_sensor_frames  raw records -> s3a://<bucket>/bronze/engine_sensor_frames/
                                                ingest_date=YYYY-MM-DD/ingest_hour=H/
    silver_engine_hf_features    features    -> s3a://<bucket>/silver/engine_hf_features/event_date=YYYY-MM-DD/
Group ``clickhouse`` (Kafka + ClickHouse), at-least-once through ``foreachBatch``:
    clickhouse_hf_features       features    -> raw_engine_hf_features (ReplacingMergeTree keyed by
                                                engine_id, telemetry_id: replays collapse), one
                                                hf_ingest_batches row per batch with frames,
                                                raw_cdc_dead_letter for rejected records

Unlike the CDC path the HF path is best effort. HF_FAIL_ON_DATA_LOSS defaults to false: frames
the topic retention (HF_TOPIC_RETENTION_MS, 2 h) deleted before a query read them are skipped
with a Spark warning instead of failing the query. Each HF query checks its own prerequisites
when it starts (the schema file, the HF ClickHouse tables), so a missing one keeps only that
query retrying while the CDC queries of its group run on.

Volume: a trigger reads at most HF_MAX_OFFSETS_PER_TRIGGER frames, so a backlog such as the
simulator's backfill is worked off in bounded micro-batches; each query then sustains up to
that many frames per trigger interval (2000 per 60 s for the lake and per 10 s for ClickHouse
with the defaults). Arrow batches of ``ARROW_MAX_RECORDS_PER_BATCH`` frames bound the memory of
the feature workers.

Raw frames expire from MinIO after HF_BRONZE_RETENTION_DAYS (lifecycle rule of minio-init).
The bronze sink's ``retention`` option (see :func:`bronze_log_retention`) drops entries older
than that, minus an hour, from ``_spark_metadata``, but only in a compaction batch (every 10
micro-batches), which runs only while frames arrive: the log stays bounded, yet after an idle
period it can list objects MinIO already expired, so Spark readers of the bronze root set
``spark.sql.files.ignoreMissingFiles=true``. The rule must spare ``_spark_metadata`` itself: a
query whose log objects expired while it was stopped fails on restart with
BATCH_METADATA_NOT_FOUND. Readers that
list the files themselves deduplicate bronze by (kafka_partition, kafka_offset) and silver by
(engine_id, telemetry_id), as for the CDC lake (see ``streaming_queries``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Protocol

from cdc_schemas import DEAD_LETTER_COLUMNS, DEAD_LETTER_TABLE
from clickhouse_rows import Row, RowWriter, dead_letter_row
from hf_decode import (
    BRONZE_PARTITION_COLUMNS,
    SILVER_PARTITION_COLUMN,
    bronze_frames,
    clickhouse_features,
    dead_letters,
    decode_frames,
    frame_features,
    hf_kafka_records,
    load_frame_schema,
    silver_features,
)
from hf_features import FEATURE_COLUMN_NAMES
from hf_rows import (
    HF_BATCH_COLUMNS,
    HF_BATCHES_TABLE,
    HF_FEATURES_TABLE,
    BatchStats,
    batch_row,
    batch_stats,
    feature_rows,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.streaming import StreamingQuery
from streaming_config import StreamingSettings, trigger_interval
from streaming_supervisor import QuerySpec

logger = logging.getLogger(__name__)

BRONZE_QUERY_NAME = "bronze_engine_sensor_frames"
SILVER_QUERY_NAME = "silver_engine_hf_features"
CLICKHOUSE_QUERY_NAME = "clickhouse_hf_features"
BRONZE_TABLE = "engine_sensor_frames"
SILVER_TABLE = "engine_hf_features"
HF_TABLES: tuple[str, ...] = (HF_FEATURES_TABLE, HF_BATCHES_TABLE)
# Kafka fetch size per partition: the HF topic's max.message.bytes. One fetch then returns a
# whole batch of ~20 KB frames; with the 1 MiB default, reading 3000 frames took 3.3 s, not 1.0 s.
HF_FETCH_MAX_BYTES = 8 * 1024 * 1024
_ERROR_POSITION = DEAD_LETTER_COLUMNS.index("error")


class HfClickHouseDestination(Protocol):
    """Schema check and writer of the HF feature query (``clickhouse_sink.ClickHouseTarget``)."""

    def check_tables(self, tables: Iterable[str]) -> None:
        """Raise unless every table exists (one attempt; the supervisor retries)."""

    @property
    def writer(self) -> RowWriter:
        """Writer bound to the connected client."""


@dataclass(frozen=True, slots=True)
class HfQuerySpecs:
    """The HF queries of each supervisor group."""

    lake: tuple[QuerySpec, ...]
    clickhouse: tuple[QuerySpec, ...]


def read_hf_kafka(spark: SparkSession, settings: StreamingSettings, query_name: str) -> DataFrame:
    """Streaming DataFrame of ``hf_decode.HF_KAFKA_COLUMNS`` over HF_TOPIC (one source per query)."""
    source = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.hf_topic)
        .option("startingOffsets", settings.kafka_starting_offsets)
        .option("failOnDataLoss", str(settings.hf_fail_on_data_loss).lower())
        .option("maxOffsetsPerTrigger", str(settings.hf_max_offsets_per_trigger))
        .option("groupIdPrefix", f"iot-engine-{query_name}")
        .option("kafka.max.partition.fetch.bytes", str(HF_FETCH_MAX_BYTES))
        .load()
    )
    return hf_kafka_records(source)


class HfFeatureBatchHandler:
    """``foreachBatch`` callback of ``clickhouse_hf_features``.

    Decodes the micro-batch, extracts the features (``mapInPandas``) and inserts them, the
    dead letters and the batch summary; an insert that still fails after its retries fails
    the batch, which Spark replays from the checkpoint after the supervisor's restart.
    """

    def __init__(
        self,
        writer: RowWriter,
        schema_json: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._writer = writer
        self._schema_json = schema_json
        self._clock = clock
        self._wall_clock = wall_clock

    def __call__(self, batch: DataFrame, batch_id: int) -> None:
        """Process one micro-batch; raises when an insert fails."""
        started_at, started = self._wall_clock(), self._clock()
        # Two collects below: read from Kafka and decode once (see hf_decode.decode_frames).
        decoded = decode_frames(batch, self._schema_json).persist()
        try:
            rejected = [dead_letter_row(record) for record in dead_letters(decoded).collect()]
            rows = feature_rows(clickhouse_features(frame_features(decoded)).collect())
        finally:
            decoded.unpersist()
        self._writer.insert_rows(HF_FEATURES_TABLE, FEATURE_COLUMN_NAMES, rows)
        self._writer.insert_rows(DEAD_LETTER_TABLE, DEAD_LETTER_COLUMNS, rejected)
        stats = batch_stats(rows)
        elapsed_s = self._clock() - started
        if stats is not None:
            summary = batch_row(batch_id, started_at, stats, elapsed_s)
            self._writer.insert_rows(HF_BATCHES_TABLE, HF_BATCH_COLUMNS, [summary])
        _log_batch(batch_id, stats, rejected, elapsed_s)


def _log_batch(batch_id: int, stats: BatchStats | None, rejected: Sequence[Row], elapsed_s: float) -> None:
    frames = stats.frames if stats else 0
    logger.info(
        "hf batch %d: frames=%d dead_letters=%d engines=%d bytes=%d duration_ms=%d rate=%.0f frames/s",
        batch_id,
        frames,
        len(rejected),
        stats.engines if stats else 0,
        stats.bytes if stats else 0,
        round(elapsed_s * 1000),
        frames / elapsed_s if elapsed_s > 0 else 0.0,
    )
    if rejected:
        reasons = sorted({str(row[_ERROR_POSITION]) for row in rejected})
        logger.warning("hf batch %d dead-lettered: %s", batch_id, "; ".join(reasons[:5]))


def build_hf_query_specs(
    spark: SparkSession, settings: StreamingSettings, clickhouse: HfClickHouseDestination
) -> HfQuerySpecs:
    """The restartable HF queries (factories only: nothing is started or read here)."""
    bronze_path = settings.lake_path("bronze", BRONZE_TABLE)
    silver_path = settings.lake_path("silver", SILVER_TABLE)
    return HfQuerySpecs(
        lake=(
            QuerySpec(
                BRONZE_QUERY_NAME,
                partial(_start_bronze_query, spark, settings),
                _lake_recovery(settings, BRONZE_QUERY_NAME, bronze_path),
            ),
            QuerySpec(
                SILVER_QUERY_NAME,
                partial(_start_silver_query, spark, settings),
                _lake_recovery(settings, SILVER_QUERY_NAME, silver_path),
            ),
        ),
        clickhouse=(
            QuerySpec(
                CLICKHOUSE_QUERY_NAME,
                partial(_start_clickhouse_query, spark, settings, clickhouse),
                _clickhouse_recovery(settings),
            ),
        ),
    )


def bronze_log_retention(retention_days: int) -> str:
    """File sink ``retention`` for raw frames that the bucket expires after ``retention_days``.

    One hour shorter than the expiry: the log forgets a file at its next compaction (every 10
    micro-batches) after this age, before the lifecycle rule can delete it.
    """
    return f"{retention_days * 24 - 1}h"


def _start_bronze_query(spark: SparkSession, settings: StreamingSettings) -> StreamingQuery:
    records = bronze_frames(read_hf_kafka(spark, settings, BRONZE_QUERY_NAME))
    return (
        # Copying bytes needs no parallelism: one file per partition directory and micro-batch.
        records.coalesce(1)
        .writeStream.queryName(BRONZE_QUERY_NAME)
        .format("parquet")
        .outputMode("append")
        .partitionBy(*BRONZE_PARTITION_COLUMNS)
        .option("path", settings.lake_path("bronze", BRONZE_TABLE))
        .option("checkpointLocation", settings.checkpoint_path(BRONZE_QUERY_NAME))
        .option("retention", bronze_log_retention(settings.hf_bronze_retention_days))
        .trigger(processingTime=trigger_interval(settings.hf_lake_trigger_seconds))
        .start()
    )


def _start_silver_query(spark: SparkSession, settings: StreamingSettings) -> StreamingQuery:
    schema_json = load_frame_schema(settings.hf_schema_path)
    decoded = decode_frames(read_hf_kafka(spark, settings, SILVER_QUERY_NAME), schema_json)
    return (
        # The features are computed in parallel (a task per Kafka partition), then gathered
        # into one file per partition directory and micro-batch.
        silver_features(frame_features(decoded))
        .repartition(1)
        .writeStream.queryName(SILVER_QUERY_NAME)
        .format("parquet")
        .outputMode("append")
        .partitionBy(SILVER_PARTITION_COLUMN)
        .option("path", settings.lake_path("silver", SILVER_TABLE))
        .option("checkpointLocation", settings.checkpoint_path(SILVER_QUERY_NAME))
        .trigger(processingTime=trigger_interval(settings.hf_lake_trigger_seconds))
        .start()
    )


def _start_clickhouse_query(
    spark: SparkSession, settings: StreamingSettings, clickhouse: HfClickHouseDestination
) -> StreamingQuery:
    schema_json = load_frame_schema(settings.hf_schema_path)
    clickhouse.check_tables(HF_TABLES)
    return (
        read_hf_kafka(spark, settings, CLICKHOUSE_QUERY_NAME)
        .writeStream.queryName(CLICKHOUSE_QUERY_NAME)
        .foreachBatch(HfFeatureBatchHandler(clickhouse.writer, schema_json))
        .option("checkpointLocation", settings.checkpoint_path(CLICKHOUSE_QUERY_NAME))
        .trigger(processingTime=trigger_interval(settings.hf_clickhouse_trigger_seconds))
        .start()
    )


def _clickhouse_recovery(settings: StreamingSettings) -> str:
    return (
        "Recovery (HF_FAIL_ON_DATA_LOSS=true only): run the job with HF_FAIL_ON_DATA_LOSS=false, the "
        "best-effort default, until the query has committed a batch past the gap, or with the job "
        f"stopped delete {settings.checkpoint_path(CLICKHOUSE_QUERY_NAME)} so that it re-reads what "
        "Kafka still retains (raw_engine_hf_features absorbs the replay)."
    )


def _lake_recovery(settings: StreamingSettings, query_name: str, path: str) -> str:
    return (
        "Recovery (HF_FAIL_ON_DATA_LOSS=true only): run the job with HF_FAIL_ON_DATA_LOSS=false, the "
        "best-effort default, until this query has committed a batch past the gap. Do not just delete "
        f"its checkpoint {settings.checkpoint_path(query_name)}: a new run starts again at batch 0 and "
        f"the file sink skips every batch id already committed in {path}/_spark_metadata."
    )
