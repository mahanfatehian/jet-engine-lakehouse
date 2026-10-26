"""Spark Structured Streaming job: Debezium CDC in Kafka -> MinIO lakehouse + ClickHouse.

Startup: wait for ClickHouse, MinIO and Kafka, check the ClickHouse tables, start the
seven queries (see ``streaming_queries``) in one local-mode driver, then supervise them.

Exit codes: 0 after a requested shutdown (SIGTERM/SIGINT), 1 when startup or any query
fails (Docker's ``restart: unless-stopped`` then restarts the job, which resumes from its
checkpoints), 2 for an invalid configuration.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import closing
from types import FrameType

import clickhouse_sink
from cdc_schemas import DEAD_LETTER_TABLE, TABLES
from pyspark.sql import SparkSession
from streaming_config import ConfigError, StreamingSettings, load_settings, spark_conf
from streaming_queries import EXIT_OK, start_queries, supervise
from streaming_readiness import wait_for_dependencies
from streaming_retry import RetryAbortedError

logger = logging.getLogger("spark_kafka_to_lakehouse")

APP_NAME = "iot-engine-cdc-lakehouse"
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2


def configure_logging(level: str) -> None:
    """Log to stdout with UTC ISO-8601 timestamps."""
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # py4j logs every gateway round-trip below WARNING.
    logging.getLogger("py4j").setLevel(logging.WARNING)


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Turn SIGTERM/SIGINT into a stop request handled by the main loop.

    The handler only sets the event: logging from a signal handler can hit a re-entrant
    write on stdout while the interrupted main thread is itself logging.
    """

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)


def build_spark_session(settings: StreamingSettings) -> SparkSession:
    """Local-mode session; driver memory takes effect because the JVM is launched here."""
    builder = SparkSession.builder.appName(APP_NAME).master("local[*]")
    for key, value in spark_conf(settings).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(settings.spark_log_level)
    return spark


def run(settings: StreamingSettings, stop_event: threading.Event) -> int:
    """Startup sequence followed by supervision of the streaming queries."""
    wait_for_dependencies(settings, stop_event.wait)
    with closing(clickhouse_sink.connect(settings, stop_event.wait)) as client:
        required_tables = [table.clickhouse_table for table in TABLES] + [DEAD_LETTER_TABLE]
        clickhouse_sink.verify_tables(client, settings.clickhouse_db, required_tables, stop_event.wait)
        spark = build_spark_session(settings)
        try:
            # Creating the SparkContext replaced our SIGINT handler with PySpark's own.
            install_signal_handlers(stop_event)
            if stop_event.is_set():
                logger.info("shutdown requested before the queries started")
                return EXIT_OK
            writer = clickhouse_sink.ClickHouseWriter(client, settings.clickhouse_db, stop_event.wait)
            queries = start_queries(spark, settings, writer)
            logger.info("started %d queries: %s", len(queries), ", ".join(query.name for query in queries))
            return supervise(spark, queries, stop_event)
        finally:
            spark.stop()


def main() -> int:
    """Process entrypoint; returns the exit code."""
    try:
        settings = load_settings(os.environ)
    except ConfigError as exc:
        configure_logging("INFO")
        logger.error("invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR
    configure_logging(settings.log_level)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    logger.info("starting with %s", settings)
    try:
        exit_code = run(settings, stop_event)
    except RetryAbortedError as exc:
        logger.info("shutdown requested during startup: %s", exc)
        return EXIT_OK
    except Exception:
        logger.exception("streaming job failed")
        return EXIT_FAILURE
    logger.info("streaming job exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
