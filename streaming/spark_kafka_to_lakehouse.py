"""Spark Structured Streaming job: Debezium CDC in Kafka -> MinIO lakehouse + ClickHouse.

Startup: write the first heartbeat, build the local-mode driver and hand the seven queries
(see ``streaming_queries``) to the supervisor (see ``streaming_supervisor``). The lake
queries start once Kafka and MinIO answer, ``clickhouse_raw`` once Kafka and ClickHouse
(with its tables) answer; a query that fails is restarted from its checkpoint with a capped
exponential backoff while the others keep running. The supervisor keeps the heartbeat file
(STREAMING_HEARTBEAT_FILE) fresh for the container healthcheck.

Exit codes: 0 after a requested shutdown (SIGTERM/SIGINT), 1 when the Spark driver cannot be
started or stops (Docker's ``restart: unless-stopped`` then restarts the job, which resumes
from its checkpoints), 2 for an invalid configuration.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import FrameType

import clickhouse_sink
from pyspark.sql import SparkSession
from streaming_config import ConfigError, StreamingSettings, load_settings, spark_conf
from streaming_heartbeat import HeartbeatWriter
from streaming_queries import build_query_groups
from streaming_supervisor import QuerySupervisor

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


def driver_alive(spark: SparkSession) -> bool:
    """False once the SparkContext has stopped or its JVM is gone: no query can run any more."""
    try:
        return not spark.sparkContext._jsc.sc().isStopped()
    except Exception:  # the py4j gateway died with the JVM
        return False


def run(settings: StreamingSettings, stop_event: threading.Event, heartbeat: HeartbeatWriter) -> int:
    """Build the driver, then supervise the streaming queries until shutdown."""
    spark = build_spark_session(settings)
    clickhouse = clickhouse_sink.ClickHouseTarget(settings, stop_event.wait)
    try:
        # Creating the SparkContext replaced our SIGINT handler with PySpark's own.
        install_signal_handlers(stop_event)
        groups = build_query_groups(spark, settings, clickhouse)
        supervisor = QuerySupervisor(groups, stop_event, heartbeat, driver_alive=partial(driver_alive, spark))
        return supervisor.run()
    finally:
        clickhouse.close()
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
    heartbeat = HeartbeatWriter(Path(settings.heartbeat_file))
    heartbeat.publish({"status": "starting", "queries": {}}, force=True)  # fresh before Spark boots
    logger.info("starting with %s", settings)
    if not settings.kafka_fail_on_data_loss:
        logger.warning(
            "KAFKA_FAIL_ON_DATA_LOSS=false: records deleted by the Kafka topic retention before a "
            "query read them are skipped with only a Spark warning"
        )
    try:
        exit_code = run(settings, stop_event, heartbeat)
    except Exception:
        logger.exception("streaming job failed")
        return EXIT_FAILURE
    logger.info("streaming job exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
