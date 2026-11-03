"""Spark Structured Streaming job: Kafka (Debezium CDC + HF sensor frames) -> MinIO + ClickHouse.

Startup: write the first heartbeat, build the local-mode driver and hand the ten queries
(seven CDC, three high-frequency, see ``streaming_queries`` and ``hf_queries``) to the
supervisor (see ``streaming_supervisor``). The lake queries start once Kafka and MinIO
answer, the ClickHouse queries once Kafka and ClickHouse (with its tables) answer; a query
that fails is restarted from its checkpoint with a capped exponential backoff while the
others keep running. The supervisor keeps the heartbeat file (STREAMING_HEARTBEAT_FILE)
fresh for the container healthcheck. A listener buffers the progress of every micro-batch, and
a thread of its own writes it to ClickHouse (``ops_stream_progress``, see ``streaming_progress``).

Exit codes: 0 after a requested shutdown (SIGTERM/SIGINT), 1 when the Spark driver cannot be
started or stops (Docker's ``restart: unless-stopped`` then restarts the job, which resumes
from its checkpoints), 2 for an invalid configuration.

``--check-dependencies`` is a reachability probe: it checks once, with the job's settings, that
Kafka accepts a TCP connection and that MinIO and ClickHouse answer HTTP, logs one line per
dependency that does not answer, and exits 0 when all of them answer, 1 otherwise. ClickHouse
credentials and tables are checked by the job itself when it starts its ClickHouse queries. It
starts no Spark driver and leaves the heartbeat file alone, so it can run next to the job:
``docker exec de_streaming python spark_kafka_to_lakehouse.py --check-dependencies``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from types import FrameType

import clickhouse_sink
from pyspark.sql import SparkSession
from streaming_config import ConfigError, StreamingSettings, load_settings, spark_conf
from streaming_heartbeat import HeartbeatWriter
from streaming_progress import ProgressBuffer
from streaming_queries import ProgressRecorder, build_query_groups
from streaming_readiness import Probe, dependency_probes, unavailable_dependencies
from streaming_supervisor import EXIT_OK, QuerySupervisor

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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Command-line options; an unknown option exits with code 2 (argparse), like a bad setting."""
    parser = argparse.ArgumentParser(description="Kafka -> MinIO + ClickHouse streaming job.")
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="reachability probe: check once that Kafka, MinIO and ClickHouse answer, report each one "
        "that does not, and exit (credentials and tables are checked when the job starts its queries)",
    )
    return parser.parse_args(argv)


def check_dependencies(probes: Sequence[Probe]) -> int:
    """Run every probe once; ``EXIT_OK`` when all answer, else one error line each and ``EXIT_FAILURE``."""
    failures = unavailable_dependencies(probes)
    for failure in failures:
        logger.error("dependency not ready: %s", failure)
    if failures:
        return EXIT_FAILURE
    logger.info("dependencies answer: %s", ", ".join(name for name, _ in probes))
    return EXIT_OK


def run(settings: StreamingSettings, stop_event: threading.Event, heartbeat: HeartbeatWriter) -> int:
    """Build the driver, then supervise the streaming queries until shutdown."""
    spark = build_spark_session(settings)
    clickhouse = clickhouse_sink.ClickHouseTarget(settings, stop_event.wait)
    progress = ProgressBuffer(lambda: clickhouse.writer)
    try:
        # Creating the SparkContext replaced our SIGINT handler with PySpark's own.
        install_signal_handlers(stop_event)
        groups = build_query_groups(spark, settings, clickhouse)
        spark.streams.addListener(ProgressRecorder(progress))
        threading.Thread(target=progress.run, args=(stop_event,), name="stream-progress", daemon=True).start()
        supervisor = QuerySupervisor(groups, stop_event, heartbeat, driver_alive=partial(driver_alive, spark))
        return supervisor.run()
    finally:
        spark.stop()  # first: it delivers the progress events still queued in Spark's listener bus
        progress.flush()
        clickhouse.close()


def main(argv: Sequence[str] = ()) -> int:
    """Process entrypoint (``argv`` without the program name); returns the exit code."""
    options = parse_args(argv)
    try:
        settings = load_settings(os.environ)
    except ConfigError as exc:
        configure_logging("INFO")
        logger.error("invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR
    configure_logging(settings.log_level)
    if options.check_dependencies:
        return check_dependencies(dependency_probes(settings))
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
    if not settings.hf_fail_on_data_loss:
        logger.info(
            "HF_FAIL_ON_DATA_LOSS=false (best effort): sensor frames deleted by the retention of %s "
            "before an HF query read them are skipped with a Spark warning",
            settings.hf_topic,
        )
    try:
        exit_code = run(settings, stop_event, heartbeat)
    except Exception:
        logger.exception("streaming job failed")
        return EXIT_FAILURE
    logger.info("streaming job exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
