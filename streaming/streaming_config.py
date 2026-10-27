"""Environment configuration of the streaming job (contract section 2).

Variable names and non-network defaults follow the contract. Network endpoints default to
the host-published ports on ``localhost`` so that the module (and its unit tests) work with
no environment set; docker compose always injects the in-network service addresses.

The job itself runs in its image (compose service ``streaming``, ``make stream``): the
image bakes the Kafka source, Avro and S3A jars (spark-sql-kafka-0-10 with its Kafka client
dependencies, spark-avro, hadoop-aws, aws-java-sdk-bundle) into ``pyspark/jars``, puts the
job's modules on the PYTHONPATH of Spark's Python workers and creates the default
``CHECKPOINT_DIR``. A pip-installed PySpark has none of these jars, so a plain host run fails
with "Failed to find data source: kafka" or a missing S3AFileSystem class. A host run
additionally needs Java 17, the requirements.txt packages, the jars resolved at startup, for
example (from ``streaming/``, so that the workers find the job's modules)::

    PKGS=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9,org.apache.spark:spark-avro_2.12:3.5.9
    PKGS=$PKGS,org.apache.hadoop:hadoop-aws:3.3.4
    PYTHONPATH=. PYSPARK_SUBMIT_ARGS="--packages $PKGS pyspark-shell" python spark_kafka_to_lakehouse.py

a writable ``CHECKPOINT_DIR`` (plus winutils on Windows), HF_SCHEMA_PATH pointing at
``infrastructure/schemas/engine_sensor_frame.avsc``, and the ``streaming`` container stopped
(or another ``LAKEHOUSE_BUCKET``): a second checkpoint lineage writing the same file-sink
paths makes Spark skip batch ids it believes are already committed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

SPARK_LOG_LEVELS = frozenset({"ALL", "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"})
PYTHON_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
SHUFFLE_PARTITIONS = 4
# mapInPandas ships the HF frames (about 20 KB each at the default rates) to the Python workers
# in Arrow batches of at most this many rows, so the memory of a feature task stays flat however
# many frames a micro-batch holds (a backfill burst included).
ARROW_MAX_RECORDS_PER_BATCH = 256
# Caps of the Avro decoder, read once when the JVM loads it: a corrupted length prefix then fails
# the decoding of its record (which is dead-lettered) instead of allocating up to 2 GB. They sit
# far above any valid frame (HF topic messages are limited to 8 MB, a frame has a few hundred
# channels at most).
AVRO_DECODER_LIMITS: tuple[str, ...] = (
    "-Dorg.apache.avro.limits.bytes.maxLength=16777216",
    "-Dorg.apache.avro.limits.string.maxLength=65536",
    "-Dorg.apache.avro.limits.collectionItems.maxLength=1000000",
)
_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})
_JVM_MEMORY = re.compile(r"^[1-9]\d*[kmgt]$", re.IGNORECASE)
_S3A_CREDENTIALS_PROVIDER = "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"


class ConfigError(ValueError):
    """An environment variable holds an unusable value."""


@dataclass(frozen=True, slots=True)
class StreamingSettings:
    """Validated runtime settings; secrets are excluded from ``repr`` so it is safe to log."""

    kafka_bootstrap_servers: str
    cdc_topic_prefix: str
    kafka_starting_offsets: str
    max_offsets_per_trigger: int
    # True: a query whose unread offsets were deleted by the topic retention fails instead of
    # silently skipping them (Spark's ``failOnDataLoss``).
    kafka_fail_on_data_loss: bool
    hf_topic: str
    hf_schema_path: str
    hf_max_offsets_per_trigger: int
    # The HF path is best effort, so the default is False: frames the topic retention deleted
    # before a query read them are skipped with a Spark warning.
    hf_fail_on_data_loss: bool
    hf_lake_trigger_seconds: float
    hf_clickhouse_trigger_seconds: float
    # Age at which MinIO expires raw frames (minio-init lifecycle rule); the bronze file sink
    # drops them from its metadata log after the same time.
    hf_bronze_retention_days: int
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str = field(repr=False)
    aws_region: str
    lakehouse_bucket: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str = field(repr=False)
    clickhouse_db: str
    checkpoint_dir: str
    heartbeat_file: str
    lake_trigger_seconds: float
    clickhouse_trigger_seconds: float
    spark_driver_memory: str
    spark_log_level: str
    log_level: str

    def topic(self, table: str) -> str:
        """Debezium topic of a PostgreSQL table in the ``public`` schema."""
        return f"{self.cdc_topic_prefix}.public.{table}"

    def lake_path(self, zone: str, table: str) -> str:
        """S3A URI of a table folder inside a medallion zone (bronze/silver)."""
        return f"s3a://{self.lakehouse_bucket}/{zone}/{table}"

    def checkpoint_path(self, query_name: str) -> str:
        """Checkpoint location of one streaming query: ``CHECKPOINT_DIR/<query_name>``."""
        return f"{self.checkpoint_dir.rstrip('/')}/{query_name}"

    @property
    def clickhouse_ping_url(self) -> str:
        """Unauthenticated liveness endpoint of the ClickHouse HTTP interface."""
        return f"http://{self.clickhouse_host}:{self.clickhouse_port}/ping"

    @property
    def s3_health_url(self) -> str:
        """MinIO liveness endpoint; any non-5xx answer proves the S3 endpoint is serving."""
        return f"{self.s3_endpoint_url.rstrip('/')}/minio/health/live"


def load_settings(env: Mapping[str, str]) -> StreamingSettings:
    """Build :class:`StreamingSettings` from ``env``, raising :class:`ConfigError` on bad values."""
    return StreamingSettings(
        kafka_bootstrap_servers=_text(env, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        cdc_topic_prefix=_text(env, "CDC_TOPIC_PREFIX", "engine_cdc"),
        kafka_starting_offsets=_text(env, "KAFKA_STARTING_OFFSETS", "earliest"),
        max_offsets_per_trigger=_positive_int(env, "MAX_OFFSETS_PER_TRIGGER", 20000),
        kafka_fail_on_data_loss=_boolean(env, "KAFKA_FAIL_ON_DATA_LOSS", True),
        hf_topic=_text(env, "HF_TOPIC", "engine_hf.sensor_frames"),
        hf_schema_path=_text(env, "HF_SCHEMA_PATH", "/schemas/engine_sensor_frame.avsc"),
        hf_max_offsets_per_trigger=_positive_int(env, "HF_MAX_OFFSETS_PER_TRIGGER", 2000),
        hf_fail_on_data_loss=_boolean(env, "HF_FAIL_ON_DATA_LOSS", False),
        hf_lake_trigger_seconds=_positive_float(env, "HF_LAKE_TRIGGER_SECONDS", 60.0),
        hf_clickhouse_trigger_seconds=_positive_float(env, "HF_CLICKHOUSE_TRIGGER_SECONDS", 10.0),
        hf_bronze_retention_days=_positive_int(env, "HF_BRONZE_RETENTION_DAYS", 1),
        s3_endpoint_url=_http_url(env, "S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=_text(env, "AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=_secret(env, "AWS_SECRET_ACCESS_KEY", "minioadmin"),
        aws_region=_text(env, "AWS_REGION", "us-east-1"),
        lakehouse_bucket=_text(env, "LAKEHOUSE_BUCKET", "lakehouse"),
        clickhouse_host=_text(env, "CLICKHOUSE_HOST", "localhost"),
        clickhouse_port=_positive_int(env, "CLICKHOUSE_PORT", 8123),
        clickhouse_user=_text(env, "CLICKHOUSE_USER", "clickhouse"),
        clickhouse_password=_secret(env, "CLICKHOUSE_PASSWORD", "clickhouse"),
        clickhouse_db=_text(env, "CLICKHOUSE_DB", "iot_engine"),
        checkpoint_dir=_text(env, "CHECKPOINT_DIR", "/opt/spark-checkpoints"),
        heartbeat_file=_text(env, "STREAMING_HEARTBEAT_FILE", "/tmp/streaming.heartbeat"),
        lake_trigger_seconds=_positive_float(env, "LAKE_TRIGGER_SECONDS", 60.0),
        clickhouse_trigger_seconds=_positive_float(env, "CLICKHOUSE_TRIGGER_SECONDS", 5.0),
        spark_driver_memory=_jvm_memory(env, "SPARK_DRIVER_MEMORY", "2g"),
        spark_log_level=_choice(env, "SPARK_LOG_LEVEL", "WARN", SPARK_LOG_LEVELS),
        log_level=_choice(env, "LOG_LEVEL", "INFO", PYTHON_LOG_LEVELS),
    )


def spark_conf(settings: StreamingSettings) -> dict[str, str]:
    """Spark properties for the local-mode driver, UTC semantics and S3A on MinIO."""
    secure = urlparse(settings.s3_endpoint_url).scheme == "https"
    return {
        "spark.driver.memory": settings.spark_driver_memory,
        # Pin the JVM default zone as well: log4j and a few Hadoop paths ignore the session zone.
        "spark.driver.extraJavaOptions": " ".join(("-Duser.timezone=UTC", *AVRO_DECODER_LIMITS)),
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.shuffle.partitions": str(SHUFFLE_PARTITIONS),
        # AQE never applies here (streaming plans reject it with a warning per query, and the
        # foreachBatch collects have no shuffle), so it is switched off explicitly.
        "spark.sql.adaptive.enabled": "false",
        # Spark's default INT96 is legacy; INT64 micros (UTC-adjusted) is what pyarrow/DuckDB expect.
        "spark.sql.parquet.outputTimestampType": "TIMESTAMP_MICROS",
        "spark.sql.execution.arrow.maxRecordsPerBatch": str(ARROW_MAX_RECORDS_PER_BATCH),
        "spark.ui.showConsoleProgress": "false",
        # A driver that runs for weeks must not accumulate UI history without bound.
        "spark.ui.retainedJobs": "200",
        "spark.ui.retainedStages": "200",
        "spark.sql.ui.retainedExecutions": "100",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.endpoint": settings.s3_endpoint_url,
        "spark.hadoop.fs.s3a.endpoint.region": settings.aws_region,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": str(secure).lower(),
        "spark.hadoop.fs.s3a.aws.credentials.provider": _S3A_CREDENTIALS_PROVIDER,
        "spark.hadoop.fs.s3a.access.key": settings.aws_access_key_id,
        "spark.hadoop.fs.s3a.secret.key": settings.aws_secret_access_key,
    }


def trigger_interval(seconds: float) -> str:
    """Spark processing-time trigger interval for ``seconds`` (at least one millisecond)."""
    return f"{max(1, round(seconds * 1000))} milliseconds"


def _raw(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None else value.strip()


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = _raw(env, name, default)
    if not value:
        raise ConfigError(f"{name} must not be empty")
    return value


def _secret(env: Mapping[str, str], name: str, default: str) -> str:
    # Secrets are taken verbatim: an explicitly empty password is a legitimate setting.
    value = env.get(name)
    return default if value is None else value


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _raw(env, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _raw(env, name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise ConfigError(f"{name} must be a finite number > 0, got {raw!r}")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _raw(env, name, str(default)).lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true or false, got {value!r}")


def _choice(env: Mapping[str, str], name: str, default: str, allowed: frozenset[str]) -> str:
    value = _raw(env, name, default).upper()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _http_url(env: Mapping[str, str], name: str, default: str) -> str:
    value = _text(env, name, default)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{name} must be an http(s) URL, got {value!r}")
    return value


def _jvm_memory(env: Mapping[str, str], name: str, default: str) -> str:
    value = _text(env, name, default)
    if not _JVM_MEMORY.match(value):
        raise ConfigError(f"{name} must look like 512m or 2g, got {value!r}")
    return value
