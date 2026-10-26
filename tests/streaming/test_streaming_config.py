"""Unit tests of the environment configuration and the derived Spark properties."""

from __future__ import annotations

import pytest
from streaming_config import ConfigError, load_settings, spark_conf

COMPOSE_ENV = {
    "KAFKA_BOOTSTRAP_SERVERS": "kafka:29092",
    "CDC_TOPIC_PREFIX": "engine_cdc",
    "S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "s3-secret",
    "AWS_REGION": "us-east-1",
    "LAKEHOUSE_BUCKET": "lakehouse",
    "CLICKHOUSE_HOST": "clickhouse",
    "CLICKHOUSE_PORT": "8123",
    "CLICKHOUSE_USER": "clickhouse",
    "CLICKHOUSE_PASSWORD": "ch-secret",
    "CLICKHOUSE_DB": "iot_engine",
    "CHECKPOINT_DIR": "/opt/spark-checkpoints",
    "LAKE_TRIGGER_SECONDS": "60",
    "CLICKHOUSE_TRIGGER_SECONDS": "5",
    "KAFKA_STARTING_OFFSETS": "earliest",
    "MAX_OFFSETS_PER_TRIGGER": "20000",
    "SPARK_DRIVER_MEMORY": "2g",
    "SPARK_LOG_LEVEL": "WARN",
}


def test_defaults_target_host_published_ports() -> None:
    settings = load_settings({})
    assert settings.kafka_bootstrap_servers == "localhost:9092"
    assert settings.s3_endpoint_url == "http://localhost:9000"
    assert (settings.clickhouse_host, settings.clickhouse_port) == ("localhost", 8123)
    assert (settings.clickhouse_user, settings.clickhouse_password) == ("clickhouse", "clickhouse")
    assert settings.clickhouse_db == "iot_engine"
    assert (settings.aws_access_key_id, settings.aws_secret_access_key) == ("minioadmin", "minioadmin")
    assert (settings.aws_region, settings.lakehouse_bucket) == ("us-east-1", "lakehouse")
    assert settings.cdc_topic_prefix == "engine_cdc"
    assert settings.checkpoint_dir == "/opt/spark-checkpoints"
    assert settings.heartbeat_file == "/tmp/streaming.heartbeat"
    assert settings.kafka_fail_on_data_loss is True
    assert (settings.lake_trigger_seconds, settings.clickhouse_trigger_seconds) == (60.0, 5.0)
    assert (settings.kafka_starting_offsets, settings.max_offsets_per_trigger) == ("earliest", 20000)
    assert (settings.spark_driver_memory, settings.spark_log_level, settings.log_level) == (
        "2g",
        "WARN",
        "INFO",
    )


def test_compose_environment_is_honoured() -> None:
    settings = load_settings(COMPOSE_ENV)
    assert settings.kafka_bootstrap_servers == "kafka:29092"
    assert settings.clickhouse_ping_url == "http://clickhouse:8123/ping"
    assert settings.s3_health_url == "http://minio:9000/minio/health/live"
    assert settings.topic("engine_telemetry") == "engine_cdc.public.engine_telemetry"
    assert settings.lake_path("silver", "maintenance_log") == "s3a://lakehouse/silver/maintenance_log"
    assert (
        settings.checkpoint_path("bronze_engine_telemetry")
        == "/opt/spark-checkpoints/bronze_engine_telemetry"
    )


def test_values_are_trimmed_and_levels_normalised() -> None:
    settings = load_settings({"CDC_TOPIC_PREFIX": " cdc ", "SPARK_LOG_LEVEL": "info", "LOG_LEVEL": "debug"})
    assert (settings.cdc_topic_prefix, settings.spark_log_level, settings.log_level) == (
        "cdc",
        "INFO",
        "DEBUG",
    )


def test_fractional_trigger_seconds_are_accepted() -> None:
    assert load_settings({"CLICKHOUSE_TRIGGER_SECONDS": "0.5"}).clickhouse_trigger_seconds == 0.5


def test_empty_password_is_kept_verbatim() -> None:
    assert load_settings({"CLICKHOUSE_PASSWORD": ""}).clickhouse_password == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        (" False ", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_fail_on_data_loss_is_a_boolean_switch(value: str, expected: bool) -> None:
    assert load_settings({"KAFKA_FAIL_ON_DATA_LOSS": value}).kafka_fail_on_data_loss is expected


def test_heartbeat_file_is_configurable() -> None:
    assert load_settings({"STREAMING_HEARTBEAT_FILE": "/run/streaming.json"}).heartbeat_file == (
        "/run/streaming.json"
    )


def test_repr_hides_secrets() -> None:
    text = repr(load_settings(COMPOSE_ENV))
    assert "s3-secret" not in text
    assert "ch-secret" not in text
    assert "kafka:29092" in text


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_OFFSETS_PER_TRIGGER", "many"),
        ("MAX_OFFSETS_PER_TRIGGER", "0"),
        ("CLICKHOUSE_PORT", "-1"),
        ("LAKE_TRIGGER_SECONDS", "soon"),
        ("LAKE_TRIGGER_SECONDS", "0"),
        ("CLICKHOUSE_TRIGGER_SECONDS", "nan"),
        ("CLICKHOUSE_TRIGGER_SECONDS", "inf"),
        ("SPARK_LOG_LEVEL", "LOUD"),
        ("LOG_LEVEL", "TRACE"),
        ("S3_ENDPOINT_URL", "minio:9000"),
        ("SPARK_DRIVER_MEMORY", "2 GB"),
        ("SPARK_DRIVER_MEMORY", "0g"),
        ("KAFKA_BOOTSTRAP_SERVERS", "   "),
        ("KAFKA_FAIL_ON_DATA_LOSS", "maybe"),
        ("KAFKA_FAIL_ON_DATA_LOSS", ""),
        ("STREAMING_HEARTBEAT_FILE", " "),
    ],
)
def test_invalid_values_raise_config_error(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


def test_spark_conf_targets_minio_with_utc_semantics() -> None:
    settings = load_settings(COMPOSE_ENV)
    conf = spark_conf(settings)
    assert conf["spark.driver.memory"] == "2g"
    assert conf["spark.sql.session.timeZone"] == "UTC"
    assert "-Duser.timezone=UTC" in conf["spark.driver.extraJavaOptions"]
    assert conf["spark.sql.shuffle.partitions"] == "4"
    assert conf["spark.sql.adaptive.enabled"] == "false"
    assert conf["spark.sql.parquet.outputTimestampType"] == "TIMESTAMP_MICROS"
    assert conf["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
    assert conf["spark.hadoop.fs.s3a.path.style.access"] == "true"
    assert conf["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "false"
    assert conf["spark.hadoop.fs.s3a.aws.credentials.provider"].endswith("SimpleAWSCredentialsProvider")
    assert conf["spark.hadoop.fs.s3a.access.key"] == "minioadmin"
    assert conf["spark.hadoop.fs.s3a.secret.key"] == "s3-secret"


def test_spark_conf_enables_ssl_for_https_endpoints() -> None:
    settings = load_settings({"S3_ENDPOINT_URL": "https://s3.example.com"})
    assert spark_conf(settings)["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "true"
