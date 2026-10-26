"""Environment configuration of the data-science jobs (interface contract, section 2).

Compose injects every variable; the defaults below let the jobs also run on a
developer host with nothing set (services reached on localhost + host ports).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

PROMOTION_POLICIES = ("better", "always")


class ConfigError(ValueError):
    """An environment variable holds an invalid value."""


@dataclass(frozen=True)
class ClickHouseSettings:
    """Connection parameters of the ClickHouse HTTP interface."""

    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str


@dataclass(frozen=True)
class S3Settings:
    """Connection parameters of the MinIO / S3 lakehouse bucket."""

    endpoint_url: str
    access_key_id: str
    secret_access_key: str = field(repr=False)
    region: str
    bucket: str


@dataclass(frozen=True)
class ModelSettings:
    """Labelling, training and serving parameters of the RUL model."""

    failure_horizon_cycles: int
    rul_cap_cycles: int
    min_training_lifecycles: int
    flight_hours_per_cycle: float
    promotion_policy: str
    artifact_prefix: str


@dataclass(frozen=True)
class Settings:
    """Complete configuration of a data-science job."""

    clickhouse: ClickHouseSettings
    s3: S3Settings
    model: ModelSettings
    snapshot_retention: int
    log_level: str


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip()
    return value or default


def _integer(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = _text(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _text(env, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not value > 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _text(env, name, default).lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value


def _model_settings(env: Mapping[str, str]) -> ModelSettings:
    horizon = _integer(env, "RUL_FAILURE_HORIZON_CYCLES", 25, minimum=1)
    cap = _integer(env, "RUL_CAP_CYCLES", 125, minimum=2)
    if cap <= horizon:
        raise ConfigError(f"RUL_CAP_CYCLES ({cap}) must exceed RUL_FAILURE_HORIZON_CYCLES ({horizon})")
    prefix = _text(env, "ML_ARTIFACT_PREFIX", "ml-artifacts/rul_model").strip("/")
    if not prefix:
        raise ConfigError("ML_ARTIFACT_PREFIX must not be empty")
    return ModelSettings(
        failure_horizon_cycles=horizon,
        rul_cap_cycles=cap,
        min_training_lifecycles=_integer(env, "MIN_TRAINING_LIFECYCLES", 6, minimum=1),
        flight_hours_per_cycle=_positive_float(env, "FLIGHT_HOURS_PER_CYCLE", 2.0),
        promotion_policy=_choice(env, "MODEL_PROMOTION_POLICY", "better", PROMOTION_POLICIES),
        artifact_prefix=prefix,
    )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read and validate the job configuration from ``env`` (defaults to os.environ)."""
    source = os.environ if env is None else env
    return Settings(
        clickhouse=ClickHouseSettings(
            host=_text(source, "CLICKHOUSE_HOST", "localhost"),
            port=_integer(source, "CLICKHOUSE_PORT", 8123, minimum=1),
            user=_text(source, "CLICKHOUSE_USER", "clickhouse"),
            password=source.get("CLICKHOUSE_PASSWORD", "clickhouse"),
            database=_text(source, "CLICKHOUSE_DB", "iot_engine"),
        ),
        s3=S3Settings(
            endpoint_url=_text(source, "S3_ENDPOINT_URL", "http://localhost:9000"),
            access_key_id=_text(source, "AWS_ACCESS_KEY_ID", "minioadmin"),
            secret_access_key=source.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            region=_text(source, "AWS_REGION", "us-east-1"),
            bucket=_text(source, "LAKEHOUSE_BUCKET", "lakehouse"),
        ),
        model=_model_settings(source),
        snapshot_retention=_integer(source, "SNAPSHOT_RETENTION", 5, minimum=1),
        log_level=_text(source, "LOG_LEVEL", "INFO"),
    )
