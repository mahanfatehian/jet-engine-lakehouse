"""Environment configuration of the data-science jobs (interface contract, section 2).

Compose injects every variable; the defaults below let the jobs also run on a
developer host with nothing set (services reached on localhost + host ports).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

PROMOTION_POLICIES = ("better", "always")
# The train/validation/test split needs one lifecycle per side, whatever
# MIN_TRAINING_LIFECYCLES says; fewer completed lifecycles skip training (exit 3).
MIN_SPLIT_LIFECYCLES = 3


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
class PromotionSettings:
    """Promotion gate: absolute quality floors plus the champion comparison tolerance.

    The floors apply to every promotion, also under policy ``always`` and when no
    champion exists; ``f1_tolerance`` is how much holdout classifier F1 a challenger
    may lose against a comparable champion while still replacing it.
    """

    min_r2: float = 0.5
    min_roc_auc: float = 0.75
    f1_tolerance: float = 0.05


@dataclass(frozen=True)
class ModelSettings:
    """Labelling, training and serving parameters of the RUL model."""

    failure_horizon_cycles: int
    rul_cap_cycles: int
    min_training_lifecycles: int
    flight_hours_per_cycle: float
    promotion_policy: str
    artifact_prefix: str
    training_max_rows: int = 600_000
    promotion: PromotionSettings = field(default_factory=PromotionSettings)


@dataclass(frozen=True)
class Settings:
    """Complete configuration of a data-science job."""

    clickhouse: ClickHouseSettings
    s3: S3Settings
    model: ModelSettings
    snapshot_retention: int
    log_level: str
    snapshot_window_days: int = 7


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


def _float_in_range(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = _text(env, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not (math.isfinite(value) and low <= value <= high):
        raise ConfigError(f"{name} must be a finite number in [{low:g}, {high:g}], got {raw!r}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _text(env, name, default).lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value


def _promotion_settings(env: Mapping[str, str]) -> PromotionSettings:
    defaults = PromotionSettings()
    return PromotionSettings(
        min_r2=_float_in_range(env, "PROMOTION_MIN_R2", defaults.min_r2, -math.inf, 1.0),
        min_roc_auc=_float_in_range(env, "PROMOTION_MIN_ROC_AUC", defaults.min_roc_auc, 0.0, 1.0),
        f1_tolerance=_float_in_range(env, "PROMOTION_F1_TOLERANCE", defaults.f1_tolerance, 0.0, 1.0),
    )


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
        training_max_rows=_integer(env, "TRAINING_MAX_ROWS", 600_000, minimum=1000),
        promotion=_promotion_settings(env),
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
        snapshot_window_days=_integer(source, "SNAPSHOT_WINDOW_DAYS", 7, minimum=1),
    )
