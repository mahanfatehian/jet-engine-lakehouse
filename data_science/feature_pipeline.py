"""ML feature contract, feature-matrix construction and the snapshot export CLI.

CLI::

    python data_science/feature_pipeline.py snapshot [--retention N] [--window-days D]

exports int_ml_features (silver: the model input of every pulse, last D days of event
time, default SNAPSHOT_WINDOW_DAYS) plus mart_rul_training_features and
fct_engine_health_status (gold) from ClickHouse to the lakehouse bucket as
``<zone>/<table>/snapshot_ts=<TS>/data.parquet`` and keeps the newest N snapshots per
dataset (default SNAPSHOT_RETENTION). Datasets with 0 rows are not published.
Exit codes: 0 done, 3 skipped (source tables missing or every source empty),
1 failure, 143 terminated.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from artifact_store import S3ArtifactStore
from ds_runtime import EXIT_OK, StopToken, configure_logging, positive_int, run_cli
from ds_settings import load_settings
from lakehouse_io import ClickHouseGateway
from lakehouse_snapshots import run_snapshot

RAW_SENSOR_COLUMNS: tuple[str, ...] = (
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
)

# The 46 numeric inputs computed from the 1 Hz telemetry pulse (contract section 7), in a fixed order.
PULSE_FEATURE_COLUMNS: tuple[str, ...] = (
    "cycle_number",
    *RAW_SENSOR_COLUMNS,
    "vibration_mm_s_mean_5m",
    "vibration_mm_s_std_5m",
    "vibration_mm_s_min_5m",
    "vibration_mm_s_max_5m",
    "vibration_mm_s_mean_15m",
    "vibration_mm_s_std_15m",
    "vibration_mm_s_min_15m",
    "vibration_mm_s_max_15m",
    "egt_c_mean_5m",
    "egt_c_std_5m",
    "egt_c_min_5m",
    "egt_c_max_5m",
    "egt_c_mean_15m",
    "egt_c_std_15m",
    "egt_c_min_15m",
    "egt_c_max_15m",
    "oil_pressure_psi_mean_5m",
    "oil_pressure_psi_std_5m",
    "oil_pressure_psi_min_5m",
    "oil_pressure_psi_max_5m",
    "oil_pressure_psi_mean_15m",
    "oil_pressure_psi_std_15m",
    "oil_pressure_psi_min_15m",
    "oil_pressure_psi_max_15m",
    "vibration_mm_s_delta_1",
    "vibration_mm_s_roc_5m",
    "vibration_mm_s_roc_15m",
    "egt_c_delta_1",
    "egt_c_roc_5m",
    "egt_c_roc_15m",
    "oil_pressure_psi_delta_1",
    "oil_pressure_psi_roc_5m",
    "oil_pressure_psi_roc_15m",
    "egt_c_drift",
    "core_temp_c_drift",
    "vibration_mm_s_ratio",
    "oil_pressure_psi_drop",
    "fuel_flow_kg_h_drift_pct",
    "rpm_drift_pct",
)
# 5-minute means of the high-frequency frame features (contract v2 section 8), in a fixed
# order. A pulse has none of them when the HF path is disabled or lagging, or dropped its frame.
SPECTRAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "fan_amp_bpfo_mean_5m",
    "fan_kurtosis_mean_5m",
    "fan_crest_factor_mean_5m",
    "fan_amp_1x_n1_mean_5m",
    "core_amp_1x_n2_mean_5m",
    "fan_band_energy_hf_mean_5m",
    "oil_temp_c_mean_5m",
    "oil_quantity_mean_5m",
    "oil_filter_dp_mean_5m",
    "egt_ab_split_mean_5m",
)
# Every numeric model input: the pulse features, then the spectral features.
FEATURE_COLUMNS: tuple[str, ...] = (*PULSE_FEATURE_COLUMNS, *SPECTRAL_FEATURE_COLUMNS)
# Only these may be missing (NULL, NaN, or non-finite which counts as missing): XGBoost and
# HistGradientBoosting route missing values natively. A missing pulse feature is an error.
NULLABLE_FEATURE_COLUMNS = frozenset(SPECTRAL_FEATURE_COLUMNS)
EXPECTED_PULSE_FEATURE_COUNT = 46
EXPECTED_SPECTRAL_FEATURE_COUNT = 10
EXPECTED_FEATURE_COUNT = EXPECTED_PULSE_FEATURE_COUNT + EXPECTED_SPECTRAL_FEATURE_COUNT
CATEGORICAL_COLUMNS: tuple[str, ...] = ("engine_model",)

IDENTIFIER_COLUMNS = ("telemetry_id", "engine_id", "engine_model", "lifecycle_id", "cycle_number", "event_ts")
LABEL_COLUMNS = (
    "failure_cycle",
    "rul_cycles",
    "rul_cycles_capped",
    "failure_within_horizon",
    "failure_mode",
    "split_group",
)
# Ground truth, operational labels, identities, timestamps and labels must never
# reach a model: they either leak the target or do not exist at inference time.
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "wear_factor",
        "status",
        "operational_status",
        "maintenance_id",
        "telemetry_id",
        "engine_id",
        "lifecycle_id",
        "event_ts",
        "created_ts",
        "feature_event_ts",
        *LABEL_COLUMNS,
    }
)
_FORBIDDEN_SUFFIXES = ("_id", "_ts", "_at")
_FORBIDDEN_PREFIXES = ("cdc_", "kafka_")
_FORBIDDEN_FRAGMENTS = ("wear", "rul", "failure", "status", "maintenance")

TRAINING_FRAME_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *IDENTIFIER_COLUMNS,
            *FEATURE_COLUMNS,
            "rul_cycles",
            "rul_cycles_capped",
            "failure_within_horizon",
            "split_group",
        )
    )
)
INFERENCE_FRAME_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys((*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS)))

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("feature_pipeline")


class FeatureLeakageError(ValueError):
    """A column that would leak labels, ground truth or identity was offered as a feature."""


class FeatureSchemaError(ValueError):
    """A feature frame does not match the feature contract."""


class FeatureValueError(ValueError):
    """A feature that must always be present holds a missing or non-finite value."""


def is_forbidden_feature(column: str) -> bool:
    """True when ``column`` must never be used as a model input."""
    name = column.lower()
    return (
        name in FORBIDDEN_FEATURE_COLUMNS
        or name.endswith(_FORBIDDEN_SUFFIXES)
        or name.startswith(_FORBIDDEN_PREFIXES)
        or any(fragment in name for fragment in _FORBIDDEN_FRAGMENTS)
    )


def assert_allowed_features(columns: Iterable[str]) -> None:
    """Raise FeatureLeakageError when any of ``columns`` is a forbidden model input."""
    offending = sorted({column for column in columns if is_forbidden_feature(column)})
    if offending:
        raise FeatureLeakageError(f"forbidden feature columns: {', '.join(offending)}")


def _check_feature_contract() -> None:
    for name, columns, expected in (
        ("PULSE_FEATURE_COLUMNS", PULSE_FEATURE_COLUMNS, EXPECTED_PULSE_FEATURE_COUNT),
        ("SPECTRAL_FEATURE_COLUMNS", SPECTRAL_FEATURE_COLUMNS, EXPECTED_SPECTRAL_FEATURE_COUNT),
        # Unique over the concatenation also proves the two groups are disjoint.
        ("FEATURE_COLUMNS", FEATURE_COLUMNS, EXPECTED_FEATURE_COUNT),
    ):
        if len(columns) != expected or len(set(columns)) != expected:
            raise FeatureSchemaError(f"{name} must hold {expected} unique columns")
    assert_allowed_features((*FEATURE_COLUMNS, *CATEGORICAL_COLUMNS))


_check_feature_contract()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureSchemaError(f"missing columns: {', '.join(missing)}")


def spectral_present(frame: pd.DataFrame) -> np.ndarray:
    """Per row: True when every spectral feature is present and finite."""
    _require_columns(frame, SPECTRAL_FEATURE_COLUMNS)
    values = frame.loc[:, list(SPECTRAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.isfinite(values).all(axis=1)


def spectral_coverage(frame: pd.DataFrame) -> float:
    """Share of rows whose spectral features are all present (NaN for an empty frame)."""
    return float(spectral_present(frame).mean()) if len(frame) else float("nan")


def fit_categories(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Learn the levels of every categorical feature from ``frame`` (sorted, as strings)."""
    _require_columns(frame, CATEGORICAL_COLUMNS)
    return {
        column: sorted({str(level) for level in frame[column].dropna().unique()})
        for column in CATEGORICAL_COLUMNS
    }


def one_hot_column(column: str, level: str) -> str:
    """Name of the indicator column of ``column == level``."""
    return f"{column}={level}"


def matrix_columns(feature_columns: Sequence[str], categories: Mapping[str, Sequence[str]]) -> list[str]:
    """Column order of the model input: numeric features, then one indicator per level."""
    indicators = [one_hot_column(column, level) for column, levels in categories.items() for level in levels]
    return [*feature_columns, *indicators]


def _check_required_values(feature_columns: Sequence[str], finite: np.ndarray) -> None:
    """Raise FeatureValueError when a non-nullable feature has a missing or non-finite value."""
    invalid = {
        column: int(count)
        for column, count in zip(feature_columns, (~finite).sum(axis=0), strict=True)
        if count and column not in NULLABLE_FEATURE_COLUMNS
    }
    if invalid:
        detail = ", ".join(f"{column} ({count} rows)" for column, count in invalid.items())
        raise FeatureValueError(
            f"missing or non-finite values in features that must always be present: {detail}; "
            "only the spectral features may be missing"
        )


def build_feature_matrix(
    frame: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return the float32 model input for ``frame``.

    Numeric features come first in ``feature_columns`` order, followed by one-hot
    indicators of the persisted category levels: unseen or missing levels encode
    as all zeros. A missing or non-finite spectral feature becomes NaN, which both
    models treat as missing; in any other feature it raises FeatureValueError.
    """
    assert_allowed_features((*feature_columns, *categories))
    _require_columns(frame, (*feature_columns, *categories))
    numeric = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, na_value=np.nan)
    finite = np.isfinite(numeric)
    _check_required_values(feature_columns, finite)
    numeric[~finite] = np.nan
    blocks = [numeric]
    for column, levels in categories.items():
        values = frame[column].astype(str).to_numpy(dtype=object)
        indicators = np.zeros((len(frame), len(levels)), dtype=np.float32)
        for position, level in enumerate(levels):
            indicators[:, position] = values == level
        blocks.append(indicators)
    return pd.DataFrame(
        np.hstack(blocks), columns=matrix_columns(feature_columns, categories), index=frame.index
    )


def column_list(columns: Iterable[str], table_alias: str = "") -> str:
    """SELECT list of ``columns``, each quoted and aliased to its own name."""
    qualifier = f"{table_alias}." if table_alias else ""
    return ", ".join(f"{qualifier}`{column}` AS `{column}`" for column in columns)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature pipeline jobs of the RUL platform.",
        epilog="Exit codes: 0 done, 3 skipped (source tables missing or all empty), 1 failure.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser(
        "snapshot",
        help="export silver/gold feature tables from ClickHouse to lakehouse Parquet snapshots",
    )
    snapshot.add_argument(
        "--retention",
        type=positive_int,
        default=None,
        help="snapshots kept per dataset (default: SNAPSHOT_RETENTION, 5)",
    )
    snapshot.add_argument(
        "--window-days",
        type=positive_int,
        default=None,
        help="days of event time exported from int_ml_features (default: SNAPSHOT_WINDOW_DAYS, 7)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``feature_pipeline.py``."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    stop = StopToken().install_signal_handlers()
    retention = args.retention or settings.snapshot_retention
    window_days = args.window_days or settings.snapshot_window_days
    store = S3ArtifactStore(settings.s3)
    with ClickHouseGateway(settings.clickhouse) as gateway:
        written = run_snapshot(
            gateway, store, retention=retention, now=datetime.now(UTC), stop=stop, window_days=window_days
        )
    logger.info("snapshot complete (%d datasets, retention %d)", len(written), retention)
    return EXIT_OK


if __name__ == "__main__":
    run_cli(main)
