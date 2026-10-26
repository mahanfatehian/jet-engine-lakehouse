"""Feature contract, feature matrix encoding and the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from feature_pipeline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FeatureLeakageError,
    FeatureSchemaError,
    assert_allowed_features,
    build_feature_matrix,
    fit_categories,
    matrix_columns,
    one_hot_column,
)


def _contract_feature_columns() -> list[str]:
    """Re-derive the 46 features from the generating rules of contract section 7."""
    sensors = ["rpm", "core_temp_c", "egt_c", "vibration_mm_s", "fuel_flow_kg_h", "oil_pressure_psi"]
    windowed = ["vibration_mm_s", "egt_c", "oil_pressure_psi"]
    rolling = [
        f"{s}_{stat}_{w}" for s in windowed for w in ("5m", "15m") for stat in ("mean", "std", "min", "max")
    ]
    deltas = [f"{s}_{kind}" for s in windowed for kind in ("delta_1", "roc_5m", "roc_15m")]
    drift = [
        "egt_c_drift",
        "core_temp_c_drift",
        "vibration_mm_s_ratio",
        "oil_pressure_psi_drop",
        "fuel_flow_kg_h_drift_pct",
        "rpm_drift_pct",
    ]
    return ["cycle_number", *sensors, *rolling, *deltas, *drift]


def test_feature_columns_match_the_contract() -> None:
    assert list(FEATURE_COLUMNS) == _contract_feature_columns()
    assert len(FEATURE_COLUMNS) == 46
    assert CATEGORICAL_COLUMNS == ("engine_model",)


def test_matrix_has_numeric_features_then_sorted_one_hot_columns(synthetic_mart: pd.DataFrame) -> None:
    categories = fit_categories(synthetic_mart)
    matrix = build_feature_matrix(synthetic_mart, categories)

    levels = sorted(synthetic_mart["engine_model"].unique())
    assert categories == {"engine_model": levels}
    assert list(matrix.columns) == [
        *FEATURE_COLUMNS,
        *(one_hot_column("engine_model", lvl) for lvl in levels),
    ]
    assert list(matrix.columns) == matrix_columns(FEATURE_COLUMNS, categories)
    assert (matrix.dtypes == np.float32).all()
    one_hot = matrix[[one_hot_column("engine_model", level) for level in levels]]
    assert (one_hot.sum(axis=1) == 1).all()
    first = synthetic_mart.iloc[0]
    assert matrix.iloc[0][one_hot_column("engine_model", first["engine_model"])] == 1.0
    np.testing.assert_allclose(matrix["egt_c"].to_numpy(), synthetic_mart["egt_c"].to_numpy(), rtol=1e-6)


def test_unseen_and_missing_categories_encode_as_all_zeros(synthetic_mart: pd.DataFrame) -> None:
    categories = fit_categories(synthetic_mart)
    rows = synthetic_mart.head(3).copy()
    rows["engine_model"] = ["PW4000-94", None, rows["engine_model"].iloc[2]]

    one_hot = build_feature_matrix(rows, categories).iloc[:, len(FEATURE_COLUMNS) :]

    assert one_hot.iloc[0].sum() == 0.0
    assert one_hot.iloc[1].sum() == 0.0
    assert one_hot.iloc[2].sum() == 1.0


def test_ground_truth_and_labels_never_reach_the_matrix(synthetic_mart: pd.DataFrame) -> None:
    assert {"wear_factor", "status", "rul_cycles", "split_group"} <= set(synthetic_mart.columns)

    matrix = build_feature_matrix(synthetic_mart, fit_categories(synthetic_mart))

    forbidden = {"wear_factor", "status", "rul_cycles", "rul_cycles_capped", "failure_within_horizon"}
    assert forbidden.isdisjoint(matrix.columns)
    assert not any(column.endswith(("_id", "_ts")) for column in matrix.columns)


@pytest.mark.parametrize(
    "column",
    [
        "wear_factor",
        "status",
        "operational_status",
        "maintenance_id",
        "rul_cycles",
        "rul_cycles_capped",
        "failure_within_horizon",
        "failure_cycle",
        "failure_mode",
        "split_group",
        "telemetry_id",
        "engine_id",
        "lifecycle_id",
        "event_ts",
        "predicted_at",
        "cdc_lsn",
    ],
)
def test_forbidden_columns_are_rejected(column: str, synthetic_mart: pd.DataFrame) -> None:
    with pytest.raises(FeatureLeakageError, match=column):
        assert_allowed_features([*FEATURE_COLUMNS, column])

    frame = synthetic_mart.head(5).assign(**{column: 1.0})
    with pytest.raises(FeatureLeakageError):
        build_feature_matrix(frame, {}, feature_columns=[*FEATURE_COLUMNS, column])


def test_contract_features_pass_the_guard() -> None:
    assert_allowed_features([*FEATURE_COLUMNS, *CATEGORICAL_COLUMNS])


def test_missing_feature_column_is_a_schema_error(synthetic_mart: pd.DataFrame) -> None:
    frame = synthetic_mart.drop(columns=["egt_c_drift"])

    with pytest.raises(FeatureSchemaError, match="egt_c_drift"):
        build_feature_matrix(frame, fit_categories(synthetic_mart))


def test_non_finite_values_become_missing(synthetic_mart: pd.DataFrame) -> None:
    frame = synthetic_mart.head(3).copy()
    frame["vibration_mm_s_ratio"] = [np.inf, -np.inf, 1.5]

    matrix = build_feature_matrix(frame, fit_categories(synthetic_mart))

    assert matrix["vibration_mm_s_ratio"].isna().tolist() == [True, True, False]
