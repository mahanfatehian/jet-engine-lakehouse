"""Feature contract, feature matrix encoding, the missing-value guard and the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from feature_pipeline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    INFERENCE_FRAME_COLUMNS,
    NULLABLE_FEATURE_COLUMNS,
    PULSE_FEATURE_COLUMNS,
    SPECTRAL_FEATURE_COLUMNS,
    TRAINING_FRAME_COLUMNS,
    FeatureLeakageError,
    FeatureSchemaError,
    FeatureValueError,
    assert_allowed_features,
    build_feature_matrix,
    fit_categories,
    matrix_columns,
    one_hot_column,
    spectral_coverage,
    spectral_present,
)

# The spectral features of contract v2 section 8 / the task's SPECTRAL_FEATURE_COLUMNS, in order.
CONTRACT_SPECTRAL_COLUMNS = [
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
]


def _contract_feature_columns() -> list[str]:
    """Re-derive the 46 pulse features from the generating rules of contract section 7."""
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
    assert list(PULSE_FEATURE_COLUMNS) == _contract_feature_columns()
    assert list(SPECTRAL_FEATURE_COLUMNS) == CONTRACT_SPECTRAL_COLUMNS
    assert list(FEATURE_COLUMNS) == [*_contract_feature_columns(), *CONTRACT_SPECTRAL_COLUMNS]
    assert (len(PULSE_FEATURE_COLUMNS), len(SPECTRAL_FEATURE_COLUMNS), len(FEATURE_COLUMNS)) == (46, 10, 56)
    assert set(CONTRACT_SPECTRAL_COLUMNS) == NULLABLE_FEATURE_COLUMNS
    assert CATEGORICAL_COLUMNS == ("engine_model",)
    assert set(FEATURE_COLUMNS) <= set(TRAINING_FRAME_COLUMNS)
    assert set(FEATURE_COLUMNS) <= set(INFERENCE_FRAME_COLUMNS)


def test_matrix_has_numeric_features_then_sorted_one_hot_columns(synthetic_mart: pd.DataFrame) -> None:
    categories = fit_categories(synthetic_mart)
    matrix = build_feature_matrix(synthetic_mart, categories)

    levels = sorted(synthetic_mart["engine_model"].unique())
    assert categories == {"engine_model": levels}
    assert list(matrix.columns) == [
        *PULSE_FEATURE_COLUMNS,
        *SPECTRAL_FEATURE_COLUMNS,
        *(one_hot_column("engine_model", lvl) for lvl in levels),
    ]
    assert list(matrix.columns) == matrix_columns(FEATURE_COLUMNS, categories)
    assert (matrix.dtypes == np.float32).all()
    one_hot = matrix[[one_hot_column("engine_model", level) for level in levels]]
    assert (one_hot.sum(axis=1) == 1).all()
    first = synthetic_mart.iloc[0]
    assert matrix.iloc[0][one_hot_column("engine_model", first["engine_model"])] == 1.0
    np.testing.assert_allclose(matrix["egt_c"].to_numpy(), synthetic_mart["egt_c"].to_numpy(), rtol=1e-6)


def test_missing_spectral_rows_stay_missing_in_the_matrix(synthetic_mart: pd.DataFrame) -> None:
    missing = synthetic_mart["fan_amp_bpfo_mean_5m"].isna().to_numpy()
    assert 0 < missing.sum() < len(synthetic_mart)

    matrix = build_feature_matrix(synthetic_mart, fit_categories(synthetic_mart))

    spectral = matrix[list(SPECTRAL_FEATURE_COLUMNS)].to_numpy()
    assert np.isnan(spectral[missing]).all() and np.isfinite(spectral[~missing]).all()
    assert np.isfinite(matrix[list(PULSE_FEATURE_COLUMNS)].to_numpy()).all()


def test_the_matrix_is_built_from_the_columns_it_is_given(synthetic_mart: pd.DataFrame) -> None:
    categories = fit_categories(synthetic_mart)
    frame = synthetic_mart.drop(columns=list(SPECTRAL_FEATURE_COLUMNS))

    matrix = build_feature_matrix(frame, categories, PULSE_FEATURE_COLUMNS)

    assert list(matrix.columns) == matrix_columns(PULSE_FEATURE_COLUMNS, categories)
    with pytest.raises(FeatureSchemaError, match="fan_amp_bpfo_mean_5m"):
        build_feature_matrix(frame, categories)


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
        "kafka_offset",
        "frame_start_ts",
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


def test_non_finite_spectral_values_become_missing(synthetic_mart: pd.DataFrame) -> None:
    frame = synthetic_mart.head(4).copy()
    frame["fan_kurtosis_mean_5m"] = [np.inf, -np.inf, np.nan, 3.5]

    matrix = build_feature_matrix(frame, fit_categories(synthetic_mart))

    assert matrix["fan_kurtosis_mean_5m"].isna().tolist() == [True, True, True, False]


@pytest.mark.parametrize(
    "dtype_case",
    ["float64-nan", "nullable-Float64", "object-None"],
)
def test_every_missing_marker_of_a_spectral_value_is_accepted(
    dtype_case: str, synthetic_mart: pd.DataFrame
) -> None:
    # clickhouse-connect returns Nullable(Float64) as pandas Float64 with pd.NA.
    frame = synthetic_mart.head(3).copy()
    values = {
        "float64-nan": pd.Series([np.nan, 0.4, 0.5], dtype="float64"),
        "nullable-Float64": pd.Series([pd.NA, 0.4, 0.5], dtype="Float64"),
        "object-None": pd.Series([None, 0.4, 0.5], dtype=object),
    }[dtype_case]
    frame["fan_amp_bpfo_mean_5m"] = values.array if dtype_case == "nullable-Float64" else values.to_numpy()

    matrix = build_feature_matrix(frame, fit_categories(synthetic_mart))

    assert matrix["fan_amp_bpfo_mean_5m"].isna().tolist() == [True, False, False]
    assert spectral_present(frame).tolist() == [False, True, True]


@pytest.mark.parametrize("bad", [np.nan, np.inf, None])
def test_a_missing_pulse_feature_is_rejected(bad, synthetic_mart: pd.DataFrame) -> None:
    frame = synthetic_mart.head(4).copy()
    frame["vibration_mm_s_ratio"] = frame["vibration_mm_s_ratio"].astype(object)
    frame.loc[frame.index[1], "vibration_mm_s_ratio"] = bad
    frame.loc[frame.index[:2], "egt_c"] = np.nan

    with pytest.raises(FeatureValueError) as raised:
        build_feature_matrix(frame, fit_categories(synthetic_mart))

    message = str(raised.value)
    assert "egt_c (2 rows)" in message and "vibration_mm_s_ratio (1 rows)" in message
    assert "only the spectral features may be missing" in message


def test_spectral_coverage_counts_rows_with_every_spectral_feature(synthetic_mart: pd.DataFrame) -> None:
    frame = synthetic_mart.dropna(subset=list(SPECTRAL_FEATURE_COLUMNS)).head(4).copy()
    frame.loc[frame.index[0], "egt_ab_split_mean_5m"] = np.nan
    frame.loc[frame.index[1], "oil_quantity_mean_5m"] = np.inf

    assert spectral_present(frame).tolist() == [False, False, True, True]
    assert spectral_coverage(frame) == pytest.approx(0.5)
    assert np.isnan(spectral_coverage(frame.iloc[0:0]))
    expected = synthetic_mart[list(SPECTRAL_FEATURE_COLUMNS)].notna().all(axis=1).mean()
    assert spectral_coverage(synthetic_mart) == pytest.approx(expected)
