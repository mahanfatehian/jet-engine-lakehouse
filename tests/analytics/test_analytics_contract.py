"""Static checks of the dbt project against the feature contract (CONTRACT.md section 7,
CONTRACT_V2.md section 8).

They need neither dbt nor ClickHouse: they parse the model YAML, whose columns are enforced as
dbt model contracts, so a drift between the docs, the physical tables and the ML interface
fails here first. The end-to-end behaviour is covered by run_dbt_integration.sh.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
ANALYTICS = REPO / "analytics"
IDENTIFIERS = ("telemetry_id", "engine_id", "engine_model", "lifecycle_id", "cycle_number", "event_ts")
RAW_SENSORS = ("rpm", "core_temp_c", "egt_c", "vibration_mm_s", "fuel_flow_kg_h", "oil_pressure_psi")
ROLLED = ("vibration_mm_s", "egt_c", "oil_pressure_psi")
WINDOWS = ("5m", "15m")
FLOAT_FEATURES = (
    *RAW_SENSORS,
    *(f"{s}_{stat}_{w}" for s in ROLLED for w in WINDOWS for stat in ("mean", "std", "min", "max")),
    *(name for s in ROLLED for name in (f"{s}_delta_1", f"{s}_roc_5m", f"{s}_roc_15m")),
    "egt_c_drift",
    "core_temp_c_drift",
    "vibration_mm_s_ratio",
    "oil_pressure_psi_drop",
    "fuel_flow_kg_h_drift_pct",
    "rpm_drift_pct",
)
# SPECTRAL_FEATURE_COLUMNS of the ML contract: 5-minute means of the HF frame features
SPECTRAL = (
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
PULSE_PREFIX = len(IDENTIFIERS) + len(FLOAT_FEATURES)
LABELS = {
    "failure_cycle": "UInt32",
    "rul_cycles": "Int32",
    "rul_cycles_capped": "Int32",
    "failure_within_horizon": "UInt8",
    "failure_mode": "String",
    "split_group": "String",
}
FORBIDDEN_IN_MART = {"wear_factor", "status", "operational_status", "maintenance_id"}


def _models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for path in (ANALYTICS / "models").rglob("*.yml"):
        for model in yaml.safe_load(path.read_text(encoding="utf-8")).get("models", []):
            models[model["name"]] = model
    return models


def _columns(model: str) -> list[dict[str, Any]]:
    return _models()[model]["columns"]


def test_numeric_feature_count_matches_contract() -> None:
    assert len(FLOAT_FEATURES) == 45
    assert len({"cycle_number", *FLOAT_FEATURES}) == 46


@pytest.mark.parametrize(
    "model", ["int_sensor_rolling_features", "int_ml_features", "mart_rul_training_features"]
)
def test_feature_columns_are_exact_and_float(model: str) -> None:
    columns = _columns(model)
    names = [column["name"] for column in columns]
    assert names[:PULSE_PREFIX] == [*IDENTIFIERS, *FLOAT_FEATURES]
    types = {column["name"]: column["data_type"] for column in columns}
    assert {types[name] for name in FLOAT_FEATURES} == {"Float64"}
    assert types["cycle_number"] == "UInt32"


def test_mart_labels_follow_contract_and_exclude_ground_truth() -> None:
    columns = _columns("mart_rul_training_features")
    names = [column["name"] for column in columns]
    # the spectral features follow the 45 Float64 features, the labels close the row
    assert names[PULSE_PREFIX:] == [*SPECTRAL, *LABELS]
    assert {column["name"]: column["data_type"] for column in columns if column["name"] in LABELS} == LABELS
    assert not FORBIDDEN_IN_MART & set(names)


@pytest.mark.parametrize(
    ("model", "position"),
    [
        ("int_vibration_features", 5),
        ("int_ml_features", PULSE_PREFIX),
        ("mart_rul_training_features", PULSE_PREFIX),
    ],
)
def test_spectral_features_are_nullable_floats_in_contract_order(model: str, position: int) -> None:
    columns = _columns(model)
    assert [column["name"] for column in columns][position : position + len(SPECTRAL)] == list(SPECTRAL)
    assert {column["data_type"] for column in columns if column["name"] in SPECTRAL} == {"Nullable(Float64)"}


def test_spectral_features_are_the_same_in_dbt_and_data_science() -> None:
    macros = (ANALYTICS / "macros" / "feature_columns.sql").read_text(encoding="utf-8")
    dbt_names = re.findall(r"\{'name': '(\w+)', 'source': '\w+'\}", macros)
    tree = ast.parse((REPO / "data_science" / "feature_pipeline.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SPECTRAL_FEATURE_COLUMNS"
    )
    assert tuple(dbt_names) == ast.literal_eval(assignment.value) == SPECTRAL


def test_ml_features_view_joins_spectral_features_onto_every_pulse() -> None:
    sql = " ".join(
        (ANALYTICS / "models" / "silver" / "int_ml_features.sql").read_text(encoding="utf-8").split()
    )
    assert "config(materialized='view')" in sql
    assert re.search(
        r"from \{\{ ref\('int_sensor_rolling_features'\) \}\} final left join \(.*"
        r"from \{\{ ref\('int_vibration_features'\) \}\} final \) as \w+ "
        r"using \(engine_id, lifecycle_id, telemetry_id\)",
        sql,
    )


def test_board_counts_the_bearing_defect_flag() -> None:
    model = _models()["fct_engine_health_status"]
    types = {column["name"]: column["data_type"] for column in model["columns"]}
    assert types["is_bearing_defect_anomaly"] == "UInt8"
    assert types["last_hf_frame_ts"] == "Nullable(DateTime64(6, 'UTC'))"
    assert {types[kpi] for kpi in ("fan_amp_bpfo_mean_5m", "fan_kurtosis_mean_5m")} == {"Nullable(Float64)"}
    expressions = [
        test["expression_is_true"]["arguments"]["expression"]
        for test in model["data_tests"]
        if "expression_is_true" in test
    ]
    assert any(
        "anomaly_count =" in expression and "is_bearing_defect_anomaly" in expression
        for expression in expressions
    )
    anomaly_count = next(column for column in model["columns"] if column["name"] == "anomaly_count")
    assert anomaly_count["data_tests"][0]["accepted_range"]["arguments"]["max_value"] == 5


def test_every_model_is_documented_with_typed_columns() -> None:
    models = _models()
    sql_models = {path.stem for path in (ANALYTICS / "models").rglob("*.sql")}
    assert sql_models == set(models)
    for name, model in models.items():
        assert model.get("description"), name
        for column in model["columns"]:
            assert column.get("description") and column.get("data_type"), f"{name}.{column['name']}"


def _vars() -> dict[str, Any]:
    return yaml.safe_load((ANALYTICS / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]


def test_contract_enforced_and_rul_vars_default_from_env() -> None:
    project = yaml.safe_load((ANALYTICS / "dbt_project.yml").read_text(encoding="utf-8"))
    assert project["models"]["iot_engine_analytics"]["+contract"]["enforced"] is True
    assert "RUL_FAILURE_HORIZON_CYCLES" in project["vars"]["rul_failure_horizon_cycles"]
    assert "RUL_CAP_CYCLES" in project["vars"]["rul_cap_cycles"]
    assert project["vars"]["baseline_cycles"] == 10


def test_feature_rows_record_their_baseline_definition() -> None:
    # not a feature: it follows the 51 contract columns, ahead of the build metadata
    names = [column["name"] for column in _columns("int_sensor_rolling_features")]
    tail = names[len(IDENTIFIERS) + len(FLOAT_FEATURES) :]
    assert tail == ["baseline_cycles", "source_ingested_at", "feature_computed_at", "is_deleted"]
    types = {column["name"]: column["data_type"] for column in _columns("int_sensor_rolling_features")}
    assert types["baseline_cycles"] == "UInt32"
    assert "baseline_cycles" not in {column["name"] for column in _columns("mart_rul_training_features")}
    guard = (ANALYTICS / "tests" / "assert_features_use_current_baseline_cycles.sql").read_text(
        encoding="utf-8"
    )
    assert "dbt run --full-refresh -s int_sensor_rolling_features+" in guard


def test_mart_training_window_defaults_to_fifty_lifecycles_per_engine() -> None:
    assert _vars()["mart_max_lifecycles_per_engine"] == 50
    mart = (ANALYTICS / "models" / "gold" / "mart_rul_training_features.sql").read_text(encoding="utf-8")
    assert "positive_int_var('mart_max_lifecycles_per_engine')" in mart


# KPI -> (anomaly threshold var, degradation full-scale var, healthy value of the KPI)
KPI_SCALES = {
    "egt_c_drift": ("anomaly_egt_drift_c", "degradation_full_scale_egt_drift_c", 0.0),
    "vibration_mm_s_ratio": ("anomaly_vibration_ratio", "degradation_full_scale_vibration_ratio", 1.0),
    "oil_pressure_psi_drop": (
        "anomaly_oil_pressure_drop_psi",
        "degradation_full_scale_oil_pressure_drop_psi",
        0.0,
    ),
}


@pytest.mark.parametrize("kpi", sorted(KPI_SCALES))
def test_anomaly_threshold_lies_between_healthy_value_and_full_scale(kpi: str) -> None:
    # a flag must be raised before the KPI's degradation component saturates at 1
    threshold, full_scale, healthy = KPI_SCALES[kpi]
    values = _vars()
    assert healthy < values[threshold] < values[full_scale]


# is_bearing_defect_anomaly: thresholds of two spectral KPIs, one rule in macros/health.sql
BEARING_THRESHOLDS = ("anomaly_fan_amp_bpfo_mm_s", "anomaly_fan_kurtosis")


def test_every_kpi_anomaly_threshold_has_a_calibration_guard() -> None:
    fct = (ANALYTICS / "models" / "gold" / "fct_engine_health_status.sql").read_text(encoding="utf-8")
    guard = (ANALYTICS / "tests" / "assert_anomaly_thresholds_calibrated.sql").read_text(encoding="utf-8")
    health = (ANALYTICS / "macros" / "health.sql").read_text(encoding="utf-8")
    kpi_thresholds = {name for name in _vars() if name.startswith("anomaly_") and "stale" not in name}
    sensor_thresholds = {threshold for threshold, _, _ in KPI_SCALES.values()}
    assert sensor_thresholds | set(BEARING_THRESHOLDS) == kpi_thresholds
    for threshold in sensor_thresholds:
        assert f"number_var('{threshold}')" in fct
        assert f"number_var('{threshold}')" in guard
    # the board raises the bearing flag with the very rule the guard judges
    for threshold in BEARING_THRESHOLDS:
        assert f"number_var('{threshold}')" in health
    assert "bearing_defect_detected(" in fct and "bearing_defect_detected(" in guard


def test_bearing_thresholds_sit_above_a_healthy_spectrum() -> None:
    values = _vars()
    # a healthy bearing has no BPFO tone; Gaussian noise has a kurtosis of 3, tones less
    assert values["anomaly_fan_amp_bpfo_mm_s"] > 0
    assert values["anomaly_fan_kurtosis"] > 3.0


def test_board_keeps_raw_predictions_and_adds_aged_ones() -> None:
    types = {column["name"]: column["data_type"] for column in _columns("fct_engine_health_status")}
    for raw in (
        "predicted_rul_cycles",
        "predicted_rul_hours",
        "failure_probability",
        "risk_band",
        "predicted_at",
    ):
        assert raw in types
    assert types["prediction_cycle_number"] == "Nullable(UInt32)"
    assert types["cycles_since_prediction"] == "Nullable(UInt32)"
    assert types["prediction_age_seconds"] == "Nullable(Float64)"
    assert types["predicted_rul_cycles_adjusted"] == "Nullable(Float64)"
    assert types["predicted_rul_hours_adjusted"] == "Nullable(Float64)"
    assert types["risk_band_adjusted"] == "Nullable(String)"
