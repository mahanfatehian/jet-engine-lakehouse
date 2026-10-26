"""Static checks of the dbt project against the feature contract (CONTRACT.md section 7).

They need neither dbt nor ClickHouse: they parse the model YAML, whose columns are enforced as
dbt model contracts, so a drift between the docs, the physical tables and the ML interface
fails here first. The end-to-end behaviour is covered by run_dbt_integration.sh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

ANALYTICS = Path(__file__).resolve().parents[2] / "analytics"
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


@pytest.mark.parametrize("model", ["int_sensor_rolling_features", "mart_rul_training_features"])
def test_feature_columns_are_exact_and_float(model: str) -> None:
    columns = _columns(model)
    names = [column["name"] for column in columns]
    assert names[: len(IDENTIFIERS) + len(FLOAT_FEATURES)] == [*IDENTIFIERS, *FLOAT_FEATURES]
    types = {column["name"]: column["data_type"] for column in columns}
    assert {types[name] for name in FLOAT_FEATURES} == {"Float64"}
    assert types["cycle_number"] == "UInt32"


def test_mart_labels_follow_contract_and_exclude_ground_truth() -> None:
    columns = _columns("mart_rul_training_features")
    names = [column["name"] for column in columns]
    assert names[len(IDENTIFIERS) + len(FLOAT_FEATURES) :] == list(LABELS)
    assert {column["name"]: column["data_type"] for column in columns if column["name"] in LABELS} == LABELS
    assert not FORBIDDEN_IN_MART & set(names)


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


def test_every_kpi_anomaly_threshold_has_a_calibration_guard() -> None:
    fct = (ANALYTICS / "models" / "gold" / "fct_engine_health_status.sql").read_text(encoding="utf-8")
    guard = (ANALYTICS / "tests" / "assert_anomaly_thresholds_calibrated.sql").read_text(encoding="utf-8")
    kpi_thresholds = {name for name in _vars() if name.startswith("anomaly_") and "stale" not in name}
    assert {threshold for threshold, _, _ in KPI_SCALES.values()} == kpi_thresholds
    for threshold in kpi_thresholds:
        assert f"number_var('{threshold}')" in fct
        assert f"number_var('{threshold}')" in guard


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
