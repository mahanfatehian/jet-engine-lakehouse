"""Inference output: schema, value ranges, risk bands, model caching and the live loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from artifact_store import LocalArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken
from model_registry import (
    ML_MODEL_RUNS_COLUMNS,
    ArtifactLayout,
    ModelArtifacts,
    ProductionPointer,
    publish_artifacts,
    write_production_pointer,
)
from predict_rul import (
    PREDICTION_COLUMNS,
    PREDICTIONS_TABLE,
    ProductionModelCache,
    assign_risk_bands,
    predict_once,
    prediction_rows,
    run_live,
    score_engines,
)
from rul_modeling import RulModelBundle, feature_importance

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
LAYOUT = ArtifactLayout("ml-artifacts/rul_model")


class CountingStore(LocalArtifactStore):
    """Local store that counts object reads."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads: list[str] = []

    def get_bytes(self, key: str) -> bytes:
        self.reads.append(key)
        return super().get_bytes(key)


class FakeGateway:
    """Serves a fixed inference frame and captures inserts."""

    def __init__(self, frame: pd.DataFrame, on_insert: Callable[[], None] = lambda: None) -> None:
        self.frame = frame
        self.inserts: list[tuple[str, list[list[object]], tuple[str, ...]]] = []
        self._on_insert = on_insert

    def table_engines(self, tables):
        return {table: "ReplacingMergeTree" for table in tables}

    def query_df(self, sql, parameters=None):
        return self.frame.copy()

    def insert_rows(self, table, rows, column_names):
        self.inserts.append((table, rows, tuple(column_names)))
        self._on_insert()


def _promote(store: LocalArtifactStore, bundle: RulModelBundle) -> None:
    artifacts = ModelArtifacts(bundle, {}, {}, feature_importance(bundle))
    publish_artifacts(store, LAYOUT, artifacts)
    pointer = ProductionPointer(
        bundle.model_version, LAYOUT.version_prefix(bundle.model_version), NOW.isoformat(), {}, []
    )
    write_production_pointer(store, LAYOUT, pointer)


def test_prediction_columns_match_the_clickhouse_ddl(ddl_columns) -> None:
    assert list(PREDICTION_COLUMNS) == ddl_columns("gold_engine_rul_predictions")
    assert list(ML_MODEL_RUNS_COLUMNS) == ddl_columns("ml_model_runs")


def test_scores_have_the_contract_schema_and_ranges(synthetic_training, inference_frame) -> None:
    bundle = synthetic_training.trained.bundle

    predictions = score_engines(bundle, inference_frame, hours_per_cycle=2.0, predicted_at=NOW)

    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert predictions["engine_id"].tolist() == inference_frame["engine_id"].tolist()
    assert predictions["telemetry_id"].tolist() == inference_frame["telemetry_id"].astype(int).tolist()
    np.testing.assert_array_equal(predictions["egt_c"], inference_frame["egt_c"])
    assert predictions["predicted_rul_cycles"].between(0, bundle.rul_cap_cycles).all()
    np.testing.assert_allclose(predictions["predicted_rul_hours"], predictions["predicted_rul_cycles"] * 2.0)
    assert predictions["failure_probability"].between(0, 1).all()
    expected_flags = (predictions["failure_probability"] >= bundle.decision_threshold).astype(int)
    assert predictions["failure_within_horizon"].tolist() == expected_flags.tolist()
    assert set(predictions["risk_band"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert (predictions["horizon_cycles"] == bundle.failure_horizon_cycles).all()
    assert (predictions["model_version"] == bundle.model_version).all()
    assert (predictions["predicted_at"] == pd.Timestamp(NOW)).all()
    assert predictions["operational_status"].tolist() == inference_frame["operational_status"].tolist()


def test_late_life_engines_score_higher_risk_than_fresh_ones(
    synthetic_training, make_mart, latest_rows_of
) -> None:
    bundle = synthetic_training.trained.bundle
    fresh = latest_rows_of(make_mart(n_engines=3, lifecycles_per_engine=1, seed=5, observed_fraction=0.2))
    worn = latest_rows_of(make_mart(n_engines=3, lifecycles_per_engine=1, seed=5, observed_fraction=0.97))

    fresh_scores = score_engines(bundle, fresh, hours_per_cycle=2.0, predicted_at=NOW)
    worn_scores = score_engines(bundle, worn, hours_per_cycle=2.0, predicted_at=NOW)

    assert (worn_scores["predicted_rul_cycles"] < fresh_scores["predicted_rul_cycles"]).all()
    assert (worn_scores["failure_probability"] > fresh_scores["failure_probability"]).all()
    assert set(worn_scores["risk_band"]) == {"HIGH"}
    assert set(fresh_scores["risk_band"]) == {"LOW"}


@pytest.mark.parametrize(
    ("rul", "probability", "band"),
    [
        (100.0, 0.90, "HIGH"),
        (20.0, 0.00, "HIGH"),
        (25.0, 0.00, "HIGH"),
        (45.0, 0.00, "MEDIUM"),
        (100.0, 0.30, "MEDIUM"),
        (100.0, 0.10, "LOW"),
        (51.0, 0.24, "LOW"),
    ],
)
def test_risk_bands(rul: float, probability: float, band: str) -> None:
    bands = assign_risk_bands(np.array([rul]), np.array([probability]), threshold=0.5, horizon_cycles=25)

    assert bands.tolist() == [band]


def test_missing_registry_status_becomes_unknown(synthetic_training, inference_frame) -> None:
    frame = inference_frame.copy()
    frame["operational_status"] = ["", None, *frame["operational_status"].iloc[2:]]

    predictions = score_engines(
        synthetic_training.trained.bundle, frame, hours_per_cycle=2.0, predicted_at=NOW
    )

    assert predictions["operational_status"].iloc[:2].tolist() == ["UNKNOWN", "UNKNOWN"]


def test_prediction_rows_hold_plain_python_values(synthetic_training, inference_frame) -> None:
    predictions = score_engines(
        synthetic_training.trained.bundle, inference_frame, hours_per_cycle=2.0, predicted_at=NOW
    )

    rows = prediction_rows(predictions)

    assert len(rows) == len(predictions)
    first = dict(zip(PREDICTION_COLUMNS, rows[0], strict=True))
    assert first["predicted_at"] == NOW
    assert isinstance(first["feature_event_ts"], datetime) and first["feature_event_ts"].tzinfo is not None
    assert first["feature_event_ts"] == inference_frame["event_ts"].iloc[0].to_pydatetime()
    assert all(type(first[name]) is int for name in ("lifecycle_id", "cycle_number", "telemetry_id"))
    assert type(first["failure_within_horizon"]) is int and type(first["horizon_cycles"]) is int
    assert all(type(first[name]) is float for name in ("rpm", "predicted_rul_cycles", "failure_probability"))
    assert all(type(first[name]) is str for name in ("engine_id", "risk_band", "model_version"))


def test_model_cache_loads_once_and_follows_promotions(tmp_path: Path, synthetic_training) -> None:
    store = CountingStore(tmp_path)
    cache = ProductionModelCache(store, LAYOUT)
    with pytest.raises(PreconditionsNotMetError, match="no production model"):
        cache.current()

    bundle = synthetic_training.trained.bundle
    _promote(store, bundle)
    assert cache.current().model_version == bundle.model_version
    assert cache.current().model_version == bundle.model_version
    assert sum(key.endswith("model_bundle.joblib") for key in store.reads) == 1

    successor = RulModelBundle(**{**vars(bundle), "model_version": "rul-20261027T000000Z"})
    _promote(store, successor)
    assert cache.current().model_version == "rul-20261027T000000Z"
    assert sum(key.endswith("model_bundle.joblib") for key in store.reads) == 2


def test_predict_once_inserts_one_row_per_engine(tmp_path: Path, synthetic_training, inference_frame) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote(store, synthetic_training.trained.bundle)
    gateway = FakeGateway(inference_frame)

    predictions = predict_once(gateway, ProductionModelCache(store, LAYOUT), hours_per_cycle=2.0)

    [(table, rows, columns)] = gateway.inserts
    assert table == PREDICTIONS_TABLE and columns == PREDICTION_COLUMNS
    assert len(rows) == len(predictions) == inference_frame["engine_id"].nunique()


def test_predict_once_skips_without_feature_rows(tmp_path: Path, synthetic_training, inference_frame) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote(store, synthetic_training.trained.bundle)

    with pytest.raises(PreconditionsNotMetError, match="no rows"):
        predict_once(
            FakeGateway(inference_frame.iloc[0:0]), ProductionModelCache(store, LAYOUT), hours_per_cycle=2.0
        )


class FlakyModelCache(ProductionModelCache):
    """First round: no model yet; second round: storage outage; then the real cache."""

    def __init__(self, store: LocalArtifactStore) -> None:
        super().__init__(store, LAYOUT)
        self.calls = 0

    def current(self) -> RulModelBundle:
        self.calls += 1
        if self.calls == 1:
            raise PreconditionsNotMetError("no production model yet")
        if self.calls == 2:
            raise ConnectionError("object store unreachable")
        return super().current()


def test_live_loop_survives_skips_and_errors_and_stops_gracefully(
    tmp_path: Path, synthetic_training, inference_frame
) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote(store, synthetic_training.trained.bundle)
    cache = FlakyModelCache(store)
    stop = StopToken()
    gateway = FakeGateway(inference_frame, on_insert=stop.request_stop)

    run_live(gateway, cache, hours_per_cycle=2.0, interval_seconds=0.01, stop=stop)

    assert cache.calls == 3
    assert len(gateway.inserts) == 1
