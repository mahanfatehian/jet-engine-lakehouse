"""Model artifacts, bundle integrity, the production pointer and whole training runs on a local store."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from artifact_store import LocalArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken, TerminationRequestedError
from feature_pipeline import FEATURE_COLUMNS
from model_promotion import ChampionStatus
from model_registry import (
    BUNDLE_FILE,
    IMPORTANCE_FILE,
    METADATA_FILE,
    METRICS_FILE,
    ML_MODEL_RUNS_COLUMNS,
    REGRESSOR_FILE,
    ArtifactLayout,
    ModelArtifacts,
    ModelIntegrityError,
    ProductionPointer,
    RunSummary,
    deserialize_bundle,
    load_production_bundle,
    publish_artifacts,
    read_production_pointer,
    write_production_pointer,
)
from rul_modeling import evaluate_bundle, feature_importance
from train_rul_model import TrainingRequest, run_training
from training_sources import prepare_training_frame

LAYOUT = ArtifactLayout("ml-artifacts/rul_model")
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def _request(
    policy: str = "better", *, skip: bool = False, dry_run: bool = False, min_lifecycles: int = 6
) -> TrainingRequest:
    return TrainingRequest(
        min_lifecycles=min_lifecycles, promotion_policy=policy, skip_if_unchanged=skip, dry_run=dry_run
    )


def _publish(store: LocalArtifactStore, synthetic_training) -> tuple:
    bundle = synthetic_training.trained.bundle
    artifacts = ModelArtifacts(
        bundle=bundle,
        metadata={"model_version": bundle.model_version},
        metrics={"holdout": synthetic_training.holdout.as_dict()},
        feature_importance=feature_importance(bundle),
    )
    return bundle, publish_artifacts(store, LAYOUT, artifacts)


def test_artifacts_round_trip_through_the_local_store(tmp_path: Path, synthetic_training) -> None:
    store = LocalArtifactStore(tmp_path)
    bundle, published = _publish(store, synthetic_training)

    prefix = LAYOUT.version_prefix(bundle.model_version)
    files = (BUNDLE_FILE, REGRESSOR_FILE, METADATA_FILE, METRICS_FILE, IMPORTANCE_FILE)
    assert store.list_keys(prefix + "/") == sorted(f"{prefix}/{name}" for name in files)
    assert published.uri.endswith(f"{prefix}/")
    bundle_bytes = store.get_bytes(LAYOUT.artifact_key(bundle.model_version, BUNDLE_FILE))
    assert published.bundle_sha256 == hashlib.sha256(bundle_bytes).hexdigest()
    metadata = store.get_json(LAYOUT.artifact_key(bundle.model_version, METADATA_FILE))
    assert metadata["bundle_sha256"] == published.bundle_sha256
    restored = deserialize_bundle(bundle_bytes)
    test = synthetic_training.frames["test"]
    matrix = bundle.feature_matrix(test)
    np.testing.assert_array_equal(restored.predict_rul_cycles(matrix), bundle.predict_rul_cycles(matrix))
    np.testing.assert_array_equal(
        restored.predict_failure_probability(matrix), bundle.predict_failure_probability(matrix)
    )
    assert (restored.decision_threshold, restored.categories, restored.calibration) == (
        bundle.decision_threshold,
        bundle.categories,
        bundle.calibration,
    )
    assert restored.library_versions["xgboost"] == xgb.__version__
    importance = pd.read_csv(
        io.BytesIO(store.get_bytes(LAYOUT.artifact_key(bundle.model_version, IMPORTANCE_FILE)))
    )
    assert list(importance.columns) == ["feature", "gain", "weight", "cover", "gain_share"]
    metrics = store.get_json(LAYOUT.artifact_key(bundle.model_version, METRICS_FILE))
    assert metrics["holdout"]["rmse"] == pytest.approx(synthetic_training.holdout.rmse)


def test_regressor_json_reproduces_the_served_predictions(tmp_path: Path, synthetic_training) -> None:
    store = LocalArtifactStore(tmp_path)
    bundle, _ = _publish(store, synthetic_training)
    booster = xgb.Booster()

    booster.load_model(bytearray(store.get_bytes(LAYOUT.artifact_key(bundle.model_version, REGRESSOR_FILE))))

    matrix = bundle.feature_matrix(synthetic_training.frames["test"])
    assert booster.feature_names == bundle.matrix_columns
    assert booster.num_boosted_rounds() == synthetic_training.trained.best_iteration + 1
    np.testing.assert_array_equal(booster.predict(xgb.DMatrix(matrix)), bundle.regressor.predict(matrix))


def test_production_pointer_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert read_production_pointer(store, LAYOUT) is None
    pointer = ProductionPointer(
        model_version="rul-20261026T120000Z",
        artifact_prefix="ml-artifacts/rul_model/rul-20261026T120000Z",
        promoted_at="2026-10-26T12:00:00+00:00",
        metrics={"rmse": 10.5, "clf_roc_auc": None},
        feature_columns=list(FEATURE_COLUMNS),
        bundle_sha256="ab" * 32,
    )

    write_production_pointer(store, LAYOUT, pointer)

    assert read_production_pointer(store, LAYOUT) == pointer
    assert store.list_keys("ml-artifacts/") == ["ml-artifacts/rul_model/production.json"]
    assert set(store.get_json(LAYOUT.production_key)) == {
        "model_version",
        "artifact_prefix",
        "promoted_at",
        "metrics",
        "feature_columns",
        "bundle_sha256",
    }
    store.put_json(LAYOUT.production_key, {"model_version": "rul-x", "artifact_prefix": "p/rul-x"})
    assert read_production_pointer(store, LAYOUT).bundle_sha256 is None


def test_malformed_production_pointer_is_rejected(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put_json(LAYOUT.production_key, {"model_version": "rul-x"})

    with pytest.raises(ValueError, match="artifact_prefix"):
        read_production_pointer(store, LAYOUT)


def _pointer_for(bundle, digest: str | None) -> ProductionPointer:
    return ProductionPointer(
        bundle.model_version, LAYOUT.version_prefix(bundle.model_version), T0.isoformat(), {}, [], digest
    )


def test_bundle_is_verified_before_it_is_unpickled(tmp_path: Path, synthetic_training, monkeypatch) -> None:
    store = LocalArtifactStore(tmp_path)
    bundle, published = _publish(store, synthetic_training)
    assert load_production_bundle(store, _pointer_for(bundle, published.bundle_sha256)).model_version == (
        bundle.model_version
    )
    unpickled: list[bool] = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: unpickled.append(True) or real_load(*args))

    with pytest.raises(ModelIntegrityError, match="no bundle_sha256"):
        load_production_bundle(store, _pointer_for(bundle, None))
    with pytest.raises(ModelIntegrityError, match="refusing to unpickle a modified"):
        load_production_bundle(store, _pointer_for(bundle, "0" * 64))
    tampered = store.get_bytes(LAYOUT.artifact_key(bundle.model_version, BUNDLE_FILE)) + b"\x00"
    store.put_bytes(LAYOUT.artifact_key(bundle.model_version, BUNDLE_FILE), tampered)
    with pytest.raises(ModelIntegrityError, match=published.bundle_sha256):
        load_production_bundle(store, _pointer_for(bundle, published.bundle_sha256))
    assert unpickled == []


def test_first_run_publishes_promotes_and_records(tmp_path: Path, synthetic_mart, make_context) -> None:
    context = make_context(tmp_path, synthetic_mart)

    outcome = run_training(_request(), context, now=T0)

    assert outcome.model_version == "rul-20261026T120000Z"
    assert outcome.champion.status is ChampionStatus.NONE and outcome.decision.promote
    pointer = read_production_pointer(context.store, LAYOUT)
    assert pointer is not None and pointer.model_version == outcome.model_version
    assert pointer.artifact_prefix == "ml-artifacts/rul_model/rul-20261026T120000Z"
    assert pointer.feature_columns == list(FEATURE_COLUMNS)
    assert pointer.metrics["rmse"] == pytest.approx(outcome.holdout.rmse)
    stored = context.store.get_bytes(LAYOUT.artifact_key(outcome.model_version, BUNDLE_FILE))
    assert pointer.bundle_sha256 == outcome.bundle_sha256 == hashlib.sha256(stored).hexdigest()
    [record] = context.runs.records
    assert record.promoted and record.model_version == outcome.model_version
    assert record.data_source == "clickhouse" and record.artifact_uri == outcome.artifact_uri
    assert record.train_lifecycles + record.test_lifecycles == synthetic_mart["split_group"].nunique()
    assert record.train_rows + record.test_rows == len(synthetic_mart)
    assert record.test_rows == outcome.holdout.rows and record.rmse == outcome.holdout.rmse
    assert len(record.as_row()) == len(ML_MODEL_RUNS_COLUMNS)
    params = json.loads(record.params_json)
    assert params["regressor"]["best_iteration"] > 0
    assert params["regressor"]["trees"] == params["regressor"]["best_iteration"] + 1
    assert params["clf_brier"] == pytest.approx(outcome.holdout.clf_brier)
    assert params["clf_ece"] == pytest.approx(outcome.holdout.clf_ece)
    assert params["calibration"]["method"] == "platt" and params["bundle_sha256"] == outcome.bundle_sha256
    assert params["training_window"]["dropped_lifecycles"] == 0
    newest = prepare_training_frame(synthetic_mart)["event_ts"].max()
    assert params["data_watermark_us"] == pd.Timestamp(newest).value // 1000
    metadata = context.store.get_json(LAYOUT.artifact_key(outcome.model_version, METADATA_FILE))
    assert metadata["lifecycles"]["test"] == list(outcome.split.test)
    assert metadata["bundle_sha256"] == outcome.bundle_sha256
    assert metadata["promotion"]["champion_status"] == "none"
    assert metadata["promotion"]["floors"] == {"min_r2": 0.5, "min_roc_auc": 0.75, "f1_tolerance": 0.05}
    assert metadata["promotion"]["promoted"] and metadata["promotion"]["champion_version"] is None


def test_champion_is_rescored_on_the_challengers_holdout(
    tmp_path: Path, synthetic_mart, make_context, run_registry, label_spec
) -> None:
    runs = run_registry()
    early_mart = synthetic_mart[synthetic_mart["engine_id"] != "ENG-004"]
    first = run_training(_request(), make_context(tmp_path, early_mart, runs), now=T0)
    context = make_context(tmp_path, synthetic_mart, runs)

    second = run_training(_request("better"), context, now=T0 + timedelta(hours=1))

    champion = second.champion
    assert champion.status is ChampionStatus.COMPARABLE and champion.model_version == first.model_version
    champion_bundle = deserialize_bundle(
        context.store.get_bytes(LAYOUT.artifact_key(first.model_version, BUNDLE_FILE))
    )
    full = prepare_training_frame(synthetic_mart)
    holdout = full[full["split_group"].isin(second.split.test)].reset_index(drop=True)
    assert champion.holdout == evaluate_bundle(champion_bundle, holdout, label_spec)
    assert champion.holdout.rows == second.holdout.rows
    # The stable split keeps the champion's fitting lifecycles out of the new holdout.
    assert set(second.split.test).isdisjoint(first.split.train + first.split.validation)
    better = second.holdout.rmse < champion.holdout.rmse
    within = second.holdout.clf_f1 >= champion.holdout.clf_f1 - 0.05
    assert second.decision.promote == (better and within)
    serving = second.model_version if second.decision.promote else first.model_version
    assert read_production_pointer(context.store, LAYOUT).model_version == serving
    assert [record.promoted for record in runs.records] == [True, second.decision.promote]


def test_policy_better_keeps_an_equal_champion_and_always_replaces_it(
    tmp_path: Path, synthetic_mart, make_context
) -> None:
    context = make_context(tmp_path, synthetic_mart)
    first = run_training(_request(), context, now=T0)

    second = run_training(_request("better"), context, now=T0 + timedelta(hours=1))

    assert second.champion.status is ChampionStatus.COMPARABLE
    assert second.holdout.rmse == second.champion.holdout.rmse
    assert not second.decision.promote
    assert read_production_pointer(context.store, LAYOUT).model_version == first.model_version
    assert context.store.list_keys(LAYOUT.version_prefix(second.model_version) + "/")

    third = run_training(_request("always"), context, now=T0 + timedelta(hours=2))

    assert third.decision.promote and "bypassed" in third.decision.reason
    assert read_production_pointer(context.store, LAYOUT).model_version == third.model_version
    assert [record.promoted for record in context.runs.records] == [True, False, True]


def test_too_few_lifecycles_skips(tmp_path: Path, synthetic_mart, make_context) -> None:
    context = make_context(tmp_path, synthetic_mart)

    with pytest.raises(PreconditionsNotMetError, match="100 required"):
        run_training(_request(min_lifecycles=100), context, now=T0)
    assert context.store.list_keys("") == [] and context.runs.records == []


@pytest.mark.parametrize(("lifecycles", "min_lifecycles"), [(2, 2), (1, 1), (2, 1)])
def test_fewer_than_three_lifecycles_skip_whatever_the_minimum(
    tmp_path: Path, synthetic_mart, make_context, lifecycles: int, min_lifecycles: int
) -> None:
    groups = sorted(synthetic_mart["split_group"].unique())[:lifecycles]
    context = make_context(tmp_path, synthetic_mart[synthetic_mart["split_group"].isin(groups)])

    floor = r"3 required \(the train/validation/test split needs at least 3\)"
    with pytest.raises(PreconditionsNotMetError, match=floor):
        run_training(_request(min_lifecycles=min_lifecycles), context, now=T0)
    assert context.store.list_keys("") == [] and context.runs.records == []


def test_dry_run_writes_nothing(tmp_path: Path, synthetic_mart, make_context) -> None:
    context = make_context(tmp_path, synthetic_mart)

    outcome = run_training(_request(dry_run=True), context, now=T0)

    assert outcome.artifact_uri is None and outcome.bundle_sha256 is None and outcome.holdout.r2 > 0.6
    assert context.store.list_keys("") == [] and context.runs.records == []


def test_termination_request_stops_before_anything_is_written(
    tmp_path: Path, synthetic_mart, make_context
) -> None:
    stop = StopToken()
    stop.request_stop()
    context = replace(make_context(tmp_path, synthetic_mart), stop=stop)

    with pytest.raises(TerminationRequestedError):
        run_training(_request(), context, now=T0)
    assert context.store.list_keys("") == [] and context.runs.records == []


def test_run_summary_defaults_describe_a_legacy_run() -> None:
    summary = RunSummary("rul-20261001T000000Z", 24)

    assert summary.data_watermark_us is None and summary.promoted is False
