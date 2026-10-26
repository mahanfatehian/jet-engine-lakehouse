"""Model artifacts, production pointer, promotion rule and whole training runs on a local store."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from artifact_store import LocalArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken, TerminationRequestedError
from feature_pipeline import FEATURE_COLUMNS, TrainingFrame, prepare_training_frame
from model_registry import (
    BUNDLE_FILE,
    IMPORTANCE_FILE,
    METADATA_FILE,
    METRICS_FILE,
    ML_MODEL_RUNS_COLUMNS,
    REGRESSOR_FILE,
    ArtifactLayout,
    ModelArtifacts,
    ModelRunRecord,
    ProductionPointer,
    RunSummary,
    decide_promotion,
    deserialize_bundle,
    publish_artifacts,
    read_production_pointer,
    write_production_pointer,
)
from rul_modeling import LabelSpec, evaluate_bundle, feature_importance
from train_rul_model import TrainingContext, TrainingRequest, run_training

LAYOUT = ArtifactLayout("ml-artifacts/rul_model")
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


class FrameSource:
    """TrainingDataSource over an in-memory mart frame."""

    name = "clickhouse"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = prepare_training_frame(frame)

    def count_completed_lifecycles(self) -> int:
        return int(self._frame["split_group"].nunique())

    def load(self) -> TrainingFrame:
        return TrainingFrame(self._frame, self.name, "memory:synthetic")


class MemoryRunRegistry:
    """RunRegistry kept in a list; ``history`` stands in for rows written by earlier runs."""

    def __init__(self, history: RunSummary | None = None) -> None:
        self.history = history
        self.records: list[ModelRunRecord] = []

    def latest_run(self) -> RunSummary | None:
        if not self.records:
            return self.history
        last = max(self.records, key=lambda record: (record.trained_at, record.model_version))
        return RunSummary(last.model_version, last.train_lifecycles + last.test_lifecycles)

    def record_run(self, record: ModelRunRecord) -> None:
        self.records.append(record)


def _context(root: Path, frame: pd.DataFrame, runs: MemoryRunRegistry, labels: LabelSpec) -> TrainingContext:
    return TrainingContext(
        source=FrameSource(frame),
        store=LocalArtifactStore(root),
        runs=runs,
        layout=LAYOUT,
        labels=labels,
        hours_per_cycle=2.0,
        stop=StopToken(),
    )


def _request(
    policy: str = "better", *, skip: bool = False, dry_run: bool = False, min_lifecycles: int = 6
) -> TrainingRequest:
    return TrainingRequest(
        min_lifecycles=min_lifecycles, promotion_policy=policy, skip_if_unchanged=skip, dry_run=dry_run
    )


def test_artifacts_round_trip_through_the_local_store(tmp_path: Path, synthetic_training) -> None:
    store = LocalArtifactStore(tmp_path)
    bundle = synthetic_training.trained.bundle
    holdout = synthetic_training.holdout
    artifacts = ModelArtifacts(
        bundle=bundle,
        metadata={"model_version": bundle.model_version},
        metrics={"holdout": holdout.as_dict()},
        feature_importance=feature_importance(bundle),
    )

    uri = publish_artifacts(store, LAYOUT, artifacts)

    prefix = LAYOUT.version_prefix(bundle.model_version)
    files = (BUNDLE_FILE, REGRESSOR_FILE, METADATA_FILE, METRICS_FILE, IMPORTANCE_FILE)
    assert store.list_keys(prefix + "/") == sorted(f"{prefix}/{name}" for name in files)
    assert uri.endswith(f"{prefix}/")
    restored = deserialize_bundle(store.get_bytes(LAYOUT.artifact_key(bundle.model_version, BUNDLE_FILE)))
    test = synthetic_training.frames["test"]
    np.testing.assert_array_equal(
        restored.predict_rul_cycles(restored.feature_matrix(test)),
        bundle.predict_rul_cycles(bundle.feature_matrix(test)),
    )
    np.testing.assert_array_equal(
        restored.predict_failure_probability(restored.feature_matrix(test)),
        bundle.predict_failure_probability(bundle.feature_matrix(test)),
    )
    assert (restored.decision_threshold, restored.categories) == (
        bundle.decision_threshold,
        bundle.categories,
    )
    assert restored.library_versions["xgboost"] == xgb.__version__
    booster = xgb.Booster()
    booster.load_model(bytearray(store.get_bytes(LAYOUT.artifact_key(bundle.model_version, REGRESSOR_FILE))))
    assert booster.feature_names == bundle.matrix_columns
    importance = pd.read_csv(
        io.BytesIO(store.get_bytes(LAYOUT.artifact_key(bundle.model_version, IMPORTANCE_FILE)))
    )
    assert list(importance.columns) == ["feature", "gain", "weight", "cover", "gain_share"]
    metrics = store.get_json(LAYOUT.artifact_key(bundle.model_version, METRICS_FILE))
    assert metrics["holdout"]["rmse"] == pytest.approx(holdout.rmse)


def test_production_pointer_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert read_production_pointer(store, LAYOUT) is None
    pointer = ProductionPointer(
        model_version="rul-20261026T120000Z",
        artifact_prefix="ml-artifacts/rul_model/rul-20261026T120000Z",
        promoted_at="2026-10-26T12:00:00+00:00",
        metrics={"rmse": 10.5, "clf_roc_auc": None},
        feature_columns=list(FEATURE_COLUMNS),
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
    }


def test_malformed_production_pointer_is_rejected(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put_json(LAYOUT.production_key, {"model_version": "rul-x"})

    with pytest.raises(ValueError, match="artifact_prefix"):
        read_production_pointer(store, LAYOUT)


@pytest.mark.parametrize(
    ("policy", "challenger_rmse", "champion_rmse", "promote"),
    [
        ("better", 12.0, None, True),
        ("always", 12.0, None, True),
        ("better", 9.0, 10.0, True),
        ("better", 10.0, 10.0, False),
        ("better", 11.0, 10.0, False),
        ("always", 11.0, 10.0, True),
    ],
)
def test_promotion_policy(
    policy: str, challenger_rmse: float, champion_rmse: float | None, promote: bool
) -> None:
    decision = decide_promotion(policy, challenger_rmse, champion_rmse)

    assert decision.promote is promote
    assert decision.reason


def test_unknown_promotion_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sometimes"):
        decide_promotion("sometimes", 1.0, 2.0)


def test_first_run_publishes_promotes_and_records(tmp_path: Path, synthetic_mart, label_spec) -> None:
    runs = MemoryRunRegistry()
    context = _context(tmp_path, synthetic_mart, runs, label_spec)

    outcome = run_training(_request(), context, now=T0)

    assert outcome.model_version == "rul-20261026T120000Z"
    assert outcome.champion is None and outcome.decision.promote
    pointer = read_production_pointer(context.store, LAYOUT)
    assert pointer is not None and pointer.model_version == outcome.model_version
    assert pointer.artifact_prefix == "ml-artifacts/rul_model/rul-20261026T120000Z"
    assert pointer.feature_columns == list(FEATURE_COLUMNS)
    assert pointer.metrics["rmse"] == pytest.approx(outcome.holdout.rmse)
    [record] = runs.records
    assert record.promoted and record.model_version == outcome.model_version
    assert record.data_source == "clickhouse" and record.artifact_uri == outcome.artifact_uri
    assert record.train_lifecycles + record.test_lifecycles == synthetic_mart["split_group"].nunique()
    assert record.train_rows + record.test_rows == len(synthetic_mart)
    assert record.test_rows == outcome.holdout.rows and record.rmse == outcome.holdout.rmse
    assert json.loads(record.params_json)["regressor"]["best_iteration"] > 0
    assert len(record.as_row()) == len(ML_MODEL_RUNS_COLUMNS)
    metadata = context.store.get_json(LAYOUT.artifact_key(outcome.model_version, METADATA_FILE))
    assert metadata["lifecycles"]["test"] == list(outcome.split.test)
    assert metadata["promotion"] == {
        "policy": "better",
        "promoted": True,
        "reason": outcome.decision.reason,
        "champion_version": None,
    }


def test_champion_is_rescored_on_the_challengers_holdout(tmp_path: Path, synthetic_mart, label_spec) -> None:
    runs = MemoryRunRegistry()
    early_mart = synthetic_mart[synthetic_mart["engine_id"] != "ENG-004"]
    first = run_training(_request(), _context(tmp_path, early_mart, runs, label_spec), now=T0)
    context = _context(tmp_path, synthetic_mart, runs, label_spec)

    second = run_training(_request("better"), context, now=T0 + timedelta(hours=1))

    assert second.champion is not None and second.champion.model_version == first.model_version
    champion_bundle = deserialize_bundle(
        context.store.get_bytes(LAYOUT.artifact_key(first.model_version, BUNDLE_FILE))
    )
    full = prepare_training_frame(synthetic_mart)
    holdout = full[full["split_group"].isin(second.split.test)].reset_index(drop=True)
    assert second.champion.holdout == evaluate_bundle(champion_bundle, holdout, label_spec)
    assert second.champion.holdout.rows == second.holdout.rows
    # The stable split keeps the champion's fitting lifecycles out of the new holdout.
    assert set(second.split.test).isdisjoint(first.split.train + first.split.validation)
    assert second.decision.promote == (second.holdout.rmse < second.champion.holdout.rmse)
    serving = second.model_version if second.decision.promote else first.model_version
    assert read_production_pointer(context.store, LAYOUT).model_version == serving
    assert [record.promoted for record in runs.records] == [True, second.decision.promote]


def test_policy_better_keeps_an_equal_champion_and_always_replaces_it(
    tmp_path: Path, synthetic_mart, label_spec
) -> None:
    runs = MemoryRunRegistry()
    context = _context(tmp_path, synthetic_mart, runs, label_spec)
    first = run_training(_request(), context, now=T0)

    second = run_training(_request("better"), context, now=T0 + timedelta(hours=1))

    assert second.champion is not None
    assert second.holdout.rmse == second.champion.holdout.rmse
    assert not second.decision.promote
    assert read_production_pointer(context.store, LAYOUT).model_version == first.model_version
    assert context.store.list_keys(LAYOUT.version_prefix(second.model_version) + "/")

    third = run_training(_request("always"), context, now=T0 + timedelta(hours=2))

    assert third.decision.promote
    assert read_production_pointer(context.store, LAYOUT).model_version == third.model_version
    assert [record.promoted for record in runs.records] == [True, False, True]


def test_skip_if_unchanged_skips_only_when_nothing_new_and_a_model_serves(
    tmp_path: Path, synthetic_mart, make_mart, label_spec
) -> None:
    lifecycles = synthetic_mart["split_group"].nunique()
    runs = MemoryRunRegistry(history=RunSummary("rul-20261001T000000Z", lifecycles))
    context = _context(tmp_path, synthetic_mart, runs, label_spec)

    bootstrap = run_training(_request(skip=True), context, now=T0)
    with pytest.raises(PreconditionsNotMetError, match="no new completed lifecycles"):
        run_training(_request(skip=True), context, now=T0 + timedelta(hours=1))
    grown = _context(tmp_path, make_mart(n_engines=5), runs, label_spec)
    retrained = run_training(_request(skip=True), grown, now=T0 + timedelta(hours=2))

    assert bootstrap.artifact_uri and retrained.artifact_uri
    assert len(runs.records) == 2


def test_too_few_lifecycles_skips(tmp_path: Path, synthetic_mart, label_spec) -> None:
    runs = MemoryRunRegistry()
    context = _context(tmp_path, synthetic_mart, runs, label_spec)

    with pytest.raises(PreconditionsNotMetError, match="100 required"):
        run_training(_request(min_lifecycles=100), context, now=T0)
    assert context.store.list_keys("") == [] and runs.records == []


def test_dry_run_writes_nothing(tmp_path: Path, synthetic_mart, label_spec) -> None:
    runs = MemoryRunRegistry()
    context = _context(tmp_path, synthetic_mart, runs, label_spec)

    outcome = run_training(_request(dry_run=True), context, now=T0)

    assert outcome.artifact_uri is None and outcome.holdout.r2 > 0.6
    assert context.store.list_keys("") == [] and runs.records == []


def test_termination_request_stops_before_anything_is_written(
    tmp_path: Path, synthetic_mart, label_spec
) -> None:
    runs = MemoryRunRegistry()
    stop = StopToken()
    stop.request_stop()
    context = replace(_context(tmp_path, synthetic_mart, runs, label_spec), stop=stop)

    with pytest.raises(TerminationRequestedError):
        run_training(_request(), context, now=T0)
    assert context.store.list_keys("") == [] and runs.records == []
