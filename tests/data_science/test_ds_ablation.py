"""Spectral ablation: same split without the spectral features, the planted bearing signal, recording."""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import feature_ablation
import numpy as np
import pandas as pd
import pytest
from feature_ablation import (
    ABLATION_BASELINE,
    AblationResult,
    AblationStatus,
    run_ablation,
    spectral_importance_share,
)
from feature_pipeline import FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS, SPECTRAL_FEATURE_COLUMNS
from model_registry import METADATA_FILE, METRICS_FILE, ArtifactLayout, read_production_pointer
from rul_metrics import HoldoutMetrics
from rul_modeling import LabelSpec, evaluate_bundle, feature_importance, train_models
from train_rul_model import TrainingContext, TrainingRequest, run_training

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
LAYOUT = ArtifactLayout("ml-artifacts/rul_model")
BEARING = "BEARING_DEGRADATION"


@pytest.fixture(scope="module")
def ablation(synthetic_training, label_spec: LabelSpec) -> AblationResult:
    return run_ablation(
        synthetic_training.frames, label_spec, synthetic_training.trained, synthetic_training.holdout
    )


def _metrics(rmse: float, f1: float) -> HoldoutMetrics:
    return HoldoutMetrics(
        rows=100,
        lifecycles=3,
        rmse=rmse,
        mae=rmse * 0.8,
        r2=0.9,
        late_life_rmse=rmse,
        nasa_score=0.5,
        nasa_score_sum=50.0,
        clf_threshold=0.4,
        clf_f1=f1,
        clf_precision=f1,
        clf_recall=f1,
        clf_roc_auc=float("nan"),
        positive_rate=0.2,
    )


def test_ablation_drops_only_the_spectral_features_and_keeps_the_split(
    ablation: AblationResult, synthetic_training
) -> None:
    test = synthetic_training.frames["test"]

    assert ablation.status is AblationStatus.COMPLETED and ablation.detail == ""
    assert ablation.dropped_columns == SPECTRAL_FEATURE_COLUMNS
    assert ablation.with_spectral == synthetic_training.holdout
    without = ablation.without_spectral
    assert without.rows == ablation.with_spectral.rows == len(test)
    assert without.lifecycles == ablation.with_spectral.lifecycles == test["split_group"].nunique()
    assert set(ablation.coverage) == {"train", "validation", "test"}
    assert all(0.9 < share <= 1.0 for share in ablation.coverage.values())
    assert ablation.seconds > 0 and ablation.baseline_best_iteration > 0
    assert 0.05 <= ablation.baseline_threshold <= 0.95


def test_ablation_measures_the_same_models_without_the_spectral_features(
    ablation: AblationResult, synthetic_training, label_spec: LabelSpec
) -> None:
    frames = synthetic_training.frames
    baseline = train_models(
        frames["train"],
        frames["validation"],
        label_spec,
        model_version="rul-20260101T000000Z",
        trained_at="2026-01-01T00:00:00+00:00",
        hours_per_cycle=2.0,
        feature_columns=PULSE_FEATURE_COLUMNS,
    )

    assert baseline.bundle.feature_columns == PULSE_FEATURE_COLUMNS
    assert evaluate_bundle(baseline.bundle, frames["test"], label_spec) == ablation.without_spectral


def test_ablation_shows_the_gain_of_the_planted_bearing_signal(
    ablation: AblationResult, synthetic_training, label_spec: LabelSpec
) -> None:
    test = synthetic_training.frames["test"]
    assert BEARING in set(test["failure_mode"]), "the holdout must contain the planted signal"
    with_spectral, without = ablation.with_spectral, ablation.without_spectral

    assert with_spectral.rmse < without.rmse - 3.0
    assert with_spectral.late_life_rmse < without.late_life_rmse - 3.0
    assert with_spectral.r2 > without.r2 and with_spectral.clf_f1 > without.clf_f1
    assert ablation.differences()["rmse"] == pytest.approx(with_spectral.rmse - without.rmse)
    importance = feature_importance(synthetic_training.trained.bundle)
    spectral = importance[importance["feature"].isin(SPECTRAL_FEATURE_COLUMNS)]
    assert spectral["feature"].iloc[0] in {"fan_amp_bpfo_mean_5m", "fan_kurtosis_mean_5m"}
    assert ablation.spectral_importance_share > 0.25
    # The gain sits on the bearing lifecycles, where only the spectral features see the defect.
    bearing = test[test["failure_mode"] == BEARING].reset_index(drop=True)
    assert evaluate_bundle(synthetic_training.trained.bundle, bearing, label_spec).rmse < 0.6 * without.rmse


def test_ablation_is_skipped_when_the_development_data_has_no_spectral_value(
    synthetic_training, label_spec: LabelSpec, hf_disabled
) -> None:
    frames = {side: hf_disabled(frame) for side, frame in synthetic_training.frames.items()}
    frames["test"] = synthetic_training.frames["test"]

    result = run_ablation(frames, label_spec, synthetic_training.trained, synthetic_training.holdout)

    assert result.status is AblationStatus.SKIPPED and result.without_spectral is None
    assert "no spectral feature value" in result.detail and result.differences() is None
    assert result.coverage["train"] == 0.0 and result.coverage["test"] > 0.9
    document = result.as_dict()
    json.dumps(document, allow_nan=False)
    assert document["status"] == "skipped" and document["with_minus_without"] is None
    assert "spectral ablation skipped" in result.describe()


def test_ablation_failure_is_recorded_and_never_raised(
    synthetic_training, label_spec: LabelSpec, monkeypatch, caplog
) -> None:
    def broken(*_args, **_kwargs):
        raise MemoryError("no room for a second model")

    monkeypatch.setattr(feature_ablation, "train_models", broken)

    with caplog.at_level(logging.ERROR, logger="feature_ablation"):
        result = run_ablation(
            synthetic_training.frames, label_spec, synthetic_training.trained, synthetic_training.holdout
        )

    assert result.status is AblationStatus.FAILED and result.without_spectral is None
    assert result.detail == "MemoryError: no room for a second model"
    assert "production model is unaffected" in caplog.text
    assert "spectral ablation failed: MemoryError" in result.describe()


def test_comparison_line_and_document(ablation: AblationResult) -> None:
    result = AblationResult(
        status=AblationStatus.COMPLETED,
        seconds=1.5,
        dropped_columns=SPECTRAL_FEATURE_COLUMNS,
        coverage={"train": 0.95, "validation": 1.0, "test": 0.97},
        spectral_importance_share=0.3,
        with_spectral=_metrics(rmse=10.0, f1=0.9),
        without_spectral=_metrics(rmse=12.0, f1=0.8),
    )

    line = result.describe()
    assert "with vs without spectral: RMSE 10.00 vs 12.00 | MAE 8.00 vs 9.60" in line
    assert "F1 0.900 vs 0.800 | ROC-AUC nan vs nan" in line
    assert "spectral share of regressor gain 30.0%" in line and line.endswith("+1.5 s")
    assert "spectral coverage train 95% / validation 100% / test 97%" in line
    document = result.as_dict()
    json.dumps(document, allow_nan=False)
    assert document["baseline"] == ABLATION_BASELINE
    assert document["with_minus_without"]["rmse"] == pytest.approx(-2.0)
    assert document["with_minus_without"]["clf_roc_auc"] is None
    assert document["dropped_columns"] == list(SPECTRAL_FEATURE_COLUMNS)
    assert set(ablation.summary()) == {"status", "seconds", "spectral_importance_share", "with_minus_without"}


def test_spectral_importance_share_sums_the_spectral_gain() -> None:
    importance = pd.DataFrame(
        {"feature": ["fan_amp_bpfo_mean_5m", "egt_c", "oil_temp_c_mean_5m"], "gain": [3.0, 5.0, 2.0]}
    )

    assert spectral_importance_share(importance) == pytest.approx(0.5)
    assert spectral_importance_share(importance.assign(gain=0.0)) == 0.0


def test_every_training_run_records_the_ablation_as_evidence_only(
    tmp_path: Path, synthetic_mart, make_context, caplog
) -> None:
    context = make_context(tmp_path, synthetic_mart, ablate=run_ablation)
    request = TrainingRequest(
        min_lifecycles=6, promotion_policy="better", skip_if_unchanged=False, dry_run=False
    )

    with caplog.at_level(logging.INFO):
        outcome = run_training(request, context, now=T0)

    document = outcome.ablation.as_dict()
    assert document["status"] == "completed"
    metrics = context.store.get_json(LAYOUT.artifact_key(outcome.model_version, METRICS_FILE))
    metadata = context.store.get_json(LAYOUT.artifact_key(outcome.model_version, METADATA_FILE))
    [record] = context.runs.records
    params = json.loads(record.params_json)
    assert (
        metrics["ablation"] == metadata["ablation"] == params["ablation"] == json.loads(json.dumps(document))
    )
    assert metadata["nullable_feature_columns"] == list(SPECTRAL_FEATURE_COLUMNS)
    # The production model keeps every feature; the ablation never changes what is served.
    assert read_production_pointer(context.store, LAYOUT).feature_columns == list(FEATURE_COLUMNS)
    assert record.rmse == outcome.holdout.rmse == document["with_spectral"]["rmse"]
    line = next(entry.getMessage() for entry in caplog.records if "spectral ablation" in entry.getMessage())
    assert "with vs without spectral: RMSE" in line and np.isfinite(outcome.ablation.seconds)
    summary = next(
        entry.getMessage() for entry in caplog.records if entry.getMessage().startswith("training summary")
    )
    assert json.loads(summary.removeprefix("training summary "))["ablation"]["status"] == "completed"


def test_production_training_contexts_always_measure_the_ablation(
    tmp_path: Path, synthetic_mart, make_context
) -> None:
    stubbed = make_context(tmp_path, synthetic_mart)

    production = TrainingContext(
        **{field.name: getattr(stubbed, field.name) for field in fields(stubbed) if field.name != "ablate"}
    )

    assert production.ablate is run_ablation and stubbed.ablate is not run_ablation
