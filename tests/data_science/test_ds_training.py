"""Lifecycle-aware split, model quality on synthetic data, metrics and threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rul_metrics import evaluate_predictions, nasa_scores
from rul_modeling import (
    F1_TOLERANCE,
    LabelSpec,
    RulModelBundle,
    feature_importance,
    split_groups,
    tune_threshold,
)
from sklearn.metrics import f1_score


def test_split_keeps_every_lifecycle_on_exactly_one_side(synthetic_mart: pd.DataFrame) -> None:
    groups = synthetic_mart["split_group"].unique()

    split = split_groups(groups)

    train, validation, test = set(split.train), set(split.validation), set(split.test)
    assert train and validation and test
    assert train.isdisjoint(test) and validation.isdisjoint(test) and train.isdisjoint(validation)
    assert train | validation | test == set(groups)


def test_split_rows_never_share_a_lifecycle(synthetic_training) -> None:
    frames = synthetic_training.frames
    test_groups = set(frames["test"]["split_group"])

    assert test_groups.isdisjoint(frames["train"]["split_group"])
    assert test_groups.isdisjoint(frames["validation"]["split_group"])
    assert sum(len(frame) for frame in frames.values()) == len(pd.concat(frames.values()))


def test_split_is_stable_as_more_lifecycles_complete() -> None:
    early = [f"ENG-{engine:03d}#{cycle}" for engine in range(1, 9) for cycle in range(1, 6)]
    later = early + [f"ENG-{engine:03d}#{cycle}" for engine in range(1, 9) for cycle in range(6, 9)]

    before, after = split_groups(early), split_groups(later)

    assert set(before.test) <= set(after.test)
    assert set(before.train) | set(before.validation) <= set(after.train) | set(after.validation)
    assert 0.1 <= len(after.test) / len(later) <= 0.4


def test_split_is_deterministic_and_order_independent() -> None:
    groups = [f"E#{index}" for index in range(30)]

    assert split_groups(groups) == split_groups(list(reversed(groups)))


def test_split_needs_three_lifecycles() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        split_groups(["A#1", "A#2"])
    assert split_groups(["A#1", "A#2", "A#3"]).train


def test_regressor_and_classifier_learn_the_planted_signal(synthetic_training) -> None:
    holdout = synthetic_training.holdout

    assert holdout.r2 > 0.6
    assert holdout.rmse < 25.0
    assert holdout.late_life_rmse < 25.0
    assert holdout.clf_f1 > 0.7
    assert holdout.clf_roc_auc > 0.9
    assert holdout.rows == len(synthetic_training.frames["test"])
    assert holdout.lifecycles == len(synthetic_training.split.test)


def test_bundle_carries_its_contract(synthetic_training, label_spec: LabelSpec) -> None:
    trained = synthetic_training.trained
    bundle: RulModelBundle = trained.bundle

    assert bundle.rul_cap_cycles == label_spec.rul_cap_cycles
    assert bundle.failure_horizon_cycles == label_spec.failure_horizon_cycles
    assert 0.0 < bundle.decision_threshold < 1.0
    assert bundle.categories == {
        "engine_model": sorted(synthetic_training.frames["train"]["engine_model"].unique())
    }
    assert 0 < trained.best_iteration < 2000
    predictions = bundle.predict_rul_cycles(bundle.feature_matrix(synthetic_training.frames["test"]))
    assert predictions.min() >= 0.0 and predictions.max() <= label_spec.rul_cap_cycles


def test_feature_importance_covers_every_model_input(synthetic_training) -> None:
    bundle = synthetic_training.trained.bundle

    importance = feature_importance(bundle)

    assert sorted(importance["feature"]) == sorted(bundle.matrix_columns)
    assert importance["gain"].is_monotonic_decreasing
    assert importance["gain_share"].sum() == pytest.approx(1.0)


def test_label_spec_targets_and_mismatch_detection(synthetic_mart: pd.DataFrame) -> None:
    labels = LabelSpec(rul_cap_cycles=125, failure_horizon_cycles=25)
    assert labels.mismatches(synthetic_mart) == {"rul_cycles_capped": 0, "failure_within_horizon": 0}

    other = LabelSpec(rul_cap_cycles=100, failure_horizon_cycles=30)
    mismatches = other.mismatches(synthetic_mart)
    assert mismatches["rul_cycles_capped"] > 0 and mismatches["failure_within_horizon"] > 0
    assert other.regression_target(synthetic_mart).max() == 100


def test_nasa_score_penalises_late_predictions_harder() -> None:
    late, early = nasa_scores(np.array([50.0, 50.0]), np.array([60.0, 40.0]))

    assert late == pytest.approx(np.exp(1.0) - 1)
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late > early > 0
    assert nasa_scores(np.array([10.0]), np.array([10.0]))[0] == 0.0


def test_metrics_use_capped_target_and_late_life_rows() -> None:
    rul_true = np.array([200.0, 125.0, 60.0, 10.0])
    rul_pred = np.array([125.0, 125.0, 50.0, 20.0])
    probability = np.array([0.1, 0.2, 0.3, 0.9])

    metrics = evaluate_predictions(
        rul_true=rul_true,
        rul_pred=rul_pred,
        failure_probability=probability,
        threshold=0.5,
        rul_cap_cycles=125,
        failure_horizon_cycles=25,
        lifecycles=1,
    )

    assert metrics.rmse == pytest.approx(np.sqrt((0 + 0 + 100 + 100) / 4))
    assert metrics.late_life_rmse == pytest.approx(np.sqrt((0 + 100 + 100) / 3))
    assert metrics.clf_f1 == 1.0 and metrics.clf_roc_auc == 1.0
    assert metrics.positive_rate == 0.25


def test_threshold_tuning_maximises_f1() -> None:
    truth = np.array([0, 0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.1, 0.2, 0.35, 0.3, 0.6, 0.9])

    threshold = tune_threshold(truth, probability)

    assert threshold == pytest.approx(0.3)
    assert tune_threshold(np.zeros(4, dtype=int), np.linspace(0, 1, 4)) == 0.5


@pytest.mark.parametrize("seed", range(5))
def test_threshold_is_the_lowest_near_optimal_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, 400)
    probability = np.clip(0.35 * truth + rng.normal(0.3, 0.2, 400), 0.0, 1.0)

    threshold = tune_threshold(truth, probability)

    candidates = np.unique(probability)
    f1 = np.array([f1_score(truth, (probability >= c).astype(int)) for c in candidates])
    assert f1_score(truth, (probability >= threshold).astype(int)) >= f1.max() - F1_TOLERANCE - 1e-12
    assert not (f1[candidates < threshold] >= f1.max() - F1_TOLERANCE).any()
