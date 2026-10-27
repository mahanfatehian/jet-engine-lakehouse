"""Lifecycle-aware split, model quality on synthetic data, metrics, calibration and the threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from failure_classifier import (
    CALIBRATION_FOLDS,
    CLASSIFIER_PARAMS,
    NEUTRAL_THRESHOLD,
    THRESHOLD_BOUNDS,
    PlattCalibration,
    fit_platt,
    group_folds,
    out_of_fold_scores,
    select_threshold,
    without_empty_columns,
)
from feature_pipeline import FEATURE_COLUMNS, SPECTRAL_FEATURE_COLUMNS, build_feature_matrix, fit_categories
from rul_metrics import brier_score, evaluate_predictions, expected_calibration_error, nasa_scores
from rul_modeling import (
    IncompatibleModelError,
    LabelSpec,
    RulModelBundle,
    evaluate_bundle,
    feature_importance,
    split_groups,
    train_models,
)
from scipy.special import expit
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
    assert holdout.clf_brier < 0.1 and holdout.clf_ece < 0.1
    assert holdout.rows == len(synthetic_training.frames["test"])
    assert holdout.lifecycles == len(synthetic_training.split.test)


def _patchy_spectral(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Spectral features missing on ~40 % of the rows and on one whole lifecycle (HF disabled)."""
    rng = np.random.default_rng(seed)
    first_lifecycle = (frame["split_group"] == frame["split_group"].iloc[0]).to_numpy()
    patchy = frame.copy()
    patchy.loc[(rng.random(len(patchy)) < 0.4) | first_lifecycle, list(SPECTRAL_FEATURE_COLUMNS)] = np.nan
    return patchy


def test_both_models_train_calibrate_and_score_with_missing_spectral_features(
    synthetic_training, label_spec: LabelSpec, hf_disabled
) -> None:
    frames = {
        side: _patchy_spectral(frame, seed)
        for seed, (side, frame) in enumerate(synthetic_training.frames.items())
    }
    assert all(frame[list(SPECTRAL_FEATURE_COLUMNS)].isna().mean().min() > 0.3 for frame in frames.values())

    trained = train_models(
        frames["train"],
        frames["validation"],
        label_spec,
        model_version="rul-20260101T000000Z",
        trained_at="2026-01-01T00:00:00+00:00",
        hours_per_cycle=2.0,
    )
    holdout = evaluate_bundle(trained.bundle, frames["test"], label_spec)

    report = trained.calibration
    assert np.isfinite([report.calibration.slope, report.calibration.intercept, report.oof_brier]).all()
    assert THRESHOLD_BOUNDS[0] <= report.threshold <= THRESHOLD_BOUNDS[1]
    assert report.rows == len(frames["train"]) + len(frames["validation"])
    assert holdout.rows == len(frames["test"]) and np.isfinite([holdout.rmse, holdout.clf_roc_auc]).all()
    matrix = trained.bundle.feature_matrix(hf_disabled(frames["test"]))
    probability = trained.bundle.predict_failure_probability(matrix)
    assert np.isfinite(trained.bundle.predict_rul_cycles(matrix)).all()
    assert ((probability > 0) & (probability < 1)).all()
    # Still above the promotion floors (R2 0.5, ROC-AUC 0.75) with 40 % of the spectral values gone.
    assert holdout.r2 > 0.5 and holdout.clf_roc_auc > 0.9


def test_models_train_when_hf_is_disabled(synthetic_training, label_spec: LabelSpec, hf_disabled) -> None:
    # scikit-learn's HistGradientBoosting cannot bin a column without any observed value.
    frames = {side: hf_disabled(frame) for side, frame in synthetic_training.frames.items()}

    trained = train_models(
        frames["train"],
        frames["validation"],
        label_spec,
        model_version="rul-20260101T000000Z",
        trained_at="2026-01-01T00:00:00+00:00",
        hours_per_cycle=2.0,
    )

    bundle = trained.bundle
    assert bundle.feature_columns == FEATURE_COLUMNS
    holdout = evaluate_bundle(bundle, frames["test"], label_spec)
    assert holdout.r2 > 0.5 and np.isfinite(trained.calibration.oof_brier)
    # With HF back, the served models still score: they simply never split on the spectral features.
    live = bundle.feature_matrix(synthetic_training.frames["test"])
    np.testing.assert_array_equal(
        bundle.predict_rul_cycles(live), bundle.predict_rul_cycles(bundle.feature_matrix(frames["test"]))
    )
    importance = feature_importance(bundle).set_index("feature")["gain"]
    assert importance[list(SPECTRAL_FEATURE_COLUMNS)].sum() == 0.0


def test_empty_columns_become_constant_for_a_classifier_fit_only() -> None:
    matrix = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan], "c": [np.nan, 2.0, np.nan]}, dtype=np.float32
    )

    filled = without_empty_columns(matrix)

    assert list(filled.columns) == ["a", "b", "c"] and (filled.dtypes == np.float32).all()
    assert filled["b"].tolist() == [0.0, 0.0, 0.0]
    pd.testing.assert_frame_equal(filled[["a", "c"]], matrix[["a", "c"]])
    assert without_empty_columns(filled) is filled and matrix["b"].isna().all()


def test_out_of_fold_scores_must_cover_every_row(synthetic_mart: pd.DataFrame, label_spec: LabelSpec) -> None:
    matrix = build_feature_matrix(synthetic_mart, fit_categories(synthetic_mart))
    assert matrix[list(SPECTRAL_FEATURE_COLUMNS)].isna().any(axis=None)
    outcome = label_spec.failure_target(synthetic_mart)
    folds = group_folds(synthetic_mart["split_group"].to_numpy())
    params = {**CLASSIFIER_PARAMS, "early_stopping": False, "max_iter": 5}

    scores = out_of_fold_scores(matrix, outcome, folds, params)

    assert np.isfinite(scores).all() and len(scores) == len(matrix)
    _, scored_rows = folds[0]
    with pytest.raises(ValueError, match=f"{len(scored_rows)} rows got no finite out-of-fold score"):
        out_of_fold_scores(matrix, outcome, folds[1:], params)


def test_bundle_carries_its_contract(synthetic_training, label_spec: LabelSpec) -> None:
    trained = synthetic_training.trained
    bundle: RulModelBundle = trained.bundle

    assert bundle.feature_columns == FEATURE_COLUMNS
    assert bundle.rul_cap_cycles == label_spec.rul_cap_cycles
    assert bundle.failure_horizon_cycles == label_spec.failure_horizon_cycles
    assert THRESHOLD_BOUNDS[0] <= bundle.decision_threshold <= THRESHOLD_BOUNDS[1]
    assert bundle.decision_threshold == trained.calibration.threshold
    assert bundle.calibration == trained.calibration.calibration and bundle.calibration.slope > 0
    assert bundle.categories == {
        "engine_model": sorted(synthetic_training.frames["train"]["engine_model"].unique())
    }
    assert 0 < trained.best_iteration < 2000
    predictions = bundle.predict_rul_cycles(bundle.feature_matrix(synthetic_training.frames["test"]))
    assert predictions.min() >= 0.0 and predictions.max() <= label_spec.rul_cap_cycles


def test_published_probability_is_the_calibrated_classifier_score(synthetic_training) -> None:
    bundle = synthetic_training.trained.bundle
    matrix = bundle.feature_matrix(synthetic_training.frames["test"])

    probability = bundle.predict_failure_probability(matrix)

    raw = bundle.classifier.decision_function(matrix)
    np.testing.assert_allclose(
        probability, expit(bundle.calibration.slope * raw + bundle.calibration.intercept)
    )
    assert ((probability > 0) & (probability < 1)).all()


def test_calibration_report_comes_from_out_of_fold_lifecycles(synthetic_training) -> None:
    report = synthetic_training.trained.calibration
    development = synthetic_training.split.train + synthetic_training.split.validation

    assert report.lifecycles == len(development)
    assert report.folds == min(CALIBRATION_FOLDS, len(development))
    assert report.rows == len(synthetic_training.frames["train"]) + len(
        synthetic_training.frames["validation"]
    )
    assert report.n_iter == synthetic_training.trained.bundle.classifier.n_iter_
    assert report.oof_roc_auc > 0.9 and 0 <= report.oof_brier < 0.1 and 0 <= report.oof_ece < 0.1
    document = report.as_dict()
    assert document["method"] == "platt" and document["threshold"] == report.threshold
    assert "out-of-fold" in document["threshold_rule"]


def test_regressor_keeps_only_the_trees_up_to_its_best_iteration(synthetic_training) -> None:
    trained = synthetic_training.trained
    booster = trained.bundle.regressor.get_booster()

    assert booster.num_boosted_rounds() == trained.best_iteration + 1


def test_feature_importance_covers_every_model_input(synthetic_training) -> None:
    bundle = synthetic_training.trained.bundle

    importance = feature_importance(bundle)

    assert sorted(importance["feature"]) == sorted(bundle.matrix_columns)
    assert importance["gain"].is_monotonic_decreasing
    assert importance["gain_share"].sum() == pytest.approx(1.0)
    trees = bundle.regressor.get_booster().trees_to_dataframe()
    assert importance["weight"].sum() == (trees["Feature"] != "Leaf").sum()


def test_bundle_payload_round_trip_and_format_check(synthetic_training) -> None:
    bundle = synthetic_training.trained.bundle
    payload = bundle.to_payload()

    restored = RulModelBundle.from_payload(payload)

    assert restored.calibration == bundle.calibration
    assert restored.decision_threshold == bundle.decision_threshold
    with pytest.raises(IncompatibleModelError, match="format 1"):
        RulModelBundle.from_payload({**payload, "bundle_format": 1})


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
    assert metrics.clf_brier == pytest.approx((0.01 + 0.04 + 0.09 + 0.01) / 4)
    assert metrics.clf_ece == pytest.approx((0.1 + 0.2 + 0.3 + 0.1) / 4)


def test_brier_score_and_calibration_error_known_values() -> None:
    assert brier_score(np.array([0, 1]), np.array([0.2, 0.6])) == pytest.approx(0.1)
    outcome = np.array([0, 0, 1, 1])
    assert expected_calibration_error(outcome, np.array([0.05, 0.15, 0.95, 0.85])) == pytest.approx(0.1)
    # Four rows stated at 25 % with one positive: perfectly calibrated.
    assert expected_calibration_error(np.array([1, 0, 0, 0]), np.full(4, 0.25)) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([1]), np.array([1.0])) == pytest.approx(0.0)


def test_platt_scaling_recovers_a_known_sigmoid() -> None:
    rng = np.random.default_rng(3)
    score = rng.normal(0.0, 2.0, 40_000)
    outcome = rng.random(score.size) < expit(1.7 * score - 0.8)

    calibration = fit_platt(score, outcome)

    assert calibration.slope == pytest.approx(1.7, abs=0.08)
    assert calibration.intercept == pytest.approx(-0.8, abs=0.08)


def test_platt_scaling_stays_finite_on_separable_scores() -> None:
    score = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    outcome = np.array([0, 0, 0, 1, 1, 1])

    calibration = fit_platt(score, outcome)
    probability = calibration.probability(score)

    assert np.isfinite([calibration.slope, calibration.intercept]).all()
    assert ((probability > 0) & (probability < 1)).all()
    assert probability[3:].min() > 0.5 > probability[:3].max()
    with pytest.raises(ValueError, match="both sides"):
        fit_platt(score, np.zeros(6))


def test_platt_payload_round_trip() -> None:
    calibration = PlattCalibration(slope=1.25, intercept=-0.5)

    assert PlattCalibration.from_payload(calibration.to_payload()) == calibration
    with pytest.raises(ValueError, match="isotonic"):
        PlattCalibration.from_payload({"method": "isotonic"})


def test_group_folds_score_every_lifecycle_exactly_once() -> None:
    groups = np.repeat([f"E#{index}" for index in range(8)], 5)

    folds = group_folds(groups)

    assert len(folds) == CALIBRATION_FOLDS
    scored = np.concatenate([scored_rows for _, scored_rows in folds])
    assert sorted(scored) == list(range(groups.size))
    for fit_rows, scored_rows in folds:
        assert set(groups[fit_rows]).isdisjoint(groups[scored_rows])
    assert len(group_folds(groups[:10])) == 2
    with pytest.raises(ValueError, match="at least 2"):
        group_folds(groups[:5])


def _f1_at(outcome: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    return float(f1_score(outcome, (probability >= threshold).astype(int), zero_division=0))


@pytest.mark.parametrize("seed", range(5))
def test_threshold_maximises_f1_within_the_bounds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    outcome = rng.integers(0, 2, 400)
    probability = np.clip(0.35 * outcome + rng.normal(0.3, 0.2, 400), 0.0, 1.0)

    threshold = select_threshold(outcome, probability)

    low, high = THRESHOLD_BOUNDS
    assert low <= threshold <= high
    grid = [low, high, *probability[(probability >= low) & (probability <= high)]]
    best = max(_f1_at(outcome, probability, candidate) for candidate in grid)
    assert _f1_at(outcome, probability, threshold) == pytest.approx(best)


def test_threshold_never_drops_to_the_probability_floor() -> None:
    # A perfect F1 would need a threshold of 0.01, right above the negatives' floor of 0.001.
    outcome = np.array([0] * 50 + [1] * 10)
    probability = np.array([0.001] * 50 + [0.01] * 5 + [0.2] * 5)

    threshold = select_threshold(outcome, probability)

    assert _f1_at(outcome, probability, 0.01) == 1.0
    # Within the bounds 0.05 and 0.2 flag the same rows; the tie goes towards 0.5.
    assert threshold == pytest.approx(0.2)
    assert _f1_at(outcome, probability, threshold) == pytest.approx(2 / 3)


def test_threshold_ties_break_towards_one_half() -> None:
    outcome = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.02, 0.08, 0.1, 0.9, 0.93, 0.99])

    assert select_threshold(outcome, probability) == NEUTRAL_THRESHOLD
    shifted = np.array([0.02, 0.08, 0.1, 0.3, 0.32, 0.35])
    assert select_threshold(outcome, shifted) == pytest.approx(0.3)


def test_threshold_without_both_classes_is_neutral() -> None:
    assert select_threshold(np.zeros(4, dtype=int), np.linspace(0, 1, 4)) == NEUTRAL_THRESHOLD
