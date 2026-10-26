"""RUL model: lifecycle-aware split, training, threshold tuning and the model bundle.

Two models share one feature matrix:
  * an XGBoost regressor of the capped remaining useful life (piecewise-linear target), and
  * a scikit-learn classifier of "fails within the horizon", whose decision threshold is
    tuned for F1 on validation lifecycles.

The classifier is deliberately unweighted: class re-weighting inflates the minority
class probability, and ``failure_probability`` is published as a probability. The
F1-tuned threshold handles the class imbalance instead.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from feature_pipeline import FEATURE_COLUMNS, build_feature_matrix, fit_categories, matrix_columns
from rul_metrics import HoldoutMetrics, evaluate_predictions
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

BUNDLE_FORMAT_VERSION = 1
SPLIT_SEED = 42
TEST_FRACTION = 0.25
VALIDATION_FRACTION = 0.2
MIN_SPLIT_GROUPS = 3
DEFAULT_DECISION_THRESHOLD = 0.5
F1_TOLERANCE = 0.01

REGRESSOR_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "eval_metric": "rmse",
    "early_stopping_rounds": 75,
    "random_state": SPLIT_SEED,
}
CLASSIFIER_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "random_state": SPLIT_SEED,
}


@dataclass(frozen=True)
class GroupSplit:
    """Assignment of whole lifecycles (split groups) to train / validation / test."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def _unit_score(group: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_groups(
    groups: Iterable[str],
    *,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> GroupSplit:
    """Deterministically assign lifecycles to train / validation / test.

    A lifecycle's side depends only on (seed, split_group), never on which other
    lifecycles exist, so a lifecycle never migrates from train to test as more
    lifecycles complete. That keeps every holdout free of the lifecycles earlier
    (champion) models were trained on, which makes champion/challenger fair.
    Small fleets fall back to the lowest-scoring lifecycle so no side is empty.
    """
    unique = sorted({str(group) for group in groups})
    if len(unique) < MIN_SPLIT_GROUPS:
        raise ValueError(f"need at least {MIN_SPLIT_GROUPS} lifecycles to split, got {len(unique)}")
    test_score = {group: _unit_score(group, f"test:{seed}") for group in unique}
    test = {group for group in unique if test_score[group] < test_fraction}
    if not test:
        test = {min(unique, key=test_score.__getitem__)}
    while len(unique) - len(test) < 2:
        test.remove(max(test, key=test_score.__getitem__))
    remaining = [group for group in unique if group not in test]
    val_score = {group: _unit_score(group, f"validation:{seed}") for group in remaining}
    validation = {group for group in remaining if val_score[group] < validation_fraction}
    if not validation:
        validation = {min(remaining, key=val_score.__getitem__)}
    if len(validation) == len(remaining):
        validation.remove(max(validation, key=val_score.__getitem__))
    return GroupSplit(
        train=tuple(group for group in remaining if group not in validation),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
    )


@dataclass(frozen=True)
class LabelSpec:
    """How both training targets derive from ``rul_cycles``."""

    rul_cap_cycles: int
    failure_horizon_cycles: int

    def regression_target(self, frame: pd.DataFrame) -> np.ndarray:
        """Capped RUL (``least(rul_cycles, cap)``)."""
        return np.minimum(frame["rul_cycles"].to_numpy(dtype=np.float64), float(self.rul_cap_cycles))

    def failure_target(self, frame: pd.DataFrame) -> np.ndarray:
        """1 when the engine fails within the horizon (``rul_cycles <= horizon``)."""
        return (frame["rul_cycles"].to_numpy(dtype=np.float64) <= self.failure_horizon_cycles).astype(np.int8)

    def mismatches(self, frame: pd.DataFrame) -> dict[str, int]:
        """Rows whose dbt-computed labels disagree with this spec (dbt vars vs trainer env)."""
        return {
            "rul_cycles_capped": int(
                np.count_nonzero(
                    frame["rul_cycles_capped"].to_numpy(np.float64) != self.regression_target(frame)
                )
            ),
            "failure_within_horizon": int(
                np.count_nonzero(
                    frame["failure_within_horizon"].to_numpy(np.int8) != self.failure_target(frame)
                )
            ),
        }


@dataclass
class RulModelBundle:
    """Both fitted models plus the feature and label contract they were trained with."""

    model_version: str
    trained_at: str
    regressor: xgb.XGBRegressor
    classifier: HistGradientBoostingClassifier
    decision_threshold: float
    feature_columns: tuple[str, ...]
    categories: dict[str, list[str]]
    failure_horizon_cycles: int
    rul_cap_cycles: int
    hours_per_cycle: float
    library_versions: dict[str, str]

    @property
    def matrix_columns(self) -> list[str]:
        """Model input columns (numeric features, then one-hot indicators)."""
        return matrix_columns(self.feature_columns, self.categories)

    def feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode ``frame`` exactly as at training time."""
        return build_feature_matrix(frame, self.categories, self.feature_columns)

    def predict_rul_cycles(self, matrix: pd.DataFrame) -> np.ndarray:
        """Remaining useful life in cycles, clipped to [0, cap]."""
        predicted = self.regressor.predict(matrix).astype(np.float64)
        return np.clip(predicted, 0.0, float(self.rul_cap_cycles))

    def predict_failure_probability(self, matrix: pd.DataFrame) -> np.ndarray:
        """Probability of failing within ``failure_horizon_cycles``."""
        positive = list(self.classifier.classes_).index(1)
        return self.classifier.predict_proba(matrix)[:, positive].astype(np.float64)

    def to_payload(self) -> dict[str, Any]:
        """Plain mapping persisted with joblib (no dependency on this class at load time)."""
        return {
            "bundle_format": BUNDLE_FORMAT_VERSION,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "regressor": self.regressor,
            "classifier": self.classifier,
            "decision_threshold": self.decision_threshold,
            "feature_columns": list(self.feature_columns),
            "categories": {column: list(levels) for column, levels in self.categories.items()},
            "failure_horizon_cycles": self.failure_horizon_cycles,
            "rul_cap_cycles": self.rul_cap_cycles,
            "hours_per_cycle": self.hours_per_cycle,
            "library_versions": dict(self.library_versions),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RulModelBundle:
        """Rebuild a bundle from :meth:`to_payload` output."""
        if payload.get("bundle_format") != BUNDLE_FORMAT_VERSION:
            raise ValueError(f"unsupported model bundle format {payload.get('bundle_format')!r}")
        return cls(
            model_version=str(payload["model_version"]),
            trained_at=str(payload["trained_at"]),
            regressor=payload["regressor"],
            classifier=payload["classifier"],
            decision_threshold=float(payload["decision_threshold"]),
            feature_columns=tuple(payload["feature_columns"]),
            categories={column: list(levels) for column, levels in payload["categories"].items()},
            failure_horizon_cycles=int(payload["failure_horizon_cycles"]),
            rul_cap_cycles=int(payload["rul_cap_cycles"]),
            hours_per_cycle=float(payload["hours_per_cycle"]),
            library_versions=dict(payload["library_versions"]),
        )


@dataclass(frozen=True)
class TrainedModel:
    """Result of fitting both models on the training lifecycles."""

    bundle: RulModelBundle
    best_iteration: int
    validation: HoldoutMetrics


def library_versions() -> dict[str, str]:
    """Versions of the libraries a bundle depends on (recorded for reproducibility)."""
    return {
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def tune_threshold(failure_true: np.ndarray, probability: np.ndarray) -> float:
    """Lowest decision threshold whose F1 on the given (validation) rows is near the best.

    The F1 curve is flat around its peak on a few validation lifecycles, so its exact
    argmax is noise. Among thresholds within F1_TOLERANCE of the best, the lowest one
    maximises recall: a missed failure costs more than a false alarm.
    """
    if np.unique(failure_true).size < 2:
        return DEFAULT_DECISION_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(failure_true, probability)
    # The last precision/recall pair has no threshold (it is the "predict nothing" point).
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    near_best = np.flatnonzero(f1 >= f1.max() - F1_TOLERANCE)
    return float(thresholds[near_best].min())


def train_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    labels: LabelSpec,
    *,
    model_version: str,
    trained_at: str,
    hours_per_cycle: float,
) -> TrainedModel:
    """Fit the regressor and the classifier; early stopping and threshold use validation rows."""
    categories = fit_categories(train_frame)
    x_train = build_feature_matrix(train_frame, categories)
    x_val = build_feature_matrix(validation_frame, categories)
    failure_train = labels.failure_target(train_frame)
    failure_val = labels.failure_target(validation_frame)
    if np.unique(failure_train).size < 2 or np.unique(failure_val).size < 2:
        raise ValueError("training and validation lifecycles must contain rows on both sides of the horizon")

    regressor = xgb.XGBRegressor(**REGRESSOR_PARAMS)
    regressor.fit(
        x_train,
        labels.regression_target(train_frame),
        eval_set=[(x_val, labels.regression_target(validation_frame))],
        verbose=False,
    )
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(x_train, failure_train, X_val=x_val, y_val=failure_val)

    bundle = RulModelBundle(
        model_version=model_version,
        trained_at=trained_at,
        regressor=regressor,
        classifier=classifier,
        decision_threshold=DEFAULT_DECISION_THRESHOLD,
        feature_columns=FEATURE_COLUMNS,
        categories=categories,
        failure_horizon_cycles=labels.failure_horizon_cycles,
        rul_cap_cycles=labels.rul_cap_cycles,
        hours_per_cycle=hours_per_cycle,
        library_versions=library_versions(),
    )
    bundle.decision_threshold = tune_threshold(failure_val, bundle.predict_failure_probability(x_val))
    return TrainedModel(
        bundle=bundle,
        best_iteration=int(regressor.best_iteration),
        validation=evaluate_bundle(bundle, validation_frame, labels),
    )


def evaluate_bundle(bundle: RulModelBundle, frame: pd.DataFrame, labels: LabelSpec) -> HoldoutMetrics:
    """Score ``bundle`` on ``frame`` against the targets defined by ``labels``."""
    matrix = bundle.feature_matrix(frame)
    return evaluate_predictions(
        rul_true=frame["rul_cycles"].to_numpy(dtype=np.float64),
        rul_pred=bundle.predict_rul_cycles(matrix),
        failure_probability=bundle.predict_failure_probability(matrix),
        threshold=bundle.decision_threshold,
        rul_cap_cycles=labels.rul_cap_cycles,
        failure_horizon_cycles=labels.failure_horizon_cycles,
        lifecycles=int(frame["split_group"].nunique()),
    )


def feature_importance(bundle: RulModelBundle) -> pd.DataFrame:
    """XGBoost importances (gain, weight, cover) of every model input, by gain descending."""
    booster = bundle.regressor.get_booster()
    importance = pd.DataFrame({"feature": bundle.matrix_columns})
    for kind in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=kind)
        importance[kind] = [float(scores.get(name, 0.0)) for name in importance["feature"]]
    total_gain = float(importance["gain"].sum())
    importance["gain_share"] = importance["gain"] / total_gain if total_gain > 0 else 0.0
    return importance.sort_values("gain", ascending=False, kind="mergesort").reset_index(drop=True)
