"""Imminent-failure classifier: calibrated failure probabilities and the decision threshold.

The published ``failure_probability`` is a calibrated probability and the alarm threshold is
chosen on lifecycles the scoring model never saw. The procedure is scikit-learn's
``CalibratedClassifierCV(method="sigmoid", ensemble=False)`` over lifecycle-grouped folds,
written out explicitly because the threshold search needs the same out-of-fold probabilities
(the estimator does not expose them, and the pinned scikit-learn rejects ``cv="prefit"``):

1. ``HistGradientBoostingClassifier`` early-stops on the validation lifecycles, which fixes
   the number of boosting iterations.
2. With that iteration count, ``GroupKFold`` over the train + validation lifecycles
   (k = min(5, lifecycles)) gives every development row a score from a model that never saw
   its lifecycle: the situation of an engine in service. Splitting rows instead of
   lifecycles would leak, because consecutive pulses of one lifecycle are near-duplicates.
3. Platt scaling - a two-parameter sigmoid of the raw log-odds, with Platt's target smoothing -
   is fitted on those out-of-fold scores. The calibration sample is effectively the number of
   lifecycles (tens), not rows: isotonic regression has a free step per distinct score and
   memorises a handful of lifecycles, whereas two sigmoid parameters are well determined and
   preserve the ranking (ROC-AUC is unchanged by calibration).
4. The decision threshold maximises F1 on the out-of-fold calibrated probabilities, searched
   only within [0.05, 0.95]; ties go to the threshold closest to 0.5. For calibrated
   probabilities the F1-optimal threshold is half the attainable F1 (Lipton, Elkan and
   Naryanaswamy, 2014), i.e. near the middle of that range; the bounds keep it off the
   probability floor, where a few rows of a few lifecycles would decide it.
5. The served classifier is refitted on all train + validation rows with the same iteration
   count, and its raw scores pass through the fitted sigmoid.

Remaining caveat: calibration holds for lifecycles like the development ones; a shift in
operating conditions (new failure behaviour, sensor drift) can still make the model
over-confident, which the holdout Brier score and expected calibration error expose.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rul_metrics import brier_score, expected_calibration_error
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

RANDOM_STATE = 42
CLASSIFIER_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "random_state": RANDOM_STATE,
}
CALIBRATION_METHOD = "platt"
CALIBRATION_FOLDS = 5
THRESHOLD_BOUNDS = (0.05, 0.95)
NEUTRAL_THRESHOLD = 0.5
THRESHOLD_RULE = (
    "maximise F1 on out-of-fold calibrated probabilities over "
    f"[{THRESHOLD_BOUNDS[0]}, {THRESHOLD_BOUNDS[1]}], ties broken towards {NEUTRAL_THRESHOLD}"
)
_TIE_TOLERANCE = 1e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlattCalibration:
    """``p = 1 / (1 + exp(-(slope * score + intercept)))`` of the classifier's raw log-odds."""

    slope: float
    intercept: float

    def probability(self, score: np.ndarray) -> np.ndarray:
        """Calibrated probability of the raw decision scores ``score``."""
        return expit(self.slope * np.asarray(score, dtype=np.float64) + self.intercept)

    def to_payload(self) -> dict[str, Any]:
        """Plain mapping stored in the model bundle and in metadata.json."""
        return {"method": CALIBRATION_METHOD, "slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlattCalibration:
        """Rebuild a calibration from :meth:`to_payload` output."""
        if payload.get("method") != CALIBRATION_METHOD:
            raise ValueError(f"unsupported calibration method {payload.get('method')!r}")
        return cls(slope=float(payload["slope"]), intercept=float(payload["intercept"]))


def fit_platt(score: np.ndarray, outcome: np.ndarray) -> PlattCalibration:
    """Fit Platt scaling to raw scores, with Platt's (1999) smoothed targets.

    The targets ``(N+ + 1) / (N+ + 2)`` and ``1 / (N- + 2)`` instead of 1 and 0 keep the fit
    finite when the scores separate the classes perfectly (scikit-learn does the same).
    """
    score = np.asarray(score, dtype=np.float64)
    outcome = np.asarray(outcome).astype(bool)
    positives = int(outcome.sum())
    negatives = int(outcome.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("calibration needs rows on both sides of the horizon")
    target = np.where(outcome, (positives + 1) / (positives + 2), 1.0 / (negatives + 2))

    def loss_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        logit = params[0] * score + params[1]
        loss = -np.mean(target * log_expit(logit) + (1.0 - target) * log_expit(-logit))
        residual = expit(logit) - target
        return float(loss), np.array([np.mean(residual * score), np.mean(residual)])

    # Start from the identity: the raw scores already are log-odds.
    result = minimize(loss_and_gradient, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not result.success:
        logger.warning("Platt calibration did not fully converge: %s", result.message)
    return PlattCalibration(slope=float(result.x[0]), intercept=float(result.x[1]))


Fold = tuple[np.ndarray, np.ndarray]


def group_folds(groups: Sequence[str] | np.ndarray, max_folds: int = CALIBRATION_FOLDS) -> list[Fold]:
    """Lifecycle-grouped (fit rows, scored rows) folds: every lifecycle is scored by exactly one fold."""
    groups = np.asarray(groups)
    lifecycles = np.unique(groups).size
    if lifecycles < 2:
        raise ValueError("out-of-fold calibration needs at least 2 development lifecycles")
    splitter = GroupKFold(n_splits=min(max_folds, lifecycles))
    return list(splitter.split(np.zeros(groups.size), groups=groups))


def out_of_fold_scores(
    matrix: pd.DataFrame, outcome: np.ndarray, folds: Sequence[Fold], params: Mapping[str, Any]
) -> np.ndarray:
    """Raw decision scores of every row from a model fitted without the row's lifecycle."""
    scores = np.full(len(outcome), np.nan)
    for fit_rows, scored_rows in folds:
        if np.unique(outcome[fit_rows]).size < 2:
            raise ValueError("every calibration fold needs rows on both sides of the horizon")
        model = HistGradientBoostingClassifier(**params).fit(matrix.iloc[fit_rows], outcome[fit_rows])
        scores[scored_rows] = model.decision_function(matrix.iloc[scored_rows])
    return scores


def select_threshold(
    outcome: np.ndarray, probability: np.ndarray, bounds: tuple[float, float] = THRESHOLD_BOUNDS
) -> float:
    """Threshold maximising F1 of ``probability >= t`` with t in ``bounds``; ties go towards 0.5.

    Candidates are the distinct probabilities inside the bounds plus the bounds and 0.5,
    which covers every distinct decision the range allows. Without both classes the
    neutral 0.5 is returned.
    """
    outcome = np.asarray(outcome).astype(bool)
    probability = np.asarray(probability, dtype=np.float64)
    if np.unique(outcome).size < 2:
        return NEUTRAL_THRESHOLD
    low, high = bounds
    inside = probability[(probability >= low) & (probability <= high)]
    candidates = np.unique(np.concatenate([inside, [low, high, NEUTRAL_THRESHOLD]]))
    order = np.argsort(probability, kind="stable")
    ranked, ranked_outcome = probability[order], outcome[order]
    # positives_from[i] = positives among the rows ranked i.. (the rows flagged by ranked[i]).
    positives_from = np.concatenate([np.cumsum(ranked_outcome[::-1])[::-1], [0]])
    first_flagged = np.searchsorted(ranked, candidates, side="left")
    flagged = probability.size - first_flagged
    f1 = 2.0 * positives_from[first_flagged] / (flagged + outcome.sum())
    tied = candidates[f1 >= f1.max() - _TIE_TOLERANCE]
    return float(min(tied, key=lambda threshold: (abs(threshold - NEUTRAL_THRESHOLD), threshold)))


@dataclass(frozen=True)
class CalibrationReport:
    """How the classifier was calibrated and how its out-of-fold probabilities score."""

    folds: int
    lifecycles: int
    rows: int
    n_iter: int
    calibration: PlattCalibration
    threshold: float
    oof_f1: float
    oof_precision: float
    oof_recall: float
    oof_roc_auc: float
    oof_brier: float
    oof_ece: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping for metadata.json, metrics.json and ml_model_runs.params_json."""
        values: dict[str, Any] = {
            **self.calibration.to_payload(),
            "folds": self.folds,
            "lifecycles": self.lifecycles,
            "rows": self.rows,
            "n_iter": self.n_iter,
            "threshold": self.threshold,
            "threshold_rule": THRESHOLD_RULE,
            "oof_f1": self.oof_f1,
            "oof_precision": self.oof_precision,
            "oof_recall": self.oof_recall,
            "oof_roc_auc": self.oof_roc_auc,
            "oof_brier": self.oof_brier,
            "oof_ece": self.oof_ece,
        }
        return {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in values.items()
        }


@dataclass(frozen=True)
class FittedClassifier:
    """The served classifier, its calibration and threshold, and how they were obtained."""

    model: HistGradientBoostingClassifier
    calibration: PlattCalibration
    threshold: float
    report: CalibrationReport


def _report(
    outcome: np.ndarray,
    probability: np.ndarray,
    *,
    folds: int,
    lifecycles: int,
    n_iter: int,
    calibration: PlattCalibration,
    threshold: float,
) -> CalibrationReport:
    flagged = probability >= threshold
    return CalibrationReport(
        folds=folds,
        lifecycles=lifecycles,
        rows=int(outcome.size),
        n_iter=n_iter,
        calibration=calibration,
        threshold=threshold,
        oof_f1=float(f1_score(outcome, flagged, zero_division=0)),
        oof_precision=float(precision_score(outcome, flagged, zero_division=0)),
        oof_recall=float(recall_score(outcome, flagged, zero_division=0)),
        oof_roc_auc=float(roc_auc_score(outcome, probability)),
        oof_brier=brier_score(outcome, probability),
        oof_ece=expected_calibration_error(outcome, probability),
    )


def fit_failure_classifier(
    train: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    validation: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> FittedClassifier:
    """Fit, calibrate and threshold the classifier; each side is (matrix, outcome, split groups)."""
    x_train, y_train, groups_train = train
    x_val, y_val, groups_val = validation
    probe = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
    probe.fit(x_train, y_train, X_val=x_val, y_val=y_val)
    n_iter = int(probe.n_iter_)
    params = {**CLASSIFIER_PARAMS, "early_stopping": False, "max_iter": n_iter}

    matrix = pd.concat([x_train, x_val], ignore_index=True)
    outcome = np.concatenate([y_train, y_val]).astype(np.int8)
    groups = np.concatenate([np.asarray(groups_train), np.asarray(groups_val)])
    folds = group_folds(groups)
    oof_scores = out_of_fold_scores(matrix, outcome, folds, params)
    calibration = fit_platt(oof_scores, outcome)
    oof_probability = calibration.probability(oof_scores)
    threshold = select_threshold(outcome, oof_probability)
    model = HistGradientBoostingClassifier(**params).fit(matrix, outcome)
    report = _report(
        outcome,
        oof_probability,
        folds=len(folds),
        lifecycles=int(np.unique(groups).size),
        n_iter=n_iter,
        calibration=calibration,
        threshold=threshold,
    )
    return FittedClassifier(model=model, calibration=calibration, threshold=threshold, report=report)
