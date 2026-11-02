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

Missing values (the spectral features) are routed by the trees natively. A feature with no
observed value at all in a fit's rows - every spectral feature when HF is disabled, or in a
fold that holds none of the few lifecycles with HF data - carries nothing to split on, but
the pinned scikit-learn cannot bin it (it raises). Such a column is made constant for that
fit only, which gives it a single bin: the trees can never split on it, exactly as if it
were absent, while every model keeps the full input schema.

Rows can carry weights: training passes the spectral-dropout copies of the development rows
(rul_modeling.py) at a fraction of a row's weight, and every step above weighs them alike -
the early stopping, each fit, the Platt fit, the threshold's F1 and the out-of-fold report.

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
from rul_metrics import brier_score, expected_calibration_error, row_weights
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


def fit_platt(score: np.ndarray, outcome: np.ndarray, weight: np.ndarray | None = None) -> PlattCalibration:
    """Fit Platt scaling to raw scores, with Platt's (1999) smoothed targets.

    The targets ``(N+ + 1) / (N+ + 2)`` and ``1 / (N- + 2)`` instead of 1 and 0 keep the fit
    finite when the scores separate the classes perfectly (scikit-learn does the same).
    ``weight`` weighs each row in the counts and the loss (default: every row counts once).
    """
    score = np.asarray(score, dtype=np.float64)
    outcome = np.asarray(outcome).astype(bool)
    weight = row_weights(outcome, weight)
    positives = float(weight[outcome].sum())
    negatives = float(weight[~outcome].sum())
    if positives == 0 or negatives == 0:
        raise ValueError("calibration needs rows on both sides of the horizon")
    target = np.where(outcome, (positives + 1) / (positives + 2), 1.0 / (negatives + 2))

    def loss_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        logit = params[0] * score + params[1]
        loss = -np.average(target * log_expit(logit) + (1.0 - target) * log_expit(-logit), weights=weight)
        residual = expit(logit) - target
        gradient = [np.average(residual * score, weights=weight), np.average(residual, weights=weight)]
        return float(loss), np.array(gradient)

    # Start from the identity: the raw scores already are log-odds.
    result = minimize(loss_and_gradient, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not result.success:
        logger.warning("Platt calibration did not fully converge: %s", result.message)
    return PlattCalibration(slope=float(result.x[0]), intercept=float(result.x[1]))


Fold = tuple[np.ndarray, np.ndarray]


def empty_columns(matrix: pd.DataFrame) -> list[str]:
    """Columns of ``matrix`` without any observed value."""
    return list(matrix.columns[matrix.isna().all(axis=0).to_numpy()])


def without_empty_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """``matrix`` with every column that has no observed value replaced by a constant (see above)."""
    empty = empty_columns(matrix)
    if not empty:
        return matrix
    return matrix.assign(**dict.fromkeys(empty, np.float32(0.0)))


class WeightedLossClassifier(HistGradientBoostingClassifier):
    """``HistGradientBoostingClassifier`` whose bin edges ignore the sample weights.

    Below 200k rows scikit-learn derives every feature's 255 bin edges from weighted
    percentiles, one full pass over the column per edge, which made the fits of a few hundred
    thousand weighted rows ten times slower than unweighted ones. The only weights here are the
    spectral-dropout copies' (rul_modeling): a copy repeats its original's pulse features and
    holds NaN, which binning ignores, in every spectral feature. Without the weights the edges
    are therefore exactly the quantiles of the original rows alone, as if no copy existed; the
    weighted percentiles differ from them only by rounding a few edges to a neighbouring value
    (well under 0.1 % of a feature's range). Loss, early stopping and everything downstream still
    see the weights. Relies on the private ``_bin_data`` of the pinned scikit-learn, and a test
    pins the edges.
    """

    def _bin_data(
        self, data: np.ndarray, sample_weight: np.ndarray | None, is_training_data: bool
    ) -> np.ndarray:
        return super()._bin_data(data, None, is_training_data)


def _fit(
    params: Mapping[str, Any], matrix: pd.DataFrame, outcome: np.ndarray, **fit_params: Any
) -> HistGradientBoostingClassifier:
    return WeightedLossClassifier(**params).fit(without_empty_columns(matrix), outcome, **fit_params)


def group_folds(groups: Sequence[str] | np.ndarray, max_folds: int = CALIBRATION_FOLDS) -> list[Fold]:
    """Lifecycle-grouped (fit rows, scored rows) folds: every lifecycle is scored by exactly one fold."""
    groups = np.asarray(groups)
    lifecycles = np.unique(groups).size
    if lifecycles < 2:
        raise ValueError("out-of-fold calibration needs at least 2 development lifecycles")
    splitter = GroupKFold(n_splits=min(max_folds, lifecycles))
    return list(splitter.split(np.zeros(groups.size), groups=groups))


def out_of_fold_scores(
    matrix: pd.DataFrame,
    outcome: np.ndarray,
    folds: Sequence[Fold],
    params: Mapping[str, Any],
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Raw decision scores of every row from a model fitted without the row's lifecycle.

    Missing feature values (the spectral features) are routed by the trees and still
    yield a finite score, so a row left without one means the folds did not cover it;
    that would poison the Platt fit, so it raises instead. ``weight`` weighs the rows of
    every fold's fit.
    """
    scores = np.full(len(outcome), np.nan)
    for fit_rows, scored_rows in folds:
        if np.unique(outcome[fit_rows]).size < 2:
            raise ValueError("every calibration fold needs rows on both sides of the horizon")
        fit_weight = None if weight is None else weight[fit_rows]
        model = _fit(params, matrix.iloc[fit_rows], outcome[fit_rows], sample_weight=fit_weight)
        scores[scored_rows] = model.decision_function(matrix.iloc[scored_rows])
    unscored = int(np.count_nonzero(~np.isfinite(scores)))
    if unscored:
        raise ValueError(f"{unscored} rows got no finite out-of-fold score; the folds must cover every row")
    return scores


def select_threshold(
    outcome: np.ndarray,
    probability: np.ndarray,
    bounds: tuple[float, float] = THRESHOLD_BOUNDS,
    weight: np.ndarray | None = None,
) -> float:
    """Threshold maximising F1 of ``probability >= t`` with t in ``bounds``; ties go towards 0.5.

    Candidates are the distinct probabilities inside the bounds plus the bounds and 0.5,
    which covers every distinct decision the range allows. Without both classes the
    neutral 0.5 is returned. ``weight`` weighs each row in the F1 (default: every row
    counts once).
    """
    outcome = np.asarray(outcome).astype(bool)
    probability = np.asarray(probability, dtype=np.float64)
    if np.unique(outcome).size < 2:
        return NEUTRAL_THRESHOLD
    weight = row_weights(outcome, weight)
    low, high = bounds
    inside = probability[(probability >= low) & (probability <= high)]
    candidates = np.unique(np.concatenate([inside, [low, high, NEUTRAL_THRESHOLD]]))
    order = np.argsort(probability, kind="stable")
    ranked, ranked_weight, ranked_positive = probability[order], weight[order], (weight * outcome)[order]
    # *_from[i] = weight of the (positive) rows ranked i.. (the rows flagged by ranked[i]).
    positives_from = np.concatenate([np.cumsum(ranked_positive[::-1])[::-1], [0.0]])
    flagged_from = np.concatenate([np.cumsum(ranked_weight[::-1])[::-1], [0.0]])
    first_flagged = np.searchsorted(ranked, candidates, side="left")
    f1 = 2.0 * positives_from[first_flagged] / (flagged_from[first_flagged] + ranked_positive.sum())
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
    weight: np.ndarray | None,
) -> CalibrationReport:
    flagged = probability >= threshold
    return CalibrationReport(
        folds=folds,
        lifecycles=lifecycles,
        rows=int(outcome.size),
        n_iter=n_iter,
        calibration=calibration,
        threshold=threshold,
        oof_f1=float(f1_score(outcome, flagged, sample_weight=weight, zero_division=0)),
        oof_precision=float(precision_score(outcome, flagged, sample_weight=weight, zero_division=0)),
        oof_recall=float(recall_score(outcome, flagged, sample_weight=weight, zero_division=0)),
        oof_roc_auc=float(roc_auc_score(outcome, probability, sample_weight=weight)),
        oof_brier=brier_score(outcome, probability, weight),
        oof_ece=expected_calibration_error(outcome, probability, weight=weight),
    )


def fit_failure_classifier(
    train: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    validation: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    *,
    weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> FittedClassifier:
    """Fit, calibrate and threshold the classifier; each side is (matrix, outcome, split groups).

    ``weights`` (train, validation) weighs every row alike in the early stopping, every fit,
    the Platt calibration, the threshold and the report (default: every row counts once).
    """
    x_train, y_train, groups_train = train
    x_val, y_val, groups_val = validation
    weight_train, weight_val = (None, None) if weights is None else weights
    probe = _fit(
        CLASSIFIER_PARAMS,
        x_train,
        y_train,
        sample_weight=weight_train,
        X_val=x_val,
        y_val=y_val,
        sample_weight_val=weight_val,
    )
    n_iter = int(probe.n_iter_)
    params = {**CLASSIFIER_PARAMS, "early_stopping": False, "max_iter": n_iter}

    matrix = pd.concat([x_train, x_val], ignore_index=True)
    outcome = np.concatenate([y_train, y_val]).astype(np.int8)
    groups = np.concatenate([np.asarray(groups_train), np.asarray(groups_val)])
    weight = None if weights is None else np.concatenate([weight_train, weight_val])
    folds = group_folds(groups)
    oof_scores = out_of_fold_scores(matrix, outcome, folds, params, weight)
    calibration = fit_platt(oof_scores, outcome, weight)
    oof_probability = calibration.probability(oof_scores)
    threshold = select_threshold(outcome, oof_probability, weight=weight)
    empty = empty_columns(matrix)
    if empty:
        logger.info("classifier: %d features have no observed value and are not used: %s", len(empty), empty)
    model = _fit(params, matrix, outcome, sample_weight=weight)
    report = _report(
        outcome,
        oof_probability,
        folds=len(folds),
        lifecycles=int(np.unique(groups).size),
        n_iter=n_iter,
        calibration=calibration,
        threshold=threshold,
        weight=weight,
    )
    return FittedClassifier(model=model, calibration=calibration, threshold=threshold, report=report)
