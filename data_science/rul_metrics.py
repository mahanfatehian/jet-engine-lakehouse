"""Holdout metrics of the RUL regressor and the imminent-failure classifier."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# Asymmetric scoring of the NASA C-MAPSS challenge: predicting failure too late
# (over-estimating RUL) is penalised harder than predicting it too early.
NASA_EARLY_DIVISOR = 13.0
NASA_LATE_DIVISOR = 10.0
# Equal-width probability bins of the expected calibration error.
CALIBRATION_BINS = 10


def nasa_scores(rul_true: np.ndarray, rul_pred: np.ndarray) -> np.ndarray:
    """Per-row C-MAPSS score, ``exp(-d/13) - 1`` if early else ``exp(d/10) - 1`` (d = pred - true)."""
    error = np.asarray(rul_pred, dtype=np.float64) - np.asarray(rul_true, dtype=np.float64)
    return np.where(error < 0, np.expm1(-error / NASA_EARLY_DIVISOR), np.expm1(error / NASA_LATE_DIVISOR))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def brier_score(outcome: np.ndarray, probability: np.ndarray) -> float:
    """Mean squared difference between predicted probabilities and 0/1 outcomes (lower is better)."""
    outcome = np.asarray(outcome, dtype=np.float64)
    if outcome.size == 0:
        return math.nan
    return float(np.mean((np.asarray(probability, dtype=np.float64) - outcome) ** 2))


def expected_calibration_error(
    outcome: np.ndarray, probability: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    """Expected calibration error over ``bins`` equal-width probability bins.

    ``sum_b (n_b / n) * |mean probability in b - observed positive rate in b|``: 0 for a
    perfectly calibrated model, and the average gap between stated and observed risk otherwise.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if outcome.size == 0:
        return math.nan
    # Bin b holds [b/bins, (b+1)/bins); probability 1.0 joins the last bin.
    index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    stated = np.bincount(index, weights=probability, minlength=bins)
    observed = np.bincount(index, weights=outcome, minlength=bins)
    return float(np.abs(stated - observed).sum() / outcome.size)


@dataclass(frozen=True)
class HoldoutMetrics:
    """Quality of a model on one set of lifecycles.

    Regression metrics compare against the capped RUL target; ``late_life_rmse``
    only covers rows whose true RUL is within the cap (where the target is exact).
    ``nasa_score`` is the mean per-row C-MAPSS score (comparable across holdout
    sizes), ``nasa_score_sum`` the classic sum. ``clf_brier`` and ``clf_ece`` measure
    how well the published failure probabilities are calibrated.
    """

    rows: int
    lifecycles: int
    rmse: float
    mae: float
    r2: float
    late_life_rmse: float
    nasa_score: float
    nasa_score_sum: float
    clf_threshold: float
    clf_f1: float
    clf_precision: float
    clf_recall: float
    clf_roc_auc: float
    positive_rate: float
    clf_brier: float = math.nan
    clf_ece: float = math.nan

    def as_dict(self) -> dict[str, float | int | None]:
        """JSON-safe mapping (NaN becomes None)."""
        return {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in asdict(self).items()
        }


def evaluate_predictions(
    *,
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    failure_probability: np.ndarray,
    threshold: float,
    rul_cap_cycles: int,
    failure_horizon_cycles: int,
    lifecycles: int,
) -> HoldoutMetrics:
    """Score RUL predictions and failure probabilities against the true (uncapped) RUL."""
    rul_true = np.asarray(rul_true, dtype=np.float64)
    rul_pred = np.asarray(rul_pred, dtype=np.float64)
    probability = np.asarray(failure_probability, dtype=np.float64)
    target = np.minimum(rul_true, float(rul_cap_cycles))
    late_life = rul_true <= rul_cap_cycles
    failure_true = (rul_true <= failure_horizon_cycles).astype(np.int8)
    failure_pred = (probability >= threshold).astype(np.int8)
    scores = nasa_scores(target, rul_pred)
    both_classes = np.unique(failure_true).size == 2
    return HoldoutMetrics(
        rows=int(rul_true.size),
        lifecycles=int(lifecycles),
        rmse=_rmse(target, rul_pred),
        mae=float(mean_absolute_error(target, rul_pred)),
        r2=float(r2_score(target, rul_pred)),
        late_life_rmse=_rmse(target[late_life], rul_pred[late_life]),
        nasa_score=float(scores.mean()),
        nasa_score_sum=float(scores.sum()),
        clf_threshold=float(threshold),
        clf_f1=float(f1_score(failure_true, failure_pred, zero_division=0)),
        clf_precision=float(precision_score(failure_true, failure_pred, zero_division=0)),
        clf_recall=float(recall_score(failure_true, failure_pred, zero_division=0)),
        clf_roc_auc=float(roc_auc_score(failure_true, probability)) if both_classes else math.nan,
        positive_rate=float(failure_true.mean()),
        clf_brier=brier_score(failure_true, probability),
        clf_ece=expected_calibration_error(failure_true, probability),
    )
