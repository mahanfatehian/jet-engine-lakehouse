"""Ablation of the spectral features: what the high-frequency path adds to the RUL models.

Every training run refits the regressor and the calibrated classifier without the
spectral features - same train/validation lifecycles, same procedure and seeds - and
scores the result on the identical holdout, next to the production model. The comparison
is evidence only: the production model always uses the full feature set and the
promotion gate never looks at the ablation. It is logged as one comparison line and
recorded as "ablation" in metrics.json, metadata.json and ml_model_runs.params_json,
together with the spectral features' share of the production regressor's gain and how
many rows had spectral features at all (coverage).

It is skipped, and recorded as such, when the development lifecycles hold no spectral
value (HF disabled, or its features not landed yet): both models would then see the same
information. A failure of the ablation is logged and recorded; it never fails training.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from feature_pipeline import SPECTRAL_FEATURE_COLUMNS, spectral_coverage
from rul_metrics import HoldoutMetrics
from rul_modeling import LabelSpec, TrainedModel, evaluate_bundle, feature_importance, train_models

ABLATION_BASELINE = "without_spectral"
ABLATION_SPLIT = "identical train/validation/test lifecycles, procedure and seeds as the production model"
# Holdout metrics compared with and without the spectral features: (key, label, format).
COMPARED_METRICS: tuple[tuple[str, str, str], ...] = (
    ("rmse", "RMSE", ".2f"),
    ("mae", "MAE", ".2f"),
    ("r2", "R2", ".3f"),
    ("late_life_rmse", "late-life RMSE", ".2f"),
    ("clf_f1", "F1", ".3f"),
    ("clf_roc_auc", "ROC-AUC", ".3f"),
)
SPLIT_SIDES = ("train", "validation", "test")

logger = logging.getLogger(__name__)


class AblationStatus(StrEnum):
    """Outcome of the spectral ablation of one training run."""

    COMPLETED = "completed"
    SKIPPED = "skipped"  # nothing to compare: no spectral value in the development lifecycles
    FAILED = "failed"  # logged and recorded; the production model is unaffected


def _json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class AblationResult:
    """The production model's holdout metrics next to those of the same models without spectral features."""

    status: AblationStatus
    seconds: float
    dropped_columns: tuple[str, ...]
    coverage: dict[str, float]  # share of rows with every spectral feature present, per split side
    spectral_importance_share: float  # spectral share of the production regressor's total gain
    with_spectral: HoldoutMetrics
    without_spectral: HoldoutMetrics | None = None
    detail: str = ""  # why it was skipped or failed
    baseline_best_iteration: int | None = None
    baseline_threshold: float | None = None

    def differences(self) -> dict[str, float | None] | None:
        """``with - without`` of every compared metric (None when undefined or not run)."""
        if self.without_spectral is None:
            return None
        return {
            key: _json_float(getattr(self.with_spectral, key) - getattr(self.without_spectral, key))
            for key, _, _ in COMPARED_METRICS
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe document for metrics.json, metadata.json and ml_model_runs.params_json."""
        return {
            "status": self.status.value,
            "detail": self.detail or None,
            "baseline": ABLATION_BASELINE,
            "dropped_columns": list(self.dropped_columns),
            "split": ABLATION_SPLIT,
            "seconds": round(self.seconds, 3),
            "spectral_coverage": {side: _json_float(share) for side, share in self.coverage.items()},
            "spectral_importance_share": _json_float(self.spectral_importance_share),
            "with_spectral": self.with_spectral.as_dict(),
            "without_spectral": None if self.without_spectral is None else self.without_spectral.as_dict(),
            "with_minus_without": self.differences(),
            "baseline_best_iteration": self.baseline_best_iteration,
            "baseline_threshold": self.baseline_threshold,
        }

    def summary(self) -> dict[str, Any]:
        """The headline numbers for the one-line JSON training summary."""
        return {
            "status": self.status.value,
            "seconds": round(self.seconds, 3),
            "spectral_importance_share": _json_float(self.spectral_importance_share),
            "with_minus_without": self.differences(),
        }

    def describe(self) -> str:
        """The comparison log line."""
        coverage = " / ".join(f"{side} {share:.0%}" for side, share in self.coverage.items())
        if self.status is not AblationStatus.COMPLETED or self.without_spectral is None:
            return (
                f"spectral ablation {self.status.value}: {self.detail} | spectral coverage {coverage} "
                f"| +{self.seconds:.1f} s"
            )
        comparisons = " | ".join(
            f"{label} {getattr(self.with_spectral, key):{fmt}} vs {getattr(self.without_spectral, key):{fmt}}"
            for key, label, fmt in COMPARED_METRICS
        )
        return (
            f"spectral ablation on the same split (evidence only) | with vs without spectral: {comparisons} "
            f"| spectral share of regressor gain {self.spectral_importance_share:.1%} "
            f"| spectral coverage {coverage} | +{self.seconds:.1f} s"
        )


# How a training run obtains its ablation: run_ablation in production (tests may stub it).
AblationRunner = Callable[
    [Mapping[str, pd.DataFrame], LabelSpec, TrainedModel, HoldoutMetrics], AblationResult
]


def spectral_importance_share(importance: pd.DataFrame) -> float:
    """Share of the regressor's total gain that falls on spectral features (0 without any gain)."""
    total = float(importance["gain"].sum())
    if total <= 0:
        return 0.0
    spectral = importance["feature"].isin(SPECTRAL_FEATURE_COLUMNS)
    return float(importance.loc[spectral, "gain"].sum()) / total


def _has_spectral_values(frame: pd.DataFrame) -> bool:
    values = frame.loc[:, list(SPECTRAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan)
    return bool(np.isfinite(values).any())


def run_ablation(
    frames: Mapping[str, pd.DataFrame],
    labels: LabelSpec,
    production: TrainedModel,
    holdout: HoldoutMetrics,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> AblationResult:
    """Refit both models without the spectral features on ``frames`` and score them on the holdout.

    ``frames`` holds the train / validation / test rows the production model was fitted
    and scored on; ``holdout`` is its holdout score.
    """
    started = clock()
    bundle = production.bundle
    dropped = tuple(column for column in bundle.feature_columns if column in SPECTRAL_FEATURE_COLUMNS)
    common = {
        "dropped_columns": dropped,
        "coverage": {side: spectral_coverage(frames[side]) for side in SPLIT_SIDES},
        "spectral_importance_share": spectral_importance_share(feature_importance(bundle)),
        "with_spectral": holdout,
    }
    skip_reason = ""
    if not dropped:
        skip_reason = "the production model uses no spectral features"
    elif not any(_has_spectral_values(frames[side]) for side in ("train", "validation")):
        skip_reason = "the development lifecycles hold no spectral feature value (HF disabled or not landed)"
    if skip_reason:
        return AblationResult(AblationStatus.SKIPPED, clock() - started, detail=skip_reason, **common)
    try:
        baseline = train_models(
            frames["train"],
            frames["validation"],
            labels,
            model_version=f"{bundle.model_version}-{ABLATION_BASELINE}",
            trained_at=bundle.trained_at,
            hours_per_cycle=bundle.hours_per_cycle,
            feature_columns=[column for column in bundle.feature_columns if column not in dropped],
        )
        without = evaluate_bundle(baseline.bundle, frames["test"], labels)
    except Exception as exc:
        logger.error("spectral ablation failed; the production model is unaffected: %s", exc, exc_info=exc)
        return AblationResult(
            AblationStatus.FAILED, clock() - started, detail=f"{type(exc).__name__}: {exc}", **common
        )
    return AblationResult(
        AblationStatus.COMPLETED,
        clock() - started,
        without_spectral=without,
        baseline_best_iteration=baseline.best_iteration,
        baseline_threshold=baseline.bundle.decision_threshold,
        **common,
    )
