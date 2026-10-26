"""What a finished training run reports: log lines, metadata.json, metrics.json and its
ml_model_runs row. Kept apart from the training flow so the flow reads top to bottom."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from feature_pipeline import TrainingFrame
from model_registry import ModelRunRecord, PromotionDecision
from rul_metrics import HoldoutMetrics
from rul_modeling import (
    CLASSIFIER_PARAMS,
    F1_TOLERANCE,
    REGRESSOR_PARAMS,
    SPLIT_SEED,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    GroupSplit,
    TrainedModel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChampionScore:
    """The production model re-scored on the challenger's holdout lifecycles."""

    model_version: str
    holdout: HoldoutMetrics


@dataclass(frozen=True)
class TrainingReport:
    """Facts of one training run, rendered into every reporting format."""

    trained_at: datetime
    dataset: TrainingFrame
    split: GroupSplit
    rows: dict[str, int]
    trained: TrainedModel
    holdout: HoldoutMetrics
    champion: ChampionScore | None
    decision: PromotionDecision
    policy: str

    @property
    def model_version(self) -> str:
        """Version of the challenger model."""
        return self.trained.bundle.model_version

    def log(self) -> None:
        """Human-readable metric lines plus one machine-readable JSON summary line."""
        holdout = self.holdout
        logger.info(
            "holdout regressor  | RMSE %.2f | MAE %.2f | R2 %.3f | "
            "late-life RMSE %.2f | NASA %.3f/row (sum %.1f)",
            holdout.rmse,
            holdout.mae,
            holdout.r2,
            holdout.late_life_rmse,
            holdout.nasa_score,
            holdout.nasa_score_sum,
        )
        logger.info(
            "holdout classifier | threshold %.3f | F1 %.3f | precision %.3f | recall %.3f | ROC-AUC %.3f",
            holdout.clf_threshold,
            holdout.clf_f1,
            holdout.clf_precision,
            holdout.clf_recall,
            holdout.clf_roc_auc,
        )
        if self.champion is not None:
            logger.info(
                "champion %s on the same holdout | RMSE %.2f | R2 %.3f | F1 %.3f",
                self.champion.model_version,
                self.champion.holdout.rmse,
                self.champion.holdout.r2,
                self.champion.holdout.clf_f1,
            )
        verdict = "promote" if self.decision.promote else "keep champion"
        logger.info("promotion: %s - %s", verdict, self.decision.reason)
        summary = {
            "model_version": self.model_version,
            "data_source": self.dataset.data_source,
            "source_uri": self.dataset.source_uri,
            "best_iteration": self.trained.best_iteration,
            "holdout": holdout.as_dict(),
            "validation": self.trained.validation.as_dict(),
            "champion": self._champion_document(),
            "promoted": self.decision.promote,
            "promotion_reason": self.decision.reason,
        }
        logger.info("training summary %s", json.dumps(summary, sort_keys=True))

    def _champion_document(self) -> dict[str, Any] | None:
        if self.champion is None:
            return None
        return {"model_version": self.champion.model_version, "holdout": self.champion.holdout.as_dict()}

    def metadata(self) -> dict[str, Any]:
        """Content of metadata.json: data, split, feature contract, parameters, promotion."""
        bundle = self.trained.bundle
        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at.isoformat(),
            "data_source": self.dataset.data_source,
            "source_uri": self.dataset.source_uri,
            "lifecycles": {
                "train": list(self.split.train),
                "validation": list(self.split.validation),
                "test": list(self.split.test),
            },
            "rows": dict(self.rows),
            "feature_columns": list(bundle.feature_columns),
            "categories": bundle.categories,
            "matrix_columns": bundle.matrix_columns,
            "labels": {
                "rul_cap_cycles": bundle.rul_cap_cycles,
                "failure_horizon_cycles": bundle.failure_horizon_cycles,
            },
            "hours_per_cycle": bundle.hours_per_cycle,
            "decision_threshold": bundle.decision_threshold,
            "decision_threshold_rule": f"lowest threshold with validation F1 >= best F1 - {F1_TOLERANCE}",
            "regressor": {"params": REGRESSOR_PARAMS, "best_iteration": self.trained.best_iteration},
            "classifier": {"params": CLASSIFIER_PARAMS, "n_iter": int(bundle.classifier.n_iter_)},
            "split": {
                "method": "sha256(seed|split_group) thresholds",
                "seed": SPLIT_SEED,
                "test_fraction": TEST_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
            },
            "library_versions": bundle.library_versions,
            "promotion": {
                "policy": self.policy,
                "promoted": self.decision.promote,
                "reason": self.decision.reason,
                "champion_version": None if self.champion is None else self.champion.model_version,
            },
        }

    def metrics(self) -> dict[str, Any]:
        """Content of metrics.json: holdout and validation metrics plus the champion comparison."""
        return {
            "model_version": self.model_version,
            "holdout": self.holdout.as_dict(),
            "validation": self.trained.validation.as_dict(),
            "champion": self._champion_document(),
            "promotion": {"promoted": self.decision.promote, "reason": self.decision.reason},
        }

    def run_record(self, artifact_uri: str) -> ModelRunRecord:
        """The ml_model_runs row; train_* counts include the validation lifecycles."""
        metadata = self.metadata()
        reproducibility_keys = (
            "regressor",
            "classifier",
            "decision_threshold",
            "decision_threshold_rule",
            "labels",
            "hours_per_cycle",
            "split",
        )
        params = {key: metadata[key] for key in reproducibility_keys}
        params.update(
            source_uri=self.dataset.source_uri,
            validation_lifecycles=len(self.split.validation),
            late_life_rmse=self.holdout.as_dict()["late_life_rmse"],
            nasa_score_sum=self.holdout.as_dict()["nasa_score_sum"],
            promotion=metadata["promotion"],
            champion_rmse=None if self.champion is None else self.champion.holdout.as_dict()["rmse"],
        )
        holdout = self.holdout
        return ModelRunRecord(
            model_version=self.model_version,
            trained_at=self.trained_at,
            data_source=self.dataset.data_source,
            train_rows=self.rows["train"] + self.rows["validation"],
            test_rows=self.rows["test"],
            train_lifecycles=len(self.split.train) + len(self.split.validation),
            test_lifecycles=len(self.split.test),
            rmse=holdout.rmse,
            mae=holdout.mae,
            r2=holdout.r2,
            nasa_score=holdout.nasa_score,
            clf_f1=holdout.clf_f1,
            clf_precision=holdout.clf_precision,
            clf_recall=holdout.clf_recall,
            clf_roc_auc=holdout.clf_roc_auc,
            promoted=self.decision.promote,
            artifact_uri=artifact_uri,
            params_json=json.dumps(params, sort_keys=True),
        )
