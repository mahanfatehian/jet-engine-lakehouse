"""What a finished training run reports: log lines, metadata.json, metrics.json and its
ml_model_runs row. Kept apart from the training flow so the flow reads top to bottom."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ds_settings import PromotionSettings
from failure_classifier import CLASSIFIER_PARAMS, THRESHOLD_RULE
from feature_ablation import AblationResult
from feature_pipeline import NULLABLE_FEATURE_COLUMNS
from model_promotion import ChampionOutcome, PromotionDecision
from model_registry import ModelRunRecord
from rul_metrics import HoldoutMetrics
from rul_modeling import (
    EARLY_STOPPING_ROUNDS,
    REGRESSOR_PARAMS,
    SPLIT_SEED,
    TEST_FRACTION,
    VALIDATION_FRACTION,
    GroupSplit,
    TrainedModel,
)
from training_window import TrainingWindow, format_micros

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetInfo:
    """Origin of the training rows: source, the window loaded from it and its data watermark."""

    data_source: str
    source_uri: str
    window: TrainingWindow
    data_watermark_us: int | None


@dataclass(frozen=True)
class TrainingReport:
    """Facts of one training run, rendered into every reporting format."""

    trained_at: datetime
    dataset: DatasetInfo
    split: GroupSplit
    rows: dict[str, int]
    trained: TrainedModel
    holdout: HoldoutMetrics
    ablation: AblationResult
    champion: ChampionOutcome
    decision: PromotionDecision
    policy: str
    gate: PromotionSettings

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
            "holdout classifier | threshold %.3f | F1 %.3f | precision %.3f | recall %.3f | ROC-AUC %.3f "
            "| Brier %.4f | ECE %.4f",
            holdout.clf_threshold,
            holdout.clf_f1,
            holdout.clf_precision,
            holdout.clf_recall,
            holdout.clf_roc_auc,
            holdout.clf_brier,
            holdout.clf_ece,
        )
        calibration = self.trained.calibration
        logger.info(
            "classifier calibration | Platt slope %.3f intercept %.3f fitted on %d out-of-fold rows "
            "(%d lifecycles, %d folds) | OOF F1 %.3f Brier %.4f ECE %.4f | threshold %.3f (%s)",
            calibration.calibration.slope,
            calibration.calibration.intercept,
            calibration.rows,
            calibration.lifecycles,
            calibration.folds,
            calibration.oof_f1,
            calibration.oof_brier,
            calibration.oof_ece,
            calibration.threshold,
            THRESHOLD_RULE,
        )
        logger.info("%s", self.ablation.describe())
        champion = self.champion.holdout
        if champion is not None:
            logger.info(
                "champion %s (%s features, challenger %d) on %s | RMSE %.2f | R2 %.3f | F1 %.3f",
                self.champion.model_version,
                self.champion.feature_count,
                len(self.trained.bundle.feature_columns),
                self.champion.detail or "the same holdout",
                champion.rmse,
                champion.r2,
                champion.clf_f1,
            )
        challenger = self.champion.challenger_holdout
        if challenger is not None:
            logger.info(
                "challenger on the same lifecycles | RMSE %.2f | R2 %.3f | F1 %.3f",
                challenger.rmse,
                challenger.r2,
                challenger.clf_f1,
            )
        verdict = "promote" if self.decision.promote else "keep champion"
        log = logger.info if self.decision.promote else logger.warning
        log("promotion: %s - %s", verdict, self.decision.reason)
        summary = {
            "model_version": self.model_version,
            "data_source": self.dataset.data_source,
            "source_uri": self.dataset.source_uri,
            "training_window": self.dataset.window.as_dict(),
            "best_iteration": self.trained.best_iteration,
            "holdout": holdout.as_dict(),
            "validation": self.trained.validation.as_dict(),
            "calibration": calibration.as_dict(),
            "ablation": self.ablation.summary(),
            "champion": self.champion.as_dict(),
            "promoted": self.decision.promote,
            "promotion_reason": self.decision.reason,
        }
        logger.info("training summary %s", json.dumps(summary, sort_keys=True))

    def _promotion_document(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "promoted": self.decision.promote,
            "reason": self.decision.reason,
            "champion_version": self.champion.model_version,
            "champion_status": self.champion.status.value,
            "champion_detail": self.champion.detail or None,
            "floors": asdict(self.gate),
            "floor_failures": list(self.decision.floor_failures),
        }

    def metadata(self) -> dict[str, Any]:
        """Content of metadata.json: data, split, feature contract, parameters, promotion."""
        bundle = self.trained.bundle
        watermark = self.dataset.data_watermark_us
        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at.isoformat(),
            "data_source": self.dataset.data_source,
            "source_uri": self.dataset.source_uri,
            "data_watermark_us": watermark,
            "data_watermark": format_micros(watermark),
            "training_window": self.dataset.window.as_dict(),
            "lifecycles": {
                "train": list(self.split.train),
                "validation": list(self.split.validation),
                "test": list(self.split.test),
            },
            "rows": dict(self.rows),
            "feature_columns": list(bundle.feature_columns),
            "nullable_feature_columns": [
                column for column in bundle.feature_columns if column in NULLABLE_FEATURE_COLUMNS
            ],
            "categories": bundle.categories,
            "matrix_columns": bundle.matrix_columns,
            "labels": {
                "rul_cap_cycles": bundle.rul_cap_cycles,
                "failure_horizon_cycles": bundle.failure_horizon_cycles,
            },
            "hours_per_cycle": bundle.hours_per_cycle,
            "decision_threshold": bundle.decision_threshold,
            "decision_threshold_rule": THRESHOLD_RULE,
            "regressor": {
                "params": REGRESSOR_PARAMS,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "best_iteration": self.trained.best_iteration,
                "trees": int(bundle.regressor.get_booster().num_boosted_rounds()),
            },
            "classifier": {"params": CLASSIFIER_PARAMS, "n_iter": int(bundle.classifier.n_iter_)},
            "calibration": self.trained.calibration.as_dict(),
            "split": {
                "method": "sha256(seed|split_group) thresholds",
                "seed": SPLIT_SEED,
                "test_fraction": TEST_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
            },
            "library_versions": bundle.library_versions,
            "ablation": self.ablation.as_dict(),
            "promotion": self._promotion_document(),
        }

    def metrics(self) -> dict[str, Any]:
        """Content of metrics.json: holdout, validation and out-of-fold metrics plus the champion."""
        return {
            "model_version": self.model_version,
            "holdout": self.holdout.as_dict(),
            # The validation lifecycles early-stopped both models and the classifier is refit
            # on them, so these are optimistic; holdout and out_of_fold are the honest numbers.
            "validation": self.trained.validation.as_dict(),
            "out_of_fold": self.trained.calibration.as_dict(),
            # Evidence only: the same models without the spectral features, on the same split.
            "ablation": self.ablation.as_dict(),
            "champion": self.champion.as_dict(),
            "promotion": {"promoted": self.decision.promote, "reason": self.decision.reason},
        }

    def run_record(self, artifact_uri: str, bundle_sha256: str) -> ModelRunRecord:
        """The ml_model_runs row; train_* counts include the validation lifecycles."""
        metadata = self.metadata()
        reproducibility_keys = (
            "regressor",
            "classifier",
            "calibration",
            "decision_threshold",
            "decision_threshold_rule",
            "labels",
            "hours_per_cycle",
            "split",
            "training_window",
            "data_watermark_us",
        )
        params = {key: metadata[key] for key in reproducibility_keys}
        holdout = self.holdout.as_dict()
        champion = self.champion.holdout
        params.update(
            source_uri=self.dataset.source_uri,
            validation_lifecycles=len(self.split.validation),
            late_life_rmse=holdout["late_life_rmse"],
            nasa_score_sum=holdout["nasa_score_sum"],
            clf_brier=holdout["clf_brier"],
            clf_ece=holdout["clf_ece"],
            bundle_sha256=bundle_sha256,
            promotion=metadata["promotion"],
            ablation=metadata["ablation"],
            champion_rmse=None if champion is None else champion.as_dict()["rmse"],
            champion_clf_f1=None if champion is None else champion.as_dict()["clf_f1"],
        )
        return ModelRunRecord(
            model_version=self.model_version,
            trained_at=self.trained_at,
            data_source=self.dataset.data_source,
            train_rows=self.rows["train"] + self.rows["validation"],
            test_rows=self.rows["test"],
            train_lifecycles=len(self.split.train) + len(self.split.validation),
            test_lifecycles=len(self.split.test),
            rmse=self.holdout.rmse,
            mae=self.holdout.mae,
            r2=self.holdout.r2,
            nasa_score=self.holdout.nasa_score,
            clf_f1=self.holdout.clf_f1,
            clf_precision=self.holdout.clf_precision,
            clf_recall=self.holdout.clf_recall,
            clf_roc_auc=self.holdout.clf_roc_auc,
            promoted=self.decision.promote,
            artifact_uri=artifact_uri,
            params_json=json.dumps(params, sort_keys=True),
        )
