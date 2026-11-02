"""RUL model: lifecycle-aware split, training and the model bundle.

Two models share one feature matrix:
  * an XGBoost regressor of the capped remaining useful life (piecewise-linear target),
    early-stopped on the validation lifecycles and trimmed to its best iteration, so the
    served model, regressor.json and the feature importances describe the same trees; and
  * a scikit-learn classifier of "fails within the horizon" whose probabilities are
    calibrated out-of-fold over lifecycles and whose decision threshold maximises F1 on
    those calibrated probabilities (see failure_classifier.py).

The classifier is deliberately not class-weighted: class re-weighting would inflate the
minority class probability, and ``failure_probability`` is published as a calibrated
probability.

Spectral dropout. A pulse is scored without spectral features whenever its HF frame is
missing (HF disabled or down, the stack just started, or the frame lagging more than
SPECTRAL_MAX_LAG_SECONDS). Both models send a missing value down a default branch, but a
branch is only fitted where training rows were missing: trained on rows that all carry
spectral values, every such branch is an untrained default and the estimate of a pulse
without them is arbitrary. So whenever the development rows hold spectral values, both
models are fitted on those rows plus a copy of every row with all spectral features
missing, which fits the missing-value branches to what the pulse features alone say. Every
row is copied, also rows already without spectral values, so the lifecycles keep their
relative weights. A copy keeps the lifecycle (``split_group``) of its row, so the
classifier's lifecycle-grouped folds never score a row with a model that saw its copy. The
regressor early-stops on the validation rows plus their copy, and the classifier's iteration
count, calibration and threshold come from both, so the published probability holds for
pulses with and without an HF frame. A model without spectral inputs (the spectral
ablation) gets no copy.

A copy weighs SPECTRAL_DROPOUT_WEIGHT of a row, in every fit, early stopping, calibration
and threshold. At full weight half of every node's rows have no value to split a spectral
feature on, which halves the gain of every spectral split: the trees then all but ignore
the spectral features, and on the live mart the regressor lost all it gained from them
(holdout RMSE as high as the pulse-only model's). A tenth of a row still fits the
missing-value branches - with every spectral value masked the models score about as well
as models of the pulse features alone - while pulses with an HF frame keep much of the
spectral gain. A larger weight buys a little accuracy without HF frames for a loss with them.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from ds_settings import MIN_SPLIT_LIFECYCLES
from failure_classifier import CalibrationReport, PlattCalibration, fit_failure_classifier
from feature_pipeline import (
    FEATURE_COLUMNS,
    SPECTRAL_FEATURE_COLUMNS,
    build_feature_matrix,
    fit_categories,
    matrix_columns,
)
from rul_metrics import HoldoutMetrics, evaluate_predictions
from sklearn.ensemble import HistGradientBoostingClassifier

# 2: calibrated classifier (raw model + Platt parameters) and a regressor trimmed to its best iteration.
BUNDLE_FORMAT_VERSION = 2
SPLIT_SEED = 42
TEST_FRACTION = 0.25
VALIDATION_FRACTION = 0.2
MIN_SPLIT_GROUPS = MIN_SPLIT_LIFECYCLES
EARLY_STOPPING_ROUNDS = 75
# Weight of a spectral-dropout copy relative to its row (see the module docstring).
SPECTRAL_DROPOUT_WEIGHT = 0.1

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
    "random_state": SPLIT_SEED,
}


class IncompatibleModelError(ValueError):
    """A stored model was built for another bundle format or feature schema (a planned change)."""


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
    """Both fitted models plus the feature, label and calibration contract they were trained with."""

    model_version: str
    trained_at: str
    regressor: xgb.XGBRegressor
    classifier: HistGradientBoostingClassifier
    calibration: PlattCalibration
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
        """Calibrated probability of failing within ``failure_horizon_cycles``."""
        return self.calibration.probability(self.classifier.decision_function(matrix))

    def to_payload(self) -> dict[str, Any]:
        """Plain mapping persisted with joblib (no dependency on this class at load time)."""
        return {
            "bundle_format": BUNDLE_FORMAT_VERSION,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "regressor": self.regressor,
            "classifier": self.classifier,
            "calibration": self.calibration.to_payload(),
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
            raise IncompatibleModelError(
                f"model bundle format {payload.get('bundle_format')!r} is not the supported "
                f"format {BUNDLE_FORMAT_VERSION}"
            )
        return cls(
            model_version=str(payload["model_version"]),
            trained_at=str(payload["trained_at"]),
            regressor=payload["regressor"],
            classifier=payload["classifier"],
            calibration=PlattCalibration.from_payload(payload["calibration"]),
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
    """Result of fitting both models on the development (train + validation) lifecycles."""

    bundle: RulModelBundle
    best_iteration: int
    validation: HoldoutMetrics
    calibration: CalibrationReport
    spectral_dropout: bool = False  # fitted on the rows plus a copy without spectral values


def library_versions() -> dict[str, str]:
    """Versions of the libraries a bundle depends on (recorded for reproducibility)."""
    return {
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


@dataclass(frozen=True)
class FitRows:
    """Model input, both targets and the lifecycles of one side (train or validation) of a fit."""

    matrix: pd.DataFrame
    rul: np.ndarray
    failure: np.ndarray
    groups: np.ndarray
    weight: np.ndarray | None = None  # per row; None: every row counts once

    @classmethod
    def of(
        cls,
        frame: pd.DataFrame,
        categories: Mapping[str, Sequence[str]],
        columns: Sequence[str],
        labels: LabelSpec,
    ) -> FitRows:
        """Encode ``frame`` and derive both targets with ``labels``."""
        return cls(
            matrix=build_feature_matrix(frame, categories, columns),
            rul=labels.regression_target(frame),
            failure=labels.failure_target(frame),
            groups=frame["split_group"].to_numpy(),
        )

    def holds_values(self, columns: Sequence[str]) -> bool:
        """True when any row has a value in any of ``columns``."""
        return bool(self.matrix.loc[:, list(columns)].notna().to_numpy().any())

    def with_spectral_dropout(
        self, columns: Sequence[str], weight: float = SPECTRAL_DROPOUT_WEIGHT
    ) -> FitRows:
        """These rows followed by a copy of every row with ``columns`` missing (see above).

        A copy keeps the targets and the lifecycle of its row and weighs ``weight`` of it.
        """
        values = self.matrix.to_numpy()
        doubled = np.concatenate([values, values])
        doubled[len(values) :, [self.matrix.columns.get_loc(column) for column in columns]] = np.nan
        row_weight = np.ones(len(values)) if self.weight is None else self.weight
        return FitRows(
            matrix=pd.DataFrame(doubled, columns=self.matrix.columns),
            rul=np.tile(self.rul, 2),
            failure=np.tile(self.failure, 2),
            groups=np.tile(self.groups, 2),
            weight=np.concatenate([row_weight, weight * row_weight]),
        )


def _fit_regressor(train: FitRows, validation: FitRows) -> xgb.XGBRegressor:
    """Early-stop on the validation lifecycles and keep only the trees up to the best iteration."""
    regressor = xgb.XGBRegressor(
        **REGRESSOR_PARAMS,
        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)],
    )
    regressor.fit(
        train.matrix,
        train.rul,
        sample_weight=train.weight,
        eval_set=[(validation.matrix, validation.rul)],
        sample_weight_eval_set=None if validation.weight is None else [validation.weight],
        verbose=False,
    )
    return regressor


def train_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    labels: LabelSpec,
    *,
    model_version: str,
    trained_at: str,
    hours_per_cycle: float,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> TrainedModel:
    """Fit the regressor and the calibrated classifier on the development lifecycles.

    The regressor trains on the train lifecycles and early-stops on the validation ones;
    the classifier's iteration count comes from the same validation lifecycles, its
    calibration and threshold from out-of-fold scores over both (failure_classifier.py).
    When the development rows hold spectral values, both sides are fitted with spectral
    dropout (see above). ``feature_columns`` defaults to the full contract; the spectral
    ablation (feature_ablation.py) passes the same columns without the spectral features.
    """
    columns = tuple(feature_columns)
    categories = fit_categories(train_frame)
    train = FitRows.of(train_frame, categories, columns, labels)
    validation = FitRows.of(validation_frame, categories, columns, labels)
    if np.unique(train.failure).size < 2 or np.unique(validation.failure).size < 2:
        raise ValueError("training and validation lifecycles must contain rows on both sides of the horizon")
    spectral = [column for column in columns if column in SPECTRAL_FEATURE_COLUMNS]
    dropout = bool(spectral) and (train.holds_values(spectral) or validation.holds_values(spectral))
    if dropout:
        train, validation = train.with_spectral_dropout(spectral), validation.with_spectral_dropout(spectral)

    regressor = _fit_regressor(train, validation)
    classifier = fit_failure_classifier(
        (train.matrix, train.failure, train.groups),
        (validation.matrix, validation.failure, validation.groups),
        weights=None if train.weight is None else (train.weight, validation.weight),
    )
    bundle = RulModelBundle(
        model_version=model_version,
        trained_at=trained_at,
        regressor=regressor,
        classifier=classifier.model,
        calibration=classifier.calibration,
        decision_threshold=classifier.threshold,
        feature_columns=columns,
        categories=categories,
        failure_horizon_cycles=labels.failure_horizon_cycles,
        rul_cap_cycles=labels.rul_cap_cycles,
        hours_per_cycle=hours_per_cycle,
        library_versions=library_versions(),
    )
    return TrainedModel(
        bundle=bundle,
        best_iteration=int(regressor.best_iteration),
        validation=evaluate_bundle(bundle, validation_frame, labels),
        calibration=classifier.report,
        spectral_dropout=dropout,
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


def evaluate_without_spectral(
    bundle: RulModelBundle, frame: pd.DataFrame, labels: LabelSpec
) -> HoldoutMetrics:
    """Score ``bundle`` on ``frame`` with every spectral feature missing, as if no HF frame had landed."""
    return evaluate_bundle(bundle, frame.assign(**dict.fromkeys(SPECTRAL_FEATURE_COLUMNS, np.nan)), labels)


def feature_importance(bundle: RulModelBundle) -> pd.DataFrame:
    """XGBoost importances of every model input, by total gain descending.

    ``total_gain`` is the loss reduction summed over every split on the feature, ``gain``
    XGBoost's average per split (``total_gain / weight``), ``weight`` the number of splits
    and ``cover`` their average coverage (rows, for squared error). Only the total adds up
    across features, so ``gain_share`` is each feature's share of the total gain. The
    regressor holds only the trees up to its best iteration, so these are the importances
    of exactly the trees that serve predictions.
    """
    booster = bundle.regressor.get_booster()
    importance = pd.DataFrame({"feature": bundle.matrix_columns})
    for kind in ("gain", "total_gain", "weight", "cover"):
        scores = booster.get_score(importance_type=kind)
        importance[kind] = [float(scores.get(name, 0.0)) for name in importance["feature"]]
    total = float(importance["total_gain"].sum())
    importance["gain_share"] = importance["total_gain"] / total if total > 0 else 0.0
    return importance.sort_values("total_gain", ascending=False, kind="mergesort").reset_index(drop=True)
