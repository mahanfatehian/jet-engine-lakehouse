"""Does the high-frequency path improve the RUL models? The spectral comparison over many splits.

Every training run compares its models with and without the 10 spectral features on its own
holdout (feature_ablation), but one holdout holds a handful of lifecycles, so a single run is
noisy evidence. This job repeats the comparison over several lifecycle splits of the same
training data (``split_groups`` with seeds ``--first-seed`` .. ``+ --splits - 1``) and scores,
on each split's holdout:

    with      the production procedure: all 56 features, spectral dropout (rul_modeling)
    masked    the same models with every spectral value missing: an engine whose frames never land
    without   models fitted on the 46 pulse features only

It reports per split and on average the holdout RMSE, late-life RMSE, R2, classifier F1 and
Brier score, the differences with - without (negative RMSE and Brier, positive R2 and F1 favour
the spectral features) and in how many splits the spectral features won. ``--stride`` keeps
every n-th pulse of a lifecycle to bound the run time (each split fits four models); with the
default 1 the rows are those of a training run (``TRAINING_MAX_ROWS``).

Evidence only: nothing is published or promoted. The table goes to the log and one JSON line
(``{"spectral_evaluation": ...}``) to stdout. Exit codes: 0 evaluated, 3 preconditions not met
(fewer than ``--min-lifecycles`` completed lifecycles, no spectral value at all), 1 failure.

    python data_science/evaluate_spectral.py --splits 5 --stride 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from ds_runtime import (
    EXIT_OK,
    PreconditionsNotMetError,
    StopToken,
    configure_logging,
    positive_int,
    run_cli,
)
from ds_settings import MIN_SPLIT_LIFECYCLES, load_settings
from feature_pipeline import FEATURE_COLUMNS, SPECTRAL_FEATURE_COLUMNS
from lakehouse_io import ClickHouseGateway
from rul_metrics import HoldoutMetrics
from rul_modeling import (
    SPLIT_SEED,
    LabelSpec,
    evaluate_bundle,
    evaluate_without_spectral,
    split_groups,
    train_models,
)
from training_sources import ClickHouseTrainingSource, TrainingDataSource
from training_window import plan_training_window

logger = logging.getLogger("evaluate_spectral")

PULSE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column not in SPECTRAL_FEATURE_COLUMNS
)
SCORINGS = ("with", "masked", "without")
# Holdout metric -> whether a higher value is better.
METRICS: dict[str, bool] = {
    "rmse": False,
    "late_life_rmse": False,
    "r2": True,
    "clf_f1": True,
    "clf_brier": False,
}


@dataclass(frozen=True)
class SplitResult:
    """The three scorings of one lifecycle split's holdout."""

    seed: int
    test_lifecycles: int
    test_rows: int
    scores: dict[str, HoldoutMetrics]

    def metric(self, scoring: str, name: str) -> float:
        return float(getattr(self.scores[scoring], name))


@dataclass(frozen=True)
class SpectralEvaluation:
    """Per-split results plus the window of training data they come from."""

    splits: tuple[SplitResult, ...]
    lifecycles: int
    rows: int
    stride: int
    spectral_coverage: float

    def mean(self, scoring: str, name: str) -> float:
        return statistics.fmean(split.metric(scoring, name) for split in self.splits)

    def differences(self, name: str, *, against: str = "without") -> list[float]:
        """Per split: with - ``against`` for metric ``name``."""
        return [split.metric("with", name) - split.metric(against, name) for split in self.splits]

    def spectral_wins(self, name: str) -> int:
        """Splits in which the spectral features (with) beat the pulse-only models (without)."""
        higher_is_better = METRICS[name]
        return sum(1 for diff in self.differences(name) if (diff > 0) == higher_is_better and diff != 0)

    def as_dict(self) -> dict[str, object]:
        summary = {}
        for name in METRICS:
            diffs = self.differences(name)
            summary[name] = {
                **{scoring: round(self.mean(scoring, name), 4) for scoring in SCORINGS},
                "with_minus_without": round(statistics.fmean(diffs), 4),
                "with_minus_without_sd": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None,
                "spectral_wins": self.spectral_wins(name),
            }
        return {
            "splits": len(self.splits),
            "seeds": [split.seed for split in self.splits],
            "lifecycles": self.lifecycles,
            "rows": self.rows,
            "stride": self.stride,
            "spectral_coverage": round(self.spectral_coverage, 4),
            "summary": summary,
            "per_split": [
                {
                    "seed": split.seed,
                    "test_lifecycles": split.test_lifecycles,
                    "test_rows": split.test_rows,
                    **{
                        scoring: {name: round(split.metric(scoring, name), 4) for name in METRICS}
                        for scoring in SCORINGS
                    },
                }
                for split in self.splits
            ],
        }


def split_frames(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Train / validation / test rows of the lifecycle split ``seed`` (whole lifecycles per side)."""
    split = split_groups(frame["split_group"].unique(), seed=seed)
    sides = {"train": split.train, "validation": split.validation, "test": split.test}
    return {
        side: frame[frame["split_group"].isin(groups)].reset_index(drop=True)
        for side, groups in sides.items()
    }


def evaluate_split(
    frame: pd.DataFrame, labels: LabelSpec, seed: int, *, hours_per_cycle: float
) -> SplitResult:
    """Fit the 56- and the 46-feature models of one split and score its holdout three ways."""
    frames = split_frames(frame, seed)
    common = {"trained_at": datetime.now(UTC).isoformat(), "hours_per_cycle": hours_per_cycle}
    full = train_models(frames["train"], frames["validation"], labels, model_version=f"eval-{seed}", **common)
    pulse_only = train_models(
        frames["train"],
        frames["validation"],
        labels,
        model_version=f"eval-{seed}-pulse",
        feature_columns=PULSE_FEATURE_COLUMNS,
        **common,
    )
    test = frames["test"]
    return SplitResult(
        seed=seed,
        test_lifecycles=int(test["split_group"].nunique()),
        test_rows=len(test),
        scores={
            "with": evaluate_bundle(full.bundle, test, labels),
            "masked": evaluate_without_spectral(full.bundle, test, labels),
            "without": evaluate_bundle(pulse_only.bundle, test, labels),
        },
    )


def evaluate_spectral(
    source: TrainingDataSource,
    labels: LabelSpec,
    *,
    seeds: Sequence[int],
    stride: int,
    max_rows: int,
    min_lifecycles: int,
    hours_per_cycle: float,
    stop: StopToken,
) -> SpectralEvaluation:
    """Load the training window (thinned to ``stride``) and evaluate every split of ``seeds``."""
    inventory = source.inventory()
    required = max(min_lifecycles, MIN_SPLIT_LIFECYCLES)
    if inventory.completed_lifecycles < required:
        raise PreconditionsNotMetError(
            f"{inventory.completed_lifecycles} completed lifecycle(s), at least {required} needed"
        )
    window = plan_training_window(inventory, max_rows=max_rows, min_lifecycles=required)
    if stride > window.stride:
        window = dataclasses.replace(window, stride=stride)
    frame = source.load(window).frame
    spectral = frame[list(SPECTRAL_FEATURE_COLUMNS)]
    coverage = float(spectral.notna().any(axis=1).mean()) if len(frame) else 0.0
    if coverage == 0.0:
        raise PreconditionsNotMetError(
            "the training data holds no spectral value (HF disabled or not landed)"
        )
    lifecycles = int(frame["split_group"].nunique())
    logger.info(
        "evaluating the spectral features on %d split(s): %d lifecycles, %d rows (pulse stride %d), "
        "spectral coverage %.1f%%",
        len(seeds),
        lifecycles,
        len(frame),
        window.stride,
        100 * coverage,
    )
    results = []
    for seed in seeds:
        stop.raise_if_stopped()
        result = evaluate_split(frame, labels, seed, hours_per_cycle=hours_per_cycle)
        logger.info("%s", describe_split(result))
        results.append(result)
    return SpectralEvaluation(tuple(results), lifecycles, len(frame), window.stride, coverage)


def describe_split(result: SplitResult) -> str:
    """One log line: the three scorings of a split."""
    parts = [
        f"{scoring} RMSE {result.metric(scoring, 'rmse'):.2f} F1 {result.metric(scoring, 'clf_f1'):.3f}"
        for scoring in SCORINGS
    ]
    return f"split seed {result.seed} ({result.test_lifecycles} holdout lifecycles): " + " | ".join(parts)


def describe(evaluation: SpectralEvaluation) -> list[str]:
    """The summary table: mean per scoring, mean difference with - without and spectral wins."""
    lines = [
        f"spectral evaluation over {len(evaluation.splits)} split(s) of {evaluation.lifecycles} lifecycles:",
        f"{'metric':<15}{'with':>9}{'masked':>9}{'without':>9}{'with-without':>14}{'spectral wins':>15}",
    ]
    for name in METRICS:
        diffs = evaluation.differences(name)
        lines.append(
            f"{name:<15}"
            + "".join(f"{evaluation.mean(scoring, name):>9.4f}" for scoring in SCORINGS)
            + f"{statistics.fmean(diffs):>+14.4f}"
            + f"{evaluation.spectral_wins(name):>11d} / {len(diffs)}"
        )
    return lines


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the RUL models with and without the spectral features over several "
        "lifecycle splits.",
        epilog="Exit codes: 0 evaluated, 3 skipped (preconditions not met), 1 failure.",
    )
    parser.add_argument(
        "--splits", type=positive_int, default=5, help="lifecycle splits to evaluate (default 5)"
    )
    parser.add_argument(
        "--first-seed", type=int, default=SPLIT_SEED, help=f"seed of the first split (default {SPLIT_SEED})"
    )
    parser.add_argument(
        "--stride", type=positive_int, default=1, help="keep every n-th pulse of a lifecycle (default 1: all)"
    )
    parser.add_argument(
        "--min-lifecycles",
        type=positive_int,
        default=None,
        help="completed lifecycles required (default: MIN_TRAINING_LIFECYCLES)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``evaluate_spectral.py``."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    stop = StopToken().install_signal_handlers()
    model = settings.model
    with ClickHouseGateway(settings.clickhouse) as gateway:
        evaluation = evaluate_spectral(
            ClickHouseTrainingSource(gateway),
            LabelSpec(model.rul_cap_cycles, model.failure_horizon_cycles),
            seeds=range(args.first_seed, args.first_seed + args.splits),
            stride=args.stride,
            max_rows=model.training_max_rows,
            min_lifecycles=args.min_lifecycles or model.min_training_lifecycles,
            hours_per_cycle=model.flight_hours_per_cycle,
            stop=stop,
        )
    for line in describe(evaluation):
        logger.info("%s", line)
    print(json.dumps({"spectral_evaluation": evaluation.as_dict()}), flush=True)
    return EXIT_OK


if __name__ == "__main__":
    run_cli(main)
