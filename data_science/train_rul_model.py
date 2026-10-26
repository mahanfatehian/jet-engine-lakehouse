"""Train the RUL regressor and the imminent-failure classifier; publish and promote them.

    python data_science/train_rul_model.py [--source clickhouse|minio] [--min-lifecycles N]
        [--promotion-policy better|always] [--skip-if-unchanged] [--dry-run]

Rows come from the dbt gold mart (ClickHouse) or its newest MinIO snapshot. Whole
lifecycles are split into train / validation / test; the challenger is scored on
the test lifecycles and, when a production model exists, so is the champion. The
challenger is promoted per MODEL_PROMOTION_POLICY. Artifacts go to MinIO and every
non-dry run is recorded in ClickHouse ml_model_runs.

Exit codes: 0 trained, 3 skipped (preconditions not met), 1 failure, 143 terminated.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from artifact_store import ArtifactStore, S3ArtifactStore
from ds_runtime import EXIT_OK, PreconditionsNotMetError, StopToken, configure_logging, positive_int, run_cli
from ds_settings import PROMOTION_POLICIES, load_settings
from feature_pipeline import ClickHouseTrainingSource, SnapshotTrainingSource, TrainingDataSource
from lakehouse_io import ClickHouseGateway
from model_registry import (
    ArtifactLayout,
    ClickHouseRunRegistry,
    ModelArtifacts,
    ProductionPointer,
    PromotionDecision,
    RunRegistry,
    decide_promotion,
    load_production_bundle,
    new_model_version,
    publish_artifacts,
    read_production_pointer,
    write_production_pointer,
)
from rul_metrics import HoldoutMetrics
from rul_modeling import (
    GroupSplit,
    LabelSpec,
    evaluate_bundle,
    feature_importance,
    split_groups,
    train_models,
)
from training_report import ChampionScore, TrainingReport

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("train_rul_model")


@dataclass(frozen=True)
class TrainingRequest:
    """CLI flags resolved against the environment defaults."""

    min_lifecycles: int
    promotion_policy: str
    skip_if_unchanged: bool
    dry_run: bool


@dataclass(frozen=True)
class TrainingContext:
    """Collaborators of a training run (injected so tests can use local fakes)."""

    source: TrainingDataSource
    store: ArtifactStore
    runs: RunRegistry
    layout: ArtifactLayout
    labels: LabelSpec
    hours_per_cycle: float
    stop: StopToken


@dataclass(frozen=True)
class TrainingOutcome:
    """Result of one training run (``artifact_uri`` is None for dry runs)."""

    model_version: str
    split: GroupSplit
    holdout: HoldoutMetrics
    champion: ChampionScore | None
    decision: PromotionDecision
    artifact_uri: str | None


def _check_preconditions(
    request: TrainingRequest, context: TrainingContext, pointer: ProductionPointer | None
) -> None:
    completed = context.source.count_completed_lifecycles()
    if completed < request.min_lifecycles:
        raise PreconditionsNotMetError(
            f"{completed} completed lifecycles available, {request.min_lifecycles} required"
        )
    if not request.skip_if_unchanged:
        return
    if pointer is None:
        logger.info("no production model yet: training despite --skip-if-unchanged")
        return
    latest = context.runs.latest_run()
    if latest is not None and latest.lifecycles == completed:
        raise PreconditionsNotMetError(
            f"no new completed lifecycles since {latest.model_version} ({completed} lifecycles)"
        )


def _split_frames(frame: pd.DataFrame, split: GroupSplit) -> dict[str, pd.DataFrame]:
    sides = {"train": split.train, "validation": split.validation, "test": split.test}
    return {
        side: frame[frame["split_group"].isin(groups)].reset_index(drop=True)
        for side, groups in sides.items()
    }


def _warn_on_label_mismatch(frame: pd.DataFrame, labels: LabelSpec) -> None:
    for column, count in labels.mismatches(frame).items():
        if count:
            logger.warning(
                "%d rows of %s disagree with RUL_CAP_CYCLES=%d / RUL_FAILURE_HORIZON_CYCLES=%d; "
                "the trainer derives labels from rul_cycles with its own settings (check the dbt vars)",
                count,
                column,
                labels.rul_cap_cycles,
                labels.failure_horizon_cycles,
            )


def _score_champion(
    context: TrainingContext, pointer: ProductionPointer | None, test_frame: pd.DataFrame
) -> ChampionScore | None:
    if pointer is None:
        return None
    try:
        champion = load_production_bundle(context.store, pointer)
        return ChampionScore(champion.model_version, evaluate_bundle(champion, test_frame, context.labels))
    except Exception:
        # An unreadable or incompatible champion must not block a fresh, working model.
        logger.warning(
            "production model %s could not be re-scored; treating it as unusable",
            pointer.model_version,
            exc_info=True,
        )
        return None


def _publish(context: TrainingContext, report: TrainingReport) -> str:
    """Write artifacts, then (on promotion) the pointer, then the run record; return the artifact URI."""
    bundle = report.trained.bundle
    artifacts = ModelArtifacts(
        bundle=bundle,
        metadata=report.metadata(),
        metrics=report.metrics(),
        feature_importance=feature_importance(bundle),
    )
    artifact_uri = publish_artifacts(context.store, context.layout, artifacts)
    logger.info("published artifacts to %s", artifact_uri)
    if report.decision.promote:
        pointer = ProductionPointer(
            model_version=bundle.model_version,
            artifact_prefix=context.layout.version_prefix(bundle.model_version),
            promoted_at=datetime.now(UTC).isoformat(),
            metrics=report.holdout.as_dict(),
            feature_columns=list(bundle.feature_columns),
        )
        write_production_pointer(context.store, context.layout, pointer)
        logger.info("production.json now points to %s", bundle.model_version)
    context.runs.record_run(report.run_record(artifact_uri))
    return artifact_uri


def run_training(request: TrainingRequest, context: TrainingContext, *, now: datetime) -> TrainingOutcome:
    """Run one training job end to end; raise PreconditionsNotMetError when it should be skipped."""
    pointer = read_production_pointer(context.store, context.layout)
    _check_preconditions(request, context, pointer)
    dataset = context.source.load()
    if dataset.completed_lifecycles < request.min_lifecycles:
        raise PreconditionsNotMetError(
            f"{dataset.completed_lifecycles} completed lifecycles loaded, {request.min_lifecycles} required"
        )
    _warn_on_label_mismatch(dataset.frame, context.labels)
    split = split_groups(dataset.frame["split_group"].unique())
    frames = _split_frames(dataset.frame, split)
    model_version = new_model_version(now)
    logger.info(
        "training %s from %s: %d/%d/%d train/validation/test lifecycles (%d/%d/%d rows)",
        model_version,
        dataset.source_uri,
        len(split.train),
        len(split.validation),
        len(split.test),
        len(frames["train"]),
        len(frames["validation"]),
        len(frames["test"]),
    )
    context.stop.raise_if_stopped()
    trained = train_models(
        frames["train"],
        frames["validation"],
        context.labels,
        model_version=model_version,
        trained_at=now.isoformat(),
        hours_per_cycle=context.hours_per_cycle,
    )
    context.stop.raise_if_stopped()
    holdout = evaluate_bundle(trained.bundle, frames["test"], context.labels)
    champion = _score_champion(context, pointer, frames["test"])
    decision = decide_promotion(
        request.promotion_policy, holdout.rmse, None if champion is None else champion.holdout.rmse
    )
    report = TrainingReport(
        trained_at=now,
        dataset=dataset,
        split=split,
        rows={side: len(frame) for side, frame in frames.items()},
        trained=trained,
        holdout=holdout,
        champion=champion,
        decision=decision,
        policy=request.promotion_policy,
    )
    report.log()
    artifact_uri = None
    if request.dry_run:
        logger.info("dry run: no artifacts, run record or production pointer written")
    else:
        # Last safe point: once publishing starts it runs to completion.
        context.stop.raise_if_stopped()
        artifact_uri = _publish(context, report)
    return TrainingOutcome(model_version, split, holdout, champion, decision, artifact_uri)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RUL regressor and imminent-failure classifier, then publish/promote them.",
        epilog="Exit codes: 0 trained, 3 skipped (preconditions not met), 1 failure.",
    )
    parser.add_argument(
        "--source",
        choices=("clickhouse", "minio"),
        default="clickhouse",
        help="clickhouse: mart_rul_training_features; minio: newest gold snapshot (default: clickhouse)",
    )
    parser.add_argument(
        "--min-lifecycles",
        type=positive_int,
        default=None,
        help="completed lifecycles required to train (default: MIN_TRAINING_LIFECYCLES, 6)",
    )
    parser.add_argument(
        "--promotion-policy",
        choices=PROMOTION_POLICIES,
        default=None,
        help="better: promote only on lower holdout RMSE; always (default: MODEL_PROMOTION_POLICY)",
    )
    parser.add_argument(
        "--skip-if-unchanged",
        action="store_true",
        help="exit 3 when no lifecycle completed since the latest ml_model_runs row",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="train and evaluate only; write no artifacts, run record or production pointer",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``train_rul_model.py``."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    stop = StopToken().install_signal_handlers()
    model = settings.model
    request = TrainingRequest(
        min_lifecycles=args.min_lifecycles or model.min_training_lifecycles,
        promotion_policy=args.promotion_policy or model.promotion_policy,
        skip_if_unchanged=args.skip_if_unchanged,
        dry_run=args.dry_run,
    )
    store = S3ArtifactStore(settings.s3)
    with ClickHouseGateway(settings.clickhouse) as gateway:
        source: TrainingDataSource = (
            ClickHouseTrainingSource(gateway)
            if args.source == "clickhouse"
            else SnapshotTrainingSource(store)
        )
        context = TrainingContext(
            source=source,
            store=store,
            runs=ClickHouseRunRegistry(gateway),
            layout=ArtifactLayout(model.artifact_prefix),
            labels=LabelSpec(model.rul_cap_cycles, model.failure_horizon_cycles),
            hours_per_cycle=model.flight_hours_per_cycle,
            stop=stop,
        )
        run_training(request, context, now=datetime.now(UTC))
    return EXIT_OK


if __name__ == "__main__":
    run_cli(main)
