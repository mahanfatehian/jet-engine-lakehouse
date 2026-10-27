"""Train the RUL regressor and the imminent-failure classifier; publish and promote them.

    python data_science/train_rul_model.py [--source clickhouse|minio] [--min-lifecycles N]
        [--promotion-policy better|always] [--skip-if-unchanged] [--dry-run]

Rows come from the dbt gold mart (ClickHouse) or its newest non-empty MinIO snapshot, at
most TRAINING_MAX_ROWS of them (the newest completed lifecycles, see training_window.py).
Whole lifecycles are split into train / validation / test; the challenger is scored on the
test lifecycles and, when a production model exists, so is the champion. Both models are
also refitted without the spectral features on the identical split: that ablation is
logged and recorded as evidence of what the HF path adds (feature_ablation.py) and never
influences promotion. The promotion gate (model_promotion.py) applies absolute quality
floors to every promotion and, under MODEL_PROMOTION_POLICY=better, the champion
comparison; it fails closed. Artifacts go to MinIO and every non-dry run is recorded in
ClickHouse ml_model_runs.

Training needs max(MIN_TRAINING_LIFECYCLES, 3) completed lifecycles: the split needs at
least one lifecycle per side. ``--skip-if-unchanged`` skips while the newest completed
lifecycle is the one the latest recorded run already saw.

Exit codes: 0 trained (promoted or not), 3 skipped (preconditions not met), 1 failure,
143 terminated.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from artifact_store import ArtifactStore, S3ArtifactStore
from champion_scoring import score_champion
from ds_runtime import EXIT_OK, PreconditionsNotMetError, StopToken, configure_logging, positive_int, run_cli
from ds_settings import MIN_SPLIT_LIFECYCLES, PROMOTION_POLICIES, PromotionSettings, load_settings
from feature_ablation import AblationResult, AblationRunner, run_ablation
from lakehouse_io import ClickHouseGateway
from model_promotion import ChampionOutcome, PromotionDecision, decide_promotion
from model_registry import (
    ArtifactLayout,
    ClickHouseRunRegistry,
    ModelArtifacts,
    ProductionPointer,
    PublishedModel,
    RunRegistry,
    RunSummary,
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
from training_report import DatasetInfo, TrainingReport
from training_sources import ClickHouseTrainingSource, SnapshotTrainingSource, TrainingDataSource
from training_window import LifecycleInventory, format_micros, plan_training_window

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("train_rul_model")


@dataclass(frozen=True)
class TrainingRequest:
    """CLI flags resolved against the environment defaults."""

    min_lifecycles: int
    promotion_policy: str
    skip_if_unchanged: bool
    dry_run: bool
    max_training_rows: int = 600_000
    gate: PromotionSettings = field(default_factory=PromotionSettings)

    @property
    def required_lifecycles(self) -> int:
        """Completed lifecycles needed: ``min_lifecycles``, but at least one per split side (3)."""
        return max(self.min_lifecycles, MIN_SPLIT_LIFECYCLES)


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
    ablate: AblationRunner = run_ablation


@dataclass(frozen=True)
class TrainingOutcome:
    """Result of one training run (``artifact_uri`` and ``bundle_sha256`` are None for dry runs)."""

    model_version: str
    split: GroupSplit
    holdout: HoldoutMetrics
    champion: ChampionOutcome
    decision: PromotionDecision
    ablation: AblationResult
    artifact_uri: str | None
    bundle_sha256: str | None = None


def _require_lifecycles(available: int, request: TrainingRequest, what: str) -> None:
    required = request.required_lifecycles
    if available < required:
        floor = (
            " (the train/validation/test split needs at least 3)" if request.min_lifecycles < required else ""
        )
        raise PreconditionsNotMetError(f"{available} completed lifecycles {what}, {required} required{floor}")


def _same_data(latest: RunSummary, inventory: LifecycleInventory) -> bool:
    if latest.data_watermark_us is not None:
        return latest.data_watermark_us == inventory.watermark_us
    # Runs recorded before data watermarks existed carry only their lifecycle count.
    return latest.lifecycles == inventory.completed_lifecycles


def _skip_reason(
    pointer: ProductionPointer | None, latest: RunSummary | None, inventory: LifecycleInventory
) -> str | None:
    """Why ``--skip-if-unchanged`` skips this run; None means train."""
    if latest is None or not _same_data(latest, inventory):
        return None
    newest = format_micros(inventory.watermark_us)
    if pointer is None:
        if latest.promoted:
            logger.info(
                "production.json is missing although %s was promoted: retraining", latest.model_version
            )
            return None
        return (
            f"no lifecycle completed since {latest.model_version} (newest ends {newest}), "
            "which was not promoted"
        )
    if not pointer.bundle_sha256:
        logger.info(
            "production model %s has no bundle digest: training despite --skip-if-unchanged so that a "
            "verifiable model can replace it",
            pointer.model_version,
        )
        return None
    return f"no lifecycle completed since {latest.model_version} (newest ends {newest})"


def _check_preconditions(
    request: TrainingRequest,
    context: TrainingContext,
    pointer: ProductionPointer | None,
    inventory: LifecycleInventory,
) -> None:
    _require_lifecycles(inventory.completed_lifecycles, request, "available")
    if request.skip_if_unchanged:
        reason = _skip_reason(pointer, context.runs.latest_run(), inventory)
        if reason is not None:
            raise PreconditionsNotMetError(reason)


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


def _publish(context: TrainingContext, report: TrainingReport) -> PublishedModel:
    """Write artifacts, then (on promotion) the pointer, then the run record."""
    bundle = report.trained.bundle
    artifacts = ModelArtifacts(
        bundle=bundle,
        metadata=report.metadata(),
        metrics=report.metrics(),
        feature_importance=feature_importance(bundle),
    )
    published = publish_artifacts(context.store, context.layout, artifacts)
    logger.info("published artifacts to %s (bundle sha256 %s)", published.uri, published.bundle_sha256)
    if report.decision.promote:
        pointer = ProductionPointer(
            model_version=bundle.model_version,
            artifact_prefix=context.layout.version_prefix(bundle.model_version),
            promoted_at=datetime.now(UTC).isoformat(),
            metrics=report.holdout.as_dict(),
            feature_columns=list(bundle.feature_columns),
            bundle_sha256=published.bundle_sha256,
        )
        write_production_pointer(context.store, context.layout, pointer)
        logger.info("production.json now points to %s", bundle.model_version)
    context.runs.record_run(report.run_record(published.uri, published.bundle_sha256))
    return published


def run_training(request: TrainingRequest, context: TrainingContext, *, now: datetime) -> TrainingOutcome:
    """Run one training job end to end; raise PreconditionsNotMetError when it should be skipped."""
    pointer = read_production_pointer(context.store, context.layout)
    inventory = context.source.inventory()
    _check_preconditions(request, context, pointer, inventory)
    window = plan_training_window(
        inventory, max_rows=request.max_training_rows, min_lifecycles=request.required_lifecycles
    )
    # Thinning pulses means TRAINING_MAX_ROWS cannot even hold the minimum number of lifecycles.
    (logger.warning if window.stride > 1 else logger.info)("%s", window.describe())
    loaded = context.source.load(window)
    _require_lifecycles(loaded.completed_lifecycles, request, "loaded")
    dataset = DatasetInfo(loaded.data_source, loaded.source_uri, window, inventory.watermark_us)
    _warn_on_label_mismatch(loaded.frame, context.labels)
    split = split_groups(loaded.frame["split_group"].unique())
    frames = _split_frames(loaded.frame, split)
    # The three split frames are all training needs: release the loaded copy.
    del loaded
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
    context.stop.raise_if_stopped()
    ablation = context.ablate(frames, context.labels, trained, holdout)
    context.stop.raise_if_stopped()
    champion = score_champion(context.store, pointer, frames["test"], trained.bundle, context.labels)
    decision = decide_promotion(request.promotion_policy, holdout, champion, request.gate)
    report = TrainingReport(
        trained_at=now,
        dataset=dataset,
        split=split,
        rows={side: len(frame) for side, frame in frames.items()},
        trained=trained,
        holdout=holdout,
        ablation=ablation,
        champion=champion,
        decision=decision,
        policy=request.promotion_policy,
        gate=request.gate,
    )
    report.log()
    if request.dry_run:
        logger.info("dry run: no artifacts, run record or production pointer written")
        return TrainingOutcome(model_version, split, holdout, champion, decision, ablation, None)
    # Last safe point: once publishing starts it runs to completion.
    context.stop.raise_if_stopped()
    published = _publish(context, report)
    return TrainingOutcome(
        model_version, split, holdout, champion, decision, ablation, published.uri, published.bundle_sha256
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RUL regressor and imminent-failure classifier, then publish/promote them.",
        epilog="Exit codes: 0 trained (promoted or not), 3 skipped (preconditions not met), 1 failure.",
    )
    parser.add_argument(
        "--source",
        choices=("clickhouse", "minio"),
        default="clickhouse",
        help="clickhouse: mart_rul_training_features; minio: newest non-empty gold snapshot "
        "(default: clickhouse)",
    )
    parser.add_argument(
        "--min-lifecycles",
        type=positive_int,
        default=None,
        help="completed lifecycles required to train (default: MIN_TRAINING_LIFECYCLES, 6); "
        "at least 3 are always required, one per train/validation/test side",
    )
    parser.add_argument(
        "--promotion-policy",
        choices=PROMOTION_POLICIES,
        default=None,
        help="better: promote when the challenger passes the quality floors and beats the champion's "
        "holdout RMSE without losing more than PROMOTION_F1_TOLERANCE classifier F1; always: skip the "
        "champion comparison, the floors still apply (default: MODEL_PROMOTION_POLICY)",
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
        max_training_rows=model.training_max_rows,
        gate=model.promotion,
    )
    store = S3ArtifactStore(settings.s3)
    with ClickHouseGateway(settings.clickhouse) as gateway, contextlib.ExitStack() as cleanup:
        source: TrainingDataSource = (
            ClickHouseTrainingSource(gateway)
            if args.source == "clickhouse"
            else cleanup.enter_context(SnapshotTrainingSource(store))
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
