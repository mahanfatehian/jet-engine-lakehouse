"""Promotion gate (floors, champion comparison, fail-closed rules) and --skip-if-unchanged."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from artifact_store import LocalArtifactStore
from botocore.exceptions import EndpointConnectionError
from ds_runtime import PreconditionsNotMetError
from ds_settings import PromotionSettings
from model_promotion import ChampionOutcome, ChampionStatus, decide_promotion, floor_failures
from model_registry import (
    BUNDLE_FILE,
    ArtifactLayout,
    ModelArtifacts,
    ProductionPointer,
    RunSummary,
    publish_artifacts,
    read_production_pointer,
    write_production_pointer,
)
from rul_metrics import HoldoutMetrics
from rul_modeling import RulModelBundle, feature_importance, split_groups
from train_rul_model import TrainingRequest, run_training

LAYOUT = ArtifactLayout("ml-artifacts/rul_model")
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
GATE = PromotionSettings()
# No model reaches this R2, so the floors reject every challenger.
STRICT = PromotionSettings(min_r2=0.9999)


def _metrics(rmse: float = 10.0, r2: float = 0.9, roc_auc: float = 0.95, f1: float = 0.9) -> HoldoutMetrics:
    return HoldoutMetrics(
        rows=100,
        lifecycles=3,
        rmse=rmse,
        mae=rmse * 0.7,
        r2=r2,
        late_life_rmse=rmse,
        nasa_score=0.5,
        nasa_score_sum=50.0,
        clf_threshold=0.45,
        clf_f1=f1,
        clf_precision=f1,
        clf_recall=f1,
        clf_roc_auc=roc_auc,
        positive_rate=0.15,
        clf_brier=0.05,
        clf_ece=0.02,
    )


def _champion(**metrics: float) -> ChampionOutcome:
    return ChampionOutcome(ChampionStatus.COMPARABLE, "rul-champion", _metrics(**metrics))


@pytest.mark.parametrize(
    ("policy", "challenger", "champion", "promote"),
    [
        ("better", _metrics(), ChampionOutcome.none(), True),
        ("better", _metrics(rmse=9.0), _champion(), True),
        ("better", _metrics(rmse=10.0), _champion(), False),
        ("better", _metrics(rmse=11.0), _champion(), False),
        ("better", _metrics(rmse=9.0, f1=0.86), _champion(f1=0.9), True),
        ("better", _metrics(rmse=9.0, f1=0.84), _champion(f1=0.9), False),
        ("always", _metrics(rmse=11.0, f1=0.5), _champion(), True),
        (
            "better",
            _metrics(),
            ChampionOutcome(ChampionStatus.INCOMPARABLE, "rul-old", detail="schema"),
            True,
        ),
        ("better", _metrics(), ChampionOutcome(ChampionStatus.FAILED, "rul-old", detail="timeout"), False),
        ("always", _metrics(), ChampionOutcome(ChampionStatus.FAILED, "rul-old", detail="timeout"), True),
        (
            "better",
            _metrics(),
            ChampionOutcome(ChampionStatus.NO_UNSEEN_HOLDOUT, "rul-old", detail="seen"),
            False,
        ),
        (
            "always",
            _metrics(),
            ChampionOutcome(ChampionStatus.NO_UNSEEN_HOLDOUT, "rul-old", detail="seen"),
            True,
        ),
        ("better", _metrics(r2=0.4), ChampionOutcome.none(), False),
        ("always", _metrics(r2=0.4), _champion(rmse=20.0), False),
        ("always", _metrics(roc_auc=0.7), ChampionOutcome.none(), False),
        ("better", _metrics(rmse=5.0, r2=0.4), _champion(r2=0.3), False),
    ],
)
def test_promotion_gate(policy: str, challenger, champion: ChampionOutcome, promote: bool) -> None:
    decision = decide_promotion(policy, challenger, champion, GATE)

    assert decision.promote is promote
    assert decision.reason


def test_floors_report_every_miss_and_undefined_metrics_fail_closed() -> None:
    failures = floor_failures(_metrics(r2=0.2, roc_auc=math.nan), GATE)

    assert failures == ("R2 0.200 < 0.5", "ROC-AUC is undefined (floor 0.75)")
    decision = decide_promotion("always", _metrics(r2=0.2, roc_auc=math.nan), ChampionOutcome.none(), GATE)
    assert not decision.promote and decision.floor_failures == failures
    assert floor_failures(_metrics(), PromotionSettings(min_r2=0.95)) == ("R2 0.900 < 0.95",)


def test_failed_champion_reason_names_the_champion_and_the_error() -> None:
    champion = ChampionOutcome(ChampionStatus.FAILED, "rul-old", detail="load failed: ConnectionError: down")

    decision = decide_promotion("better", _metrics(), champion, GATE)

    assert "rul-old" in decision.reason and "ConnectionError" in decision.reason
    assert "fail closed" in decision.reason


def test_comparison_uses_the_challengers_metrics_on_the_same_lifecycles() -> None:
    champion = ChampionOutcome(
        ChampionStatus.COMPARABLE,
        "rul-old",
        _metrics(rmse=10.0),
        detail="the 2 of 3 holdout lifecycles it was not fitted on",
        challenger_holdout=_metrics(rmse=9.0),
    )

    decision = decide_promotion("better", _metrics(rmse=12.0), champion, GATE)

    assert decision.promote and "RMSE 9.000 vs 10.000" in decision.reason
    assert "the 2 of 3 holdout lifecycles" in decision.reason


def test_unknown_promotion_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sometimes"):
        decide_promotion("sometimes", _metrics(), ChampionOutcome.none(), GATE)


def _request(
    policy: str = "better", *, skip: bool = False, gate: PromotionSettings = GATE
) -> TrainingRequest:
    return TrainingRequest(
        min_lifecycles=6, promotion_policy=policy, skip_if_unchanged=skip, dry_run=False, gate=gate
    )


class RecordingStore(LocalArtifactStore):
    """Local store that records every object read."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads: list[str] = []

    def get_bytes(self, key: str) -> bytes:
        self.reads.append(key)
        return super().get_bytes(key)


class UnreachableBundleStore(LocalArtifactStore):
    """Local store whose model bundles cannot be downloaded (object store outage)."""

    def get_bytes(self, key: str) -> bytes:
        if key.endswith(BUNDLE_FILE):
            raise EndpointConnectionError(endpoint_url="http://minio:9000")
        return super().get_bytes(key)


def _promote_bundle(
    store: LocalArtifactStore,
    bundle: RulModelBundle,
    *,
    with_digest: bool = True,
    fitted_on: tuple[list[str], list[str]] | None = None,
) -> None:
    """Publish ``bundle`` and point production.json at it (without a digest: a legacy pointer).

    ``fitted_on`` (train, validation) lifecycles go to metadata.json; without it the
    metadata does not list lifecycles and the whole holdout is compared.
    """
    metadata = (
        {} if fitted_on is None else {"lifecycles": {"train": fitted_on[0], "validation": fitted_on[1]}}
    )
    artifacts = ModelArtifacts(bundle, metadata, {}, feature_importance(bundle))
    published = publish_artifacts(store, LAYOUT, artifacts)
    pointer = ProductionPointer(
        bundle.model_version,
        LAYOUT.version_prefix(bundle.model_version),
        T0.isoformat(),
        {},
        list(bundle.feature_columns),
        published.bundle_sha256 if with_digest else None,
    )
    write_production_pointer(store, LAYOUT, pointer)


def test_unreachable_champion_keeps_it_and_still_records_the_run(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context, caplog
) -> None:
    store = UnreachableBundleStore(tmp_path)
    _promote_bundle(store, synthetic_training.trained.bundle)
    context = make_context(tmp_path, synthetic_mart, store=store)

    with caplog.at_level(logging.ERROR, logger="champion_scoring"):
        outcome = run_training(_request(), context, now=T0)

    assert outcome.champion.status is ChampionStatus.FAILED
    assert not outcome.decision.promote and "EndpointConnectionError" in outcome.decision.reason
    assert (
        read_production_pointer(store, LAYOUT).model_version
        == synthetic_training.trained.bundle.model_version
    )
    [record] = context.runs.records
    assert record.promoted is False
    assert any(
        entry.levelno == logging.ERROR and "could not load" in entry.getMessage() for entry in caplog.records
    )


def test_tampered_champion_bundle_is_not_unpickled_and_keeps_the_champion(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context
) -> None:
    store = LocalArtifactStore(tmp_path)
    champion = synthetic_training.trained.bundle
    _promote_bundle(store, champion)
    key = LAYOUT.artifact_key(champion.model_version, BUNDLE_FILE)
    store.put_bytes(key, store.get_bytes(key) + b"tampered")

    outcome = run_training(_request(), make_context(tmp_path, synthetic_mart, store=store), now=T0)

    assert outcome.champion.status is ChampionStatus.FAILED
    assert "ModelIntegrityError" in outcome.champion.detail and not outcome.decision.promote


def test_policy_always_promotes_past_a_failed_champion_but_not_past_the_floors(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context
) -> None:
    store = UnreachableBundleStore(tmp_path)
    _promote_bundle(store, synthetic_training.trained.bundle)
    context = make_context(tmp_path, synthetic_mart, store=store)

    strict = run_training(_request("always", gate=STRICT), context, now=T0)
    promoted = run_training(_request("always"), context, now=T0 + timedelta(hours=1))

    assert not strict.decision.promote and strict.decision.floor_failures
    assert promoted.decision.promote
    assert read_production_pointer(store, LAYOUT).model_version == promoted.model_version
    assert [record.promoted for record in context.runs.records] == [False, True]


def test_champion_needing_a_column_the_holdout_lacks_is_incomparable(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context
) -> None:
    store = LocalArtifactStore(tmp_path)
    champion = synthetic_training.trained.bundle
    # A future feature this mart does not have: the champion cannot be scored on the holdout.
    _promote_bundle(
        store, replace(champion, feature_columns=(*champion.feature_columns, "fan_envelope_bpfo_mean_5m"))
    )

    outcome = run_training(_request(), make_context(tmp_path, synthetic_mart, store=store), now=T0)

    assert outcome.champion.status is ChampionStatus.INCOMPARABLE
    assert "fan_envelope_bpfo_mean_5m" in outcome.champion.detail
    assert outcome.champion.feature_count == len(champion.feature_columns) + 1
    assert outcome.decision.promote and "floors alone" in outcome.decision.reason


def test_legacy_pointer_without_digest_is_replaced_without_unpickling(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context, run_registry
) -> None:
    store = RecordingStore(tmp_path)
    legacy = synthetic_training.trained.bundle
    _promote_bundle(store, legacy, with_digest=False)
    lifecycles = synthetic_mart["split_group"].nunique()
    runs = run_registry(history=RunSummary("rul-20261001T000000Z", lifecycles, True))
    context = make_context(tmp_path, synthetic_mart, runs, store=store)

    # --skip-if-unchanged would skip (same lifecycle count), but the unverifiable pointer must go.
    outcome = run_training(_request(skip=True), context, now=T0)

    assert outcome.champion.status is ChampionStatus.INCOMPARABLE and outcome.decision.promote
    assert LAYOUT.artifact_key(legacy.model_version, BUNDLE_FILE) not in store.reads
    pointer = read_production_pointer(store, LAYOUT)
    assert pointer.model_version == outcome.model_version and pointer.bundle_sha256 == outcome.bundle_sha256


def test_floors_block_a_weak_bootstrap_model_and_the_run_is_recorded(
    tmp_path: Path, synthetic_mart, make_context
) -> None:
    context = make_context(tmp_path, synthetic_mart)

    outcome = run_training(_request(gate=STRICT), context, now=T0)

    assert not outcome.decision.promote and "floors" in outcome.decision.reason
    assert read_production_pointer(context.store, LAYOUT) is None
    [record] = context.runs.records
    assert not record.promoted and record.artifact_uri == outcome.artifact_uri


def test_skip_if_unchanged_follows_the_data_watermark(
    tmp_path: Path, synthetic_mart, make_mart, make_context, run_registry
) -> None:
    runs = run_registry()
    context = make_context(tmp_path, synthetic_mart, runs)

    bootstrap = run_training(_request(skip=True), context, now=T0)
    with pytest.raises(PreconditionsNotMetError, match="no lifecycle completed since"):
        run_training(_request(skip=True), context, now=T0 + timedelta(hours=1))
    grown = make_context(tmp_path, make_mart(n_engines=5), runs)
    retrained = run_training(_request(skip=True), grown, now=T0 + timedelta(hours=2))

    assert bootstrap.artifact_uri and retrained.artifact_uri
    assert len(runs.records) == 2


def test_skip_if_unchanged_trains_when_a_bounded_mart_rolls_over(
    tmp_path: Path, synthetic_mart, make_context, run_registry
) -> None:
    # A bounded mart keeps the lifecycle count constant: one new lifecycle in, the oldest out.
    by_end = synthetic_mart.groupby("split_group")["event_ts"].max().sort_values().index
    older = synthetic_mart[synthetic_mart["split_group"].isin(by_end[:-1])]
    newer = synthetic_mart[synthetic_mart["split_group"].isin(by_end[1:])]
    runs = run_registry()
    run_training(_request(skip=True), make_context(tmp_path, older, runs), now=T0)

    rolled = run_training(
        _request(skip=True), make_context(tmp_path, newer, runs), now=T0 + timedelta(hours=1)
    )

    assert rolled.artifact_uri and len(runs.records) == 2


def test_skip_if_unchanged_waits_for_new_data_after_a_rejected_bootstrap(
    tmp_path: Path, synthetic_mart, make_context, run_registry
) -> None:
    runs = run_registry()
    context = make_context(tmp_path, synthetic_mart, runs)
    run_training(_request(skip=True, gate=STRICT), context, now=T0)

    with pytest.raises(PreconditionsNotMetError, match="which was not promoted"):
        run_training(_request(skip=True, gate=STRICT), context, now=T0 + timedelta(hours=1))


def test_skip_if_unchanged_falls_back_to_lifecycle_counts_of_legacy_runs(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context, run_registry
) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote_bundle(store, synthetic_training.trained.bundle)
    lifecycles = synthetic_mart["split_group"].nunique()
    legacy = run_registry(history=RunSummary("rul-20261001T000000Z", lifecycles, True))

    with pytest.raises(PreconditionsNotMetError, match="rul-20261001T000000Z"):
        run_training(_request(skip=True), make_context(tmp_path, synthetic_mart, legacy, store=store), now=T0)
    grown = run_registry(history=RunSummary("rul-20261001T000000Z", lifecycles - 1, True))
    assert run_training(
        _request(skip=True), make_context(tmp_path, synthetic_mart, grown, store=store), now=T0
    )


def test_holdout_lifecycles_the_champion_was_fitted_on_are_left_out(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context
) -> None:
    store = LocalArtifactStore(tmp_path)
    holdout = sorted(split_groups(synthetic_mart["split_group"].unique()).test)
    assert len(holdout) >= 2
    _promote_bundle(store, synthetic_training.trained.bundle, fitted_on=([holdout[0]], []))

    outcome = run_training(_request(), make_context(tmp_path, synthetic_mart, store=store), now=T0)

    champion = outcome.champion
    assert champion.status is ChampionStatus.COMPARABLE
    assert (
        champion.detail == f"the {len(holdout) - 1} of {len(holdout)} holdout lifecycles it was not fitted on"
    )
    assert champion.holdout.lifecycles == champion.challenger_holdout.lifecycles == len(holdout) - 1
    assert champion.holdout.rows == champion.challenger_holdout.rows < outcome.holdout.rows


def test_champion_fitted_on_every_holdout_lifecycle_is_kept_until_a_fair_comparison_exists(
    tmp_path: Path, synthetic_mart, synthetic_training, make_context
) -> None:
    store = LocalArtifactStore(tmp_path)
    champion = synthetic_training.trained.bundle
    everything = sorted(synthetic_mart["split_group"].unique())
    _promote_bundle(store, champion, fitted_on=(everything, []))
    context = make_context(tmp_path, synthetic_mart, store=store)

    kept = run_training(_request(), context, now=T0)
    forced = run_training(_request("always"), context, now=T0 + timedelta(hours=1))

    assert kept.champion.status is ChampionStatus.NO_UNSEEN_HOLDOUT
    assert "among its training lifecycles" in kept.champion.detail
    assert not kept.decision.promote and "cannot be compared fairly" in kept.decision.reason
    assert forced.decision.promote
    assert read_production_pointer(store, LAYOUT).model_version == forced.model_version
