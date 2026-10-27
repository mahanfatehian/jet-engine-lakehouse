"""The v1 -> v2 transition: a production model with the 46 pulse features meets a spectral challenger.

The promotion gate lets the floors decide alone only for a champion it cannot score. A v1
champion's inputs are all still in the v2 mart, so it is re-scored on its own columns of
the identical holdout and compared like any other champion.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from artifact_store import LocalArtifactStore
from ds_settings import PromotionSettings
from feature_ablation import AblationStatus, run_ablation
from feature_pipeline import FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS
from model_promotion import ChampionStatus
from model_registry import (
    ArtifactLayout,
    ModelArtifacts,
    ProductionPointer,
    publish_artifacts,
    read_production_pointer,
    write_production_pointer,
)
from rul_modeling import LabelSpec, RulModelBundle, evaluate_bundle, feature_importance, train_models
from train_rul_model import TrainingRequest, run_training

LAYOUT = ArtifactLayout("ml-artifacts/rul_model")
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
V1_VERSION = "rul-20261001T000000Z"
GATE = PromotionSettings()
REQUEST = TrainingRequest(min_lifecycles=6, promotion_policy="better", skip_if_unchanged=False, dry_run=False)


@pytest.fixture(scope="module")
def v1_champion(synthetic_training, label_spec: LabelSpec) -> RulModelBundle:
    """A model trained like v1 (pulse features only) on the same development lifecycles."""
    frames = synthetic_training.frames
    return train_models(
        frames["train"],
        frames["validation"],
        label_spec,
        model_version=V1_VERSION,
        trained_at="2026-10-01T00:00:00+00:00",
        hours_per_cycle=2.0,
        feature_columns=PULSE_FEATURE_COLUMNS,
    ).bundle


def _promote_v1(store: LocalArtifactStore, bundle: RulModelBundle, synthetic_training) -> None:
    split = synthetic_training.split
    metadata = {"lifecycles": {"train": list(split.train), "validation": list(split.validation)}}
    published = publish_artifacts(
        store, LAYOUT, ModelArtifacts(bundle, metadata, {}, feature_importance(bundle))
    )
    pointer = ProductionPointer(
        V1_VERSION,
        LAYOUT.version_prefix(V1_VERSION),
        T0.isoformat(),
        {},
        list(bundle.feature_columns),
        published.bundle_sha256,
    )
    write_production_pointer(store, LAYOUT, pointer)


def test_v1_champion_is_compared_on_its_own_columns_and_replaced_when_worse(
    tmp_path: Path, synthetic_mart, synthetic_training, v1_champion, make_context, label_spec, caplog
) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote_v1(store, v1_champion, synthetic_training)
    assert v1_champion.feature_columns == PULSE_FEATURE_COLUMNS

    with caplog.at_level(logging.INFO, logger="training_report"):
        outcome = run_training(REQUEST, make_context(tmp_path, synthetic_mart, store=store), now=T0)

    champion = outcome.champion
    assert champion.status is ChampionStatus.COMPARABLE and champion.detail == ""
    assert champion.feature_count == 46 and champion.as_dict()["feature_count"] == 46
    assert champion.holdout == evaluate_bundle(v1_champion, synthetic_training.frames["test"], label_spec)
    assert champion.holdout.rows == outcome.holdout.rows
    # The planted spectral signal makes the challenger better, so the comparison promotes it.
    assert outcome.holdout.rmse < champion.holdout.rmse
    assert outcome.decision.promote and f"challenger is better than {V1_VERSION}" in outcome.decision.reason
    assert read_production_pointer(store, LAYOUT).feature_columns == list(FEATURE_COLUMNS)
    assert f"champion {V1_VERSION} (46 features, challenger 56)" in caplog.text


def test_without_hf_data_a_v1_champion_is_still_compared_not_replaced_on_the_floors(
    tmp_path: Path, synthetic_mart, synthetic_training, v1_champion, make_context, hf_disabled
) -> None:
    store = LocalArtifactStore(tmp_path)
    _promote_v1(store, v1_champion, synthetic_training)

    context = make_context(tmp_path, hf_disabled(synthetic_mart), store=store, ablate=run_ablation)

    outcome = run_training(REQUEST, context, now=T0)

    champion = outcome.champion
    assert champion.status is ChampionStatus.COMPARABLE and champion.feature_count == 46
    assert (
        outcome.ablation.status is AblationStatus.SKIPPED
        and "no spectral feature value" in outcome.ablation.detail
    )
    better = outcome.holdout.rmse < champion.holdout.rmse
    within = outcome.holdout.clf_f1 >= champion.holdout.clf_f1 - GATE.f1_tolerance
    assert outcome.decision.promote == (better and within)
    assert "floors alone" not in outcome.decision.reason
    serving = outcome.model_version if outcome.decision.promote else V1_VERSION
    assert read_production_pointer(store, LAYOUT).model_version == serving
