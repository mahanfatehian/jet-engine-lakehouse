"""Re-scoring the production model (champion) on a challenger's holdout for the promotion gate.

Every way this can go wrong is classified for model_promotion.decide_promotion:

* INCOMPARABLE - a planned change: the champion's bundle format or feature schema differs
  from the challenger's, or its production.json predates bundle digests (it is then never
  unpickled). The quality floors decide alone.
* NO_UNSEEN_HOLDOUT - the champion was fitted on every holdout lifecycle (a small training
  window whose split fell back to an old lifecycle): it is kept until a fair comparison exists.
* FAILED - anything else (storage errors, digest mismatch, corrupt bundle, ...): the gate
  keeps the champion (fails closed), the error is logged and the run is still recorded.

Only holdout lifecycles the champion was not fitted on take part in the comparison, so it
can never be flattered by lifecycles it was trained or early-stopped on.
"""

from __future__ import annotations

import logging

import pandas as pd
from artifact_store import ArtifactNotFoundError, ArtifactStore
from feature_pipeline import FeatureSchemaError
from model_promotion import ChampionOutcome, ChampionStatus
from model_registry import METADATA_FILE, ProductionPointer, load_production_bundle
from rul_modeling import IncompatibleModelError, LabelSpec, RulModelBundle, evaluate_bundle

logger = logging.getLogger(__name__)


def _incomparable(version: str, detail: str) -> ChampionOutcome:
    logger.warning(
        "production model %s is not comparable with the challenger (%s); the quality floors decide alone",
        version,
        detail,
    )
    return ChampionOutcome(ChampionStatus.INCOMPARABLE, version, detail=detail)


def _failed(version: str, action: str, exc: Exception) -> ChampionOutcome:
    logger.error("could not %s production model %s: %s", action, version, exc, exc_info=exc)
    return ChampionOutcome(
        ChampionStatus.FAILED, version, detail=f"{action} failed: {type(exc).__name__}: {exc}"
    )


def _champion_fit_lifecycles(store: ArtifactStore, pointer: ProductionPointer) -> set[str] | None:
    """Lifecycles the champion was fitted on (train + validation), from its metadata.json.

    None when the metadata does not list them; storage errors propagate.
    """
    try:
        lifecycles = store.get_json(f"{pointer.artifact_prefix}/{METADATA_FILE}")["lifecycles"]
        return {str(group) for side in ("train", "validation") for group in lifecycles[side]}
    except (ArtifactNotFoundError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "metadata.json of production model %s does not list its lifecycles (%s): "
            "comparing on the whole holdout",
            pointer.model_version,
            exc,
        )
        return None


def score_champion(
    store: ArtifactStore,
    pointer: ProductionPointer | None,
    test_frame: pd.DataFrame,
    challenger: RulModelBundle,
    labels: LabelSpec,
) -> ChampionOutcome:
    """Re-score the production model on the holdout lifecycles it was not fitted on.

    Classifies every outcome (see the module docstring): COMPARABLE with metrics, or
    INCOMPARABLE, NO_UNSEEN_HOLDOUT or FAILED with the reason in ``detail``.
    """
    if pointer is None:
        return ChampionOutcome.none()
    version = pointer.model_version
    if not pointer.bundle_sha256:
        return _incomparable(
            version, "production.json predates bundle digests, so its bundle is not unpickled"
        )
    try:
        champion = load_production_bundle(store, pointer)
        fitted_on = _champion_fit_lifecycles(store, pointer) or set()
    except IncompatibleModelError as exc:
        return _incomparable(version, str(exc))
    except Exception as exc:
        return _failed(version, "load", exc)
    holdout_groups = set(test_frame["split_group"])
    unseen = holdout_groups - fitted_on
    if not unseen:
        detail = f"all {len(holdout_groups)} holdout lifecycles are among its training lifecycles"
        logger.warning("production model %s cannot be compared fairly: %s", version, detail)
        return ChampionOutcome(ChampionStatus.NO_UNSEEN_HOLDOUT, version, detail=detail)
    if unseen == holdout_groups:
        frame, detail = test_frame, ""
    else:
        frame = test_frame[test_frame["split_group"].isin(unseen)].reset_index(drop=True)
        detail = f"the {len(unseen)} of {len(holdout_groups)} holdout lifecycles it was not fitted on"
    try:
        return ChampionOutcome(
            ChampionStatus.COMPARABLE,
            version,
            evaluate_bundle(champion, frame, labels),
            detail=detail,
            challenger_holdout=evaluate_bundle(challenger, frame, labels) if detail else None,
        )
    except FeatureSchemaError as exc:
        if tuple(champion.feature_columns) != tuple(challenger.feature_columns):
            return _incomparable(version, f"its feature schema differs from the challenger's: {exc}")
        return _failed(version, "re-score", exc)
    except Exception as exc:
        return _failed(version, "re-score", exc)
