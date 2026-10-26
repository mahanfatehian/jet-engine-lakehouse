"""Champion/challenger promotion gate. It fails closed.

1. Floors: the challenger's holdout R2 must reach PROMOTION_MIN_R2 and its classifier
   ROC-AUC PROMOTION_MIN_ROC_AUC (an undefined metric fails). They apply to every
   promotion: at bootstrap, against an incomparable champion and under policy ``always``.
2. Policy ``always`` then promotes: it bypasses the champion comparison, not the floors.
3. Policy ``better`` compares against the champion re-scored on the challenger's holdout:
   the challenger needs a lower RMSE and may lose at most PROMOTION_F1_TOLERANCE of
   classifier F1 (each model at its own decision threshold). Only holdout lifecycles the
   champion was not fitted on take part (its metadata.json lists them). When it was fitted
   on all of them no fair comparison exists yet, and the champion is kept until a lifecycle
   it has not seen reaches the holdout.
4. A champion that cannot be compared because it was built for another feature schema or
   bundle format (a planned change), or whose production.json predates bundle digests (so it
   cannot be verified and is never unpickled), is "incomparable": the floors decide alone.
   Any other failure to re-score it (storage errors, digest mismatch, corrupt bundle, ...)
   keeps the champion: nothing is promoted until it can be scored again or an operator
   promotes explicitly with ``--promotion-policy always``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ds_settings import PROMOTION_POLICIES, PromotionSettings
from rul_metrics import HoldoutMetrics


class ChampionStatus(StrEnum):
    """What re-scoring the production model on the challenger's holdout produced."""

    NONE = "none"  # no production model yet
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"  # planned schema/format change or unverifiable legacy pointer
    NO_UNSEEN_HOLDOUT = "no_unseen_holdout"  # fitted on every holdout lifecycle: keep it for now
    FAILED = "failed"  # any other error: fail closed


@dataclass(frozen=True)
class ChampionOutcome:
    """The production model's holdout metrics, or why there are none.

    ``challenger_holdout`` is set when the comparison had to leave out holdout lifecycles
    the champion was fitted on: the challenger's metrics on the same remaining lifecycles.
    """

    status: ChampionStatus
    model_version: str | None = None
    holdout: HoldoutMetrics | None = None
    detail: str = ""
    challenger_holdout: HoldoutMetrics | None = None

    @classmethod
    def none(cls) -> ChampionOutcome:
        """No production model exists."""
        return cls(ChampionStatus.NONE)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary for metadata.json, metrics.json and ml_model_runs."""
        return {
            "status": self.status.value,
            "model_version": self.model_version,
            "holdout": None if self.holdout is None else self.holdout.as_dict(),
            "detail": self.detail or None,
            "challenger_holdout": None
            if self.challenger_holdout is None
            else self.challenger_holdout.as_dict(),
        }


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the promotion gate and the checks behind it."""

    promote: bool
    reason: str
    floor_failures: tuple[str, ...] = ()


def floor_failures(holdout: HoldoutMetrics, gate: PromotionSettings) -> tuple[str, ...]:
    """Every absolute quality floor the challenger misses (an undefined metric misses)."""
    failures = []
    for name, value, floor in (
        ("R2", holdout.r2, gate.min_r2),
        ("ROC-AUC", holdout.clf_roc_auc, gate.min_roc_auc),
    ):
        if not math.isfinite(value):
            failures.append(f"{name} is undefined (floor {floor:g})")
        elif value < floor:
            failures.append(f"{name} {value:.3f} < {floor:g}")
    return tuple(failures)


def _compare(challenger: HoldoutMetrics, champion: HoldoutMetrics, tolerance: float) -> tuple[bool, str]:
    rmse_ok = challenger.rmse < champion.rmse
    f1_ok = challenger.clf_f1 >= champion.clf_f1 - tolerance
    text = (
        f"RMSE {challenger.rmse:.3f} vs {champion.rmse:.3f} ({'lower' if rmse_ok else 'not lower'}), "
        f"F1 {challenger.clf_f1:.3f} vs {champion.clf_f1:.3f} "
        f"({'within' if f1_ok else 'beyond'} tolerance {tolerance:g})"
    )
    return rmse_ok and f1_ok, text


def decide_promotion(
    policy: str, challenger: HoldoutMetrics, champion: ChampionOutcome, gate: PromotionSettings
) -> PromotionDecision:
    """Apply the floors, then MODEL_PROMOTION_POLICY, to holdout metrics of the same lifecycles."""
    if policy not in PROMOTION_POLICIES:
        raise ValueError(f"unknown promotion policy {policy!r}")
    failures = floor_failures(challenger, gate)
    if failures:
        return PromotionDecision(
            False, f"challenger misses the quality floors: {'; '.join(failures)}", failures
        )
    if policy == "always":
        return PromotionDecision(True, "policy 'always': floors passed, champion comparison bypassed")
    version = champion.model_version
    if champion.status is ChampionStatus.NONE:
        return PromotionDecision(True, "no production model yet: floors passed")
    if champion.status is ChampionStatus.INCOMPARABLE:
        return PromotionDecision(
            True, f"champion {version} is not comparable ({champion.detail}): promoted on the floors alone"
        )
    if champion.status is ChampionStatus.NO_UNSEEN_HOLDOUT:
        return PromotionDecision(
            False,
            f"champion {version} was fitted on every holdout lifecycle ({champion.detail}), so it cannot be "
            "compared fairly yet: keeping it until a lifecycle it has not seen reaches the holdout",
        )
    holdout = champion.holdout
    if champion.status is not ChampionStatus.COMPARABLE or holdout is None:
        return PromotionDecision(
            False, f"champion {version} could not be re-scored ({champion.detail}): keeping it (fail closed)"
        )
    better, comparison = _compare(champion.challenger_holdout or challenger, holdout, gate.f1_tolerance)
    if champion.detail:
        comparison += f", on {champion.detail}"
    verdict = "challenger is better" if better else "challenger is not better"
    return PromotionDecision(better, f"{verdict} than {version}: {comparison}")
