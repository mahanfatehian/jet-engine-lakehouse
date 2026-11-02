"""The spectral evaluation over several lifecycle splits: scorings, summary, preconditions."""

from __future__ import annotations

import json
import math
import statistics

import pytest
from ds_runtime import PreconditionsNotMetError, StopToken
from evaluate_spectral import (
    METRICS,
    PULSE_FEATURE_COLUMNS,
    SCORINGS,
    SpectralEvaluation,
    SplitResult,
    _parse_args,
    describe,
    evaluate_spectral,
)
from feature_pipeline import FEATURE_COLUMNS, SPECTRAL_FEATURE_COLUMNS
from rul_metrics import HoldoutMetrics
from rul_modeling import SPLIT_SEED, LabelSpec


def metrics(**values: float) -> HoldoutMetrics:
    base = {
        "rows": 100,
        "lifecycles": 2,
        "rmse": 10.0,
        "mae": 8.0,
        "r2": 0.9,
        "late_life_rmse": 12.0,
        "nasa_score": 1.0,
        "nasa_score_sum": 100.0,
        "clf_threshold": 0.5,
        "clf_f1": 0.8,
        "clf_precision": 0.8,
        "clf_recall": 0.8,
        "clf_roc_auc": 0.95,
        "positive_rate": 0.1,
        "clf_brier": 0.03,
    }
    return HoldoutMetrics(**(base | values))


def split(seed: int, with_rmse: float, without_rmse: float, masked_rmse: float = 30.0) -> SplitResult:
    return SplitResult(
        seed=seed,
        test_lifecycles=2,
        test_rows=100,
        scores={
            "with": metrics(rmse=with_rmse, clf_f1=0.9),
            "masked": metrics(rmse=masked_rmse, clf_f1=0.7),
            "without": metrics(rmse=without_rmse, clf_f1=0.85),
        },
    )


def test_pulse_features_are_the_contract_without_the_spectral_ones() -> None:
    assert len(PULSE_FEATURE_COLUMNS) == len(FEATURE_COLUMNS) - len(SPECTRAL_FEATURE_COLUMNS)
    assert not set(PULSE_FEATURE_COLUMNS) & set(SPECTRAL_FEATURE_COLUMNS)
    assert list(PULSE_FEATURE_COLUMNS) == [c for c in FEATURE_COLUMNS if c not in SPECTRAL_FEATURE_COLUMNS]


def test_summary_counts_spectral_wins_in_the_direction_each_metric_improves() -> None:
    evaluation = SpectralEvaluation(
        (split(1, 10.0, 11.0), split(2, 12.0, 11.5), split(3, 9.0, 9.6)),
        lifecycles=30,
        rows=3000,
        stride=2,
        spectral_coverage=0.99,
    )

    assert evaluation.differences("rmse") == pytest.approx([-1.0, 0.5, -0.6])
    assert evaluation.spectral_wins("rmse") == 2, "lower RMSE wins"
    assert evaluation.spectral_wins("clf_f1") == 3, "higher F1 wins: 0.9 vs 0.85 in every split"
    assert evaluation.spectral_wins("r2") == 0, "a tie is no win"
    assert evaluation.mean("masked", "rmse") == pytest.approx(30.0)
    assert evaluation.differences("rmse", against="masked") == pytest.approx([-20.0, -18.0, -21.0])

    document = evaluation.as_dict()
    json.dumps(document, allow_nan=False)
    assert document["seeds"] == [1, 2, 3] and document["stride"] == 2
    assert document["summary"]["rmse"] == {
        "with": pytest.approx(31.0 / 3, abs=1e-4),
        "masked": 30.0,
        "without": pytest.approx(32.1 / 3, abs=1e-4),
        "with_minus_without": pytest.approx(-0.3667, abs=1e-4),
        "with_minus_without_sd": pytest.approx(statistics.stdev([-1.0, 0.5, -0.6]), abs=1e-4),
        "spectral_wins": 2,
    }
    assert set(document["per_split"][0]) == {"seed", "test_lifecycles", "test_rows", *SCORINGS}


def test_one_split_has_no_spread() -> None:
    evaluation = SpectralEvaluation((split(1, 10.0, 11.0),), 9, 900, 1, 1.0)

    assert evaluation.as_dict()["summary"]["rmse"]["with_minus_without_sd"] is None


def test_the_table_has_one_line_per_metric() -> None:
    lines = describe(SpectralEvaluation((split(1, 10.0, 11.0), split(2, 12.0, 11.5)), 30, 3000, 1, 1.0))

    assert lines[0] == "spectral evaluation over 2 split(s) of 30 lifecycles:"
    assert len(lines) == 2 + len(METRICS)
    rmse = next(line for line in lines if line.startswith("rmse "))
    assert rmse.split()[1:4] == ["11.0000", "30.0000", "11.2500"] and rmse.endswith("1 / 2")


def test_evaluates_every_split_on_the_full_procedure_the_masked_holdout_and_the_pulse_features(
    make_mart, frame_source, label_spec: LabelSpec
) -> None:
    mart = make_mart(n_engines=3, lifecycles_per_engine=3, seed=11)
    source = frame_source(mart)

    evaluation = evaluate_spectral(
        source,
        label_spec,
        seeds=(SPLIT_SEED, SPLIT_SEED + 1),
        stride=2,
        max_rows=1_000_000,
        min_lifecycles=6,
        hours_per_cycle=2.0,
        stop=StopToken(),
    )

    assert [result.seed for result in evaluation.splits] == [SPLIT_SEED, SPLIT_SEED + 1]
    assert [window.stride for window in source.windows] == [2], "one load, thinned to every 2nd pulse"
    assert 0 < evaluation.rows < len(mart) and evaluation.lifecycles == 9
    assert 0.0 < evaluation.spectral_coverage <= 1.0
    for result in evaluation.splits:
        assert set(result.scores) == set(SCORINGS) and result.test_lifecycles >= 1
        for scoring in SCORINGS:
            assert all(math.isfinite(result.metric(scoring, name)) for name in ("rmse", "r2", "clf_brier"))
        # The masked scoring uses the same models with every spectral value missing: a different
        # input, so a different estimate than the full one.
        assert result.metric("masked", "rmse") != result.metric("with", "rmse")
    assert evaluation.as_dict()["splits"] == 2


def test_a_planned_stride_above_the_requested_one_is_kept(
    make_mart, frame_source, label_spec: LabelSpec
) -> None:
    mart = make_mart(n_engines=3, lifecycles_per_engine=2, seed=5)
    source = frame_source(mart)
    rows = len(mart)

    evaluate_spectral(
        source,
        label_spec,
        seeds=(SPLIT_SEED,),
        stride=1,
        max_rows=rows // 3,
        min_lifecycles=6,
        hours_per_cycle=2.0,
        stop=StopToken(),
    )

    assert source.windows[0].stride >= 3, "TRAINING_MAX_ROWS still bounds the rows"


def test_too_few_lifecycles_skip(make_mart, frame_source, label_spec: LabelSpec) -> None:
    source = frame_source(make_mart(n_engines=1, lifecycles_per_engine=2, seed=3))

    with pytest.raises(PreconditionsNotMetError, match="2 completed lifecycle"):
        evaluate_spectral(
            source,
            label_spec,
            seeds=(SPLIT_SEED,),
            stride=1,
            max_rows=1_000_000,
            min_lifecycles=1,
            hours_per_cycle=2.0,
            stop=StopToken(),
        )
    assert source.windows == [], "nothing is loaded"


def test_data_without_any_spectral_value_skip(
    make_mart, hf_disabled, frame_source, label_spec: LabelSpec
) -> None:
    source = frame_source(hf_disabled(make_mart(n_engines=3, lifecycles_per_engine=2, seed=4)))

    with pytest.raises(PreconditionsNotMetError, match="no spectral value"):
        evaluate_spectral(
            source,
            label_spec,
            seeds=(SPLIT_SEED,),
            stride=1,
            max_rows=1_000_000,
            min_lifecycles=3,
            hours_per_cycle=2.0,
            stop=StopToken(),
        )


def test_cli_defaults_and_overrides() -> None:
    defaults = _parse_args([])
    assert (defaults.splits, defaults.first_seed, defaults.stride, defaults.min_lifecycles) == (
        5,
        SPLIT_SEED,
        1,
        None,
    )
    chosen = _parse_args(["--splits", "8", "--first-seed", "1", "--stride", "4", "--min-lifecycles", "12"])
    assert (chosen.splits, chosen.first_seed, chosen.stride, chosen.min_lifecycles) == (8, 1, 4, 12)
    with pytest.raises(SystemExit):
        _parse_args(["--splits", "0"])
