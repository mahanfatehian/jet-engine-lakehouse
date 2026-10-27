"""Bounded training data: lifecycle inventory, window planning, thinning and the training queries."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from feature_pipeline import SPECTRAL_FEATURE_COLUMNS
from train_rul_model import TrainingRequest, run_training
from training_sources import ClickHouseTrainingSource, training_select_sql
from training_window import (
    LifecycleInventory,
    LifecycleStats,
    apply_window,
    epoch_micros,
    format_micros,
    inventory_from_frame,
    plan_training_window,
)

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def _inventory(*sizes: int) -> LifecycleInventory:
    """Lifecycles L0, L1, ... of the given sizes, each ending later than the previous one."""
    return LifecycleInventory(
        tuple(LifecycleStats(f"L{index}", rows, 1_000 * (index + 1)) for index, rows in enumerate(sizes))
    )


def test_inventory_summarises_each_lifecycle(synthetic_mart: pd.DataFrame) -> None:
    inventory = inventory_from_frame(synthetic_mart)

    sizes = synthetic_mart.groupby("split_group").size()
    assert inventory.completed_lifecycles == len(sizes)
    assert inventory.total_rows == len(synthetic_mart)
    assert {stats.split_group: stats.rows for stats in inventory.lifecycles} == sizes.to_dict()
    assert inventory.watermark_us == epoch_micros(synthetic_mart["event_ts"].max())
    assert inventory_from_frame(synthetic_mart.iloc[0:0]).watermark_us is None


def test_epoch_micros_treats_naive_timestamps_as_utc() -> None:
    aware = epoch_micros(pd.Timestamp("2026-10-26T12:00:00.123456Z"))

    assert aware == epoch_micros(datetime(2026, 10, 26, 12, 0, 0, 123456))
    assert format_micros(aware) == "2026-10-26T12:00:00.123456+00:00"
    assert format_micros(None) is None


def test_window_keeps_everything_that_fits() -> None:
    window = plan_training_window(_inventory(100, 100, 100), max_rows=300, min_lifecycles=3)

    assert window.is_complete and window.groups == ("L0", "L1", "L2") and window.stride == 1
    assert window.rows == 300 and window.dropped_rows == 0
    assert "all 3 completed lifecycles" in window.describe()


def test_window_keeps_the_newest_lifecycles_that_fit() -> None:
    window = plan_training_window(_inventory(100, 100, 100, 100, 100), max_rows=350, min_lifecycles=3)

    assert window.groups == ("L2", "L3", "L4") and window.stride == 1
    assert (window.rows, window.dropped_lifecycles, window.dropped_rows) == (300, 2, 200)
    assert window.as_dict() == {
        "max_rows": 350,
        "source_lifecycles": 5,
        "source_rows": 500,
        "lifecycles": 3,
        "rows": 300,
        "stride": 1,
        "dropped_lifecycles": 2,
        "dropped_rows": 200,
    }
    assert "dropping 2 older lifecycles" in window.describe()


def test_window_stops_at_the_first_lifecycle_that_does_not_fit() -> None:
    # The newest lifecycle that overflows ends the window, so it stays a contiguous recent range.
    window = plan_training_window(_inventory(10, 10, 10, 500, 10, 10, 10), max_rows=100, min_lifecycles=3)

    assert window.groups == ("L4", "L5", "L6")


def test_window_thins_pulses_when_the_minimum_lifecycles_do_not_fit() -> None:
    window = plan_training_window(_inventory(1000, 1000, 1000, 1000), max_rows=1000, min_lifecycles=3)

    # Stride 3 would keep 3 * 334 = 1002 rows, one lifecycle-rounding too many.
    assert window.groups == ("L1", "L2", "L3")
    assert window.stride == 4 and window.rows == 3 * 250
    assert "1 in 4 pulses" in window.describe()


def test_window_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        plan_training_window(_inventory(1), max_rows=0, min_lifecycles=1)


def test_applying_a_window_selects_groups_and_every_nth_pulse(synthetic_mart: pd.DataFrame) -> None:
    inventory = inventory_from_frame(synthetic_mart)
    window = plan_training_window(inventory, max_rows=len(synthetic_mart) // 10, min_lifecycles=4)

    selected = apply_window(synthetic_mart, window)

    assert window.stride > 1 and len(selected) == window.rows <= len(synthetic_mart) // 10
    assert set(selected["split_group"]) == set(window.groups)
    first = synthetic_mart[synthetic_mart["split_group"] == window.groups[0]].sort_values("event_ts")
    expected = first["telemetry_id"].iloc[:: window.stride].tolist()
    assert selected[selected["split_group"] == window.groups[0]]["telemetry_id"].tolist() == expected


@pytest.mark.parametrize("budget_share", [1.0, 0.1])
def test_windows_keep_pulses_whose_spectral_features_are_missing(
    synthetic_mart: pd.DataFrame, budget_share: float
) -> None:
    # The newest lifecycle is always in the window: give it no HF data at all.
    newest = synthetic_mart.groupby("split_group")["event_ts"].max().idxmax()
    frame = synthetic_mart.copy()
    frame.loc[frame["split_group"] == newest, list(SPECTRAL_FEATURE_COLUMNS)] = np.nan
    window = plan_training_window(
        inventory_from_frame(frame), max_rows=int(len(frame) * budget_share), min_lifecycles=4
    )

    selected = apply_window(frame, window)

    assert newest in window.groups and len(selected) == window.rows
    expected = frame.set_index("telemetry_id").loc[selected["telemetry_id"], list(SPECTRAL_FEATURE_COLUMNS)]
    pd.testing.assert_frame_equal(
        selected[list(SPECTRAL_FEATURE_COLUMNS)].reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert (
        selected.loc[selected["split_group"] == newest, list(SPECTRAL_FEATURE_COLUMNS)].isna().all(axis=None)
    )


def test_training_query_selects_the_spectral_features() -> None:
    window = plan_training_window(_inventory(100, 100, 100), max_rows=300, min_lifecycles=3)

    sql, _ = training_select_sql("`mart_rul_training_features` FINAL", window)

    assert all(f"`{column}` AS `{column}`" in sql for column in SPECTRAL_FEATURE_COLUMNS)


def test_training_query_filters_lifecycles_and_thins_inside_clickhouse() -> None:
    window = plan_training_window(_inventory(1000, 1000, 1000, 1000), max_rows=1000, min_lifecycles=3)
    plain = plan_training_window(_inventory(100, 100, 100, 100), max_rows=300, min_lifecycles=3)

    sql, parameters = training_select_sql("`mart_rul_training_features` FINAL", window)
    plain_sql, plain_parameters = training_select_sql("`mart_rul_training_features` FINAL", plain)

    assert parameters == {"groups": ["L1", "L2", "L3"], "stride": 4}
    assert "split_group IN {groups:Array(String)}" in sql and "% {stride:UInt32} = 0" in sql
    assert "PARTITION BY split_group ORDER BY event_ts, telemetry_id" in sql
    assert plain_parameters == {"groups": ["L1", "L2", "L3"]} and "stride" not in plain_sql
    assert plain_sql.startswith("SELECT `telemetry_id` AS `telemetry_id`")


class FakeMartGateway:
    """ClickHouse stand-in serving the inventory query and capturing the load query."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.queries: list[tuple[str, dict]] = []

    def table_engines(self, tables):
        return {table: "ReplacingMergeTree" for table in tables}

    def query_rows(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        return [
            (stats.split_group, stats.rows, stats.last_event_us)
            for stats in inventory_from_frame(self.frame).lifecycles
        ]

    def query_df(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        return self.frame[self.frame["split_group"].isin(parameters["groups"])]


def test_clickhouse_source_reads_the_inventory_then_only_the_window(synthetic_mart: pd.DataFrame) -> None:
    gateway = FakeMartGateway(synthetic_mart)
    source = ClickHouseTrainingSource(gateway)

    inventory = source.inventory()
    window = plan_training_window(inventory, max_rows=len(synthetic_mart) // 2, min_lifecycles=3)
    loaded = source.load(window)

    inventory_sql = gateway.queries[0][0]
    assert "toUnixTimestamp64Micro(max(event_ts))" in inventory_sql and "FINAL" in inventory_sql
    assert inventory == inventory_from_frame(synthetic_mart)
    assert gateway.queries[1][1]["groups"] == list(window.groups)
    assert set(loaded.frame["split_group"]) == set(window.groups)
    assert loaded.source_uri == "clickhouse:mart_rul_training_features"


def test_training_run_honours_the_row_budget_and_records_the_window(
    tmp_path: Path, synthetic_mart: pd.DataFrame, make_context, caplog
) -> None:
    budget = int(len(synthetic_mart) * 0.7)
    context = make_context(tmp_path, synthetic_mart)
    request = TrainingRequest(
        min_lifecycles=6,
        promotion_policy="better",
        skip_if_unchanged=False,
        dry_run=False,
        max_training_rows=budget,
    )

    with caplog.at_level(logging.INFO, logger="train_rul_model"):
        outcome = run_training(request, context, now=T0)

    [window] = context.source.windows
    assert window.dropped_lifecycles > 0 and window.stride == 1 and window.rows <= budget
    [record] = context.runs.records
    assert record.train_rows + record.test_rows == window.rows
    assert record.train_lifecycles + record.test_lifecycles == len(window.groups)
    params = json.loads(record.params_json)
    assert params["training_window"] == window.as_dict()
    # The watermark covers every completed lifecycle, also those outside the window.
    assert params["data_watermark_us"] == epoch_micros(synthetic_mart["event_ts"].max())
    assert f"dropping {window.dropped_lifecycles} older lifecycles" in caplog.text
    assert outcome.artifact_uri
