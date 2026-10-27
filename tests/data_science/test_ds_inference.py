"""Inference feature vectors: the int_ml_features query and the carry-forward of lagging spectral values."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from ds_runtime import PreconditionsNotMetError
from feature_pipeline import INFERENCE_FRAME_COLUMNS, SPECTRAL_FEATURE_COLUMNS
from inference_features import (
    INFERENCE_SOURCE_TABLE,
    LIFECYCLES_TABLE,
    REGISTRY_TABLE,
    SpectralFreshness,
    carry_forward_spectral,
    load_inference_frame,
)
from predict_rul import score_engines
from training_window import epoch_micros

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
VIEW_ENGINES = {
    INFERENCE_SOURCE_TABLE: "View",
    LIFECYCLES_TABLE: "ReplacingMergeTree",
    REGISTRY_TABLE: "ReplacingMergeTree",
}


class ViewGateway:
    """ClickHouse stand-in: answers the newest-pulse query and the donor query, records both."""

    def __init__(
        self, latest: pd.DataFrame, donors: pd.DataFrame | None = None, engines: dict[str, str] | None = None
    ) -> None:
        self.latest = latest
        self.donors = latest.iloc[0:0] if donors is None else donors
        self.engines = VIEW_ENGINES if engines is None else engines
        self.queries: list[tuple[str, dict]] = []

    def table_engines(self, tables):
        return {table: engine for table, engine in self.engines.items() if table in set(tables)}

    def query_df(self, sql, parameters=None):
        self.queries.append((sql, dict(parameters or {})))
        return (self.donors if "isFinite" in sql else self.latest).copy()


def _scenario(inference_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six engines; 0: frame landed (one value missing), 1-5: no frame yet, whose donors are
    1: 10 s older, 2: from the previous lifecycle, 3: 120 s older, 4: 5 s older and missing
    its last value, 5: none."""
    latest = inference_frame.copy()
    spectral = list(SPECTRAL_FEATURE_COLUMNS)
    donors = latest.loc[:, ["engine_id", "lifecycle_id", "event_ts", *spectral]].copy()
    donors[spectral] = donors[spectral] + 1.0
    donors["event_ts"] = latest["event_ts"] - pd.to_timedelta([0, 10, 5, 120, 5, 0], unit="s")
    donors.loc[2, "lifecycle_id"] = latest.loc[2, "lifecycle_id"] - 1
    donors.loc[4, spectral[-1]] = np.nan
    latest.loc[0, spectral[-1]] = np.nan
    latest.loc[1:5, spectral] = np.nan
    donors = donors.drop(index=[0, 5]).reset_index(drop=True)
    return latest, donors


def test_inference_reads_the_newest_pulse_of_int_ml_features_with_the_registry_status(
    inference_frame: pd.DataFrame,
) -> None:
    gateway = ViewGateway(inference_frame)

    frame = load_inference_frame(gateway)

    [(sql, parameters)] = gateway.queries
    assert "FROM `int_ml_features`\n" in sql and "`int_ml_features` FINAL" not in sql
    assert "ORDER BY event_ts DESC, telemetry_id DESC" in sql and "LIMIT 1 BY engine_id" in sql
    assert "WHERE (engine_id, lifecycle_id) IN (" in sql and "FROM `int_engine_lifecycles` FINAL" in sql
    assert "ORDER BY engine_id, lifecycle_id DESC" in sql and "LIMIT 2 BY engine_id" in sql
    assert "argMax(operational_status, cdc_lsn)" in sql and f"FROM `{REGISTRY_TABLE}`" in sql
    assert all(f"`{column}` AS `{column}`" in sql for column in INFERENCE_FRAME_COLUMNS)
    assert "int_sensor_rolling_features" not in sql and parameters == {}
    pd.testing.assert_frame_equal(frame, inference_frame)


def test_a_materialized_int_ml_features_is_read_with_final(inference_frame: pd.DataFrame) -> None:
    gateway = ViewGateway(
        inference_frame, engines={**VIEW_ENGINES, INFERENCE_SOURCE_TABLE: "ReplacingMergeTree"}
    )

    load_inference_frame(gateway)

    assert "FROM `int_ml_features` FINAL" in gateway.queries[0][0]


@pytest.mark.parametrize("absent", [INFERENCE_SOURCE_TABLE, LIFECYCLES_TABLE])
def test_inference_skips_until_its_dbt_models_exist(inference_frame: pd.DataFrame, absent: str) -> None:
    engines = {table: engine for table, engine in VIEW_ENGINES.items() if table != absent}
    gateway = ViewGateway(inference_frame, engines=engines)

    with pytest.raises(PreconditionsNotMetError, match=absent):
        load_inference_frame(gateway)
    assert gateway.queries == []


def test_lagging_spectral_values_are_carried_forward_within_the_lifecycle_and_the_lag(
    inference_frame: pd.DataFrame, caplog
) -> None:
    latest, donors = _scenario(inference_frame)
    gateway = ViewGateway(latest, donors)

    with caplog.at_level(logging.INFO, logger="inference_features"):
        frame = load_inference_frame(gateway, spectral_max_lag_seconds=60)

    spectral = list(SPECTRAL_FEATURE_COLUMNS)
    expected = inference_frame[spectral].to_numpy(dtype=float)
    values = frame[spectral].to_numpy(dtype=float)
    # A landed frame is scored as it is: its missing value is a measurement gap, not lag.
    np.testing.assert_allclose(values[0][:-1], expected[0][:-1])
    assert np.isnan(values[0][-1])
    np.testing.assert_allclose(values[1], expected[1] + 1.0)  # the donor 10 s earlier
    np.testing.assert_allclose(values[4][:-1], expected[4][:-1] + 1.0)  # the donor 5 s earlier ...
    assert np.isnan(values[4][-1])  # ... whose missing value stays missing
    assert np.isnan(values[2]).all() and np.isnan(values[3]).all() and np.isnan(values[5]).all()
    # The scored pulse never changes: identifiers and pulse features stay the newest ones.
    pd.testing.assert_frame_equal(frame.drop(columns=spectral), latest.drop(columns=spectral))
    assert "1 current, 2 carried forward (oldest 10.0 s), 3 missing" in caplog.text
    [_, (donor_sql, parameters)] = gateway.queries
    assert parameters["engines"] == sorted(latest.loc[1:5, "engine_id"])
    lagging_since = min(epoch_micros(stamp) for stamp in latest.loc[1:5, "event_ts"])
    assert parameters["since_us"] == lagging_since - 60_000_000
    landed = " OR ".join(f"isFinite(`{column}`)" for column in SPECTRAL_FEATURE_COLUMNS)
    assert f"AND ({landed})" in donor_sql
    assert (
        "fromUnixTimestamp64Micro({since_us:Int64}, 'UTC')" in donor_sql
        and "LIMIT 1 BY engine_id" in donor_sql
    )


def test_carry_forward_reports_where_each_value_came_from(inference_frame: pd.DataFrame) -> None:
    latest, donors = _scenario(inference_frame)

    _, freshness = carry_forward_spectral(latest, donors, max_lag_seconds=60)
    _, longer = carry_forward_spectral(latest, donors, max_lag_seconds=300)
    _, disabled = carry_forward_spectral(latest, donors, max_lag_seconds=0)

    assert freshness == SpectralFreshness(current=1, carried=2, missing=3, max_carried_lag_seconds=10.0)
    assert (longer.carried, longer.missing, longer.max_carried_lag_seconds) == (3, 2, 120.0)
    assert disabled == SpectralFreshness(current=1, carried=0, missing=5)
    assert "1 current, 2 carried forward" in freshness.describe(60)


def test_no_donor_query_runs_when_nothing_lags_or_the_carry_forward_is_off(
    inference_frame: pd.DataFrame,
) -> None:
    latest, donors = _scenario(inference_frame)
    current = ViewGateway(inference_frame, donors)
    off = ViewGateway(latest, donors)

    load_inference_frame(current, spectral_max_lag_seconds=60)
    frame = load_inference_frame(off, spectral_max_lag_seconds=0)

    assert len(current.queries) == len(off.queries) == 1
    assert frame[list(SPECTRAL_FEATURE_COLUMNS)].isna().all(axis=1).sum() == 5


def test_donor_values_may_come_back_as_nullable_extension_dtypes(inference_frame: pd.DataFrame) -> None:
    # clickhouse-connect returns Nullable(Float64) columns as pandas Float64 with pd.NA.
    latest, donors = _scenario(inference_frame)
    for column in SPECTRAL_FEATURE_COLUMNS:
        latest[column] = latest[column].astype("Float64")
        donors[column] = donors[column].astype("Float64")

    frame, freshness = carry_forward_spectral(latest, donors, max_lag_seconds=60)

    assert (freshness.carried, freshness.missing) == (2, 3)
    assert frame["fan_amp_bpfo_mean_5m"].dtype == np.float64


def test_carrying_spectral_features_forward_keeps_the_bearing_evidence(
    synthetic_training, make_mart, latest_rows_of, hf_disabled
) -> None:
    bundle = synthetic_training.trained.bundle
    worn = latest_rows_of(
        make_mart(
            n_engines=3,
            lifecycles_per_engine=1,
            seed=5,
            observed_fraction=0.95,
            modes=("BEARING_DEGRADATION",),
        )
    )
    lagging = hf_disabled(worn)
    donors = worn.loc[:, ["engine_id", "lifecycle_id", "event_ts", *SPECTRAL_FEATURE_COLUMNS]].assign(
        event_ts=worn["event_ts"] - pd.Timedelta(seconds=10)
    )

    complete = score_engines(bundle, worn, hours_per_cycle=2.0, predicted_at=NOW)
    blind = score_engines(bundle, lagging, hours_per_cycle=2.0, predicted_at=NOW)
    carried = score_engines(
        bundle, load_inference_frame(ViewGateway(lagging, donors)), hours_per_cycle=2.0, predicted_at=NOW
    )

    # Scored as missing, the newest pulse of a failing bearing looks healthy: a late warning.
    assert (blind["predicted_rul_cycles"] > complete["predicted_rul_cycles"] + 20).all()
    assert set(complete["risk_band"]) == {"HIGH"} and set(blind["risk_band"]) != {"HIGH"}
    pd.testing.assert_frame_equal(carried, complete)
