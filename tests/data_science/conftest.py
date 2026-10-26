"""Fixtures of the data-science tests: synthetic mart-like data with a planted degradation signal.

Sensor drift grows with a hidden wear w = (cycle / failure_cycle) ** 1.7 and each
lifecycle's failure mode amplifies its signature sensors. Rolling, delta and drift
features follow the dbt definitions (time-based windows per lifecycle, baseline =
mean of the first 10 cycles), so the frames match the feature contract exactly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from artifact_store import LocalArtifactStore
from ds_runtime import StopToken
from feature_pipeline import (
    FEATURE_COLUMNS,
    INFERENCE_FRAME_COLUMNS,
    RAW_SENSOR_COLUMNS,
    TRAINING_FRAME_COLUMNS,
)
from model_registry import ArtifactLayout, ModelRunRecord, RunSummary
from rul_metrics import HoldoutMetrics
from rul_modeling import GroupSplit, LabelSpec, TrainedModel, evaluate_bundle, split_groups, train_models
from train_rul_model import TrainingContext
from training_sources import TrainingFrame, prepare_training_frame
from training_window import LifecycleInventory, TrainingWindow, apply_window, inventory_from_frame

ENGINE_MODELS = ("CFM56-7B", "GE90-115B", "JT9D-7R4")
FAILURE_MODES = ("HPT_BLADE_EROSION", "BEARING_DEGRADATION", "OIL_SYSTEM_LEAK", "COMPRESSOR_FOULING")
# Healthy sensor levels per model, in RAW_SENSOR_COLUMNS order.
BASELINE_LEVELS = {
    "CFM56-7B": (14400.0, 575.0, 650.0, 1.1, 1250.0, 45.0),
    "GE90-115B": (2350.0, 610.0, 700.0, 1.4, 3100.0, 52.0),
    "JT9D-7R4": (3750.0, 540.0, 620.0, 1.8, 2400.0, 40.0),
}
WINDOW_SENSORS = ("vibration_mm_s", "egt_c", "oil_pressure_psi")
WINDOWS = (("5m", 300), ("15m", 900))
BASELINE_CYCLES = 10
RUL_CAP = 125
HORIZON = 25
START = datetime(2026, 1, 1, tzinfo=UTC)
CLICKHOUSE_DDL = Path(__file__).resolve().parents[2] / "infrastructure" / "clickhouse" / "init-db.sql"


def _sensors(rng: np.random.Generator, model: str, mode: str, wear: np.ndarray) -> dict[str, np.ndarray]:
    base = dict(zip(RAW_SENSOR_COLUMNS, BASELINE_LEVELS[model], strict=True))
    boost = {name: float(name == mode) for name in FAILURE_MODES}
    n = wear.size

    def noise(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, n)

    return {
        "rpm": base["rpm"] * (1 - 0.01 * wear - 0.015 * wear * boost["COMPRESSOR_FOULING"])
        + noise(base["rpm"] * 0.002),
        "core_temp_c": base["core_temp_c"] + 30 * wear + 25 * wear * boost["HPT_BLADE_EROSION"] + noise(1.5),
        "egt_c": base["egt_c"]
        + 45 * wear
        + 40 * wear * boost["HPT_BLADE_EROSION"]
        + 20 * wear * boost["COMPRESSOR_FOULING"]
        + noise(2.5),
        "vibration_mm_s": base["vibration_mm_s"]
        * (1 + 0.8 * wear + 1.5 * wear * boost["BEARING_DEGRADATION"])
        + noise(0.04),
        "fuel_flow_kg_h": base["fuel_flow_kg_h"]
        * (1 + 0.03 * wear + 0.04 * wear * boost["COMPRESSOR_FOULING"])
        + noise(base["fuel_flow_kg_h"] * 0.004),
        "oil_pressure_psi": base["oil_pressure_psi"]
        - 4 * wear
        - 8 * wear * boost["OIL_SYSTEM_LEAK"]
        + noise(0.3),
    }


def _add_window_features(lifecycle: pd.DataFrame) -> pd.DataFrame:
    seconds = (lifecycle["event_ts"] - lifecycle["event_ts"].iloc[0]).dt.total_seconds().to_numpy()
    for sensor in WINDOW_SENSORS:
        values = lifecycle[sensor].to_numpy(dtype=np.float64)
        indexed = pd.Series(values, index=pd.DatetimeIndex(lifecycle["event_ts"]))
        for label, span in WINDOWS:
            window = indexed.rolling(f"{span}s", closed="both")
            lifecycle[f"{sensor}_mean_{label}"] = window.mean().to_numpy()
            lifecycle[f"{sensor}_std_{label}"] = window.std(ddof=0).fillna(0.0).to_numpy()
            lifecycle[f"{sensor}_min_{label}"] = window.min().to_numpy()
            lifecycle[f"{sensor}_max_{label}"] = window.max().to_numpy()
            first = np.searchsorted(seconds, seconds - span, side="left")
            elapsed_minutes = (seconds - seconds[first]) / 60.0
            change = values - values[first]
            lifecycle[f"{sensor}_roc_{label}"] = np.divide(
                change, elapsed_minutes, out=np.zeros_like(change), where=elapsed_minutes > 0
            )
        lifecycle[f"{sensor}_delta_1"] = np.diff(values, prepend=values[0])
    baseline = lifecycle.loc[lifecycle["cycle_number"] <= BASELINE_CYCLES, list(RAW_SENSOR_COLUMNS)].mean()
    lifecycle["egt_c_drift"] = lifecycle["egt_c_mean_5m"] - baseline["egt_c"]
    lifecycle["core_temp_c_drift"] = lifecycle["core_temp_c"] - baseline["core_temp_c"]
    lifecycle["vibration_mm_s_ratio"] = lifecycle["vibration_mm_s_mean_5m"] / baseline["vibration_mm_s"]
    lifecycle["oil_pressure_psi_drop"] = baseline["oil_pressure_psi"] - lifecycle["oil_pressure_psi_mean_5m"]
    lifecycle["fuel_flow_kg_h_drift_pct"] = (
        100 * (lifecycle["fuel_flow_kg_h"] - baseline["fuel_flow_kg_h"]) / baseline["fuel_flow_kg_h"]
    )
    lifecycle["rpm_drift_pct"] = 100 * (lifecycle["rpm"] - baseline["rpm"]) / baseline["rpm"]
    return lifecycle


def make_lifecycles(
    *,
    n_engines: int = 4,
    lifecycles_per_engine: int = 4,
    seed: int = 7,
    pulses_per_cycle: int = 2,
    pulse_seconds: int = 10,
    observed_fraction: float | None = None,
) -> pd.DataFrame:
    """Telemetry-level feature rows of synthetic lifecycles.

    With ``observed_fraction`` each lifecycle is cut off at that share of its life
    (an engine still in service); otherwise every lifecycle runs to failure and
    carries the mart's label columns. Ground truth (wear_factor, status) is always
    included so tests can prove it never reaches a model.
    """
    rng = np.random.default_rng(seed)
    frames = []
    telemetry_id = 1
    for engine_index in range(n_engines):
        engine_id = f"ENG-{engine_index + 1:03d}"
        model = ENGINE_MODELS[engine_index % len(ENGINE_MODELS)]
        clock = START + pd.Timedelta(days=engine_index)
        for lifecycle_id in range(1, lifecycles_per_engine + 1):
            failure_cycle = int(rng.integers(90, 171))
            mode = FAILURE_MODES[int(rng.integers(len(FAILURE_MODES)))]
            last_cycle = (
                failure_cycle
                if observed_fraction is None
                else max(20, int(failure_cycle * observed_fraction))
            )
            cycles = np.repeat(np.arange(1, last_cycle + 1), pulses_per_cycle)
            wear = np.clip((cycles / failure_cycle) ** 1.7, 0.0, 1.0)
            lifecycle = pd.DataFrame(
                {
                    "telemetry_id": np.arange(telemetry_id, telemetry_id + cycles.size, dtype=np.uint64),
                    "engine_id": engine_id,
                    "engine_model": model,
                    "lifecycle_id": np.uint32(lifecycle_id),
                    "cycle_number": cycles.astype(np.uint32),
                    "event_ts": pd.date_range(clock, periods=cycles.size, freq=f"{pulse_seconds}s"),
                    **_sensors(rng, model, mode, wear),
                    "wear_factor": wear,
                    "status": np.select(
                        [wear >= 1.0, wear >= 0.8, wear >= 0.5], ["FAILURE", "CRITICAL", "WARNING"], "HEALTHY"
                    ),
                }
            )
            telemetry_id += cycles.size
            clock = lifecycle["event_ts"].iloc[-1] + pd.Timedelta(minutes=30)
            lifecycle = _add_window_features(lifecycle)
            if observed_fraction is None:
                rul = failure_cycle - lifecycle["cycle_number"].astype(np.int64)
                lifecycle["failure_cycle"] = np.uint32(failure_cycle)
                lifecycle["rul_cycles"] = rul.astype(np.int32)
                lifecycle["rul_cycles_capped"] = np.minimum(rul, RUL_CAP).astype(np.int32)
                lifecycle["failure_within_horizon"] = (rul <= HORIZON).astype(np.uint8)
                lifecycle["failure_mode"] = mode
                lifecycle["split_group"] = f"{engine_id}#{lifecycle_id}"
            frames.append(lifecycle)
    return pd.concat(frames, ignore_index=True)


def latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Latest row per engine with an operational_status, shaped like load_inference_frame()."""
    latest = frame.sort_values("event_ts").groupby("engine_id", as_index=False).tail(1)
    inference = latest.loc[:, list(INFERENCE_FRAME_COLUMNS)].reset_index(drop=True)
    inference["operational_status"] = latest["status"].to_numpy()
    return inference


@dataclass(frozen=True)
class SyntheticTraining:
    """One training run on the synthetic mart, shared by several tests."""

    frames: dict[str, pd.DataFrame]
    split: GroupSplit
    trained: TrainedModel
    holdout: HoldoutMetrics


@pytest.fixture(scope="session")
def synthetic_mart() -> pd.DataFrame:
    """Completed lifecycles with every mart column (plus ground truth to be ignored)."""
    frame = make_lifecycles()
    assert set(TRAINING_FRAME_COLUMNS) <= set(frame.columns)
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    return frame


@pytest.fixture(scope="session")
def make_mart() -> Callable[..., pd.DataFrame]:
    """Factory for synthetic lifecycles with custom size / seed."""
    return make_lifecycles


@pytest.fixture(scope="session")
def latest_rows_of() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Turns synthetic lifecycles into an inference frame (latest row per engine)."""
    return latest_rows


@pytest.fixture(scope="session")
def inference_frame() -> pd.DataFrame:
    """Latest feature row of engines that are part-way through a lifecycle."""
    return latest_rows(make_lifecycles(n_engines=6, lifecycles_per_engine=1, seed=99, observed_fraction=0.8))


@pytest.fixture(scope="session")
def label_spec() -> LabelSpec:
    """Contract defaults: cap 125 cycles, horizon 25 cycles."""
    return LabelSpec(rul_cap_cycles=RUL_CAP, failure_horizon_cycles=HORIZON)


@pytest.fixture(scope="session")
def synthetic_training(synthetic_mart: pd.DataFrame, label_spec: LabelSpec) -> SyntheticTraining:
    """Train once on the synthetic mart and score the holdout lifecycles."""
    split = split_groups(synthetic_mart["split_group"].unique())
    frames = {
        side: synthetic_mart[synthetic_mart["split_group"].isin(groups)].reset_index(drop=True)
        for side, groups in (("train", split.train), ("validation", split.validation), ("test", split.test))
    }
    trained = train_models(
        frames["train"],
        frames["validation"],
        label_spec,
        model_version="rul-20260101T000000Z",
        trained_at="2026-01-01T00:00:00+00:00",
        hours_per_cycle=2.0,
    )
    return SyntheticTraining(
        frames, split, trained, evaluate_bundle(trained.bundle, frames["test"], label_spec)
    )


class FrameSource:
    """TrainingDataSource over an in-memory mart frame; records the windows it was asked for."""

    name = "clickhouse"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = prepare_training_frame(frame)
        self.windows: list[TrainingWindow] = []

    def inventory(self) -> LifecycleInventory:
        return inventory_from_frame(self._frame)

    def load(self, window: TrainingWindow) -> TrainingFrame:
        self.windows.append(window)
        return TrainingFrame(apply_window(self._frame, window), self.name, "memory:synthetic")


class MemoryRunRegistry:
    """RunRegistry kept in a list; ``history`` stands in for rows written by earlier runs."""

    def __init__(self, history: RunSummary | None = None) -> None:
        self.history = history
        self.records: list[ModelRunRecord] = []

    def latest_run(self) -> RunSummary | None:
        if not self.records:
            return self.history
        last = max(self.records, key=lambda record: (record.trained_at, record.model_version))
        # Same facts ClickHouseRunRegistry extracts from the row and its params_json.
        return RunSummary(
            last.model_version,
            last.train_lifecycles + last.test_lifecycles,
            last.promoted,
            json.loads(last.params_json).get("data_watermark_us"),
        )

    def record_run(self, record: ModelRunRecord) -> None:
        self.records.append(record)


ARTIFACT_LAYOUT = ArtifactLayout("ml-artifacts/rul_model")


@pytest.fixture
def make_context(label_spec: LabelSpec) -> Callable[..., TrainingContext]:
    """Factory of TrainingContexts over a local store, an in-memory mart and run registry."""

    def build(
        root: Path, frame: pd.DataFrame, runs: MemoryRunRegistry | None = None, store=None
    ) -> TrainingContext:
        return TrainingContext(
            source=FrameSource(frame),
            store=store if store is not None else LocalArtifactStore(root),
            runs=runs if runs is not None else MemoryRunRegistry(),
            layout=ARTIFACT_LAYOUT,
            labels=label_spec,
            hours_per_cycle=2.0,
            stop=StopToken(),
        )

    return build


@pytest.fixture
def run_registry() -> Callable[..., MemoryRunRegistry]:
    """Factory of in-memory run registries (optionally seeded with an earlier run)."""
    return MemoryRunRegistry


@pytest.fixture(scope="session")
def ddl_columns() -> Callable[[str], list[str]]:
    """Column names, in order, of a table in infrastructure/clickhouse/init-db.sql."""
    if not CLICKHOUSE_DDL.is_file():
        pytest.skip("infrastructure/clickhouse/init-db.sql is not available")
    ddl = CLICKHOUSE_DDL.read_text(encoding="utf-8")

    def columns(table: str) -> list[str]:
        match = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\)\s*ENGINE", ddl, re.DOTALL)
        assert match, f"{table} not found in {CLICKHOUSE_DDL}"
        lines = (line.strip() for line in match.group(1).splitlines())
        return [line.split()[0] for line in lines if line and not line.startswith(("INDEX", "--"))]

    return columns
