"""Score the latest feature row of every engine with the production RUL model.

    python data_science/predict_rul.py [--mode batch|live] [--interval-seconds 30]

Reads production.json and its model bundle from MinIO, the latest row per engine
of int_sensor_rolling_features (plus the registry's operational status) from
ClickHouse, and appends one prediction per engine to gold_engine_rul_predictions.
``--mode live`` repeats every ``--interval-seconds`` until SIGTERM/SIGINT.

The bundle is a pickle: its SHA-256 must match the ``bundle_sha256`` production.json
records, checked before anything is unpickled. A pointer without a digest or a bundle
that does not match is refused (exit 1); see model_registry.py for the trust model.

Exit codes: 0 done, 3 skipped (batch: no production model or no feature rows),
1 failure (including a model bundle that fails verification).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from artifact_store import ArtifactStore, S3ArtifactStore
from ds_runtime import (
    EXIT_FAILURE,
    EXIT_OK,
    PreconditionsNotMetError,
    StopToken,
    configure_logging,
    positive_float,
    run_cli,
)
from ds_settings import load_settings
from feature_pipeline import RAW_SENSOR_COLUMNS, load_inference_frame
from lakehouse_io import ClickHouseGateway
from model_registry import (
    ArtifactLayout,
    ModelIntegrityError,
    load_production_bundle,
    read_production_pointer,
)
from rul_modeling import RulModelBundle

PREDICTIONS_TABLE = "gold_engine_rul_predictions"
PREDICTION_COLUMNS: tuple[str, ...] = (
    "predicted_at",
    "engine_id",
    "engine_model",
    "lifecycle_id",
    "cycle_number",
    "telemetry_id",
    "feature_event_ts",
    "operational_status",
    *RAW_SENSOR_COLUMNS,
    "predicted_rul_cycles",
    "predicted_rul_hours",
    "failure_probability",
    "failure_within_horizon",
    "horizon_cycles",
    "risk_band",
    "model_version",
)
_TIMESTAMP_COLUMNS = frozenset({"predicted_at", "feature_event_ts"})
_INTEGER_COLUMNS = frozenset(
    {"lifecycle_id", "cycle_number", "telemetry_id", "failure_within_horizon", "horizon_cycles"}
)
_FLOAT_COLUMNS = frozenset(
    {*RAW_SENSOR_COLUMNS, "predicted_rul_cycles", "predicted_rul_hours", "failure_probability"}
)
RISK_HIGH, RISK_MEDIUM, RISK_LOW = "HIGH", "MEDIUM", "LOW"
UNKNOWN_STATUS = "UNKNOWN"
DEFAULT_INTERVAL_SECONDS = 30.0

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("predict_rul")


def assign_risk_bands(
    rul_cycles: np.ndarray, failure_probability: np.ndarray, threshold: float, horizon_cycles: int
) -> np.ndarray:
    """HIGH: failure expected within the horizon; MEDIUM: approaching it; LOW otherwise."""
    within_horizon = failure_probability >= threshold
    high = within_horizon | (rul_cycles <= horizon_cycles)
    medium = (failure_probability >= threshold / 2) | (rul_cycles <= 2 * horizon_cycles)
    return np.select([high, medium], [RISK_HIGH, RISK_MEDIUM], default=RISK_LOW)


def _operational_status(frame: pd.DataFrame) -> np.ndarray:
    if "operational_status" not in frame.columns:
        return np.full(len(frame), UNKNOWN_STATUS, dtype=object)
    status = frame["operational_status"].fillna("").astype(str).str.strip()
    return status.where(status != "", UNKNOWN_STATUS).to_numpy(dtype=object)


def score_engines(
    bundle: RulModelBundle, frame: pd.DataFrame, *, hours_per_cycle: float, predicted_at: datetime
) -> pd.DataFrame:
    """One prediction row per feature row, with exactly the PREDICTION_COLUMNS."""
    matrix = bundle.feature_matrix(frame)
    rul_cycles = bundle.predict_rul_cycles(matrix)
    probability = np.clip(bundle.predict_failure_probability(matrix), 0.0, 1.0)
    within_horizon = probability >= bundle.decision_threshold
    predictions = pd.DataFrame(
        {
            "predicted_at": [predicted_at] * len(frame),
            "engine_id": frame["engine_id"].astype(str).to_numpy(),
            "engine_model": frame["engine_model"].astype(str).to_numpy(),
            "lifecycle_id": frame["lifecycle_id"].to_numpy(dtype=np.int64),
            "cycle_number": frame["cycle_number"].to_numpy(dtype=np.int64),
            "telemetry_id": frame["telemetry_id"].to_numpy(dtype=np.int64),
            "feature_event_ts": pd.to_datetime(frame["event_ts"], utc=True).array,
            "operational_status": _operational_status(frame),
            **{column: frame[column].to_numpy(dtype=np.float64) for column in RAW_SENSOR_COLUMNS},
            "predicted_rul_cycles": rul_cycles,
            "predicted_rul_hours": rul_cycles * hours_per_cycle,
            "failure_probability": probability,
            "failure_within_horizon": within_horizon.astype(np.int64),
            "horizon_cycles": np.full(len(frame), bundle.failure_horizon_cycles, dtype=np.int64),
            "risk_band": assign_risk_bands(
                rul_cycles, probability, bundle.decision_threshold, bundle.failure_horizon_cycles
            ),
            "model_version": bundle.model_version,
        }
    )
    return predictions.loc[:, list(PREDICTION_COLUMNS)]


def _native(column: str, value: Any) -> Any:
    if column in _TIMESTAMP_COLUMNS:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
        return stamp.to_pydatetime()
    if column in _INTEGER_COLUMNS:
        return int(value)
    if column in _FLOAT_COLUMNS:
        return float(value)
    return str(value)


def prediction_rows(predictions: pd.DataFrame) -> list[list[Any]]:
    """Rows of plain Python values in PREDICTION_COLUMNS order, ready for insertion."""
    columns = [predictions[column].tolist() for column in PREDICTION_COLUMNS]
    return [
        [_native(column, value) for column, value in zip(PREDICTION_COLUMNS, row, strict=True)]
        for row in zip(*columns, strict=True)
    ]


class ProductionModelCache:
    """Holds the verified production bundle in memory; reloads it only when production.json moves."""

    def __init__(self, store: ArtifactStore, layout: ArtifactLayout) -> None:
        self._store = store
        self._layout = layout
        self._bundle: RulModelBundle | None = None
        self._loaded_from: tuple[str, str | None] | None = None

    def current(self) -> RulModelBundle:
        """Return the production bundle.

        Raises PreconditionsNotMetError when no model was promoted yet and
        ModelIntegrityError when the bundle cannot be verified against production.json.
        """
        pointer = read_production_pointer(self._store, self._layout)
        if pointer is None:
            raise PreconditionsNotMetError("no production model yet (production.json does not exist)")
        identity = (pointer.model_version, pointer.bundle_sha256)
        if self._bundle is None or self._loaded_from != identity:
            self._bundle = load_production_bundle(self._store, pointer)
            self._loaded_from = identity
            logger.info(
                "loaded production model %s from %s (bundle sha256 verified)",
                pointer.model_version,
                pointer.artifact_prefix,
            )
        return self._bundle


def predict_once(
    gateway: ClickHouseGateway, models: ProductionModelCache, *, hours_per_cycle: float
) -> pd.DataFrame:
    """Score every engine once and append the predictions; return them."""
    bundle = models.current()
    frame = load_inference_frame(gateway)
    if frame.empty:
        raise PreconditionsNotMetError("int_sensor_rolling_features has no rows yet")
    predictions = score_engines(
        bundle, frame, hours_per_cycle=hours_per_cycle, predicted_at=datetime.now(UTC)
    )
    gateway.insert_rows(PREDICTIONS_TABLE, prediction_rows(predictions), PREDICTION_COLUMNS)
    for row in predictions.itertuples(index=False):
        logger.info(
            "%s cycle %d: RUL %.1f cycles (%.1f h) | P(fail within %d) %.2f | risk %s | status %s",
            row.engine_id,
            row.cycle_number,
            row.predicted_rul_cycles,
            row.predicted_rul_hours,
            row.horizon_cycles,
            row.failure_probability,
            row.risk_band,
            row.operational_status,
        )
    logger.info("wrote %d predictions with model %s", len(predictions), bundle.model_version)
    return predictions


def run_live(
    gateway: ClickHouseGateway,
    models: ProductionModelCache,
    *,
    hours_per_cycle: float,
    interval_seconds: float,
    stop: StopToken,
) -> None:
    """Predict every ``interval_seconds`` until a stop is requested; errors never end the loop."""
    logger.info("live prediction every %.0fs (SIGTERM stops gracefully)", interval_seconds)
    while not stop.stop_requested:
        try:
            predict_once(gateway, models, hours_per_cycle=hours_per_cycle)
        except PreconditionsNotMetError as exc:
            logger.info("nothing scored: %s", exc)
        except ModelIntegrityError as exc:
            logger.error("nothing scored: %s", exc)
        except Exception:
            logger.exception("prediction cycle failed; next attempt in %.0fs", interval_seconds)
        if stop.wait(interval_seconds):
            break
    logger.info("live prediction stopped")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict remaining useful life for every engine with the production model.",
        epilog="Exit codes: 0 done, 3 skipped (no production model or no feature rows), 1 failure.",
    )
    parser.add_argument(
        "--mode",
        choices=("batch", "live"),
        default="batch",
        help="batch: score once and exit; live: score every --interval-seconds (default: batch)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=positive_float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"pause between live scoring rounds (default: {DEFAULT_INTERVAL_SECONDS:.0f})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``predict_rul.py``."""
    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    stop = StopToken().install_signal_handlers()
    models = ProductionModelCache(
        S3ArtifactStore(settings.s3), ArtifactLayout(settings.model.artifact_prefix)
    )
    hours_per_cycle = settings.model.flight_hours_per_cycle
    with ClickHouseGateway(settings.clickhouse) as gateway:
        if args.mode == "live":
            run_live(
                gateway,
                models,
                hours_per_cycle=hours_per_cycle,
                interval_seconds=args.interval_seconds,
                stop=stop,
            )
        else:
            try:
                predict_once(gateway, models, hours_per_cycle=hours_per_cycle)
            except ModelIntegrityError as exc:
                logger.error("refusing to serve the production model: %s", exc)
                return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    run_cli(main)
