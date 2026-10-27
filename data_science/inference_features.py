"""The feature vectors predict_rul.py scores: the newest pulse of every engine.

Rows come from the dbt view int_ml_features (pulse and spectral features of every pulse),
newest pulse per engine (``LIMIT 1 BY engine_id``) of its newest lifecycles in
int_engine_lifecycles, joined with the engine's latest registry status.

Spectral features lag the pulse: HF frames reach ClickHouse on their own micro-batch
cadence, so when dbt builds the view's sources the newest pulses of an engine usually have
no HF frame yet, and the view gives them no spectral value at all. The models learned
their missing-value branches from the few pulses that were missing in training, so scoring
every newest pulse that way would silently drop the spectral evidence exactly where it
matters. The spectral features of such a pulse are therefore carried forward from the
engine's newest pulse whose frame has landed (an as-of join), provided that pulse belongs
to the same lifecycle and is at most SPECTRAL_MAX_LAG_SECONDS older; the features are
5-minute means, so seconds of lag barely move them. Beyond that (HF disabled or down) they
stay missing and the models' missing-value branches decide. A pulse whose frame landed is
scored as it is, also when one of its values is missing (a measurement gap, not lag), and
the scored pulse itself never changes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from ds_runtime import PreconditionsNotMetError
from feature_pipeline import INFERENCE_FRAME_COLUMNS, SPECTRAL_FEATURE_COLUMNS, column_list
from lakehouse_io import final_modifier
from training_window import epoch_micros

INFERENCE_SOURCE_TABLE = "int_ml_features"
LIFECYCLES_TABLE = "int_engine_lifecycles"
REGISTRY_TABLE = "raw_engine_status_registry"
DEFAULT_SPECTRAL_MAX_LAG_SECONDS = 60.0
_DONOR_COLUMNS = ("engine_id", "lifecycle_id", "event_ts", *SPECTRAL_FEATURE_COLUMNS)

logger = logging.getLogger(__name__)


class InferenceGateway(Protocol):
    """The ClickHouse capabilities the inference loader needs (ClickHouseGateway satisfies it)."""

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        """Map existing tables to their engine names."""
        ...

    def query_df(self, sql: str, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Run a SELECT and return the result as a DataFrame."""
        ...


@dataclass(frozen=True)
class SpectralFreshness:
    """Where the spectral features of the scored pulses came from."""

    current: int  # the newest pulse's own HF frame had landed
    carried: int  # taken from an earlier pulse of the lifecycle within the lag limit
    missing: int  # left missing: HF disabled, down or lagging beyond the limit
    max_carried_lag_seconds: float = 0.0

    def describe(self, max_lag_seconds: float) -> str:
        """One log line for the scoring round."""
        return (
            f"spectral features of the scored pulses: {self.current} current, {self.carried} carried "
            f"forward (oldest {self.max_carried_lag_seconds:.1f} s), {self.missing} missing "
            f"(HF disabled or lagging more than {max_lag_seconds:g} s)"
        )


def _as_float(values: pd.Series) -> np.ndarray:
    """A writable float64 copy of ``values`` with every missing marker (None, pd.NA, NaN) as NaN."""
    return values.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def without_hf_frame(frame: pd.DataFrame) -> np.ndarray:
    """Per row: True when no spectral feature is present, i.e. the pulse's HF frame has not landed."""
    present = np.column_stack([np.isfinite(_as_float(frame[column])) for column in SPECTRAL_FEATURE_COLUMNS])
    return ~present.any(axis=1)


def latest_rows_sql(source: str, lifecycles: str) -> str:
    """Newest pulse of every engine from ``source`` with the registry's latest status.

    The scan is bounded by the sorting-key prefix (engine_id, lifecycle_id) to each engine's
    two newest lifecycles in ``lifecycles``, instead of the whole feature history every round.
    The newest pulse is in the newest lifecycle, or in the one before while a dbt run has
    already registered an engine reset in int_engine_lifecycles but not yet built the new
    lifecycle's features.
    """
    return f"""
        SELECT {column_list(INFERENCE_FRAME_COLUMNS, "f")}, r.operational_status AS operational_status
        FROM (
            SELECT {column_list(INFERENCE_FRAME_COLUMNS)}
            FROM {source}
            WHERE (engine_id, lifecycle_id) IN (
                SELECT engine_id, lifecycle_id
                FROM {lifecycles}
                ORDER BY engine_id, lifecycle_id DESC
                LIMIT 2 BY engine_id
            )
            ORDER BY event_ts DESC, telemetry_id DESC
            LIMIT 1 BY engine_id
        ) AS f
        LEFT JOIN (
            SELECT engine_id, argMax(operational_status, cdc_lsn) AS operational_status
            FROM `{REGISTRY_TABLE}`
            WHERE is_deleted = 0
            GROUP BY engine_id
        ) AS r ON r.engine_id = f.engine_id
        ORDER BY engine_id
    """


def latest_frame_query(
    source: str, lagging: pd.DataFrame, max_lag_seconds: float
) -> tuple[str, dict[str, Any]]:
    """Newest pulse whose HF frame landed (any spectral value present), per engine of ``lagging``.

    Only pulses within ``max_lag_seconds`` of the oldest lagging pulse are read; whether
    a donor is close enough for its own engine is decided by carry_forward_spectral.
    """
    since_us = min(epoch_micros(stamp) for stamp in lagging["event_ts"]) - int(max_lag_seconds * 1e6)
    landed = " OR ".join(f"isFinite(`{column}`)" for column in SPECTRAL_FEATURE_COLUMNS)
    sql = f"""
        SELECT {column_list(_DONOR_COLUMNS)}
        FROM {source}
        WHERE engine_id IN {{engines:Array(String)}}
          AND event_ts >= fromUnixTimestamp64Micro({{since_us:Int64}}, 'UTC')
          AND ({landed})
        ORDER BY event_ts DESC, telemetry_id DESC
        LIMIT 1 BY engine_id
    """
    engines = sorted({str(engine) for engine in lagging["engine_id"]})
    return sql, {"engines": engines, "since_us": since_us}


def carry_forward_spectral(
    latest: pd.DataFrame, donors: pd.DataFrame, *, max_lag_seconds: float
) -> tuple[pd.DataFrame, SpectralFreshness]:
    """Give pulses without an HF frame the spectral features of their engine's donor pulse.

    The donor (the engine's newest pulse whose frame landed) is used when it belongs to the
    same lifecycle and is between 0 and ``max_lag_seconds`` older; its values are taken as
    they are, a value missing in the donor stays missing.
    """
    frame = latest.copy()
    lagging = without_hf_frame(frame)
    if not lagging.any() or donors.empty or max_lag_seconds <= 0:
        return frame, SpectralFreshness(int((~lagging).sum()), 0, int(lagging.sum()))
    matched = donors.drop_duplicates("engine_id").set_index("engine_id").reindex(frame["engine_id"])
    lag_seconds = (
        pd.to_datetime(frame["event_ts"], utc=True).to_numpy()
        - pd.to_datetime(matched["event_ts"], utc=True).to_numpy()
    ) / np.timedelta64(1, "s")
    same_lifecycle = _as_float(matched["lifecycle_id"]) == _as_float(frame["lifecycle_id"])
    with np.errstate(invalid="ignore"):
        within = (lag_seconds >= 0) & (lag_seconds <= max_lag_seconds)
    eligible = lagging & same_lifecycle & within
    for column in SPECTRAL_FEATURE_COLUMNS:
        values = _as_float(frame[column])
        values[eligible] = _as_float(matched[column])[eligible]
        frame[column] = values
    carried = int(eligible.sum())
    freshness = SpectralFreshness(
        current=int((~lagging).sum()),
        carried=carried,
        missing=int(lagging.sum()) - carried,
        max_carried_lag_seconds=float(lag_seconds[eligible].max()) if carried else 0.0,
    )
    return frame, freshness


def load_inference_frame(
    gateway: InferenceGateway, *, spectral_max_lag_seconds: float = DEFAULT_SPECTRAL_MAX_LAG_SECONDS
) -> pd.DataFrame:
    """Newest pulse of every engine with its registry status, lagging spectral features carried forward.

    ``operational_status`` is empty for engines without a registry row. Raises
    PreconditionsNotMetError while int_ml_features, int_engine_lifecycles or the registry
    does not exist.
    """
    required = (INFERENCE_SOURCE_TABLE, LIFECYCLES_TABLE, REGISTRY_TABLE)
    engines = gateway.table_engines(required)
    missing = [table for table in required if table not in engines]
    if missing:
        raise PreconditionsNotMetError(f"tables do not exist yet: {', '.join(missing)}")
    source = f"`{INFERENCE_SOURCE_TABLE}` {final_modifier(engines[INFERENCE_SOURCE_TABLE])}".rstrip()
    lifecycles = f"`{LIFECYCLES_TABLE}` {final_modifier(engines[LIFECYCLES_TABLE])}".rstrip()
    latest = gateway.query_df(latest_rows_sql(source, lifecycles))
    if latest.empty:
        return latest
    lagging = without_hf_frame(latest)
    if lagging.any() and spectral_max_lag_seconds > 0:
        sql, parameters = latest_frame_query(source, latest[lagging], spectral_max_lag_seconds)
        donors = gateway.query_df(sql, parameters)
    else:
        donors = pd.DataFrame(columns=list(_DONOR_COLUMNS))
    frame, freshness = carry_forward_spectral(latest, donors, max_lag_seconds=spectral_max_lag_seconds)
    logger.info("%s", freshness.describe(spectral_max_lag_seconds))
    return frame
