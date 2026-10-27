"""Pure-Python ClickHouse rows of the HF feature query (no Spark, no ClickHouse).

The query collects ``raw_engine_hf_features`` rows in ``FEATURE_COLUMNS`` order with
``frame_start_ts`` as epoch microseconds: ``collect()`` would render a TimestampType in the
Python process' local timezone (see ``clickhouse_rows``). :func:`feature_rows` rebuilds the
exact, timezone-aware UTC instant and restores the name order of ``process_means``, which
``collect()`` loses (maps travel through a Java HashMap). :func:`batch_row` summarises a
micro-batch for ``hf_ingest_batches``, the throughput log of the HF path.

An undefined feature (a flat-line or missing channel, a NaN sample) is NaN in ``hf_features``,
but PySpark's Arrow serializer of ``mapInPandas`` stores every NaN as NULL, inside the
``process_means`` map too. The ClickHouse columns are non-Nullable Float32, where a NULL fails
the whole insert and the replayed batch with it, so :func:`feature_rows` turns NULL floats back
into NaN (and a NULL map into an empty one); dbt's staging reads NaN as missing again. The
silver Parquet sink keeps the NULLs: its columns are nullable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from clickhouse_rows import Row, convert_row
from hf_features import FEATURE_COLUMN_NAMES, FEATURE_COLUMNS

HF_FEATURES_TABLE = "raw_engine_hf_features"
HF_BATCHES_TABLE = "hf_ingest_batches"
HF_BATCH_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "batch_started_at",
    "frames",
    "bytes",
    "engines",
    "min_frame_ts",
    "max_frame_ts",
    "duration_ms",
)
UINT16_MAX = 65_535

_FRAME_TS = FEATURE_COLUMN_NAMES.index("frame_start_ts")
_ENGINE_ID = FEATURE_COLUMN_NAMES.index("engine_id")
_FRAME_BYTES = FEATURE_COLUMN_NAMES.index("frame_bytes")
_PROCESS_MEANS = FEATURE_COLUMN_NAMES.index("process_means")
_FLOAT_POSITIONS = frozenset(
    position for position, (_, kind) in enumerate(FEATURE_COLUMNS) if kind == "float"
)
NAN = float("nan")


@dataclass(frozen=True, slots=True)
class BatchStats:
    """Volume and frame-time span of the feature rows of one micro-batch."""

    frames: int
    bytes: int
    engines: int
    min_frame_ts: datetime
    max_frame_ts: datetime


def feature_rows(collected: Iterable[Sequence[Any]]) -> list[Row]:
    """Insert rows for ``raw_engine_hf_features``: aware UTC ``frame_start_ts``, sorted map keys.

    An undefined float is NaN, never NULL, inside ``process_means`` too (see the module docstring).
    """
    positions = frozenset({_FRAME_TS})
    return [_process_means_as_inserted(_nan_for_null(convert_row(values, positions))) for values in collected]


def batch_stats(rows: Sequence[Row]) -> BatchStats | None:
    """Frames, Kafka value bytes, distinct engines and frame-time span; None without rows.

    ``engines`` saturates at the UInt16 maximum of its column.
    """
    if not rows:
        return None
    instants = [row[_FRAME_TS] for row in rows]
    return BatchStats(
        frames=len(rows),
        bytes=sum(row[_FRAME_BYTES] for row in rows),
        engines=min(len({row[_ENGINE_ID] for row in rows}), UINT16_MAX),
        min_frame_ts=min(instants),
        max_frame_ts=max(instants),
    )


def batch_row(batch_id: int, started_at: datetime, stats: BatchStats, duration_s: float) -> Row:
    """One ``hf_ingest_batches`` row in ``HF_BATCH_COLUMNS`` order."""
    return (
        batch_id,
        started_at,
        stats.frames,
        stats.bytes,
        stats.engines,
        stats.min_frame_ts,
        stats.max_frame_ts,
        max(round(duration_s * 1000), 0),
    )


def _nan_for_null(row: Row) -> Row:
    return tuple(
        NAN if value is None and position in _FLOAT_POSITIONS else value for position, value in enumerate(row)
    )


def _process_means_as_inserted(row: Row) -> Row:
    means = row[_PROCESS_MEANS] or {}
    ordered = {name: NAN if mean is None else mean for name, mean in sorted(means.items())}
    return (*row[:_PROCESS_MEANS], ordered, *row[_PROCESS_MEANS + 1 :])
