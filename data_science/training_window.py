"""Bounded training data: which completed lifecycles (and pulses) a training run loads.

The mart grows by ~0.5 M rows per simulated day and the trainer needs ~2 KiB of memory per
row, so a run never loads more than TRAINING_MAX_ROWS rows. The window keeps the newest
completed lifecycles that fit. Should fewer lifecycles fit than training needs, it keeps
that minimum of newest lifecycles and thins each one evenly (every n-th pulse in event
order): consecutive pulses of a cycle are near-duplicates, so thinning loses far less
information than dropping lifecycles. Both choices are deterministic.

``LifecycleInventory.watermark_us`` (the end of the newest completed lifecycle) only ever
grows, also once a bounded mart drops old lifecycles, so ``--skip-if-unchanged`` compares
it instead of lifecycle counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MICROSECOND = pd.Timedelta(microseconds=1)


def epoch_micros(value: Any) -> int:
    """Microseconds since the Unix epoch of a timestamp (naive values are taken as UTC)."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return int((stamp - _EPOCH) // _ONE_MICROSECOND)


def format_micros(micros: int | None) -> str | None:
    """ISO-8601 UTC rendering of :func:`epoch_micros` output (for logs and metadata)."""
    if micros is None:
        return None
    return (_UNIX_EPOCH + timedelta(microseconds=micros)).isoformat()


@dataclass(frozen=True)
class LifecycleStats:
    """Size and end of one completed lifecycle (split group) in the training source."""

    split_group: str
    rows: int
    last_event_us: int


@dataclass(frozen=True)
class LifecycleInventory:
    """Every completed lifecycle a training source offers."""

    lifecycles: tuple[LifecycleStats, ...]

    @property
    def completed_lifecycles(self) -> int:
        """Number of completed lifecycles."""
        return len(self.lifecycles)

    @property
    def total_rows(self) -> int:
        """Rows over all completed lifecycles."""
        return sum(stats.rows for stats in self.lifecycles)

    @property
    def watermark_us(self) -> int | None:
        """End of the newest completed lifecycle (epoch microseconds); None when empty."""
        return max((stats.last_event_us for stats in self.lifecycles), default=None)

    def newest_first(self) -> list[LifecycleStats]:
        """Lifecycles ordered newest first (ties broken by split group, so the order is total)."""
        return sorted(
            self.lifecycles, key=lambda stats: (stats.last_event_us, stats.split_group), reverse=True
        )


def inventory_from_frame(frame: pd.DataFrame) -> LifecycleInventory:
    """Inventory of an in-memory frame with ``split_group`` and ``event_ts`` columns."""
    if frame.empty:
        return LifecycleInventory(())
    grouped = frame.groupby("split_group", sort=True)["event_ts"].agg(["size", "max"])
    return LifecycleInventory(
        tuple(
            LifecycleStats(str(group), int(row["size"]), epoch_micros(row["max"]))
            for group, row in grouped.iterrows()
        )
    )


@dataclass(frozen=True)
class TrainingWindow:
    """The lifecycles a run loads and the pulse stride applied within each of them."""

    groups: tuple[str, ...]
    stride: int
    rows: int
    source_lifecycles: int
    source_rows: int
    max_rows: int

    @property
    def dropped_lifecycles(self) -> int:
        """Completed lifecycles left out of the run."""
        return self.source_lifecycles - len(self.groups)

    @property
    def dropped_rows(self) -> int:
        """Rows left out of the run (older lifecycles plus thinned pulses)."""
        return self.source_rows - self.rows

    @property
    def is_complete(self) -> bool:
        """True when the run loads every row of the source."""
        return self.dropped_lifecycles == 0 and self.stride == 1

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary for metadata.json and ml_model_runs.params_json."""
        return {
            "max_rows": self.max_rows,
            "source_lifecycles": self.source_lifecycles,
            "source_rows": self.source_rows,
            "lifecycles": len(self.groups),
            "rows": self.rows,
            "stride": self.stride,
            "dropped_lifecycles": self.dropped_lifecycles,
            "dropped_rows": self.dropped_rows,
        }

    def describe(self) -> str:
        """One log line explaining what the window keeps and drops."""
        if self.is_complete:
            return (
                f"training on all {self.source_lifecycles} completed lifecycles "
                f"({self.source_rows} rows <= TRAINING_MAX_ROWS={self.max_rows})"
            )
        thinning = f", 1 in {self.stride} pulses of each" if self.stride > 1 else ""
        return (
            f"source holds {self.source_rows} rows in {self.source_lifecycles} completed lifecycles "
            f"(TRAINING_MAX_ROWS={self.max_rows}): keeping the newest {len(self.groups)} lifecycles"
            f"{thinning} ({self.rows} rows), dropping {self.dropped_lifecycles} older lifecycles "
            f"and {self.dropped_rows} rows"
        )


def thinned_rows(rows: int, stride: int) -> int:
    """Rows kept from a lifecycle of ``rows`` pulses when every ``stride``-th pulse is kept."""
    return math.ceil(rows / stride)


def plan_training_window(
    inventory: LifecycleInventory, *, max_rows: int, min_lifecycles: int
) -> TrainingWindow:
    """Choose the newest lifecycles (and, if needed, a pulse stride) that fit ``max_rows``."""
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    newest = inventory.newest_first()
    kept: list[LifecycleStats] = []
    rows = 0
    for stats in newest:
        if rows + stats.rows > max_rows:
            break
        kept.append(stats)
        rows += stats.rows
    stride = 1
    if len(kept) < min(min_lifecycles, len(newest)):
        kept = newest[:min_lifecycles]
        stride = max(1, math.ceil(sum(stats.rows for stats in kept) / max_rows))
        while sum(thinned_rows(stats.rows, stride) for stats in kept) > max_rows:
            stride += 1
        rows = sum(thinned_rows(stats.rows, stride) for stats in kept)
    return TrainingWindow(
        groups=tuple(sorted(stats.split_group for stats in kept)),
        stride=stride,
        rows=rows,
        source_lifecycles=inventory.completed_lifecycles,
        source_rows=inventory.total_rows,
        max_rows=max_rows,
    )


def apply_window(frame: pd.DataFrame, window: TrainingWindow) -> pd.DataFrame:
    """Rows of an in-memory frame selected by ``window`` (same rule as the ClickHouse query).

    Within a lifecycle, pulses are numbered in (event_ts, telemetry_id) order from 0 and
    every pulse whose number is a multiple of the stride is kept.
    """
    selected = frame[frame["split_group"].isin(window.groups)]
    if window.stride == 1:
        return selected.reset_index(drop=True)
    ordered = selected.sort_values(["split_group", "event_ts", "telemetry_id"], kind="mergesort")
    position = ordered.groupby("split_group", sort=False).cumcount()
    return ordered[position.to_numpy() % window.stride == 0].reset_index(drop=True)
