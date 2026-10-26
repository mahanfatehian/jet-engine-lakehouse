"""Where training rows come from: the dbt gold mart in ClickHouse or its newest non-empty
gold snapshot in the lakehouse.

Every source works in two steps so a run never holds more than TRAINING_MAX_ROWS rows:
``inventory()`` returns one line per completed lifecycle (rows, end of life), the trainer
plans a TrainingWindow from it, and ``load(window)`` reads only the rows of that window.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow.parquet as pq
from artifact_store import ArtifactStore
from ds_runtime import PreconditionsNotMetError
from feature_pipeline import TRAINING_FRAME_COLUMNS, FeatureSchemaError, column_list
from lakehouse_io import ClickHouseGateway, final_modifier
from lakehouse_snapshots import (
    GOLD_TRAINING_FEATURES,
    SNAPSHOT_FILE_NAME,
    SnapshotRef,
    latest_nonempty_snapshot,
)
from training_window import (
    LifecycleInventory,
    LifecycleStats,
    TrainingWindow,
    apply_window,
    inventory_from_frame,
)

TRAINING_SOURCE_TABLE = "mart_rul_training_features"
_INVENTORY_COLUMNS = ("split_group", "event_ts")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingFrame:
    """Training rows (one per telemetry pulse of a completed lifecycle) and their origin."""

    frame: pd.DataFrame
    data_source: str
    source_uri: str

    @property
    def completed_lifecycles(self) -> int:
        """Number of distinct lifecycles (split groups) in the frame."""
        return int(self.frame["split_group"].nunique())


class TrainingDataSource(Protocol):
    """Where training rows come from (ClickHouse mart or a MinIO gold snapshot)."""

    name: str

    def inventory(self) -> LifecycleInventory:
        """Every completed lifecycle available for training (one line per lifecycle)."""
        ...

    def load(self, window: TrainingWindow) -> TrainingFrame:
        """Load the rows ``window`` selects."""
        ...


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw training rows and order them deterministically."""
    missing = [column for column in TRAINING_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureSchemaError(f"missing columns: {', '.join(missing)}")
    ordered = frame.loc[:, list(TRAINING_FRAME_COLUMNS)]
    for column in ("engine_id", "engine_model", "split_group"):
        ordered[column] = ordered[column].astype(str)
    return ordered.sort_values(
        ["engine_id", "lifecycle_id", "event_ts", "telemetry_id"], kind="mergesort"
    ).reset_index(drop=True)


def training_select_sql(source: str, window: TrainingWindow) -> tuple[str, dict[str, Any]]:
    """SELECT of the training columns restricted to ``window`` (query text and parameters).

    With a stride above 1, pulses are numbered per lifecycle in (event_ts, telemetry_id)
    order from 0 and every stride-th one is kept, exactly like ``apply_window``.
    """
    parameters: dict[str, Any] = {"groups": list(window.groups)}
    condition = "split_group IN {groups:Array(String)}"
    if window.stride > 1:
        parameters["stride"] = window.stride
        condition += (
            " AND telemetry_id IN (SELECT telemetry_id FROM (SELECT telemetry_id, row_number() OVER "
            "(PARTITION BY split_group ORDER BY event_ts, telemetry_id) - 1 AS pulse_number "
            f"FROM {source} WHERE split_group IN {{groups:Array(String)}}) "
            "WHERE pulse_number % {stride:UInt32} = 0)"
        )
    return f"SELECT {column_list(TRAINING_FRAME_COLUMNS)} FROM {source} WHERE {condition}", parameters


class ClickHouseTrainingSource:
    """Training rows straight from the dbt gold mart in ClickHouse."""

    name = "clickhouse"

    def __init__(self, gateway: ClickHouseGateway) -> None:
        self._gateway = gateway

    def _source(self) -> str:
        engines = self._gateway.table_engines([TRAINING_SOURCE_TABLE])
        if TRAINING_SOURCE_TABLE not in engines:
            raise PreconditionsNotMetError(
                f"{TRAINING_SOURCE_TABLE} does not exist yet (dbt has not built it)"
            )
        return f"`{TRAINING_SOURCE_TABLE}` {final_modifier(engines[TRAINING_SOURCE_TABLE])}".rstrip()

    def inventory(self) -> LifecycleInventory:
        """Rows and end of life of every completed lifecycle in the mart."""
        rows = self._gateway.query_rows(
            f"SELECT split_group, count(), toUnixTimestamp64Micro(max(event_ts)) FROM {self._source()} "
            "GROUP BY split_group ORDER BY split_group"
        )
        return LifecycleInventory(
            tuple(LifecycleStats(str(group), int(count), int(last)) for group, count, last in rows)
        )

    def load(self, window: TrainingWindow) -> TrainingFrame:
        """Load the mart rows of ``window`` (filtering and thinning run inside ClickHouse)."""
        sql, parameters = training_select_sql(self._source(), window)
        frame = self._gateway.query_df(sql, parameters)
        return TrainingFrame(prepare_training_frame(frame), self.name, f"clickhouse:{TRAINING_SOURCE_TABLE}")


class SnapshotTrainingSource:
    """Training rows from the newest gold snapshot that holds rows (streamed to a temp file once).

    Use it as a context manager (or call ``close``) to delete the local copy.
    """

    name = "minio"

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._snapshot: tuple[SnapshotRef, Path] | None = None

    def __enter__(self) -> SnapshotTrainingSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Delete the downloaded snapshot."""
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
            self._snapshot = None

    def _local_snapshot(self) -> tuple[SnapshotRef, Path]:
        if self._snapshot is None:
            ref = latest_nonempty_snapshot(self._store, GOLD_TRAINING_FEATURES)
            if ref is None:
                raise PreconditionsNotMetError(
                    f"no gold snapshot of {TRAINING_SOURCE_TABLE} with rows in the lakehouse yet"
                )
            logger.info("training from gold snapshot %s (%d rows)", self._store.uri(ref.key), ref.rows)
            self._workdir = tempfile.TemporaryDirectory(prefix="ds-train-snapshot-")
            path = Path(self._workdir.name) / SNAPSHOT_FILE_NAME
            self._store.download_file(ref.key, path)
            available = set(pq.ParquetFile(path).schema_arrow.names)
            missing = sorted(set(TRAINING_FRAME_COLUMNS) - available)
            if missing:
                raise FeatureSchemaError(
                    f"snapshot {self._store.uri(ref.key)} lacks columns: {', '.join(missing)}"
                )
            self._snapshot = (ref, path)
        return self._snapshot

    def inventory(self) -> LifecycleInventory:
        """Rows and end of life of every completed lifecycle in the snapshot."""
        _, path = self._local_snapshot()
        return inventory_from_frame(pq.read_table(path, columns=list(_INVENTORY_COLUMNS)).to_pandas())

    def load(self, window: TrainingWindow) -> TrainingFrame:
        """Read only the window's lifecycles (row-group filtering), then thin them if needed."""
        ref, path = self._local_snapshot()
        table = pq.read_table(
            path, columns=list(TRAINING_FRAME_COLUMNS), filters=[("split_group", "in", list(window.groups))]
        )
        frame = apply_window(table.to_pandas(), window)
        return TrainingFrame(prepare_training_frame(frame), self.name, self._store.uri(ref.key))
