"""Snapshot exports of ClickHouse models to the lakehouse bucket: naming, the export
job, discovery and retention.

Layout (interface contract, section 5)::

    <zone>/<table>/snapshot_ts=<YYYYMMDDTHHMMSSZ>/data.parquet

A snapshot exists once its ``data.parquet`` object exists; that object is
written with a single atomic PUT, so readers never see a partial file.

A dataset whose export has no rows is never published: its newest snapshot keeps
holding data (an empty export at stack start-up, before CDC rows have landed, would
otherwise shadow it). The silver dataset is the dbt view int_ml_features - the complete
model input (pulse and spectral features) of every pulse - exported for the last
SNAPSHOT_WINDOW_DAYS of event time only, so each snapshot stays bounded however long the
stack runs; ClickHouse streams the export to a local file, never into memory. It replaced
the v1 export of int_sensor_rolling_features (whose features it contains): snapshots
under silver/int_sensor_rolling_features/ are no longer written or pruned.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from artifact_store import ArtifactNotFoundError, ArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken
from lakehouse_io import final_modifier

SNAPSHOT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
SNAPSHOT_FILE_NAME = "data.parquet"
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
DEFAULT_WINDOW_DAYS = 7
# One suffix-range GET of this size returns the whole footer of typical snapshots.
FOOTER_PROBE_BYTES = 64 * 1024
_PARQUET_MAGIC = b"PAR1"
_SNAPSHOT_TS_PATTERN = re.compile(r"\d{8}T\d{6}Z")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotDataset:
    """A ClickHouse model exported to a medallion zone of the bucket.

    ``window_column`` limits the export to the last SNAPSHOT_WINDOW_DAYS of that
    timestamp column (measured back from its newest value); None exports every row.
    """

    zone: str
    table: str
    window_column: str | None = None

    @property
    def prefix(self) -> str:
        """Key prefix shared by every snapshot of this dataset."""
        return f"{self.zone}/{self.table}/snapshot_ts="


SILVER_ML_FEATURES = SnapshotDataset("silver", "int_ml_features", window_column="event_ts")
GOLD_TRAINING_FEATURES = SnapshotDataset("gold", "mart_rul_training_features")
GOLD_HEALTH_STATUS = SnapshotDataset("gold", "fct_engine_health_status")
SNAPSHOT_DATASETS = (SILVER_ML_FEATURES, GOLD_TRAINING_FEATURES, GOLD_HEALTH_STATUS)


@dataclass(frozen=True)
class SnapshotRef:
    """A published snapshot and its row count (read from the Parquet footer)."""

    key: str
    snapshot_ts: str
    rows: int


def format_snapshot_ts(moment: datetime) -> str:
    """Render ``moment`` as a snapshot id (UTC, ``YYYYMMDDTHHMMSSZ``)."""
    if moment.tzinfo is None:
        raise ValueError("snapshot timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime(SNAPSHOT_TS_FORMAT)


def snapshot_key(dataset: SnapshotDataset, snapshot_ts: str) -> str:
    """Return the object key of the Parquet file of one snapshot."""
    if not _SNAPSHOT_TS_PATTERN.fullmatch(snapshot_ts):
        raise ValueError(f"invalid snapshot id {snapshot_ts!r}, expected YYYYMMDDTHHMMSSZ")
    return f"{dataset.prefix}{snapshot_ts}/{SNAPSHOT_FILE_NAME}"


def _snapshot_id(dataset: SnapshotDataset, key: str) -> str | None:
    """Extract the snapshot id from a key below ``dataset.prefix`` (None if malformed)."""
    if not key.startswith(dataset.prefix):
        return None
    snapshot_ts, _, _ = key[len(dataset.prefix) :].partition("/")
    return snapshot_ts if _SNAPSHOT_TS_PATTERN.fullmatch(snapshot_ts) else None


def list_snapshot_ids(store: ArtifactStore, dataset: SnapshotDataset, *, complete_only: bool) -> list[str]:
    """Return the snapshot ids of ``dataset``, oldest first.

    With ``complete_only`` only snapshots whose Parquet file exists are returned;
    otherwise every ``snapshot_ts=`` prefix counts (used for pruning leftovers).
    """
    ids = set()
    for key in store.list_keys(dataset.prefix):
        snapshot_ts = _snapshot_id(dataset, key)
        if snapshot_ts is None:
            continue
        if complete_only and key != snapshot_key(dataset, snapshot_ts):
            continue
        ids.add(snapshot_ts)
    return sorted(ids)


def latest_snapshot_key(store: ArtifactStore, dataset: SnapshotDataset) -> str | None:
    """Return the key of the newest complete snapshot of ``dataset`` (None if there is none)."""
    ids = list_snapshot_ids(store, dataset, complete_only=True)
    return snapshot_key(dataset, ids[-1]) if ids else None


def parquet_row_count(store: ArtifactStore, key: str) -> int:
    """Rows of the Parquet object at ``key``, read from its footer (one or two range GETs)."""
    tail = store.get_tail(key, FOOTER_PROBE_BYTES)
    if len(tail) < 12 or tail[-4:] != _PARQUET_MAGIC:
        raise ValueError(f"{store.uri(key)} is not a Parquet file")
    footer_length = int.from_bytes(tail[-8:-4], "little")
    if footer_length + 8 > len(tail):
        tail = store.get_tail(key, footer_length + 8)
    return pq.read_metadata(pa.BufferReader(tail[-(footer_length + 8) :])).num_rows


def latest_nonempty_snapshot(store: ArtifactStore, dataset: SnapshotDataset) -> SnapshotRef | None:
    """Newest complete snapshot of ``dataset`` that holds rows; newer empty or unreadable ones are skipped."""
    skipped: list[str] = []
    for snapshot_ts in reversed(list_snapshot_ids(store, dataset, complete_only=True)):
        key = snapshot_key(dataset, snapshot_ts)
        try:
            rows = parquet_row_count(store, key)
        except (ValueError, pa.ArrowException, ArtifactNotFoundError) as exc:
            logger.warning("ignoring unreadable snapshot %s: %s", store.uri(key), exc)
            skipped.append(snapshot_ts)
            continue
        if rows == 0:
            skipped.append(snapshot_ts)
            continue
        if skipped:
            logger.info(
                "skipped %d newer empty or unreadable snapshot(s) of %s/%s: %s",
                len(skipped),
                dataset.zone,
                dataset.table,
                ", ".join(skipped),
            )
        return SnapshotRef(key=key, snapshot_ts=snapshot_ts, rows=rows)
    if skipped:
        logger.info("every snapshot of %s/%s is empty or unreadable", dataset.zone, dataset.table)
    return None


def snapshot_ids_to_prune(snapshot_ids: list[str], retention: int) -> list[str]:
    """Return the ids that fall outside the newest ``retention`` snapshots."""
    if retention < 1:
        raise ValueError("retention must keep at least one snapshot")
    ordered = sorted(set(snapshot_ids))
    return ordered[: max(0, len(ordered) - retention)]


def prune_snapshots(store: ArtifactStore, dataset: SnapshotDataset, retention: int) -> list[str]:
    """Delete every object of the snapshots beyond the newest ``retention``; return their ids."""
    keys = store.list_keys(dataset.prefix)
    expired = set(
        snapshot_ids_to_prune(
            [sid for sid in (_snapshot_id(dataset, key) for key in keys) if sid is not None], retention
        )
    )
    doomed = [key for key in keys if _snapshot_id(dataset, key) in expired]
    if doomed:
        store.delete_keys(doomed)
        logger.info(
            "pruned %d snapshot(s) of %s/%s: %s",
            len(expired),
            dataset.zone,
            dataset.table,
            ", ".join(sorted(expired)),
        )
    return sorted(expired)


def snapshot_select_sql(dataset: SnapshotDataset, engine: str, window_days: int) -> str:
    """The export query of ``dataset`` (FINAL where the engine needs it, windowed if configured)."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    source = f"`{dataset.table}` {final_modifier(engine)}".rstrip()
    sql = f"SELECT * FROM {source}"
    if dataset.window_column is not None:
        column = f"`{dataset.window_column}`"
        sql += f" WHERE {column} >= (SELECT max({column}) FROM {source}) - INTERVAL {int(window_days)} DAY"
    return sql


class SnapshotExporter(Protocol):
    """The ClickHouse capabilities a snapshot needs (ClickHouseGateway satisfies it)."""

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        """Map existing tables to their engine names."""
        ...

    def export_parquet(self, sql: str, destination: Path) -> int:
        """Write the result of ``sql`` as one Parquet file."""
        ...


def run_snapshot(
    exporter: SnapshotExporter,
    store: ArtifactStore,
    *,
    retention: int,
    now: datetime,
    stop: StopToken,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, str]:
    """Export every non-empty snapshot dataset under one snapshot id, then apply retention.

    Returns the URI written per table. Raises PreconditionsNotMetError when a source
    table is missing or when every export is empty; nothing is written or pruned then.
    """
    engines = exporter.table_engines(dataset.table for dataset in SNAPSHOT_DATASETS)
    missing = [dataset.table for dataset in SNAPSHOT_DATASETS if dataset.table not in engines]
    if missing:
        raise PreconditionsNotMetError(f"source tables do not exist yet: {', '.join(missing)}")
    snapshot_ts = format_snapshot_ts(now)
    written: dict[str, str] = {}
    empty: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ds-snapshot-") as workdir:
        for dataset in SNAPSHOT_DATASETS:
            stop.raise_if_stopped()
            local_file = Path(workdir) / f"{dataset.table}.parquet"
            sql = snapshot_select_sql(dataset, engines[dataset.table], window_days)
            size = exporter.export_parquet(sql, local_file)
            rows = pq.read_metadata(local_file).num_rows
            scope = (
                f"last {window_days} days of {dataset.window_column}" if dataset.window_column else "all rows"
            )
            if rows == 0:
                local_file.unlink()
                empty.append(dataset.table)
                logger.info("skipped %s (%s): 0 rows, nothing published", dataset.table, scope)
                continue
            key = snapshot_key(dataset, snapshot_ts)
            store.put_file(key, local_file, PARQUET_CONTENT_TYPE)
            local_file.unlink()
            written[dataset.table] = store.uri(key)
            logger.info(
                "exported %s (%s): %d rows, %.2f MiB -> %s",
                dataset.table,
                scope,
                rows,
                size / 2**20,
                written[dataset.table],
            )
    if not written:
        raise PreconditionsNotMetError(
            f"every snapshot source is empty ({', '.join(empty)}); nothing published"
        )
    for dataset in SNAPSHOT_DATASETS:
        prune_snapshots(store, dataset, retention)
    return written
