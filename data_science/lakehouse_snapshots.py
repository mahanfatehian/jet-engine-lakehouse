"""Snapshot exports of ClickHouse models to the lakehouse bucket: naming, the export
job, discovery and retention.

Layout (interface contract, section 5)::

    <zone>/<table>/snapshot_ts=<YYYYMMDDTHHMMSSZ>/data.parquet

A snapshot exists once its ``data.parquet`` object exists; that object is
written with a single atomic PUT, so readers never see a partial file.
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

import pyarrow.parquet as pq
from artifact_store import ArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken
from lakehouse_io import final_modifier

SNAPSHOT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
SNAPSHOT_FILE_NAME = "data.parquet"
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
_SNAPSHOT_TS_PATTERN = re.compile(r"\d{8}T\d{6}Z")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotDataset:
    """A ClickHouse model exported to a medallion zone of the bucket."""

    zone: str
    table: str

    @property
    def prefix(self) -> str:
        """Key prefix shared by every snapshot of this dataset."""
        return f"{self.zone}/{self.table}/snapshot_ts="


SILVER_ROLLING_FEATURES = SnapshotDataset("silver", "int_sensor_rolling_features")
GOLD_TRAINING_FEATURES = SnapshotDataset("gold", "mart_rul_training_features")
GOLD_HEALTH_STATUS = SnapshotDataset("gold", "fct_engine_health_status")
SNAPSHOT_DATASETS = (SILVER_ROLLING_FEATURES, GOLD_TRAINING_FEATURES, GOLD_HEALTH_STATUS)


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
) -> dict[str, str]:
    """Export every snapshot dataset under one snapshot id, then apply retention.

    Returns the URI written per table. Raises PreconditionsNotMetError when a source table is missing.
    """
    engines = exporter.table_engines(dataset.table for dataset in SNAPSHOT_DATASETS)
    missing = [dataset.table for dataset in SNAPSHOT_DATASETS if dataset.table not in engines]
    if missing:
        raise PreconditionsNotMetError(f"source tables do not exist yet: {', '.join(missing)}")
    snapshot_ts = format_snapshot_ts(now)
    written: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ds-snapshot-") as workdir:
        for dataset in SNAPSHOT_DATASETS:
            stop.raise_if_stopped()
            local_file = Path(workdir) / f"{dataset.table}.parquet"
            size = exporter.export_parquet(
                f"SELECT * FROM `{dataset.table}` {final_modifier(engines[dataset.table])}", local_file
            )
            rows = pq.read_metadata(local_file).num_rows
            key = snapshot_key(dataset, snapshot_ts)
            store.put_file(key, local_file, PARQUET_CONTENT_TYPE)
            local_file.unlink()
            written[dataset.table] = store.uri(key)
            logger.info(
                "exported %s: %d rows, %.2f MiB -> %s",
                dataset.table,
                rows,
                size / 2**20,
                written[dataset.table],
            )
    for dataset in SNAPSHOT_DATASETS:
        prune_snapshots(store, dataset, retention)
    return written
