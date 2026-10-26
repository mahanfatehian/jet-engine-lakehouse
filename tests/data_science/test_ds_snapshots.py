"""Snapshot naming, discovery, retention, the snapshot job and the local artifact store."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from artifact_store import ArtifactNotFoundError, LocalArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken
from feature_pipeline import INFERENCE_FRAME_COLUMNS, SnapshotTrainingSource
from lakehouse_snapshots import (
    GOLD_HEALTH_STATUS,
    GOLD_TRAINING_FEATURES,
    SILVER_ROLLING_FEATURES,
    SNAPSHOT_DATASETS,
    format_snapshot_ts,
    latest_snapshot_key,
    list_snapshot_ids,
    prune_snapshots,
    run_snapshot,
    snapshot_ids_to_prune,
    snapshot_key,
)


class FakeExporter:
    """Serves in-memory tables the way ClickHouseGateway exports them."""

    def __init__(self, tables: dict[str, pd.DataFrame], engines: dict[str, str]) -> None:
        self._tables = tables
        self._engines = engines
        self.queries: list[str] = []

    def table_engines(self, tables):
        return {table: self._engines[table] for table in tables if table in self._tables}

    def export_parquet(self, sql: str, destination: Path) -> int:
        self.queries.append(sql)
        table = re.search(r"FROM `(\w+)`", sql).group(1)
        self._tables[table].to_parquet(destination, index=False)
        return destination.stat().st_size


def _tables(mart: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "int_sensor_rolling_features": mart.loc[:, list(INFERENCE_FRAME_COLUMNS)],
        "mart_rul_training_features": mart.drop(columns=["wear_factor", "status"]),
        "fct_engine_health_status": pd.DataFrame({"engine_id": ["ENG-001"], "risk_band": ["LOW"]}),
    }


ENGINES = {
    "int_sensor_rolling_features": "ReplacingMergeTree",
    "mart_rul_training_features": "ReplacingMergeTree",
    "fct_engine_health_status": "MergeTree",
}


def test_snapshot_keys_follow_the_lakehouse_layout() -> None:
    assert (
        snapshot_key(GOLD_TRAINING_FEATURES, "20261026T120000Z")
        == "gold/mart_rul_training_features/snapshot_ts=20261026T120000Z/data.parquet"
    )
    assert (
        snapshot_key(SILVER_ROLLING_FEATURES, "20261026T120000Z")
        == "silver/int_sensor_rolling_features/snapshot_ts=20261026T120000Z/data.parquet"
    )
    assert snapshot_key(GOLD_HEALTH_STATUS, "20261026T120000Z").startswith("gold/fct_engine_health_status/")
    with pytest.raises(ValueError, match="snapshot id"):
        snapshot_key(GOLD_TRAINING_FEATURES, "2026-10-26")


def test_snapshot_ids_are_utc() -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))

    assert format_snapshot_ts(datetime(2026, 10, 26, 15, 30, 5, tzinfo=tehran)) == "20261026T120005Z"
    with pytest.raises(ValueError, match="timezone-aware"):
        format_snapshot_ts(datetime(2026, 10, 26, 12, 0))


def test_retention_keeps_the_newest_snapshots() -> None:
    ids = ["20261026T120000Z", "20261024T120000Z", "20261025T120000Z", "20261025T120000Z"]

    assert snapshot_ids_to_prune(ids, 2) == ["20261024T120000Z"]
    assert snapshot_ids_to_prune(ids, 3) == []
    with pytest.raises(ValueError, match="at least one"):
        snapshot_ids_to_prune(ids, 0)


def test_pruning_removes_whole_expired_snapshots_of_one_dataset(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ids = [f"202610{day:02d}T000000Z" for day in range(1, 8)]
    for snapshot_id in ids:
        for dataset in SNAPSHOT_DATASETS:
            store.put_bytes(snapshot_key(dataset, snapshot_id), b"parquet")
    leftover = f"{GOLD_TRAINING_FEATURES.prefix}{ids[0]}/data.parquet.partial"
    store.put_bytes(leftover, b"interrupted upload")

    pruned = prune_snapshots(store, GOLD_TRAINING_FEATURES, retention=5)

    assert pruned == ids[:2]
    assert list_snapshot_ids(store, GOLD_TRAINING_FEATURES, complete_only=False) == ids[2:]
    assert store.list_keys(leftover) == []
    assert list_snapshot_ids(store, GOLD_HEALTH_STATUS, complete_only=True) == ids


def test_latest_snapshot_ignores_incomplete_and_foreign_prefixes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert latest_snapshot_key(store, GOLD_TRAINING_FEATURES) is None
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261025T000000Z"), b"complete")
    store.put_bytes(f"{GOLD_TRAINING_FEATURES.prefix}20261026T000000Z/part-0.tmp", b"partial")
    store.put_bytes(f"{GOLD_TRAINING_FEATURES.prefix}latest/data.parquet", b"foreign")

    assert latest_snapshot_key(store, GOLD_TRAINING_FEATURES) == snapshot_key(
        GOLD_TRAINING_FEATURES, "20261025T000000Z"
    )


def test_snapshot_job_exports_every_dataset_and_applies_retention(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    exporter = FakeExporter(_tables(synthetic_mart), ENGINES)
    stop = StopToken()

    for hour in range(4):
        written = run_snapshot(
            exporter, store, retention=2, now=datetime(2026, 10, 26, hour, 0, 0, tzinfo=UTC), stop=stop
        )

    assert set(written) == {dataset.table for dataset in SNAPSHOT_DATASETS}
    for dataset in SNAPSHOT_DATASETS:
        assert list_snapshot_ids(store, dataset, complete_only=False) == [
            "20261026T020000Z",
            "20261026T030000Z",
        ]
    assert all("FINAL" in query for query in exporter.queries if "int_sensor_rolling_features" in query)
    assert all("FINAL" not in query for query in exporter.queries if "fct_engine_health_status" in query)

    source = SnapshotTrainingSource(store)
    assert source.count_completed_lifecycles() == synthetic_mart["split_group"].nunique()
    loaded = source.load()
    assert loaded.data_source == "minio"
    assert loaded.source_uri.endswith("snapshot_ts=20261026T030000Z/data.parquet")
    assert len(loaded.frame) == len(synthetic_mart)


def test_snapshot_job_skips_until_every_source_table_exists(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    tables = _tables(synthetic_mart)
    del tables["fct_engine_health_status"]

    with pytest.raises(PreconditionsNotMetError, match="fct_engine_health_status"):
        run_snapshot(
            FakeExporter(tables, ENGINES), store, retention=5, now=datetime.now(UTC), stop=StopToken()
        )
    assert store.list_keys("") == []


def test_training_from_lakehouse_skips_without_snapshots(tmp_path: Path) -> None:
    with pytest.raises(PreconditionsNotMetError, match="no gold snapshot"):
        SnapshotTrainingSource(LocalArtifactStore(tmp_path)).count_completed_lifecycles()


def test_local_store_writes_atomically_and_rejects_unsafe_keys(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put_bytes("a/b/data.bin", b"first")
    store.put_bytes("a/b/data.bin", b"second")
    store.put_json("a/doc.json", {"value": 1})

    assert store.get_bytes("a/b/data.bin") == b"second"
    assert store.get_json("a/doc.json") == {"value": 1}
    assert store.list_keys("a/") == ["a/b/data.bin", "a/doc.json"]
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]
    with pytest.raises(ArtifactNotFoundError):
        store.get_bytes("a/missing.bin")
    for unsafe in ("/etc/passwd", "../outside", "a/../../outside", "", "a/"):
        with pytest.raises(ValueError, match="invalid artifact key"):
            store.put_bytes(unsafe, b"x")
    store.delete_keys(["a/b/data.bin", "a/never-existed.bin"])
    assert store.list_keys("a/") == ["a/doc.json"]
