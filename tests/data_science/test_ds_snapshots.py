"""Snapshot naming, discovery, retention, the snapshot job, the lakehouse training source and the store."""

from __future__ import annotations

import io
import logging
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from artifact_store import ArtifactNotFoundError, LocalArtifactStore
from ds_runtime import PreconditionsNotMetError, StopToken
from feature_pipeline import INFERENCE_FRAME_COLUMNS, SPECTRAL_FEATURE_COLUMNS, FeatureSchemaError
from lakehouse_snapshots import (
    GOLD_HEALTH_STATUS,
    GOLD_TRAINING_FEATURES,
    SILVER_ML_FEATURES,
    SNAPSHOT_DATASETS,
    format_snapshot_ts,
    latest_nonempty_snapshot,
    latest_snapshot_key,
    list_snapshot_ids,
    parquet_row_count,
    prune_snapshots,
    run_snapshot,
    snapshot_ids_to_prune,
    snapshot_key,
    snapshot_select_sql,
)
from training_sources import SnapshotTrainingSource
from training_window import plan_training_window


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


class TailOnlyStore(LocalArtifactStore):
    """Local store that refuses whole-object reads, proving a reader only touches the footer."""

    def get_bytes(self, key: str) -> bytes:
        raise AssertionError(f"whole object read of {key}")


def _tables(mart: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "int_ml_features": mart.loc[:, list(INFERENCE_FRAME_COLUMNS)],
        "mart_rul_training_features": mart.drop(columns=["wear_factor", "status"]),
        "fct_engine_health_status": pd.DataFrame({"engine_id": ["ENG-001"], "risk_band": ["LOW"]}),
    }


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


ENGINES = {
    "int_ml_features": "View",
    "mart_rul_training_features": "ReplacingMergeTree",
    "fct_engine_health_status": "MergeTree",
}
NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def test_snapshot_keys_follow_the_lakehouse_layout() -> None:
    assert (
        snapshot_key(GOLD_TRAINING_FEATURES, "20261026T120000Z")
        == "gold/mart_rul_training_features/snapshot_ts=20261026T120000Z/data.parquet"
    )
    assert (
        snapshot_key(SILVER_ML_FEATURES, "20261026T120000Z")
        == "silver/int_ml_features/snapshot_ts=20261026T120000Z/data.parquet"
    )
    assert [dataset.table for dataset in SNAPSHOT_DATASETS] == [
        "int_ml_features",
        "mart_rul_training_features",
        "fct_engine_health_status",
    ]
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


def test_snapshot_query_windows_only_the_silver_features() -> None:
    silver = snapshot_select_sql(SILVER_ML_FEATURES, "View", 7)
    materialized = snapshot_select_sql(SILVER_ML_FEATURES, "ReplacingMergeTree", 2)
    gold = snapshot_select_sql(GOLD_HEALTH_STATUS, "MergeTree", 7)

    assert silver == (
        "SELECT * FROM `int_ml_features` WHERE `event_ts` >= "
        "(SELECT max(`event_ts`) FROM `int_ml_features`) - INTERVAL 7 DAY"
    )
    assert materialized == (
        "SELECT * FROM `int_ml_features` FINAL WHERE `event_ts` >= "
        "(SELECT max(`event_ts`) FROM `int_ml_features` FINAL) - INTERVAL 2 DAY"
    )
    assert gold == "SELECT * FROM `fct_engine_health_status`"
    with pytest.raises(ValueError, match="window_days"):
        snapshot_select_sql(SILVER_ML_FEATURES, "View", 0)


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
    silver_queries = [query for query in exporter.queries if "int_ml_features" in query]
    assert len(silver_queries) == 4 and all("FINAL" not in query for query in silver_queries)
    assert all("INTERVAL 7 DAY" in query for query in silver_queries)
    assert all("FINAL" not in query for query in exporter.queries if "fct_engine_health_status" in query)

    with SnapshotTrainingSource(store) as source:
        inventory = source.inventory()
        assert inventory.completed_lifecycles == synthetic_mart["split_group"].nunique()
        window = plan_training_window(inventory, max_rows=10**6, min_lifecycles=3)
        loaded = source.load(window)
    assert loaded.data_source == "minio"
    assert loaded.source_uri.endswith("snapshot_ts=20261026T030000Z/data.parquet")
    assert len(loaded.frame) == len(synthetic_mart)


def test_empty_datasets_are_not_published(tmp_path: Path, synthetic_mart, caplog) -> None:
    store = LocalArtifactStore(tmp_path)
    tables = _tables(synthetic_mart)
    run_snapshot(FakeExporter(dict(tables), ENGINES), store, retention=5, now=NOW, stop=StopToken())
    tables["mart_rul_training_features"] = tables["mart_rul_training_features"].iloc[0:0]

    with caplog.at_level(logging.INFO, logger="lakehouse_snapshots"):
        written = run_snapshot(
            FakeExporter(tables, ENGINES), store, retention=5, now=NOW + timedelta(hours=1), stop=StopToken()
        )

    assert set(written) == {"int_ml_features", "fct_engine_health_status"}
    assert list_snapshot_ids(store, GOLD_TRAINING_FEATURES, complete_only=False) == ["20261026T120000Z"]
    assert list_snapshot_ids(store, GOLD_HEALTH_STATUS, complete_only=True)[-1] == "20261026T130000Z"
    assert "skipped mart_rul_training_features" in caplog.text


def test_snapshot_job_skips_when_every_source_is_empty(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    old = [f"2026102{day}T000000Z" for day in range(1, 4)]
    for snapshot_id in old:
        store.put_bytes(snapshot_key(GOLD_HEALTH_STATUS, snapshot_id), b"parquet")
    empty = {table: frame.iloc[0:0] for table, frame in _tables(synthetic_mart).items()}

    with pytest.raises(PreconditionsNotMetError, match="every snapshot source is empty"):
        run_snapshot(FakeExporter(empty, ENGINES), store, retention=1, now=NOW, stop=StopToken())

    # Nothing published and nothing pruned.
    assert list_snapshot_ids(store, GOLD_HEALTH_STATUS, complete_only=False) == old
    assert store.list_keys("silver/") == [] and store.list_keys("gold/mart") == []


def test_snapshot_job_skips_until_every_source_table_exists(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    tables = _tables(synthetic_mart)
    del tables["fct_engine_health_status"]

    with pytest.raises(PreconditionsNotMetError, match="fct_engine_health_status"):
        run_snapshot(
            FakeExporter(tables, ENGINES), store, retention=5, now=datetime.now(UTC), stop=StopToken()
        )
    assert store.list_keys("") == []


def test_row_count_comes_from_the_parquet_footer(tmp_path: Path, synthetic_mart) -> None:
    store = TailOnlyStore(tmp_path)
    key = snapshot_key(GOLD_TRAINING_FEATURES, "20261026T120000Z")
    store.put_bytes(key, _parquet(synthetic_mart.drop(columns=["status"])))

    assert parquet_row_count(store, key) == len(synthetic_mart)
    store.put_bytes(key, b"not a parquet file")
    with pytest.raises(ValueError, match="not a Parquet file"):
        parquet_row_count(store, key)


def test_newest_snapshot_with_rows_wins_over_empty_and_unreadable_ones(
    tmp_path: Path, synthetic_mart, caplog
) -> None:
    store = TailOnlyStore(tmp_path)
    mart = synthetic_mart.drop(columns=["status"])
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T100000Z"), _parquet(mart.head(7)))
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T110000Z"), b"garbage")
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T120000Z"), _parquet(mart.iloc[0:0]))

    with caplog.at_level(logging.INFO, logger="lakehouse_snapshots"):
        found = latest_nonempty_snapshot(store, GOLD_TRAINING_FEATURES)

    assert found is not None and (found.snapshot_ts, found.rows) == ("20261026T100000Z", 7)
    assert "20261026T120000Z, 20261026T110000Z" in caplog.text
    assert latest_nonempty_snapshot(store, GOLD_HEALTH_STATUS) is None


def test_training_from_lakehouse_uses_the_newest_snapshot_with_rows(
    tmp_path: Path, synthetic_mart, caplog
) -> None:
    store = LocalArtifactStore(tmp_path)
    mart = synthetic_mart.drop(columns=["wear_factor", "status"])
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T100000Z"), _parquet(mart))
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T120000Z"), _parquet(mart.iloc[0:0]))
    source = SnapshotTrainingSource(store)

    with caplog.at_level(logging.INFO):
        inventory = source.inventory()
        window = plan_training_window(inventory, max_rows=len(mart) // 2, min_lifecycles=3)
        loaded = source.load(window)
    workdir = Path(source._workdir.name)
    source.close()

    assert inventory.completed_lifecycles == mart["split_group"].nunique()
    assert loaded.source_uri.endswith("snapshot_ts=20261026T100000Z/data.parquet")
    assert set(loaded.frame["split_group"]) == set(window.groups) and len(loaded.frame) <= len(mart) // 2
    assert "training from gold snapshot" in caplog.text and "20261026T100000Z" in caplog.text
    assert not workdir.exists()


def test_missing_spectral_values_survive_the_snapshot_round_trip(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    run_snapshot(
        FakeExporter(_tables(synthetic_mart), ENGINES), store, retention=5, now=NOW, stop=StopToken()
    )

    with SnapshotTrainingSource(store) as source:
        window = plan_training_window(source.inventory(), max_rows=10**6, min_lifecycles=3)
        loaded = source.load(window).frame

    by_pulse = synthetic_mart.set_index("telemetry_id").loc[loaded["telemetry_id"]]
    missing = by_pulse["fan_amp_bpfo_mean_5m"].isna().to_numpy()
    assert missing.any() and np.array_equal(loaded["fan_amp_bpfo_mean_5m"].isna().to_numpy(), missing)
    np.testing.assert_allclose(
        loaded["oil_temp_c_mean_5m"].to_numpy()[~missing], by_pulse["oil_temp_c_mean_5m"].to_numpy()[~missing]
    )


def test_training_from_a_snapshot_written_before_the_spectral_features(
    tmp_path: Path, synthetic_mart, caplog
) -> None:
    store = LocalArtifactStore(tmp_path)
    v1_mart = synthetic_mart.drop(columns=["wear_factor", "status", *SPECTRAL_FEATURE_COLUMNS])
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T100000Z"), _parquet(v1_mart))

    with caplog.at_level(logging.WARNING, logger="training_sources"), SnapshotTrainingSource(store) as source:
        window = plan_training_window(source.inventory(), max_rows=len(v1_mart) // 2, min_lifecycles=3)
        loaded = source.load(window)

    frame = loaded.frame
    assert set(frame["split_group"]) == set(window.groups) and len(frame) == window.rows
    assert frame[list(SPECTRAL_FEATURE_COLUMNS)].isna().all(axis=None)
    assert "predates 10 spectral feature columns" in caplog.text


def test_a_snapshot_without_a_pulse_feature_is_a_schema_error(tmp_path: Path, synthetic_mart) -> None:
    store = LocalArtifactStore(tmp_path)
    broken = synthetic_mart.drop(columns=["wear_factor", "status", "egt_c_drift", "fan_amp_bpfo_mean_5m"])
    store.put_bytes(snapshot_key(GOLD_TRAINING_FEATURES, "20261026T100000Z"), _parquet(broken))

    with SnapshotTrainingSource(store) as source, pytest.raises(FeatureSchemaError) as raised:
        source.inventory()
    assert str(raised.value).endswith("lacks columns: egt_c_drift")


def test_training_from_lakehouse_skips_without_snapshots(tmp_path: Path) -> None:
    with (
        SnapshotTrainingSource(LocalArtifactStore(tmp_path)) as source,
        pytest.raises(PreconditionsNotMetError, match="no gold snapshot"),
    ):
        source.inventory()


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


def test_local_store_tail_reads_and_downloads(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.put_bytes("a/data.bin", b"0123456789")

    assert store.get_tail("a/data.bin", 3) == b"789"
    assert store.get_tail("a/data.bin", 100) == b"0123456789"
    store.download_file("a/data.bin", tmp_path / "copy.bin")
    assert (tmp_path / "copy.bin").read_bytes() == b"0123456789"
    with pytest.raises(ArtifactNotFoundError):
        store.get_tail("a/missing.bin", 3)
    with pytest.raises(ValueError, match="length"):
        store.get_tail("a/data.bin", 0)
