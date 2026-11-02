"""Disaster recovery: rebuild the ClickHouse raw tables from the silver lake, the system of record.

CLI::

    python data_science/restore_from_lake.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]
        [--tables TABLE [TABLE ...]] [--dry-run] [--no-reconcile]

Spark writes every CDC image twice from one parsed column set (``cdc_transforms``): exactly-once
to silver Parquet and at-least-once into the raw tables, so the lake holds every column of a raw
row except the server-filled ``ingested_at``; the HF frame features likewise land in
silver/engine_hf_features and raw_engine_hf_features. ClickHouse loses nothing the lake cannot
give back: this job puts silver into raw with one ``INSERT ... SELECT ... FROM s3()`` per table and
day of event time, run by ClickHouse itself, so no row passes through Python.

    raw table                    silver dataset           identity in raw              day of
    raw_engine_telemetry         engine_telemetry         (engine_id, telemetry_id)    event_ts
    raw_engine_status_registry   engine_status_registry   (engine_id, cdc_lsn)         updated_ts
    raw_maintenance_log          maintenance_log          (engine_id, maintenance_id)  failure_ts
    raw_engine_hf_features       engine_hf_features       (engine_id, telemetry_id)    frame_start_ts

The identity is the table's sorting key: the rows its ReplacingMergeTree collapses. A silver
partition ``event_date=D`` holds the rows whose day column falls on the UTC day D (Spark runs in
UTC), the same rows a raw read of [D, D + 1 day) on that column sees.

Which rows: per identity the lake's newest version, deduplicated as every reader of the files must
(``streaming_queries``): an interrupted micro-batch leaves copies, and the highest ``cdc_lsn`` is
the current image. Copies of one version go to the last Kafka delivery, the copy raw keeps when
they reach it in separate micro-batches (see below). A row is inserted only when raw lacks it:
raw does not hold the identity, or
(``cdc_lsn``-versioned tables) only an older version. The restored row carries the key and
``cdc_lsn`` the streaming writer used, so it is idempotent by construction: a restore on a
complete table writes nothing, a second restore finds nothing to add, and a copy the streaming job
inserts at the same time collapses with it under FINAL. Deletes are restored as the delete images
(``is_deleted`` = 1) the streaming job writes.

Snapshot: the run fixes the instant it starts and reads only the Parquet files written before it
(``_time``), so the plan, the inserts and the verification count the same lake while the lake
queries keep writing. The rows those queries add later reach raw through the streaming job.

Columns: each column is read from Parquet with the raw table's own type and written by name:

    UInt*/Int* from Parquet INT32/INT64 (negative values were dead-lettered before silver)
    DateTime64(6, 'UTC') from TIMESTAMP_MICROS UTC instants, to the microsecond
    cdc_source_ts, cdc_emitted_ts: whole milliseconds, exact as DateTime64(3, 'UTC')
    is_deleted, cdc_op, cdc_lsn, kafka_partition, kafka_offset as Spark parsed them
    Float32 HF features: Arrow turned NaN into NULL in silver; NULL becomes NaN again, as
    ``hf_rows.feature_rows`` does for ClickHouse; process_means likewise, its keys sorted
    ingested_at: the server default now64(3), as for a streaming insert

``ingested_at`` is the one column a restore cannot reproduce: the lake never records when a row
first reached ClickHouse (silver's ``processed_at`` is the lake query's own micro-batch time). It
is harmless and even required: the CDC tables are versioned by ``cdc_lsn``, so it only orders
identical copies; raw_engine_hf_features is versioned by it, but a restore only adds frames raw
does not hold; and dbt's incremental models select raw rows by ``ingested_at`` above their
watermark, which a restored row's fresh stamp passes, so the next dbt run picks it up (versions
staging already published are skipped). The restore window shows up as ingestion lag and an
ingest burst on the Data Platform dashboard.

A change event Debezium delivered twice (it re-emits from its last committed offset after a
connector restart) differs between its copies only in ``kafka_offset`` and ``cdc_emitted_ts``.
Raw keeps the later copy when the two arrive in separate micro-batches, the usual case (the
ReplacingMergeTree keeps the last insert of equal versions), but the first when they share one
(``clickhouse_rows.deduplicate``), which the lake does not record; the restore always takes the
later. The image, ``cdc_op``, ``cdc_lsn`` and ``cdc_source_ts`` are the same either way, so FINAL
readers and dbt see the same row.

Rows older than a raw table's TTL (raw_engine_hf_features: 90 days) are not restored: the next
TTL merge would drop them again.

After a restore the job counts again (every restored day must show nothing left to add) and runs
the reconciliation audit (``reconcile_lakehouse.run_reconciliation``, recorded in
ops_reconciliation like every audit) over the restored event time that had settled
RECONCILE_LAG_SECONDS before the snapshot (at most its last 168 hours). Its verdict on the raw
layer is the restore's: raw must hold every pulse the source committed. Staging is reported but
does not fail the restore: dbt publishes the restored rows in its next run, after which
``make reconcile`` passes. The audit needs staging to exist: right after a rebuild of an empty
ClickHouse it is skipped with that reason until dbt has run. ``--no-reconcile`` leaves it out,
e.g. while PostgreSQL is unreachable.

ClickHouse resolves S3_ENDPOINT_URL (and, for the audit, POSTGRES_HOST) itself. The credentials
travel as query parameters whose names make ClickHouse mask them in its logs, as in
``reconcile_lakehouse``.

Exit codes: 0 restored (or dry run) and raw passed the audit, or the audit was skipped or not
requested; 3 skipped (a raw table, the bucket or every silver partition of the range does not
exist); 65 rows of the snapshot are still missing after the restore, or the audit found raw
cells that differ from the source; 1 failure (also a raw table whose columns differ from the ones
restored here); 143 terminated.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol

from clickhouse_connect.driver.exceptions import DatabaseError
from ds_runtime import EXIT_OK, PreconditionsNotMetError, StopToken, configure_logging, run_cli
from ds_settings import S3Settings, load_settings
from lakehouse_io import ClickHouseGateway
from reconcile_lakehouse import (
    EXIT_DATA_MISMATCH,
    MAX_WINDOW_HOURS,
    Layer,
    ReconcileSettings,
    ReconcileWindow,
    ReconciliationCell,
    ReconciliationRun,
    load_reconcile_settings,
    run_reconciliation,
)

SILVER_ZONE = "silver"
INGESTED_AT = "ingested_at"
# Rows that expire within this margin are left out too: a TTL merge must not drop a restored
# row between the insert and the verification count.
TTL_MARGIN = timedelta(hours=1)
# ClickHouse error code S3_ERROR; MinIO names a missing bucket NoSuchBucket.
_S3_ERROR_CODE = 499
_MISSING_BUCKET_MARKER = "NoSuchBucket"
_ONE_DAY = timedelta(days=1)
# Copies of one row version: the last Kafka delivery first (see the module docstring).
_LAST_DELIVERY_FIRST = "kafka_partition DESC, kafka_offset DESC"

_CDC_METADATA: tuple[tuple[str, str], ...] = (
    ("cdc_op", "LowCardinality(String)"),
    ("cdc_lsn", "UInt64"),
    ("cdc_source_ts", "DateTime64(3, 'UTC')"),
    ("cdc_emitted_ts", "DateTime64(3, 'UTC')"),
    ("is_deleted", "UInt8"),
    ("kafka_partition", "Int32"),
    ("kafka_offset", "Int64"),
)
_INSTANT = "DateTime64(6, 'UTC')"
_HF_CHANNEL_FEATURES = (
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "amp_1x_n1",
    "amp_1x_n2",
    "amp_bpfo",
    "band_energy_hf",
    "spectral_centroid_hz",
)
_HF_SPOOLS = ("n1_rpm", "n2_rpm", "bpfo_order")
_HF_CHANNELS = tuple(f"{prefix}_{feature}" for prefix in ("fan", "core") for feature in _HF_CHANNEL_FEATURES)
_HF_FLOATS: tuple[str, ...] = (*_HF_SPOOLS, *_HF_CHANNELS, "egt_ab_split")
_PROCESS_MEANS = "process_means"


@dataclass(frozen=True)
class RestoreTable:
    """A raw table, the silver dataset it is rebuilt from, and how both identify a row.

    ``columns`` lists every column of the raw table but ``ingested_at``, in table order, with its
    ClickHouse type. ``version`` names the column the table's engine keeps the highest of per
    ``key`` (None: the key alone identifies a row). ``nan_columns`` are Float32 columns that silver
    stores as NULL where ClickHouse holds NaN; ``nan_map`` is the Map column whose values do so.
    """

    name: str
    dataset: str
    columns: tuple[tuple[str, str], ...]
    key: tuple[str, ...]
    event_time: str
    version: str | None = None
    ttl_days: int | None = None
    nan_columns: frozenset[str] = frozenset()
    nan_map: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        """The insert column list (``ingested_at`` stays server-filled)."""
        return tuple(name for name, _ in self.columns)

    @property
    def structure(self) -> str:
        """s3() structure: the raw types, nullable where silver keeps NULL for NaN."""
        return ", ".join(f"{name} {self._read_type(name, kind)}" for name, kind in self.columns)

    def projection(self) -> tuple[str, ...]:
        """SELECT expressions turning a Parquet row into the raw row, in column order."""
        return tuple(self._expression(name) for name in self.column_names)

    def _read_type(self, name: str, kind: str) -> str:
        if name in self.nan_columns:
            return "Nullable(Float32)"
        if name == self.nan_map:
            return "Map(String, Nullable(Float32))"
        return kind

    def _expression(self, name: str) -> str:
        if name in self.nan_columns:
            return f"ifNull({name}, toFloat32(nan)) AS {name}"
        if name == self.nan_map:
            return f"mapSort(mapApply((k, v) -> (k, ifNull(v, toFloat32(nan))), {name})) AS {name}"
        return name


RAW_ENGINE_TELEMETRY = RestoreTable(
    name="raw_engine_telemetry",
    dataset="engine_telemetry",
    columns=(
        ("telemetry_id", "UInt64"),
        ("engine_id", "LowCardinality(String)"),
        ("lifecycle_id", "UInt32"),
        ("cycle_number", "UInt32"),
        ("event_ts", _INSTANT),
        ("rpm", "Float64"),
        ("core_temp_c", "Float64"),
        ("egt_c", "Float64"),
        ("vibration_mm_s", "Float64"),
        ("fuel_flow_kg_h", "Float64"),
        ("oil_pressure_psi", "Float64"),
        ("wear_factor", "Float64"),
        ("status", "LowCardinality(String)"),
        ("maintenance_id", "Nullable(UInt64)"),
        ("created_ts", _INSTANT),
        *_CDC_METADATA,
    ),
    key=("engine_id", "telemetry_id"),
    event_time="event_ts",
    version="cdc_lsn",
)
RAW_ENGINE_STATUS_REGISTRY = RestoreTable(
    name="raw_engine_status_registry",
    dataset="engine_status_registry",
    columns=(
        ("engine_id", "LowCardinality(String)"),
        ("engine_model", "LowCardinality(String)"),
        ("operational_status", "LowCardinality(String)"),
        ("current_lifecycle_id", "UInt32"),
        ("current_cycle", "UInt32"),
        ("wear_factor", "Float64"),
        ("cycles_since_new", "UInt32"),
        ("lifecycles_completed", "UInt32"),
        ("last_maintenance_ts", f"Nullable({_INSTANT})"),
        ("status_changed_ts", _INSTANT),
        ("last_telemetry_ts", f"Nullable({_INSTANT})"),
        ("updated_ts", _INSTANT),
        ("registered_ts", _INSTANT),
        *_CDC_METADATA,
    ),
    # The full change history: every CDC event is its own row.
    key=("engine_id", "cdc_lsn"),
    event_time="updated_ts",
)
RAW_MAINTENANCE_LOG = RestoreTable(
    name="raw_maintenance_log",
    dataset="maintenance_log",
    columns=(
        ("maintenance_id", "UInt64"),
        ("engine_id", "LowCardinality(String)"),
        ("lifecycle_id", "UInt32"),
        ("failure_cycle", "UInt32"),
        ("failure_ts", _INSTANT),
        ("failure_mode", "LowCardinality(String)"),
        ("maintenance_status", "LowCardinality(String)"),
        ("action_taken", "Nullable(String)"),
        ("opened_ts", _INSTANT),
        ("completed_ts", f"Nullable({_INSTANT})"),
        ("downtime_seconds", "Nullable(Int32)"),
        ("warning_rows_flagged", "Nullable(Int32)"),
        ("notes", "Nullable(String)"),
        *_CDC_METADATA,
    ),
    key=("engine_id", "maintenance_id"),
    event_time="failure_ts",
    version="cdc_lsn",
)
RAW_ENGINE_HF_FEATURES = RestoreTable(
    name="raw_engine_hf_features",
    dataset="engine_hf_features",
    columns=(
        ("engine_id", "LowCardinality(String)"),
        ("engine_model", "LowCardinality(String)"),
        ("lifecycle_id", "UInt32"),
        ("cycle_number", "UInt32"),
        ("telemetry_id", "UInt64"),
        ("frame_seq", "UInt64"),
        ("frame_start_ts", _INSTANT),
        ("frame_duration_ms", "UInt32"),
        *((name, "Float32") for name in _HF_SPOOLS),
        ("vibration_sample_rate_hz", "UInt32"),
        ("vibration_channels", "UInt16"),
        ("process_channels", "UInt16"),
        *((name, "Float32") for name in _HF_CHANNELS),
        ("egt_ab_split", "Float32"),
        (_PROCESS_MEANS, "Map(LowCardinality(String), Float32)"),
        ("frame_bytes", "UInt32"),
        ("kafka_partition", "Int32"),
        ("kafka_offset", "Int64"),
    ),
    # Versioned by ingested_at, which the lake cannot give back: a frame raw holds stays as it is.
    key=("engine_id", "telemetry_id"),
    event_time="frame_start_ts",
    ttl_days=90,
    nan_columns=frozenset(_HF_FLOATS),
    nan_map=_PROCESS_MEANS,
)
RESTORE_TABLES: tuple[RestoreTable, ...] = (
    RAW_ENGINE_TELEMETRY,
    RAW_ENGINE_STATUS_REGISTRY,
    RAW_MAINTENANCE_LOG,
    RAW_ENGINE_HF_FEATURES,
)
TABLES_BY_NAME: dict[str, RestoreTable] = {table.name: table for table in RESTORE_TABLES}

# Lake partitions per dataset: format One lists the files without reading them.
PARTITIONS_SQL = """
SELECT
    extract(_path, '/silver/([^/]+)/event_date=') AS dataset,
    toDate(extract(_path, '/event_date=([0-9-]+)/')) AS event_date,
    count() AS files
FROM s3({silver_files:String}, {s3_access_key_id:String}, {s3_secret_access_key:String}, 'One')
WHERE _time < {snapshot:DateTime('UTC')}
GROUP BY dataset, event_date
ORDER BY dataset, event_date
"""

# The newest image per identity among the day's files of the snapshot (see the module docstring).
_LAKE_ROWS = """
    SELECT
        {projection}
    FROM s3(
        {{silver_files:String}}, {{s3_access_key_id:String}}, {{s3_secret_access_key:String}}, 'Parquet',
        {{structure:String}}
    )
    WHERE _time < {{snapshot:DateTime('UTC')}}{retention}
    ORDER BY {order}
    LIMIT 1 BY {key}"""
_RETENTION = "\n      AND {event_time} >= {{kept_since:DateTime64(6, 'UTC')}}"
# The raw rows of the day: the lake partition event_date=D holds the rows of [D, D + 1 day).
_IN_DAY = (
    "{event_time} >= {{day_start:DateTime64(6, 'UTC')}}\n"
    "      AND {event_time} < {{day_end:DateTime64(6, 'UTC')}}"
)
# Raw's newest version per identity of the day; any version counts for an unversioned table.
_RAW_VERSIONS = """
    SELECT {key}, max({version}) AS raw_version, 1 AS in_raw
    FROM {table}
    WHERE {in_day}
    GROUP BY {key}"""
_RAW_KEYS = """
    SELECT DISTINCT {key}, 1 AS in_raw
    FROM {table}
    WHERE {in_day}"""
# Per table and day: the lake's identities, raw's identities, the lake rows raw lacks.
_DAY_COUNTS = """
SELECT
    count() AS lake_rows,
    (
        SELECT uniqExact({key})
        FROM {table}
        WHERE {in_day}
    ) AS raw_rows,
    countIf({missing}) AS to_restore
FROM
({lake}
) AS lake
LEFT JOIN
({raw}
) AS raw USING ({key})
"""
_DAY_RESTORE = """
INSERT INTO {table} ({columns})
SELECT {columns}
FROM
({lake}
) AS lake
LEFT JOIN
({raw}
) AS raw USING ({key})
WHERE {missing}
"""

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("restore_from_lake")


class SchemaMismatchError(RuntimeError):
    """A raw table's columns are not the ones this job restores: it would write wrong values."""


@dataclass(frozen=True)
class DateRange:
    """Event dates [first, last] (UTC, inclusive) to restore; None leaves that side open."""

    first: date | None = None
    last: date | None = None

    def __post_init__(self) -> None:
        if self.first is not None and self.last is not None and self.first > self.last:
            raise ValueError(f"--from {self.first} is after --to {self.last}")

    def __contains__(self, day: object) -> bool:
        if not isinstance(day, date):
            return False
        return (self.first is None or day >= self.first) and (self.last is None or day <= self.last)

    def describe(self) -> str:
        """``first..last``, ``*`` for an open side."""
        return f"{self.first or '*'}..{self.last or '*'}"


@dataclass(frozen=True)
class RestoreRequest:
    """What to restore (``--from``/``--to``, ``--tables``) and whether to write and audit."""

    tables: tuple[RestoreTable, ...] = RESTORE_TABLES
    dates: DateRange = field(default_factory=DateRange)
    dry_run: bool = False
    reconcile: bool = True


@dataclass(frozen=True)
class DayCounts:
    """One table and day: lake identities (deduplicated), raw identities, lake rows raw lacks."""

    table: str
    day: date
    lake_rows: int
    raw_rows: int
    to_restore: int

    def describe(self) -> str:
        """``table day: lake N, raw M, to restore K`` for the report."""
        return (
            f"{self.table} {self.day}: lake {self.lake_rows}, raw {self.raw_rows}, "
            f"to restore {self.to_restore}"
        )


@dataclass(frozen=True)
class RestoreOutcome:
    """The plan of a run, the recount after its inserts, and the reconciliation that followed."""

    snapshot: datetime
    plan: tuple[DayCounts, ...]
    dry_run: bool
    remaining: tuple[DayCounts, ...] = ()
    reconciliation: ReconciliationRun | None = None

    @property
    def to_restore(self) -> int:
        """Rows the plan found missing from raw (what the inserts add)."""
        return sum(counts.to_restore for counts in self.plan)

    @property
    def incomplete(self) -> list[DayCounts]:
        """Days that still miss rows of the snapshot after the restore."""
        return [counts for counts in self.remaining if counts.to_restore]

    @property
    def raw_mismatches(self) -> list[ReconciliationCell]:
        """Raw cells of the reconciliation that differ from the source, largest first."""
        if self.reconciliation is None:
            return []
        return [cell for cell in self.reconciliation.mismatches if cell.layer is Layer.RAW]

    @property
    def exit_code(self) -> int:
        """0, or 65 when rows are still missing or the reconciliation found raw cells that differ."""
        return EXIT_DATA_MISMATCH if self.incomplete or self.raw_mismatches else EXIT_OK


class RestoreGateway(Protocol):
    """The ClickHouse capabilities the restore and its reconciliation need (ClickHouseGateway)."""

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        """Map existing tables to their engine names."""
        ...

    def query_rows(self, sql: str, parameters: Mapping[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a SELECT and return its rows."""
        ...

    def command(self, statement: str, parameters: Mapping[str, Any] | None = None) -> None:
        """Run a statement without a result (the INSERT ... SELECT)."""
        ...

    def insert_rows(self, table: str, rows: Sequence[Sequence[Any]], column_names: Sequence[str]) -> None:
        """Insert ``rows`` into ``table`` as one batch (the reconciliation's results)."""
        ...


# ------------------------------------------------------------------------------------ SQL


def silver_files_url(s3: S3Settings, datasets: Sequence[str], day: date | None = None) -> str:
    """s3() glob of the silver Parquet files of ``datasets``: every partition, or the one of ``day``."""
    if not datasets:
        raise ValueError("at least one dataset is needed")
    names = datasets[0] if len(datasets) == 1 else "{" + ",".join(datasets) + "}"
    partition = "*" if day is None else day.isoformat()
    return f"{s3.endpoint_url.rstrip('/')}/{s3.bucket}/{SILVER_ZONE}/{names}/event_date={partition}/*.parquet"


def partitions_query(
    s3: S3Settings, tables: Sequence[RestoreTable], snapshot: datetime
) -> tuple[str, dict[str, Any]]:
    """PARTITIONS_SQL with its parameters: (dataset, event_date, files) of the snapshot."""
    parameters = {
        "silver_files": silver_files_url(s3, [table.dataset for table in tables]),
        "snapshot": snapshot,
        **_s3_parameters(s3),
    }
    return PARTITIONS_SQL, parameters


def day_counts_sql(table: RestoreTable) -> str:
    """Count query of one table and day: lake_rows, raw_rows, to_restore."""
    return _DAY_COUNTS.format(**_fragments(table))


def restore_sql(table: RestoreTable) -> str:
    """INSERT ... SELECT of one table and day: the lake rows raw lacks."""
    return _DAY_RESTORE.format(**_fragments(table))


def day_parameters(table: RestoreTable, s3: S3Settings, day: date, snapshot: datetime) -> dict[str, Any]:
    """Parameters of the count and restore statements of ``table`` and ``day``."""
    start = datetime.combine(day, time(), UTC)
    parameters: dict[str, Any] = {
        "silver_files": silver_files_url(s3, [table.dataset], day),
        "structure": table.structure,
        "snapshot": snapshot,
        "day_start": start,
        "day_end": start + _ONE_DAY,
        **_s3_parameters(s3),
    }
    kept_since = retention_start(table, snapshot)
    if kept_since is not None:
        parameters["kept_since"] = kept_since
    return parameters


def retention_start(table: RestoreTable, snapshot: datetime) -> datetime | None:
    """Oldest event time worth restoring into a table with a TTL (None: the table keeps all)."""
    if table.ttl_days is None:
        return None
    return snapshot - timedelta(days=table.ttl_days) + TTL_MARGIN


def _fragments(table: RestoreTable) -> dict[str, str]:
    key = ", ".join(table.key)
    version = f"{table.version} DESC, " if table.version else ""
    order = f"{key}, {version}{_LAST_DELIVERY_FIRST}"
    retention = _RETENTION.format(event_time=table.event_time) if table.ttl_days is not None else ""
    lake = _LAKE_ROWS.format(
        projection=",\n        ".join(table.projection()), retention=retention, order=order, key=key
    )
    in_day = _IN_DAY.format(event_time=table.event_time)
    if table.version:
        raw = _RAW_VERSIONS.format(key=key, version=table.version, table=table.name, in_day=in_day)
        missing = f"raw.in_raw = 0 OR raw.raw_version < lake.{table.version}"
    else:
        raw = _RAW_KEYS.format(key=key, table=table.name, in_day=in_day)
        missing = "raw.in_raw = 0"
    return {
        "table": table.name,
        "columns": ", ".join(table.column_names),
        "key": key,
        "in_day": in_day,
        "lake": lake,
        "raw": raw,
        "missing": missing,
    }


def _s3_parameters(s3: S3Settings) -> dict[str, str]:
    return {"s3_access_key_id": s3.access_key_id, "s3_secret_access_key": s3.secret_access_key}


# ------------------------------------------------------------------------------ the job


def require_tables(gateway: RestoreGateway, tables: Sequence[RestoreTable]) -> None:
    """Skip while a raw table is missing; fail when one has other columns than restored here."""
    names = [table.name for table in tables]
    existing = gateway.table_engines(names)
    missing = [name for name in names if name not in existing]
    if missing:
        raise PreconditionsNotMetError(
            f"tables do not exist yet: {', '.join(missing)} (init-db.sql creates them: make ch-init)"
        )
    rows = gateway.query_rows(
        "SELECT table, name, type FROM system.columns "
        "WHERE database = currentDatabase() AND table IN {tables:Array(String)} ORDER BY table, position",
        {"tables": names},
    )
    actual: dict[str, list[tuple[str, str]]] = {}
    for table, name, kind in rows:
        actual.setdefault(str(table), []).append((str(name), str(kind)))
    for table in tables:
        expected = [*table.columns, (INGESTED_AT, "DateTime64(3, 'UTC')")]
        if actual.get(table.name) != expected:
            differing = sorted(set(actual.get(table.name, [])) ^ set(expected))
            raise SchemaMismatchError(
                f"{table.name} does not have the columns this job restores (differing: "
                f"{', '.join(f'{name} {kind}' for name, kind in differing)}); re-apply init-db.sql "
                "(make ch-init) or align restore_from_lake.py with it"
            )


def lake_partitions(
    gateway: RestoreGateway, s3: S3Settings, tables: Sequence[RestoreTable], snapshot: datetime
) -> dict[str, tuple[date, ...]]:
    """Event dates of every table's silver partitions in the snapshot (PreconditionsNotMet: no bucket)."""
    try:
        rows = gateway.query_rows(*partitions_query(s3, tables, snapshot))
    except DatabaseError as exc:
        if exc.code == _S3_ERROR_CODE and _MISSING_BUCKET_MARKER in str(exc):
            raise PreconditionsNotMetError(f"the lakehouse bucket {s3.bucket!r} does not exist") from exc
        raise
    days: dict[str, list[date]] = {table.name: [] for table in tables}
    datasets = {table.dataset: table.name for table in tables}
    for dataset, day, _files in rows:
        if str(dataset) in datasets:
            days[datasets[str(dataset)]].append(_as_date(day))
    return {name: tuple(sorted(set(found))) for name, found in days.items()}


def plan_days(
    request: RestoreRequest, partitions: Mapping[str, Sequence[date]], snapshot: datetime
) -> dict[str, tuple[date, ...]]:
    """The days of each table to restore: lake partitions inside the range and within the TTL."""
    planned: dict[str, tuple[date, ...]] = {}
    for table in request.tables:
        days = [day for day in partitions.get(table.name, ()) if day in request.dates]
        kept_since = retention_start(table, snapshot)
        if kept_since is not None:
            expired = [day for day in days if datetime.combine(day + _ONE_DAY, time(), UTC) <= kept_since]
            if expired:
                logger.info(
                    "%s: %d day(s) up to %s are older than its %d-day TTL and are not restored",
                    table.name,
                    len(expired),
                    expired[-1],
                    table.ttl_days,
                )
            days = [day for day in days if day not in expired]
        planned[table.name] = tuple(days)
    if not any(planned.values()):
        datasets = ", ".join(f"{SILVER_ZONE}/{table.dataset}" for table in request.tables)
        raise PreconditionsNotMetError(
            f"no Parquet partition of {datasets} with an event date in {request.dates.describe()} "
            f"written before {_iso(snapshot)}: nothing to restore"
        )
    return planned


def count_days(
    gateway: RestoreGateway,
    s3: S3Settings,
    request: RestoreRequest,
    days: Mapping[str, Sequence[date]],
    snapshot: datetime,
    stop: StopToken,
) -> tuple[DayCounts, ...]:
    """Lake rows, raw rows and rows to restore of every planned table and day."""
    counts = []
    for table in request.tables:
        sql = day_counts_sql(table)
        for day in days.get(table.name, ()):
            stop.raise_if_stopped()
            [(lake_rows, raw_rows, to_restore)] = gateway.query_rows(
                sql, day_parameters(table, s3, day, snapshot)
            )
            counts.append(DayCounts(table.name, day, int(lake_rows), int(raw_rows), int(to_restore)))
    return tuple(counts)


def run_restore(
    gateway: RestoreGateway,
    request: RestoreRequest,
    s3: S3Settings,
    reconcile_settings: ReconcileSettings | None,
    *,
    now: datetime,
    stop: StopToken,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RestoreOutcome:
    """Plan, restore, recount and reconcile (see the module docstring); logs every step.

    ``now`` fixes the lake snapshot, ``clock`` stamps the reconciliation run.
    ``reconcile_settings`` is needed unless ``request.reconcile`` is False or it is a dry run.
    Raises PreconditionsNotMetError (exit 3) while a raw table, the bucket or every planned
    partition is missing.
    """
    if now.tzinfo is None:
        raise ValueError("the restore snapshot needs a timezone-aware 'now'")
    snapshot = now.astimezone(UTC).replace(microsecond=0)
    require_tables(gateway, request.tables)
    days = plan_days(request, lake_partitions(gateway, s3, request.tables, snapshot), snapshot)
    plan = count_days(gateway, s3, request, days, snapshot, stop)
    _log_counts("plan", plan, snapshot)
    outcome = RestoreOutcome(snapshot, plan, request.dry_run)
    if request.dry_run:
        logger.info(
            "dry run: nothing written; a restore would add %d row(s) to %s",
            outcome.to_restore,
            ", ".join(sorted({counts.table for counts in plan if counts.to_restore})) or "no table",
        )
        return outcome
    restored = _restore_days(gateway, s3, request, plan, snapshot, stop)
    remaining = count_days(gateway, s3, request, restored, snapshot, stop)
    outcome = replace(outcome, remaining=remaining)
    _log_verification(outcome)
    if not request.reconcile:
        logger.info("reconciliation not requested (--no-reconcile)")
        return outcome
    if reconcile_settings is None:
        raise ValueError("the reconciliation after a restore needs its settings")
    telemetry_days = days.get(RAW_ENGINE_TELEMETRY.name, ())
    window = reconcile_window(telemetry_days, snapshot, reconcile_settings.lag_seconds)
    run = _reconcile(gateway, window, telemetry_days, reconcile_settings, s3, clock(), stop)
    outcome = replace(outcome, reconciliation=run)
    if run is not None:
        _log_audit(outcome, run)
    return outcome


def _restore_days(
    gateway: RestoreGateway,
    s3: S3Settings,
    request: RestoreRequest,
    plan: Sequence[DayCounts],
    snapshot: datetime,
    stop: StopToken,
) -> dict[str, tuple[date, ...]]:
    """Insert every planned day that misses rows; returns those days per table."""
    restored: dict[str, list[date]] = {}
    for table in request.tables:
        sql = restore_sql(table)
        for counts in plan:
            if counts.table != table.name or not counts.to_restore:
                continue
            stop.raise_if_stopped()
            gateway.command(sql, day_parameters(table, s3, counts.day, snapshot))
            logger.info("restored %s %s: %d row(s) from the lake", table.name, counts.day, counts.to_restore)
            restored.setdefault(table.name, []).append(counts.day)
    return {name: tuple(days) for name, days in restored.items()}


def reconcile_window(
    telemetry_days: Sequence[date], snapshot: datetime, lag_seconds: int
) -> ReconcileWindow | None:
    """Event time of the restored telemetry that had settled ``lag_seconds`` before the snapshot.

    Whole hours that end at the earlier of the snapshot minus the lag and the end of the last
    restored day, starting no earlier than the first restored day, at most MAX_WINDOW_HOURS long;
    None when not one hour of it has settled.
    """
    if not telemetry_days:
        return None
    range_start = datetime.combine(min(telemetry_days), time(), UTC)
    range_end = datetime.combine(max(telemetry_days) + _ONE_DAY, time(), UTC)
    end = min(snapshot - timedelta(seconds=lag_seconds), range_end).replace(microsecond=0)
    hours = min(MAX_WINDOW_HOURS, (end - range_start) // timedelta(hours=1))
    if hours < 1:
        return None
    return ReconcileWindow(end - timedelta(hours=hours), end)


def _reconcile(
    gateway: RestoreGateway,
    window: ReconcileWindow | None,
    telemetry_days: Sequence[date],
    settings: ReconcileSettings,
    s3: S3Settings,
    now: datetime,
    stop: StopToken,
) -> ReconciliationRun | None:
    """Audit ``window`` with reconcile_lakehouse at ``now``; None when it cannot run (logged)."""
    if window is None:
        reason = (
            "raw_engine_telemetry was not restored"
            if not telemetry_days
            else f"none of the restored telemetry settled {settings.lag_seconds}s before the snapshot"
        )
        logger.info("reconciliation skipped: %s (it audits the telemetry pulses)", reason)
        return None
    # run_reconciliation ends its window ``lag_seconds`` before ``now``, cut to the whole second:
    # the lag, rounded down, puts that end at window.end while run_at stays the time of the audit.
    now = max(now.astimezone(UTC), window.end)
    hours = (window.end - window.start) // timedelta(hours=1)
    audit = replace(settings, window_hours=hours, lag_seconds=int((now - window.end).total_seconds()))
    logger.info("reconciling the restored telemetry of %s", window.describe())
    try:
        return run_reconciliation(gateway, audit, s3, now=now, stop=stop)
    except PreconditionsNotMetError as exc:
        logger.warning(
            "reconciliation skipped: %s; the restore stands, run `make reconcile` once that holds "
            "(after a rebuild from an empty ClickHouse: after the first dbt run)",
            exc,
        )
        return None


def _log_counts(phase: str, counts: Sequence[DayCounts], snapshot: datetime) -> None:
    for day in counts:
        logger.info("%s: %s", phase, day.describe())
    for table in dict.fromkeys(day.table for day in counts):
        days = [day for day in counts if day.table == table]
        logger.info(
            "%s: %s, %d day(s) of the lake as of %s: lake %d, raw %d, to restore %d",
            phase,
            table,
            len(days),
            _iso(snapshot),
            sum(day.lake_rows for day in days),
            sum(day.raw_rows for day in days),
            sum(day.to_restore for day in days),
        )


def _log_audit(outcome: RestoreOutcome, run: ReconciliationRun) -> None:
    raw = outcome.raw_mismatches
    if raw:
        logger.error(
            "the restored raw layer differs from the source in %d cell(s), worst: %s",
            len(raw),
            raw[0].describe(),
        )
    else:
        logger.info("the restored raw layer matches the source in every cell of %s", run.window.describe())
    staging = [cell for cell in run.mismatches if cell.layer is Layer.STAGING]
    if staging:
        logger.warning(
            "staging differs in %d cell(s): dbt publishes the restored rows in its next run "
            "(make dbt-run), after which make reconcile passes",
            len(staging),
        )
    silver = [cell for cell in run.mismatches if cell.layer is Layer.SILVER]
    if silver:
        logger.warning(
            "silver differs from the source in %d cell(s): the lake itself, which a restore cannot "
            "repair (see the reconciliation troubleshooting)",
            len(silver),
        )


def _log_verification(outcome: RestoreOutcome) -> None:
    incomplete = outcome.incomplete
    if not outcome.remaining:
        logger.info("nothing to restore: raw already holds every lake row of the snapshot")
        return
    if incomplete:
        logger.error(
            "restore incomplete: rows of the snapshot are still missing from raw: %s",
            "; ".join(counts.describe() for counts in incomplete),
        )
        return
    logger.info(
        "restore complete: %d row(s) added to %d table-day(s); every restored day now holds every "
        "lake row of the snapshot",
        outcome.to_restore,
        len(outcome.remaining),
    )


def _as_date(value: Any) -> date:
    return value.date() if isinstance(value, datetime) else value


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------ CLI


def _event_date(text: str) -> date:
    """argparse type: an ISO date YYYY-MM-DD."""
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a date YYYY-MM-DD, got {text!r}") from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the ClickHouse raw tables from the silver lake (idempotent).",
        epilog="Exit codes: 0 restored (or dry run), 3 skipped (a table, the bucket or the silver "
        "partitions do not exist), 65 rows still missing or the reconciliation found mismatches, "
        "1 failure.",
    )
    parser.add_argument(
        "--from",
        dest="first",
        type=_event_date,
        help="first event date (UTC) to restore, YYYY-MM-DD (default: the oldest lake partition)",
    )
    parser.add_argument(
        "--to",
        dest="last",
        type=_event_date,
        help="last event date (UTC) to restore, inclusive (default: the newest lake partition)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLES_BY_NAME),
        default=list(TABLES_BY_NAME),
        metavar="TABLE",
        help=f"raw tables to restore (default: all of {', '.join(TABLES_BY_NAME)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report per table and day the lake rows, the raw rows and what a restore would add, "
        "without writing",
    )
    parser.add_argument(
        "--no-reconcile", action="store_true", help="do not run the reconciliation audit after the restore"
    )
    args = parser.parse_args(argv)
    if args.first and args.last and args.first > args.last:
        parser.error(f"--from {args.first} is after --to {args.last}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``restore_from_lake.py``."""
    args = _parse_args(argv)
    platform = load_settings()
    configure_logging(platform.log_level)
    request = RestoreRequest(
        tables=tuple(table for table in RESTORE_TABLES if table.name in args.tables),
        dates=DateRange(args.first, args.last),
        dry_run=args.dry_run,
        reconcile=not args.no_reconcile,
    )
    audits = request.reconcile and not request.dry_run
    reconcile_settings = load_reconcile_settings() if audits else None
    stop = StopToken().install_signal_handlers()
    with ClickHouseGateway(platform.clickhouse) as gateway:
        outcome = run_restore(
            gateway, request, platform.s3, reconcile_settings, now=datetime.now(UTC), stop=stop
        )
    return outcome.exit_code


if __name__ == "__main__":
    run_cli(main)
