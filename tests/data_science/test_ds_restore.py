"""Restore of the raw tables from the silver lake: contracts, SQL builders, plan, restore and audit.

The INSERT ... SELECT ... FROM s3() runs on ClickHouse only; here a fake gateway keeps per table
and day what the lake holds and what raw lacks, answers the count queries from that state and
"restores" a day when its INSERT arrives. The SQL text is checked for its contract: every
placeholder bound, no credential in the text, the columns and keys of the raw tables, the dedup
readers of silver must apply, and the snapshot every statement shares.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import restore_from_lake
from cdc_schemas import TABLES as CDC_TABLES
from clickhouse_connect.driver.exceptions import DatabaseError
from ds_runtime import (
    EXIT_OK,
    EXIT_SKIPPED,
    PreconditionsNotMetError,
    StopToken,
    TerminationRequestedError,
    run_cli,
)
from ds_settings import S3Settings
from hf_features import FEATURE_COLUMNS
from reconcile_lakehouse import (
    EXIT_DATA_MISMATCH,
    LAYER_COUNTS_SQL,
    REQUIRED_TABLES,
    RESULTS_TABLE,
    SILVER_PROBE_SQL,
    PostgresSource,
    ReconcileSettings,
)
from restore_from_lake import (
    INGESTED_AT,
    PARTITIONS_SQL,
    RAW_ENGINE_HF_FEATURES,
    RAW_ENGINE_STATUS_REGISTRY,
    RAW_ENGINE_TELEMETRY,
    RAW_MAINTENANCE_LOG,
    RESTORE_TABLES,
    TABLES_BY_NAME,
    TTL_MARGIN,
    DateRange,
    DayCounts,
    RestoreOutcome,
    RestoreRequest,
    RestoreTable,
    SchemaMismatchError,
    day_counts_sql,
    day_parameters,
    lake_partitions,
    partitions_query,
    plan_days,
    reconcile_window,
    require_tables,
    restore_sql,
    run_restore,
    silver_files_url,
)

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 2, 6, 30, 16, 517000, tzinfo=UTC)
SNAPSHOT = datetime(2026, 11, 2, 6, 30, 16, tzinfo=UTC)
AUDIT_AT = datetime(2026, 11, 2, 6, 31, 2, 250000, tzinfo=UTC)
S3_KEY_ID = "lake-key-id"
S3_SECRET = "lake-secret-key"
PG_PASSWORD = "pg-secret-password"
S3 = S3Settings("http://minio:9000/", S3_KEY_ID, S3_SECRET, "us-east-1", "lakehouse")
RECONCILE = ReconcileSettings(
    window_hours=6, lag_seconds=600, postgres=PostgresSource("postgres", 5432, "db", "admin", PG_PASSWORD)
)
DAYS = (date(2026, 10, 26), date(2026, 10, 27), date(2026, 11, 1), date(2026, 11, 2))
PLACEHOLDER = re.compile(r"\{(\w+):([^{}]+)\}")
# ClickHouse hides query parameters with one of these in their name (see test_ds_reconcile).
MASKED_NAME_PARTS = ("password", "secret", "key", "token", "credential")
CDC_SPECS = {spec.clickhouse_table: spec for spec in CDC_TABLES}


# ------------------------------------------------------------------------------------ fakes


@dataclass
class DayState:
    """What the lake holds for one table and day, and how many of those rows raw lacks."""

    lake: int
    raw: int
    missing: int


@dataclass
class FakeLake:
    """RestoreGateway over an in-memory lake and raw state; records every statement.

    ``existing`` are the ClickHouse tables; ``lossy`` days keep missing their rows after the
    INSERT (a restore that did not land); ``hourly`` answers the reconciliation's federated count
    query (rows as in test_ds_reconcile).
    """

    days: dict[tuple[str, date], DayState] = field(default_factory=dict)
    existing: tuple[str, ...] = (*TABLES_BY_NAME, *REQUIRED_TABLES)
    columns: dict[str, list[tuple[str, str]]] | None = None
    partition_error: Exception | None = None
    lossy: frozenset[tuple[str, date]] = frozenset()
    hourly: list[tuple[Any, ...]] = field(default_factory=list)
    queries: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    commands: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    inserts: list[tuple[str, list[tuple[Any, ...]], list[str]]] = field(default_factory=list)

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        wanted = set(tables)
        return {table: "ReplacingMergeTree" for table in self.existing if table in wanted}

    def query_rows(self, sql: str, parameters: Mapping[str, Any] | None = None) -> list[tuple[Any, ...]]:
        parameters = dict(parameters or {})
        self.queries.append((sql, parameters))
        if "FROM system.columns" in sql:
            return self._columns(parameters["tables"])
        if sql == PARTITIONS_SQL:
            if self.partition_error is not None:
                raise self.partition_error
            return self._partitions()
        if sql == SILVER_PROBE_SQL:
            return [(1,)]
        if sql == LAYER_COUNTS_SQL:
            return list(self.hourly)
        table, day = self._day_of(sql, parameters, day_counts_sql)
        state = self.days[(table.name, day)]
        return [(state.lake, state.raw, state.missing)]

    def command(self, statement: str, parameters: Mapping[str, Any] | None = None) -> None:
        parameters = dict(parameters or {})
        self.commands.append((statement, parameters))
        table, day = self._day_of(statement, parameters, restore_sql)
        if (table.name, day) not in self.lossy:
            state = self.days[(table.name, day)]
            state.raw, state.missing = state.raw + state.missing, 0

    def insert_rows(self, table: str, rows: Sequence[Sequence[Any]], column_names: Sequence[str]) -> None:
        self.inserts.append((table, [tuple(row) for row in rows], list(column_names)))

    @property
    def restored(self) -> list[tuple[str, date]]:
        return [self._day_of(sql, parameters, restore_sql) for sql, parameters in self.commands]

    def _columns(self, names: Sequence[str]) -> list[tuple[Any, ...]]:
        columns = self.columns or {
            table.name: [*table.columns, (INGESTED_AT, "DateTime64(3, 'UTC')")] for table in RESTORE_TABLES
        }
        return [(table, name, kind) for table in names for name, kind in columns.get(table, [])]

    def _partitions(self) -> list[tuple[Any, ...]]:
        datasets = {table.name: table.dataset for table in RESTORE_TABLES}
        return sorted((datasets[table], day, 3) for table, day in self.days)

    @staticmethod
    def _day_of(sql: str, parameters: Mapping[str, Any], build: Any) -> tuple[RestoreTable, date]:
        [table] = [table for table in RESTORE_TABLES if build(table) == sql]
        return table, parameters["day_start"].date()


def lake(
    tables: Iterable[RestoreTable] = RESTORE_TABLES,
    days: Iterable[date] = DAYS,
    *,
    lake_rows: int = 100,
    missing: int = 0,
    **options: Any,
) -> FakeLake:
    """A lake where every table and day holds ``lake_rows`` rows, ``missing`` of them not in raw."""
    state = {
        (table.name, day): DayState(lake_rows, lake_rows - missing, missing)
        for table in tables
        for day in days
    }
    return FakeLake(days=state, **options)


def hourly(engine: str = "E1", hour: int = 5, **held: int) -> tuple[Any, ...]:
    """A LAYER_COUNTS_SQL row of 2026-11-02 where every layer holds the 3600 source pulses unless set."""
    source = 3600
    return (
        engine,
        datetime(2026, 11, 2, hour),
        source,
        source,
        held.get("raw", source),
        held.get("staging", source),
        held.get("silver", source),
        held.get("frames", source),
    )


def _restore(
    gateway: FakeLake, request: RestoreRequest | None = None, *, stop: StopToken | None = None
) -> RestoreOutcome:
    return run_restore(
        gateway,
        request or RestoreRequest(),
        S3,
        RECONCILE,
        now=NOW,
        stop=stop or StopToken(),
        clock=lambda: AUDIT_AT,
    )


def _ddl_columns(table: str) -> list[tuple[str, str]]:
    """(name, type) of every column of ``table`` in init-db.sql, DEFAULT clauses dropped."""
    body = _ddl_body(table)
    columns = []
    for line in body.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("INDEX", "--")):
            continue
        name, kind = line.split(None, 1)
        columns.append((name, kind.split(" DEFAULT ")[0].strip()))
    return columns


def _ddl_body(table: str) -> re.Match[str]:
    ddl = (REPO / "infrastructure" / "clickhouse" / "init-db.sql").read_text(encoding="utf-8")
    body = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\n(.*?);", ddl, re.DOTALL)
    assert body, f"{table} not found in init-db.sql"
    return body


def _engine_clause(table: str) -> str:
    return _ddl_body(table).group(2)


# -------------------------------------------------------------------------- the contracts


@pytest.mark.parametrize("table", RESTORE_TABLES, ids=lambda table: table.name)
def test_every_column_but_ingested_at_has_the_raw_tables_type(table: RestoreTable) -> None:
    ddl = _ddl_columns(table.name)

    assert ddl[-1] == (INGESTED_AT, "DateTime64(3, 'UTC')")
    assert list(table.columns) == ddl[:-1]
    assert INGESTED_AT not in table.column_names


@pytest.mark.parametrize("table", RESTORE_TABLES, ids=lambda table: table.name)
def test_the_identity_is_the_sorting_key_the_engine_collapses_on(table: RestoreTable) -> None:
    engine = _engine_clause(table.name)

    assert f"ORDER BY ({', '.join(table.key)})" in engine
    versioned = re.search(r"ENGINE = ReplacingMergeTree\((\w+)\)", engine)
    if table.version is None:
        # Unversioned, or versioned by ingested_at, which the lake cannot give back.
        assert versioned is None or versioned.group(1) == INGESTED_AT
    else:
        assert versioned is not None and versioned.group(1) == table.version


@pytest.mark.parametrize("table", RESTORE_TABLES, ids=lambda table: table.name)
def test_rows_beyond_the_raw_tables_ttl_are_not_restored(table: RestoreTable) -> None:
    ttl = re.search(
        rf"TTL toDateTime\({table.event_time}\) \+ INTERVAL (\d+) DAY", _engine_clause(table.name)
    )

    assert (int(ttl.group(1)) if ttl else None) == table.ttl_days


@pytest.mark.parametrize("raw_table", list(CDC_SPECS))
def test_the_cdc_tables_restore_exactly_the_columns_the_streaming_job_inserts(raw_table: str) -> None:
    spec, table = CDC_SPECS[raw_table], TABLES_BY_NAME[raw_table]

    assert table.column_names == spec.clickhouse_columns
    assert table.dataset == spec.name
    assert table.event_time == spec.event_time_field  # the column of the silver partition
    assert set(spec.primary_key) <= set(table.key)


def test_the_hf_table_restores_the_feature_columns_and_turns_null_floats_into_nan() -> None:
    table = RAW_ENGINE_HF_FEATURES

    assert table.column_names == tuple(name for name, _ in FEATURE_COLUMNS)
    assert table.nan_columns == {name for name, kind in FEATURE_COLUMNS if kind == "float"}
    assert table.nan_map == next(name for name, kind in FEATURE_COLUMNS if kind.startswith("map"))
    assert table.event_time == "frame_start_ts"


def test_the_catalogue_covers_the_four_raw_tables_in_order() -> None:
    assert [table.name for table in RESTORE_TABLES] == [
        "raw_engine_telemetry",
        "raw_engine_status_registry",
        "raw_maintenance_log",
        "raw_engine_hf_features",
    ]
    assert RAW_ENGINE_STATUS_REGISTRY.key == ("engine_id", "cdc_lsn")  # the full change history
    assert RAW_ENGINE_TELEMETRY.version == RAW_MAINTENANCE_LOG.version == "cdc_lsn"


def test_the_job_imports_nothing_the_data_platform_image_lacks() -> None:
    # The image ships data_science/ and orchestration/ only: no streaming module may be imported.
    tree = ast.parse((REPO / "data_science" / "restore_from_lake.py").read_text(encoding="utf-8"))
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    local = {path.stem for path in (REPO / "data_science").glob("*.py")}
    standard = {"__future__", "argparse", "collections", "dataclasses", "datetime", "logging", "typing"}

    assert modules - local <= standard | {"clickhouse_connect"}
    assert {"cdc_schemas", "hf_features", "hf_rows", "clickhouse_rows"}.isdisjoint(modules)


# ---------------------------------------------------------------------------- SQL builders


def _statements() -> list[tuple[str, tuple[str, dict[str, Any]]]]:
    day = date(2026, 11, 1)
    built = [("partitions", partitions_query(S3, RESTORE_TABLES, SNAPSHOT))]
    for table in RESTORE_TABLES:
        parameters = day_parameters(table, S3, day, SNAPSHOT)
        built.append((f"counts-{table.name}", (day_counts_sql(table), parameters)))
        built.append((f"restore-{table.name}", (restore_sql(table), parameters)))
    return built


STATEMENTS = _statements()


@pytest.mark.parametrize(("_name", "statement"), STATEMENTS, ids=[name for name, _ in STATEMENTS])
def test_every_placeholder_is_bound_and_every_parameter_used(_name: str, statement: Any) -> None:
    sql, parameters = statement

    assert {name for name, _ in PLACEHOLDER.findall(sql)} == set(parameters)


@pytest.mark.parametrize(("_name", "statement"), STATEMENTS, ids=[name for name, _ in STATEMENTS])
def test_credentials_travel_as_masked_parameters_never_in_the_sql(_name: str, statement: Any) -> None:
    sql, parameters = statement

    credentials = {name for name, value in parameters.items() if value in (S3_KEY_ID, S3_SECRET)}
    assert credentials == {"s3_access_key_id", "s3_secret_access_key"}
    for name in credentials:
        assert any(part in name for part in MASKED_NAME_PARTS), name
    for secret in (S3_KEY_ID, S3_SECRET):
        assert secret not in sql


@pytest.mark.parametrize(("_name", "statement"), STATEMENTS, ids=[name for name, _ in STATEMENTS])
def test_every_statement_reads_only_the_files_written_before_the_snapshot(_name: str, statement: Any) -> None:
    sql, parameters = statement

    assert "_time < {snapshot:DateTime('UTC')}" in sql
    assert parameters["snapshot"] == SNAPSHOT


def test_the_silver_globs_name_the_datasets_and_the_day() -> None:
    root = "http://minio:9000/lakehouse/silver"

    assert silver_files_url(S3, ["engine_telemetry"]) == f"{root}/engine_telemetry/event_date=*/*.parquet"
    assert silver_files_url(S3, ["maintenance_log"], date(2026, 11, 1)) == (
        f"{root}/maintenance_log/event_date=2026-11-01/*.parquet"
    )
    assert silver_files_url(S3, ["engine_telemetry", "engine_hf_features"]) == (
        f"{root}/{{engine_telemetry,engine_hf_features}}/event_date=*/*.parquet"
    )
    with pytest.raises(ValueError, match="dataset"):
        silver_files_url(S3, [])


def test_the_partition_listing_reads_no_data_and_names_dataset_and_day() -> None:
    sql, parameters = partitions_query(S3, [RAW_ENGINE_TELEMETRY, RAW_MAINTENANCE_LOG], SNAPSHOT)

    assert "'One')" in sql
    assert "extract(_path, '/silver/([^/]+)/event_date=') AS dataset" in sql
    assert parameters["silver_files"].endswith(
        "/silver/{engine_telemetry,maintenance_log}/event_date=*/*.parquet"
    )


def test_the_day_parameters_bound_one_utc_day_of_one_partition() -> None:
    parameters = day_parameters(RAW_ENGINE_TELEMETRY, S3, date(2026, 11, 1), SNAPSHOT)

    assert parameters["day_start"] == datetime(2026, 11, 1, tzinfo=UTC)
    assert parameters["day_end"] == datetime(2026, 11, 2, tzinfo=UTC)
    assert parameters["silver_files"].endswith("/silver/engine_telemetry/event_date=2026-11-01/*.parquet")
    assert parameters["structure"] == RAW_ENGINE_TELEMETRY.structure
    assert "kept_since" not in parameters


def test_the_hf_rows_start_one_ttl_after_the_snapshot_with_a_margin() -> None:
    parameters = day_parameters(RAW_ENGINE_HF_FEATURES, S3, date(2026, 11, 1), SNAPSHOT)

    assert parameters["kept_since"] == SNAPSHOT - timedelta(days=90) + TTL_MARGIN
    assert "frame_start_ts >= {kept_since:DateTime64(6, 'UTC')}" in restore_sql(RAW_ENGINE_HF_FEATURES)


@pytest.mark.parametrize("table", RESTORE_TABLES, ids=lambda table: table.name)
def test_the_restore_inserts_every_column_but_ingested_at_by_name(table: RestoreTable) -> None:
    sql = restore_sql(table)
    columns = ", ".join(table.column_names)

    assert f"INSERT INTO {table.name} ({columns})\nSELECT {columns}\n" in sql
    assert INGESTED_AT not in sql


@pytest.mark.parametrize("table", RESTORE_TABLES, ids=lambda table: table.name)
def test_silver_is_deduplicated_to_the_newest_image_per_identity(table: RestoreTable) -> None:
    sql = restore_sql(table)
    key = ", ".join(table.key)
    version = f"{table.version} DESC, " if table.version else ""

    # Copies of one version: the last Kafka delivery, the copy raw keeps across micro-batches.
    assert f"ORDER BY {key}, {version}kafka_partition DESC, kafka_offset DESC\n    LIMIT 1 BY {key}" in sql
    assert f") AS raw USING ({key})" in sql
    # The count reads the lake exactly like the insert.
    lake_rows = sql[sql.index("FROM\n(") : sql.index(") AS lake")]
    assert lake_rows in day_counts_sql(table)


@pytest.mark.parametrize(
    ("table", "missing"),
    [
        (RAW_ENGINE_TELEMETRY, "WHERE raw.in_raw = 0 OR raw.raw_version < lake.cdc_lsn"),
        (RAW_MAINTENANCE_LOG, "WHERE raw.in_raw = 0 OR raw.raw_version < lake.cdc_lsn"),
        (RAW_ENGINE_STATUS_REGISTRY, "WHERE raw.in_raw = 0"),
        (RAW_ENGINE_HF_FEATURES, "WHERE raw.in_raw = 0"),
    ],
    ids=lambda value: getattr(value, "name", "condition"),
)
def test_only_rows_raw_lacks_are_inserted(table: RestoreTable, missing: str) -> None:
    sql = restore_sql(table)

    assert sql.rstrip().endswith(missing)
    assert f"countIf({missing.removeprefix('WHERE ')}) AS to_restore" in day_counts_sql(table)
    in_day = f"{table.event_time} >= {{day_start:DateTime64(6, 'UTC')}}"
    assert in_day in sql and f"{table.event_time} < {{day_end:DateTime64(6, 'UTC')}}" in sql


def test_raw_versions_are_the_newest_per_identity_of_the_day() -> None:
    assert "SELECT engine_id, telemetry_id, max(cdc_lsn) AS raw_version, 1 AS in_raw" in restore_sql(
        RAW_ENGINE_TELEMETRY
    )
    assert "SELECT DISTINCT engine_id, cdc_lsn, 1 AS in_raw" in restore_sql(RAW_ENGINE_STATUS_REGISTRY)


def test_the_counts_compare_identities_of_lake_and_raw() -> None:
    sql = day_counts_sql(RAW_ENGINE_HF_FEATURES)

    assert "count() AS lake_rows" in sql
    assert "SELECT uniqExact(engine_id, telemetry_id)\n        FROM raw_engine_hf_features" in sql


def test_hf_floats_read_as_nullable_and_null_becomes_nan() -> None:
    table = RAW_ENGINE_HF_FEATURES
    projection = table.projection()

    assert "fan_kurtosis Nullable(Float32), " in table.structure
    assert "frame_start_ts DateTime64(6, 'UTC'), " in table.structure
    assert "ifNull(fan_kurtosis, toFloat32(nan)) AS fan_kurtosis" in projection
    assert "ifNull(egt_ab_split, toFloat32(nan)) AS egt_ab_split" in projection
    assert (
        "mapSort(mapApply((k, v) -> (k, ifNull(v, toFloat32(nan))), process_means)) AS process_means"
        in projection
    )
    assert "process_means Map(String, Nullable(Float32))" in table.structure
    assert "frame_bytes" in projection  # integers pass through unchanged


def test_cdc_columns_read_with_the_raw_types_unchanged() -> None:
    table = RAW_ENGINE_TELEMETRY

    assert table.projection() == table.column_names
    assert table.structure == ", ".join(f"{name} {kind}" for name, kind in table.columns)
    assert "cdc_source_ts DateTime64(3, 'UTC')" in table.structure
    assert "event_ts DateTime64(6, 'UTC')" in table.structure
    assert "maintenance_id Nullable(UInt64)" in table.structure


# ------------------------------------------------------------------------------ date range


def test_a_date_range_is_inclusive_and_open_where_unset() -> None:
    first_of_november = DateRange(date(2026, 11, 1))
    october = DateRange(date(2026, 10, 1), date(2026, 10, 31))

    assert date(2026, 11, 1) in first_of_november and date(2026, 10, 31) not in first_of_november
    assert date(2026, 10, 31) in october and date(2026, 11, 1) not in october
    assert date(1999, 1, 1) in DateRange()
    assert "2026-11-01" not in october
    assert first_of_november.describe() == "2026-11-01..*"


def test_a_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="after --to"):
        DateRange(date(2026, 11, 2), date(2026, 11, 1))


def test_the_plan_keeps_the_partitions_inside_the_range() -> None:
    request = RestoreRequest(dates=DateRange(date(2026, 10, 27), date(2026, 11, 1)))
    partitions = {table.name: DAYS for table in RESTORE_TABLES}

    planned = plan_days(request, partitions, SNAPSHOT)

    assert planned == {table.name: (date(2026, 10, 27), date(2026, 11, 1)) for table in RESTORE_TABLES}


def test_hf_days_wholly_beyond_the_ttl_are_skipped(caplog: pytest.LogCaptureFixture) -> None:
    kept_since = SNAPSHOT - timedelta(days=90) + TTL_MARGIN
    expired, boundary = kept_since.date() - timedelta(days=1), kept_since.date()
    request = RestoreRequest(tables=(RAW_ENGINE_TELEMETRY, RAW_ENGINE_HF_FEATURES))
    partitions = {table.name: (expired, boundary) for table in request.tables}

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        planned = plan_days(request, partitions, SNAPSHOT)

    assert planned == {
        RAW_ENGINE_TELEMETRY.name: (expired, boundary),
        RAW_ENGINE_HF_FEATURES.name: (boundary,),
    }
    assert "older than its 90-day TTL" in caplog.text


def test_nothing_in_the_range_skips_the_run() -> None:
    request = RestoreRequest(tables=(RAW_MAINTENANCE_LOG,), dates=DateRange(date(2027, 1, 1)))

    with pytest.raises(PreconditionsNotMetError, match=r"silver/maintenance_log .* 2027-01-01\.\.\*"):
        plan_days(request, {RAW_MAINTENANCE_LOG.name: DAYS}, SNAPSHOT)


# ------------------------------------------------------------------------- preconditions


def test_missing_raw_tables_skip_the_run_before_reading_the_lake() -> None:
    gateway = lake(existing=("raw_engine_telemetry",))

    with pytest.raises(PreconditionsNotMetError) as skipped:
        _restore(gateway)

    assert "raw_engine_status_registry, raw_maintenance_log, raw_engine_hf_features" in str(skipped.value)
    assert "make ch-init" in str(skipped.value)
    assert gateway.queries == [] and gateway.commands == []


def test_a_raw_table_with_other_columns_fails_the_run() -> None:
    columns = {
        table.name: [*table.columns, (INGESTED_AT, "DateTime64(3, 'UTC')")] for table in RESTORE_TABLES
    }
    columns["raw_maintenance_log"][4] = ("failure_ts", "DateTime64(3, 'UTC')")

    with pytest.raises(SchemaMismatchError, match=r"raw_maintenance_log .*failure_ts DateTime64\(3, 'UTC'\)"):
        require_tables(lake(columns=columns), RESTORE_TABLES)


def test_a_missing_bucket_skips_the_run() -> None:
    error = DatabaseError(
        "S3 exception: `NoSuchBucket`, message: 'The specified bucket does not exist'", code=499
    )

    with pytest.raises(PreconditionsNotMetError, match="bucket 'lakehouse' does not exist"):
        lake_partitions(lake(partition_error=error), S3, RESTORE_TABLES, SNAPSHOT)


def test_other_lake_errors_fail_the_run() -> None:
    error = DatabaseError("S3 exception: `SignatureDoesNotMatch`", code=499)

    with pytest.raises(DatabaseError):
        lake_partitions(lake(partition_error=error), S3, RESTORE_TABLES, SNAPSHOT)


def test_partitions_are_mapped_from_datasets_to_raw_tables() -> None:
    gateway = lake(tables=[RAW_ENGINE_TELEMETRY], days=[date(2026, 11, 1), date(2026, 10, 26)])

    partitions = lake_partitions(gateway, S3, [RAW_ENGINE_TELEMETRY, RAW_MAINTENANCE_LOG], SNAPSHOT)

    assert partitions == {
        "raw_engine_telemetry": (date(2026, 10, 26), date(2026, 11, 1)),
        "raw_maintenance_log": (),
    }


# --------------------------------------------------------------------------------- dry run


def test_a_dry_run_reports_every_table_and_day_and_writes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=0)
    gateway.days[("raw_engine_telemetry", date(2026, 11, 1))] = DayState(lake=167415, raw=0, missing=167415)

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway, RestoreRequest(dry_run=True))

    assert gateway.commands == [] and gateway.inserts == []
    assert outcome.exit_code == EXIT_OK and outcome.to_restore == 167415
    assert len(outcome.plan) == len(RESTORE_TABLES) * len(DAYS)
    assert DayCounts("raw_engine_telemetry", date(2026, 11, 1), 167415, 0, 167415) in outcome.plan
    assert "plan: raw_engine_telemetry 2026-11-01: lake 167415, raw 0, to restore 167415" in caplog.text
    # The whole message: make dr-drill (infrastructure/clickhouse/dr-drill.sh) parses these totals.
    assert (
        "plan: raw_engine_telemetry, 4 day(s) of the lake as of 2026-11-02T06:30:16Z: "
        "lake 167715, raw 300, to restore 167415"
    ) in caplog.messages
    assert (
        "dry run: nothing written; a restore would add 167415 row(s) to raw_engine_telemetry" in caplog.text
    )
    assert not any(sql == LAYER_COUNTS_SQL for sql, _ in gateway.queries)


def test_the_snapshot_is_fixed_once_for_every_statement() -> None:
    gateway = lake(missing=5, hourly=[hourly()])

    outcome = _restore(gateway)

    assert outcome.snapshot == SNAPSHOT
    lake_reads = [
        parameters for sql, parameters in [*gateway.queries, *gateway.commands] if "snapshot" in parameters
    ]
    assert lake_reads and {parameters["snapshot"] for parameters in lake_reads} == {SNAPSHOT}


# --------------------------------------------------------------------------------- restore


def test_a_restore_inserts_only_the_days_that_miss_rows_then_recounts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = lake(hourly=[hourly()])
    gateway.days[("raw_engine_telemetry", date(2026, 11, 1))] = DayState(lake=100, raw=60, missing=40)
    gateway.days[("raw_engine_hf_features", date(2026, 10, 27))] = DayState(lake=100, raw=93, missing=7)

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert gateway.restored == [
        (RAW_ENGINE_TELEMETRY, date(2026, 11, 1)),
        (RAW_ENGINE_HF_FEATURES, date(2026, 10, 27)),
    ]
    assert [(counts.table, counts.day, counts.to_restore) for counts in outcome.remaining] == [
        ("raw_engine_telemetry", date(2026, 11, 1), 0),
        ("raw_engine_hf_features", date(2026, 10, 27), 0),
    ]
    assert outcome.exit_code == EXIT_OK and outcome.to_restore == 47
    assert "restored raw_engine_telemetry 2026-11-01: 40 row(s) from the lake" in caplog.text
    assert "restore complete: 47 row(s) added to 2 table-day(s)" in caplog.text


def test_a_complete_raw_is_left_alone() -> None:
    gateway = lake(hourly=[hourly()])

    outcome = _restore(gateway)

    assert gateway.commands == [] and outcome.remaining == ()
    assert outcome.exit_code == EXIT_OK


def test_a_second_restore_finds_nothing_to_add() -> None:
    gateway = lake(missing=30, hourly=[hourly()])

    first = _restore(gateway)
    second = _restore(gateway)

    assert first.to_restore == 30 * len(RESTORE_TABLES) * len(DAYS)
    assert second.to_restore == 0 and len(gateway.commands) == len(RESTORE_TABLES) * len(DAYS)


def test_rows_still_missing_after_the_restore_exit_65(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(
        missing=3, lossy=frozenset({("raw_maintenance_log", date(2026, 11, 1))}), hourly=[hourly()]
    )

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert outcome.incomplete == [DayCounts("raw_maintenance_log", date(2026, 11, 1), 100, 97, 3)]
    assert outcome.exit_code == EXIT_DATA_MISMATCH
    assert "restore incomplete: rows of the snapshot are still missing from raw" in caplog.text


def test_only_the_requested_tables_and_days_are_touched() -> None:
    gateway = lake(missing=2)
    request = RestoreRequest(
        tables=(RAW_ENGINE_HF_FEATURES,),
        dates=DateRange(date(2026, 11, 1), date(2026, 11, 1)),
        reconcile=False,
    )

    outcome = _restore(gateway, request)

    assert gateway.restored == [(RAW_ENGINE_HF_FEATURES, date(2026, 11, 1))]
    assert [(counts.table, counts.day) for counts in outcome.plan] == [
        ("raw_engine_hf_features", date(2026, 11, 1))
    ]


def test_a_stop_request_ends_the_run_before_any_insert() -> None:
    gateway = lake(missing=1)
    stop = StopToken()
    stop.request_stop()

    with pytest.raises(TerminationRequestedError):
        _restore(gateway, stop=stop)

    assert gateway.commands == []


def test_credentials_never_reach_the_log(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=1, hourly=[hourly(raw=3599)])

    with caplog.at_level(logging.DEBUG):
        _restore(gateway)

    for secret in (S3_KEY_ID, S3_SECRET, PG_PASSWORD):
        assert secret not in caplog.text


# ---------------------------------------------------------------------- the reconciliation


def test_the_audit_covers_the_restored_telemetry_settled_before_the_snapshot() -> None:
    gateway = lake(missing=1, hourly=[hourly()])

    outcome = _restore(gateway)

    [parameters] = [parameters for sql, parameters in gateway.queries if sql == LAYER_COUNTS_SQL]
    end = SNAPSHOT - timedelta(seconds=600)
    assert parameters["window_end"] == end
    assert parameters["window_start"] == end - timedelta(hours=168)  # capped at a week
    assert outcome.reconciliation is not None and outcome.reconciliation.run_at == AUDIT_AT
    [(table, rows, _)] = gateway.inserts
    assert table == RESULTS_TABLE and {row[1] for row in rows} == {AUDIT_AT}


@pytest.mark.parametrize(
    ("days", "lag", "window"),
    [
        (DAYS, 600, ("2026-10-26T06:20:16Z", "2026-11-02T06:20:16Z")),
        ((date(2026, 10, 26), date(2026, 10, 27)), 600, ("2026-10-26T00:00:00Z", "2026-10-28T00:00:00Z")),
        ((date(2026, 11, 2),), 600, ("2026-11-02T00:20:16Z", "2026-11-02T06:20:16Z")),
        ((date(2026, 11, 2),), 0, ("2026-11-02T00:30:16Z", "2026-11-02T06:30:16Z")),
    ],
    ids=["a-week-at-most", "a-past-range-ends-with-its-last-day", "today", "no-lag"],
)
def test_the_audit_window_spans_whole_hours_of_the_restored_range(
    days: tuple[date, ...], lag: int, window: tuple[str, str]
) -> None:
    found = reconcile_window(days, SNAPSHOT, lag)

    assert found is not None and found.describe() == "..".join(window)


def test_no_audit_while_not_one_hour_of_the_range_settled() -> None:
    assert reconcile_window((date(2026, 11, 2),), datetime(2026, 11, 2, 0, 50, tzinfo=UTC), 600) is None
    assert reconcile_window((), SNAPSHOT, 600) is None


def test_raw_cells_that_differ_from_the_source_fail_the_restore(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=1, hourly=[hourly("E1"), hourly("E2", raw=3590)])

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert outcome.exit_code == EXIT_DATA_MISMATCH
    assert [cell.engine_id for cell in outcome.raw_mismatches] == ["E2"]
    assert "the restored raw layer differs from the source in 1 cell(s)" in caplog.text


def test_staging_behind_the_restored_raw_waits_for_dbt(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=1, hourly=[hourly(staging=0), hourly(hour=4, staging=1200)])

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert outcome.reconciliation is not None and not outcome.reconciliation.reconciled
    assert outcome.exit_code == EXIT_OK
    assert "the restored raw layer matches the source in every cell" in caplog.text
    assert "staging differs in 2 cell(s): dbt publishes the restored rows in its next run" in caplog.text


def test_a_silver_gap_is_reported_but_is_not_the_restores_failure(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=1, hourly=[hourly(silver=3500)])

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert outcome.exit_code == EXIT_OK
    assert "silver differs from the source in 1 cell(s)" in caplog.text


def test_the_audit_is_skipped_until_dbt_built_staging(caplog: pytest.LogCaptureFixture) -> None:
    gateway = lake(missing=1, hourly=[hourly()], existing=(*TABLES_BY_NAME, RESULTS_TABLE))

    with caplog.at_level(logging.INFO, logger="restore_from_lake"):
        outcome = _restore(gateway)

    assert outcome.reconciliation is None and outcome.exit_code == EXIT_OK
    assert "reconciliation skipped: tables do not exist yet: stg_engine_telemetry" in caplog.text
    assert len(gateway.commands) == len(RESTORE_TABLES) * len(DAYS)  # the restore itself stands


def test_no_audit_when_asked_or_when_telemetry_was_not_restored(caplog: pytest.LogCaptureFixture) -> None:
    for request in (
        RestoreRequest(reconcile=False),
        RestoreRequest(tables=(RAW_ENGINE_STATUS_REGISTRY, RAW_MAINTENANCE_LOG)),
    ):
        gateway = lake(missing=1, hourly=[hourly(raw=0)])
        with caplog.at_level(logging.INFO, logger="restore_from_lake"):
            outcome = _restore(gateway, request)
        assert outcome.reconciliation is None and outcome.exit_code == EXIT_OK
        assert not any(sql == LAYER_COUNTS_SQL for sql, _ in gateway.queries)
    assert "reconciliation not requested (--no-reconcile)" in caplog.text
    assert "reconciliation skipped: raw_engine_telemetry was not restored" in caplog.text


# --------------------------------------------------------------------------------------- CLI


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> list[FakeLake]:
    """Point main() at the fake lake appended to the returned list (the last one wins)."""
    gateways: list[FakeLake] = []

    class _NoSignals(StopToken):
        def install_signal_handlers(self) -> StopToken:
            return self

    monkeypatch.setattr(
        restore_from_lake, "ClickHouseGateway", lambda _settings: contextlib.nullcontext(gateways[-1])
    )
    monkeypatch.setattr(restore_from_lake, "configure_logging", lambda _level: None)
    monkeypatch.setattr(restore_from_lake, "StopToken", _NoSignals)
    for name in ("RECONCILE_WINDOW_HOURS", "RECONCILE_LAG_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", S3_SECRET)
    return gateways


@pytest.mark.parametrize(
    ("held", "code"),
    [({}, EXIT_OK), ({"raw": 3000}, EXIT_DATA_MISMATCH), ({"staging": 0}, EXIT_OK)],
    ids=["reconciled", "raw-differs", "staging-waits-for-dbt"],
)
def test_main_exits_with_the_verdict_on_raw(cli: list[FakeLake], held: dict[str, int], code: int) -> None:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    cli.append(lake(days=[yesterday], missing=4, hourly=[hourly(**held)]))

    assert restore_from_lake.main([]) == code
    assert cli[-1].commands


def test_main_passes_range_tables_and_dry_run(cli: list[FakeLake]) -> None:
    cli.append(lake(missing=4))

    code = restore_from_lake.main(
        ["--from", "2026-11-01", "--to", "2026-11-01", "--tables", "raw_maintenance_log", "--dry-run"]
    )

    assert code == EXIT_OK and cli[-1].commands == []
    counted = [
        parameters for sql, parameters in cli[-1].queries if sql == day_counts_sql(RAW_MAINTENANCE_LOG)
    ]
    assert [parameters["day_start"].date() for parameters in counted] == [date(2026, 11, 1)]
    assert {parameters["s3_secret_access_key"] for parameters in counted} == {S3_SECRET}


def test_main_skips_with_exit_3_through_run_cli(cli: list[FakeLake]) -> None:
    cli.append(lake(missing=4))

    with pytest.raises(SystemExit) as exited:
        run_cli(lambda: restore_from_lake.main(["--from", "2030-01-01"]))

    assert exited.value.code == EXIT_SKIPPED
    assert cli[-1].commands == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--from", "2026-11-02", "--to", "2026-11-01"],
        ["--from", "01.11.2026"],
        ["--tables", "raw_cdc_dead_letter"],
        ["--tables"],
    ],
    ids=["reversed-range", "not-iso", "unknown-table", "no-table"],
)
def test_bad_arguments_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        restore_from_lake.main(argv)

    assert exited.value.code == 2
    assert "error:" in capsys.readouterr().err
