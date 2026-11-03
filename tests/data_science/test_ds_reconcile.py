"""Reconciliation audit: settings, window, SQL builders, classification and the job end to end.

The federated SQL itself runs on ClickHouse only; here a fake gateway answers it with the
hourly counts ClickHouse would return, and the SQL text is checked for its contract: every
placeholder bound, no credential in the text, every layer under the same window and commit cutoff.
"""

from __future__ import annotations

import contextlib
import logging
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import reconcile_lakehouse
import yaml
from clickhouse_connect.driver.exceptions import DatabaseError
from ds_runtime import (
    EXIT_OK,
    EXIT_SKIPPED,
    PreconditionsNotMetError,
    StopToken,
    TerminationRequestedError,
    run_cli,
)
from ds_settings import ConfigError, S3Settings
from feature_pipeline import RAW_SENSOR_COLUMNS
from reconcile_lakehouse import (
    COUNT_COLUMNS,
    EXIT_DATA_MISMATCH,
    HF_TABLE,
    LAYER_COUNTS_SQL,
    RAW_TABLE,
    REQUIRED_TABLES,
    RESULT_COLUMNS,
    RESULTS_TABLE,
    SILVER_PROBE_SQL,
    SOURCE_TABLE,
    STAGING_TABLE,
    CellStatus,
    HourlyCounts,
    Layer,
    PostgresSource,
    ReconcileSettings,
    ReconcileWindow,
    ReconciliationCell,
    ReconciliationRun,
    classify,
    layer_counts_query,
    load_reconcile_settings,
    run_reconciliation,
    silver_files_url,
    silver_probe_query,
)

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 1, 13, 45, 30, 123456, tzinfo=UTC)
PG_PASSWORD = "pg-secret-password"
S3_KEY_ID = "lake-key-id"
S3_SECRET = "lake-secret-key"
S3 = S3Settings("http://minio:9000/", S3_KEY_ID, S3_SECRET, "us-east-1", "lakehouse")
POSTGRES = PostgresSource("postgres", 5432, "jet_engine_db", "auditor", PG_PASSWORD)
SETTINGS = ReconcileSettings(window_hours=6, lag_seconds=600, postgres=POSTGRES)
WINDOW = ReconcileWindow.before(NOW, hours=6, lag_seconds=600)
PLACEHOLDER = re.compile(r"\{(\w+):([^{}]+)\}")
# ClickHouse writes [HIDDEN] for query parameters with one of these in their name, also in the
# request URI its trace log records (checked against 26.8; "pass" alone is not enough).
MASKED_NAME_PARTS = ("password", "secret", "key", "token", "credential")


def hourly(
    engine: str = "JT9D-001",
    hour: int = 12,
    *,
    source: int = 3600,
    finite: int | None = None,
    raw: int | None = None,
    staging: int | None = None,
    silver: int | None = None,
    frames: int | None = None,
) -> tuple[Any, ...]:
    """A result row of LAYER_COUNTS_SQL; unset layers hold exactly what they must (naive UTC hour)."""
    finite = source if finite is None else finite
    return (
        engine,
        datetime(2026, 11, 1, hour),
        source,
        finite,
        source if raw is None else raw,
        finite if staging is None else staging,
        source if silver is None else silver,
        source if frames is None else frames,
    )


class FakeGateway:
    """ReconciliationGateway over canned answers; records every query and insert."""

    def __init__(
        self,
        counts: Iterable[Sequence[Any]] = (),
        *,
        tables: Iterable[str] = tuple(REQUIRED_TABLES),
        silver_files: int = 1,
        probe_error: Exception | None = None,
    ) -> None:
        self.tables = dict.fromkeys(tables, "MergeTree")
        self.counts = [tuple(row) for row in counts]
        self.silver_files = silver_files
        self.probe_error = probe_error
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.inserts: list[tuple[str, list[tuple[Any, ...]], list[str]]] = []

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        wanted = set(tables)
        return {table: engine for table, engine in self.tables.items() if table in wanted}

    def query_rows(self, sql: str, parameters: Mapping[str, Any] | None = None) -> list[tuple[Any, ...]]:
        self.queries.append((sql, dict(parameters or {})))
        if sql == SILVER_PROBE_SQL:
            if self.probe_error is not None:
                raise self.probe_error
            return [(self.silver_files,)]
        if sql == LAYER_COUNTS_SQL:
            return list(self.counts)
        raise AssertionError(f"unexpected query: {sql}")

    def insert_rows(self, table: str, rows: Sequence[Sequence[Any]], column_names: Sequence[str]) -> None:
        self.inserts.append((table, [tuple(row) for row in rows], list(column_names)))


class _NoSignals(StopToken):
    def install_signal_handlers(self) -> StopToken:
        return self


def _reconcile(gateway: FakeGateway, stop: StopToken | None = None) -> ReconciliationRun:
    return run_reconciliation(gateway, SETTINGS, S3, now=NOW, stop=stop or StopToken())


def _cell(layer: Layer, source: int, held: int, engine: str = "E1", hour: int = 12) -> ReconciliationCell:
    return ReconciliationCell(layer, engine, datetime(2026, 11, 1, hour, tzinfo=UTC), source, held)


def _run(cells: Iterable[ReconciliationCell], run_id: uuid.UUID | None = None) -> ReconciliationRun:
    return ReconciliationRun(run_id or uuid.uuid4(), NOW, WINDOW, tuple(cells))


def _cells(*rows: tuple[Any, ...]) -> list[ReconciliationCell]:
    return [cell for row in rows for cell in HourlyCounts.from_row(row).cells()]


def _branches(sql: str) -> list[str]:
    """The five per-layer SELECTs inside the outer aggregation of LAYER_COUNTS_SQL."""
    inner = sql[sql.index("FROM\n(") : sql.rindex("\n)\nGROUP BY")]
    return inner.split("UNION ALL")


def _table_columns(table: str) -> list[str]:
    ddl = (REPO / "infrastructure" / "clickhouse" / "init-db.sql").read_text(encoding="utf-8")
    body = re.search(rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)", ddl, re.DOTALL)
    assert body, f"{table} not found in init-db.sql"
    return [line.split()[0] for line in body.group(1).splitlines() if line.strip()]


# --------------------------------------------------------------------------------- settings


def test_settings_outside_compose_default_to_the_superuser_fallback() -> None:
    settings = load_reconcile_settings({})

    assert (settings.window_hours, settings.lag_seconds) == (6, 600)
    assert settings.postgres == PostgresSource("postgres", 5432, "jet_engine_db", "admin", "adminpassword")
    assert settings.postgres.address == "postgres:5432"


def test_settings_come_from_the_environment() -> None:
    env = {
        "RECONCILE_WINDOW_HOURS": "24",
        "RECONCILE_LAG_SECONDS": "0",
        "POSTGRES_HOST": "source-db",
        "POSTGRES_PORT": "15432",
        "POSTGRES_DB": "fleet",
        "POSTGRES_USER": "admin",
        "POSTGRES_PASSWORD": "superuser-password",
        "POSTGRES_AUDIT_USER": "auditor",
        "POSTGRES_AUDIT_PASSWORD": PG_PASSWORD,
    }

    settings = load_reconcile_settings(env)

    assert (settings.window_hours, settings.lag_seconds) == (24, 0)
    assert settings.postgres == PostgresSource("source-db", 15432, "fleet", "auditor", PG_PASSWORD)
    assert PG_PASSWORD not in repr(settings)


@pytest.mark.parametrize("audit_user", [None, " "], ids=["unset", "blank"])
def test_an_environment_without_the_audit_role_reads_as_postgres_user(audit_user: str | None) -> None:
    env = {"POSTGRES_USER": "admin", "POSTGRES_PASSWORD": "superuser-password"}
    if audit_user is not None:
        env |= {"POSTGRES_AUDIT_USER": audit_user, "POSTGRES_AUDIT_PASSWORD": PG_PASSWORD}

    postgres = load_reconcile_settings(env).postgres

    # An old .env or a run outside compose: the superuser's login, never its name with another password.
    assert (postgres.user, postgres.password) == ("admin", "superuser-password")


def test_compose_hands_the_audit_the_role_postgres_init_creates_not_the_superuser() -> None:
    compose = yaml.safe_load((REPO / "infrastructure" / "docker-compose.yml").read_text(encoding="utf-8"))
    audit = compose["services"]["orchestrator"]["environment"]
    roles = compose["services"]["postgres-init"]["environment"]

    for name in ("POSTGRES_AUDIT_USER", "POSTGRES_AUDIT_PASSWORD"):
        assert audit[name] == roles[name], name
    assert not {"POSTGRES_USER", "POSTGRES_PASSWORD"} & audit.keys()


def test_blank_values_fall_back_to_the_defaults() -> None:
    names = ("RECONCILE_WINDOW_HOURS", "RECONCILE_LAG_SECONDS", "POSTGRES_HOST", "POSTGRES_PORT")
    env = dict.fromkeys(names, " ")

    settings = load_reconcile_settings(env)

    assert (settings.window_hours, settings.lag_seconds) == (6, 600)
    assert settings.postgres.address == "postgres:5432"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RECONCILE_WINDOW_HOURS", "0"),
        ("RECONCILE_WINDOW_HOURS", "169"),
        ("RECONCILE_WINDOW_HOURS", "six"),
        ("RECONCILE_LAG_SECONDS", "-1"),
        ("RECONCILE_LAG_SECONDS", "1.5"),
        ("POSTGRES_PORT", "0"),
        ("POSTGRES_PORT", "70000"),
    ],
)
def test_invalid_settings_name_the_variable(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_reconcile_settings({name: value})


# ----------------------------------------------------------------------------------- window


def test_the_window_ends_the_lag_before_now_on_a_whole_second() -> None:
    assert WINDOW.end == datetime(2026, 11, 1, 13, 35, 30, tzinfo=UTC)
    assert WINDOW.start == datetime(2026, 11, 1, 7, 35, 30, tzinfo=UTC)
    assert WINDOW.event_dates == (date(2026, 11, 1),)
    assert WINDOW.describe() == "2026-11-01T07:35:30Z..2026-11-01T13:35:30Z"


def test_a_window_across_midnight_reads_both_days() -> None:
    window = ReconcileWindow.before(datetime(2026, 11, 1, 3, 0, tzinfo=UTC), hours=6, lag_seconds=600)

    assert window.event_dates == (date(2026, 10, 31), date(2026, 11, 1))


def test_a_window_ending_at_midnight_does_not_read_the_next_day() -> None:
    window = ReconcileWindow.before(datetime(2026, 11, 1, 0, 10, tzinfo=UTC), hours=6, lag_seconds=600)

    assert window.end == datetime(2026, 11, 1, tzinfo=UTC)
    assert window.event_dates == (date(2026, 10, 31),)


def test_a_week_long_window_reads_every_day_it_touches() -> None:
    window = ReconcileWindow.before(NOW, hours=168, lag_seconds=0)

    assert window.event_dates == tuple(date(2026, 10, 25) + timedelta(days=day) for day in range(8))


def test_the_window_is_in_utc_whatever_the_zone_of_now() -> None:
    tehran = timezone(timedelta(hours=3, minutes=30))

    window = ReconcileWindow.before(datetime(2026, 11, 1, 3, 0, tzinfo=tehran), hours=1, lag_seconds=0)

    assert window.end == datetime(2026, 10, 31, 23, 30, tzinfo=UTC)
    assert window.event_dates == (date(2026, 10, 31),)


def test_the_window_needs_an_aware_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ReconcileWindow.before(datetime(2026, 11, 1, 12, 0), hours=6, lag_seconds=600)


def test_the_silver_glob_names_every_partition_or_the_days_of_the_window() -> None:
    root = "http://minio:9000/lakehouse/silver/engine_telemetry"

    assert silver_files_url(S3) == f"{root}/event_date=*/*.parquet"
    assert silver_files_url(S3, (date(2026, 10, 31), date(2026, 11, 1))) == (
        f"{root}/event_date={{2026-10-31,2026-11-01}}/*.parquet"
    )
    with pytest.raises(ValueError, match="event date"):
        silver_files_url(S3, ())


# ------------------------------------------------------------------------------ SQL builders


@pytest.mark.parametrize(
    "build", [lambda: silver_probe_query(S3), lambda: _counts_query()], ids=["silver-probe", "layer-counts"]
)
def test_every_placeholder_is_bound_and_every_parameter_used(build: Any) -> None:
    sql, parameters = build()

    assert {name for name, _ in PLACEHOLDER.findall(sql)} == set(parameters)


@pytest.mark.parametrize(
    "build", [lambda: silver_probe_query(S3), lambda: _counts_query()], ids=["silver-probe", "layer-counts"]
)
def test_credentials_travel_as_masked_parameters_never_in_the_sql(build: Any) -> None:
    sql, parameters = build()

    credentials = {name for name, value in parameters.items() if value in (PG_PASSWORD, S3_KEY_ID, S3_SECRET)}
    assert credentials
    for name in credentials:
        assert any(part in name for part in MASKED_NAME_PARTS), name
    for secret in (PG_PASSWORD, S3_KEY_ID, S3_SECRET):
        assert secret not in sql


def _counts_query() -> tuple[str, dict[str, Any]]:
    return layer_counts_query(WINDOW, POSTGRES, S3)


def test_the_counts_query_binds_the_window_the_source_and_the_windows_silver_days() -> None:
    window = ReconcileWindow.before(datetime(2026, 11, 1, 3, 0, tzinfo=UTC), hours=6, lag_seconds=600)

    _, parameters = layer_counts_query(window, POSTGRES, S3)

    assert (parameters["window_start"], parameters["window_end"]) == (window.start, window.end)
    assert (parameters["pg_address"], parameters["pg_database"]) == ("postgres:5432", "jet_engine_db")
    assert (parameters["pg_user"], parameters["pg_password"]) == ("auditor", PG_PASSWORD)
    assert parameters["silver_files"] == silver_files_url(S3, (date(2026, 10, 31), date(2026, 11, 1)))
    assert (parameters["s3_access_key_id"], parameters["s3_secret_access_key"]) == (S3_KEY_ID, S3_SECRET)


def test_the_counts_query_selects_the_hourly_counts_in_order() -> None:
    outer = LAYER_COUNTS_SQL[: LAYER_COUNTS_SQL.index("FROM\n(")]
    items = [item.strip() for item in outer.removeprefix("\nSELECT").split(",\n")]

    assert [item.rsplit(" AS ", 1)[-1] for item in items] == list(COUNT_COLUMNS)


def test_every_layer_reads_its_table_under_the_same_window_and_commit_cutoff() -> None:
    branches = _branches(LAYER_COUNTS_SQL)
    reads = {
        "source": f"'{SOURCE_TABLE}'",
        "raw": f"FROM {RAW_TABLE} FINAL",
        "staging": f"FROM {STAGING_TABLE} FINAL",
        "silver": "FROM s3(",
        "hf_frames": f"FROM {HF_TABLE}",
    }

    assert len(branches) == len(reads)
    for branch, (layer, read) in zip(branches, reads.items(), strict=True):
        assert f"'{layer}'" in branch
        assert read in branch
        assert re.search(r"_ts >= \{window_start:DateTime64\(6, 'UTC'\)\}", branch), layer
        assert re.search(r"_ts < \{window_end:DateTime64\(6, 'UTC'\)\}", branch), layer
        # The commit cutoff: the source's insert time (created_ts) before the window end.
        assert re.search(r"(created|committed)_ts < \{window_end:DateTime64\(6, 'UTC'\)\}", branch), layer


def test_staging_is_held_to_the_readings_dbt_requires_to_be_finite() -> None:
    macros = REPO / "analytics" / "macros"
    dbt_sensors = re.search(
        r"macro raw_sensor_columns\(\) %\}\s*\{\{ return\(\[(.*?)\]\) \}\}",
        (macros / "feature_columns.sql").read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert dbt_sensors, "raw_sensor_columns() not found in analytics/macros/feature_columns.sql"
    quality = (macros / "telemetry_quality.sql").read_text(encoding="utf-8")
    assert "for sensor in raw_sensor_columns()" in quality

    checked = re.findall(r"isFinite\((\w+)\)", LAYER_COUNTS_SQL)

    assert tuple(checked) == tuple(re.findall(r"'(\w+)'", dbt_sensors.group(1))) == RAW_SENSOR_COLUMNS


def test_silver_keeps_the_newest_image_of_each_pulse_and_drops_deletes() -> None:
    silver = _branches(LAYER_COUNTS_SQL)[3]

    for column in ("engine_id", "event_ts", "created_ts", "is_deleted"):
        assert f"argMax({column}, cdc_lsn)" in silver
    assert "GROUP BY telemetry_id" in silver
    assert "WHERE deleted = 0" in silver


# ----------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("source", "held", "status"),
    [
        (100, 100, CellStatus.OK),
        (100, 99, CellStatus.MISSING),
        (100, 101, CellStatus.EXTRA),
        (0, 0, CellStatus.OK),
        (0, 5, CellStatus.EXTRA),
        (5, 0, CellStatus.MISSING),
    ],
)
def test_classification(source: int, held: int, status: CellStatus) -> None:
    assert classify(source, held) is status
    assert _cell(Layer.RAW, source, held).difference == held - source


def test_each_layer_is_held_to_the_source_pulses_it_must_hold() -> None:
    counts = HourlyCounts.from_row(hourly(source=100, finite=98, raw=100, staging=98, silver=100, frames=97))

    cells = {cell.layer: cell for cell in counts.cells()}

    assert list(cells) == list(Layer)
    assert (cells[Layer.STAGING].source_rows, cells[Layer.STAGING].status) == (98, CellStatus.OK)
    assert (cells[Layer.RAW].source_rows, cells[Layer.RAW].status) == (100, CellStatus.OK)
    assert (cells[Layer.SILVER].source_rows, cells[Layer.SILVER].status) == (100, CellStatus.OK)
    assert (cells[Layer.HF_FRAMES].status, cells[Layer.HF_FRAMES].difference) == (CellStatus.MISSING, -3)


def test_a_staging_layer_that_kept_a_non_finite_pulse_is_extra() -> None:
    counts = HourlyCounts.from_row(hourly(source=100, finite=99, staging=100))

    staging = next(cell for cell in counts.cells() if cell.layer is Layer.STAGING)

    assert (staging.status, staging.difference) == (CellStatus.EXTRA, 1)


def test_rows_from_clickhouse_become_typed_utc_counts() -> None:
    counts = HourlyCounts.from_row(("JT9D-001", datetime(2026, 11, 1, 12), 5, 5, 5, 5, 5, 5))

    assert counts.event_hour == datetime(2026, 11, 1, 12, tzinfo=UTC)
    assert counts.engine_id == "JT9D-001"
    assert counts.hf_frame_rows == 5


def test_hf_frames_are_reported_but_never_fail_a_run() -> None:
    frames = _cell(Layer.HF_FRAMES, 100, 60)
    raw = _cell(Layer.RAW, 100, 100)

    assert frames.informational and frames.status is CellStatus.MISSING and not frames.is_mismatch
    run = _run((raw, frames))
    assert run.reconciled and run.mismatches == []


def test_mismatches_come_largest_first_and_fail_the_run() -> None:
    cells = (
        _cell(Layer.RAW, 100, 99, engine="E1"),
        _cell(Layer.SILVER, 100, 70, engine="E2"),
        _cell(Layer.STAGING, 100, 104, engine="E3"),
        _cell(Layer.HF_FRAMES, 100, 0, engine="E4"),
        _cell(Layer.RAW, 100, 100, engine="E5"),
    )
    run = _run(cells)

    assert not run.reconciled
    worst_first = [(cell.engine_id, cell.difference) for cell in run.mismatches]
    assert worst_first == [("E2", -30), ("E3", 4), ("E1", -1)]


# ------------------------------------------------------------------------------- the results


def test_result_rows_follow_the_columns_of_ops_reconciliation() -> None:
    run_id = uuid.uuid4()
    run = _run((_cell(Layer.SILVER, 100, 97), _cell(Layer.HF_FRAMES, 100, 90)), run_id)

    rows = run.result_rows()

    assert list(RESULT_COLUMNS) == _table_columns(RESULTS_TABLE)
    assert rows[0] == (
        run_id,
        NOW,
        WINDOW.start,
        WINDOW.end,
        "silver",
        "E1",
        datetime(2026, 11, 1, 12, tzinfo=UTC),
        100,
        97,
        -3,
        "MISSING",
        0,
    )
    assert rows[1][-3:] == (-10, "MISSING", 1)


def test_the_summary_of_a_clean_run() -> None:
    run = _run(_cells(hourly("E1", 11), hourly("E1", 12), hourly("E2", 12)), uuid.UUID(int=7))

    assert run.summary() == (
        "reconciliation 00000000-0000-0000-0000-000000000007 of 2026-11-01T07:35:30Z..2026-11-01T13:35:30Z: "
        "raw, staging, silver against 10800 source pulses of 2 engine(s) in 2 hour(s): all 9 cells match "
        "the source; HF frames cover 100.0% of the pulses (informational)"
    )


def test_the_summary_names_the_worst_cells_and_how_many_more_differ() -> None:
    run = _run(
        _cells(
            hourly("E1", source=100, silver=60),
            hourly("E2", source=100, raw=90, frames=50),
            hourly("E3", source=100, staging=103),
            hourly("E4", source=100, silver=99),
        )
    )

    summary = run.summary()

    assert "4 of 12 cells differ from the source, worst: " in summary
    assert (
        "silver E1 2026-11-01T12:00:00Z MISSING 40 (layer 60, source 100); "
        "raw E2 2026-11-01T12:00:00Z MISSING 10 (layer 90, source 100); "
        "staging E3 2026-11-01T12:00:00Z EXTRA 3 (layer 103, source 100) (and 1 more)"
    ) in summary
    assert summary.endswith("HF frames cover 87.5% of the pulses (informational)")


def test_the_summary_without_source_pulses_has_no_coverage() -> None:
    run = _run(_cells(hourly(source=0, raw=4, silver=0, frames=0)))

    assert run.summary().endswith("HF frame coverage n/a (informational)")
    assert [cell.layer for cell in run.mismatches] == [Layer.RAW]


# ------------------------------------------------------------------------- the job end to end


def test_a_run_that_matches_is_recorded_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    gateway = FakeGateway([hourly("E1", 11), hourly("E1", 12), hourly("E2", 11), hourly("E2", 12)])

    with caplog.at_level(logging.INFO, logger="reconcile_lakehouse"):
        run = _reconcile(gateway)

    assert run.reconciled
    assert [sql for sql, _ in gateway.queries] == [SILVER_PROBE_SQL, LAYER_COUNTS_SQL]
    assert gateway.queries[1][1]["window_end"] == datetime(2026, 11, 1, 13, 35, 30, tzinfo=UTC)
    [(table, rows, columns)] = gateway.inserts
    assert (table, columns) == (RESULTS_TABLE, list(RESULT_COLUMNS))
    assert len(rows) == 4 * len(Layer)
    assert {row[0] for row in rows} == {run.run_id}
    assert {row[1] for row in rows} == {NOW}
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert "all 12 cells match the source" in caplog.text


def test_a_run_with_gaps_is_recorded_and_fails(caplog: pytest.LogCaptureFixture) -> None:
    gateway = FakeGateway([hourly("E1", source=3600, staging=3480), hourly("E2", source=3600)])

    with caplog.at_level(logging.INFO, logger="reconcile_lakehouse"):
        run = _reconcile(gateway)

    assert not run.reconciled
    [(_, rows, _)] = gateway.inserts
    assert [row[4:7] for row in rows if row[10] != "OK"] == [
        ("staging", "E1", datetime(2026, 11, 1, 12, tzinfo=UTC))
    ]
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "staging E1 2026-11-01T12:00:00Z MISSING 120 (layer 3480, source 3600)" in caplog.text


def test_credentials_never_reach_the_log(caplog: pytest.LogCaptureFixture) -> None:
    gateway = FakeGateway([hourly("E1", source=10, silver=7)])

    with caplog.at_level(logging.DEBUG):
        _reconcile(gateway)

    for secret in (PG_PASSWORD, S3_KEY_ID, S3_SECRET):
        assert secret not in caplog.text


@pytest.mark.parametrize(
    ("missing", "reasons"),
    [
        ((STAGING_TABLE,), ("the first dbt run builds it",)),
        ((RESULTS_TABLE, HF_TABLE), ("init-db.sql creates it: make ch-init",)),
        ((RAW_TABLE, STAGING_TABLE), ("init-db.sql creates it: make ch-init", "the first dbt run builds it")),
    ],
    ids=["staging-not-built-yet", "init-db-not-applied", "both"],
)
def test_missing_tables_skip_the_run_before_any_query(
    missing: tuple[str, ...], reasons: tuple[str, ...]
) -> None:
    gateway = FakeGateway([hourly()], tables=[table for table in REQUIRED_TABLES if table not in missing])

    with pytest.raises(PreconditionsNotMetError) as skipped:
        _reconcile(gateway)

    for name in (*missing, *reasons):
        assert name in str(skipped.value)
    assert gateway.queries == [] and gateway.inserts == []


def test_a_missing_bucket_skips_the_run() -> None:
    error = DatabaseError(
        "Could not list objects in bucket 'lakehouse' with prefix 'silver/engine_telemetry/event_date=', "
        "S3 exception: `NoSuchBucket`, message: 'The specified bucket does not exist'. (S3_ERROR)",
        code=499,
        name="S3_ERROR",
    )
    gateway = FakeGateway([hourly()], probe_error=error)

    with pytest.raises(PreconditionsNotMetError, match="bucket 'lakehouse' does not exist"):
        _reconcile(gateway)

    assert gateway.inserts == []


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("S3 exception: `SignatureDoesNotMatch`, message: 'The request signature ...'. (S3_ERROR)", 499),
        ("S3 exception: `NoSuchBucket` mentioned by an unrelated error", 60),
    ],
    ids=["bad-credentials", "unrelated-error-code"],
)
def test_other_lake_errors_fail_the_run(message: str, code: int) -> None:
    gateway = FakeGateway([hourly()], probe_error=DatabaseError(message, code=code))

    with pytest.raises(DatabaseError):
        _reconcile(gateway)

    assert gateway.inserts == []


def test_a_lake_without_silver_files_skips_the_run() -> None:
    gateway = FakeGateway([hourly()], silver_files=0)

    with pytest.raises(PreconditionsNotMetError, match="no Parquet file under lakehouse/"):
        _reconcile(gateway)

    assert [sql for sql, _ in gateway.queries] == [SILVER_PROBE_SQL]


def test_a_window_without_settled_pulses_skips_the_run_without_writing() -> None:
    gateway = FakeGateway([])

    with pytest.raises(PreconditionsNotMetError, match="nothing to reconcile yet"):
        _reconcile(gateway)

    assert gateway.inserts == []


def test_a_stop_request_ends_the_run_before_the_federated_query() -> None:
    gateway = FakeGateway([hourly()])
    stop = StopToken()
    stop.request_stop()

    with pytest.raises(TerminationRequestedError):
        _reconcile(gateway, stop)

    assert [sql for sql, _ in gateway.queries] == [SILVER_PROBE_SQL]
    assert gateway.inserts == []


# --------------------------------------------------------------------------------------- CLI


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point main() at a fake gateway; returns a setter for the counts ClickHouse would answer."""
    gateways: list[FakeGateway] = []

    def answer(*rows: tuple[Any, ...], **options: Any) -> list[FakeGateway]:
        gateways.append(FakeGateway(rows, **options))
        return gateways

    monkeypatch.setattr(
        reconcile_lakehouse, "ClickHouseGateway", lambda _settings: contextlib.nullcontext(gateways[-1])
    )
    monkeypatch.setattr(reconcile_lakehouse, "configure_logging", lambda _level: None)
    monkeypatch.setattr(reconcile_lakehouse, "StopToken", _NoSignals)
    for name in ("RECONCILE_WINDOW_HOURS", "RECONCILE_LAG_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POSTGRES_AUDIT_USER", "auditor")
    monkeypatch.setenv("POSTGRES_AUDIT_PASSWORD", PG_PASSWORD)
    return answer


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (hourly("E1"), EXIT_OK),
        (hourly("E1", silver=3599), EXIT_DATA_MISMATCH),
        (hourly("E1", frames=0), EXIT_OK),
    ],
    ids=["reconciled", "silver-missing-a-pulse", "only-hf-frames-missing"],
)
def test_main_exits_0_when_the_layers_match_and_65_when_they_do_not(
    cli: Any, row: tuple[Any, ...], code: int
) -> None:
    [gateway] = cli(row)

    assert reconcile_lakehouse.main([]) == code
    assert len(gateway.inserts) == 1


def test_the_command_line_overrides_the_window(cli: Any) -> None:
    [gateway] = cli(hourly())
    before = datetime.now(UTC)

    assert reconcile_lakehouse.main(["--window-hours", "24", "--lag-seconds", "0"]) == EXIT_OK

    parameters = gateway.queries[1][1]
    assert parameters["window_end"] - parameters["window_start"] == timedelta(hours=24)
    assert before - timedelta(seconds=1) <= parameters["window_end"] <= datetime.now(UTC)
    assert (parameters["pg_user"], parameters["pg_password"]) == ("auditor", PG_PASSWORD)


def test_the_environment_sets_the_window_of_a_scheduled_run(
    cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECONCILE_WINDOW_HOURS", "2")
    monkeypatch.setenv("RECONCILE_LAG_SECONDS", "900")
    [gateway] = cli(hourly())
    before = datetime.now(UTC)

    reconcile_lakehouse.main([])

    parameters = gateway.queries[1][1]
    assert parameters["window_end"] - parameters["window_start"] == timedelta(hours=2)
    assert parameters["window_end"] <= before - timedelta(seconds=899)


def test_a_skip_exits_3_through_the_cli_wrapper(cli: Any) -> None:
    cli(hourly(), tables=[RAW_TABLE, HF_TABLE, RESULTS_TABLE])

    with pytest.raises(SystemExit) as exited:
        run_cli(lambda: reconcile_lakehouse.main([]))

    assert exited.value.code == EXIT_SKIPPED


@pytest.mark.parametrize(
    "argv",
    [["--window-hours", "0"], ["--window-hours", "169"], ["--lag-seconds", "-5"], ["--lag-seconds", "x"]],
    ids=["no-hours", "over-a-week", "negative-lag", "lag-not-a-number"],
)
def test_out_of_range_options_are_rejected(cli: Any, argv: list[str]) -> None:
    cli(hourly())

    with pytest.raises(SystemExit) as exited:
        reconcile_lakehouse.main(argv)

    assert exited.value.code == 2
