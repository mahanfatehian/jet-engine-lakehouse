"""End-to-end reconciliation audit: every layer must hold exactly the telemetry the source committed.

CLI::

    python data_science/reconcile_lakehouse.py [--window-hours H] [--lag-seconds S]

ClickHouse federates the three systems in one query and counts, per engine and hour of event
time, the telemetry pulses of every layer:

    source     PostgreSQL engine_telemetry, through the postgresql() table function
    raw        raw_engine_telemetry FINAL: the current version of each pulse, unless a delete
    staging    the dbt model stg_engine_telemetry FINAL. It also retracts pulses with a
               non-finite reading, so it is held to the source pulses whose readings are finite
    silver     the lake's silver/engine_telemetry Parquet, through the s3() table function: the
               newest image of a telemetry_id (highest cdc_lsn) counts unless it is a delete,
               which also absorbs the copies an interrupted micro-batch leaves behind
    hf_frames  the window's pulses whose HF frame landed in raw_engine_hf_features;
               informational only, because the HF path is best effort

Each (layer, engine, hour) cell is OK, MISSING (the layer holds fewer pulses than the source)
or EXTRA (more), every cell of a run is appended to ops_reconciliation, and one log line sums
the run up. The window is the last RECONCILE_WINDOW_HOURS of event time up to now minus
RECONCILE_LAG_SECONDS, and a pulse only counts once the source committed it before that
instant (created_ts): data still in flight is never reported, not even a first-boot backfill,
whose old event times are committed at start-up.

ClickHouse resolves POSTGRES_HOST/POSTGRES_PORT and S3_ENDPOINT_URL itself, so they must be
addresses inside the compose network. The credentials travel as query parameters, never in
the SQL text: ClickHouse masks them in its query log and, as their names contain password,
secret or key, also in the request URIs its trace log records. The window reaches PostgreSQL
as UTC wall-clock literals, so the database must run with TimeZone UTC (the image default).

Exit codes: 0 every layer matches the source, 3 skipped (a table, the bucket or the silver
dataset does not exist yet, or no pulse of the window has been committed long enough ago),
65 mismatches found (sysexits EX_DATAERR), 1 failure, 143 terminated.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from clickhouse_connect.driver.exceptions import DatabaseError
from ds_runtime import EXIT_OK, PreconditionsNotMetError, StopToken, configure_logging, run_cli
from ds_settings import ConfigError, S3Settings, load_settings
from lakehouse_io import ClickHouseGateway

EXIT_DATA_MISMATCH = 65  # sysexits EX_DATAERR: a layer disagrees with the source
DEFAULT_WINDOW_HOURS = 6
MAX_WINDOW_HOURS = 168
DEFAULT_LAG_SECONDS = 600
WORST_CELLS_LOGGED = 3

RESULTS_TABLE = "ops_reconciliation"
RAW_TABLE = "raw_engine_telemetry"
STAGING_TABLE = "stg_engine_telemetry"
HF_TABLE = "raw_engine_hf_features"
SOURCE_TABLE = "engine_telemetry"
SILVER_DATASET = "silver/engine_telemetry"
# Why each ClickHouse table can be missing, named in the skip message.
REQUIRED_TABLES: dict[str, str] = {
    RAW_TABLE: "init-db.sql creates it: make ch-init",
    HF_TABLE: "init-db.sql creates it: make ch-init",
    RESULTS_TABLE: "init-db.sql creates it: make ch-init",
    STAGING_TABLE: "the first dbt run builds it",
}
RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_at",
    "window_start",
    "window_end",
    "layer",
    "engine_id",
    "event_hour",
    "source_rows",
    "layer_rows",
    "difference",
    "status",
    "informational",
)
# ClickHouse error code S3_ERROR; MinIO names a missing bucket NoSuchBucket.
_S3_ERROR_CODE = 499
_MISSING_BUCKET_MARKER = "NoSuchBucket"

# Whether the lake holds any silver telemetry file at all: format One lists files without
# reading them, and LIMIT 1 stops the listing at the first match.
SILVER_PROBE_SQL = """
SELECT count()
FROM
(
    SELECT 1
    FROM s3({silver_files:String}, {s3_access_key_id:String}, {s3_secret_access_key:String}, 'One')
    LIMIT 1
)
"""

# One row per engine and hour of event time with the pulses of every layer (HourlyCounts).
# Every layer applies the same predicate: event_ts inside the window and created_ts, the
# source's insert time, before the window end. The finite readings are the six columns of
# telemetry_readings_are_finite() in analytics/macros, which decides what staging retracts.
LAYER_COUNTS_SQL = """
SELECT
    engine_id,
    event_hour,
    sumIf(pulses, layer = 'source') AS source_rows,
    sumIf(finite_pulses, layer = 'source') AS source_finite_rows,
    sumIf(pulses, layer = 'raw') AS raw_rows,
    sumIf(pulses, layer = 'staging') AS staging_rows,
    sumIf(pulses, layer = 'silver') AS silver_rows,
    sumIf(pulses, layer = 'hf_frames') AS hf_frame_rows
FROM
(
    -- source: ClickHouse pushes the WHERE down, so only the window's rows leave PostgreSQL
    SELECT
        'source' AS layer,
        engine_id,
        toStartOfHour(event_ts, 'UTC') AS event_hour,
        count() AS pulses,
        countIf(
            isFinite(rpm) AND isFinite(core_temp_c) AND isFinite(egt_c)
            AND isFinite(vibration_mm_s) AND isFinite(fuel_flow_kg_h) AND isFinite(oil_pressure_psi)
        ) AS finite_pulses
    FROM postgresql(
        {pg_address:String}, {pg_database:String}, 'engine_telemetry',
        {pg_user:String}, {pg_password:String}
    )
    WHERE event_ts >= {window_start:DateTime64(6, 'UTC')}
      AND event_ts < {window_end:DateTime64(6, 'UTC')}
      AND created_ts < {window_end:DateTime64(6, 'UTC')}
    GROUP BY engine_id, event_hour

    UNION ALL

    -- raw: the current version of every pulse, unless that version is a delete
    SELECT 'raw', engine_id, toStartOfHour(event_ts, 'UTC') AS event_hour, count(), toUInt64(0)
    FROM raw_engine_telemetry FINAL
    WHERE event_ts >= {window_start:DateTime64(6, 'UTC')}
      AND event_ts < {window_end:DateTime64(6, 'UTC')}
      AND created_ts < {window_end:DateTime64(6, 'UTC')}
      AND is_deleted = 0
    GROUP BY engine_id, event_hour

    UNION ALL

    -- staging: FINAL hides what dbt retracted (deletes and pulses with a non-finite reading)
    SELECT 'staging', engine_id, toStartOfHour(event_ts, 'UTC') AS event_hour, count(), toUInt64(0)
    FROM stg_engine_telemetry FINAL
    WHERE event_ts >= {window_start:DateTime64(6, 'UTC')}
      AND event_ts < {window_end:DateTime64(6, 'UTC')}
      AND created_ts < {window_end:DateTime64(6, 'UTC')}
    GROUP BY engine_id, event_hour

    UNION ALL

    -- silver: the newest image of each pulse across every file of the window's partitions
    SELECT 'silver', engine, toStartOfHour(pulse_ts, 'UTC') AS event_hour, count(), toUInt64(0)
    FROM
    (
        SELECT
            telemetry_id,
            argMax(engine_id, cdc_lsn) AS engine,
            argMax(event_ts, cdc_lsn) AS pulse_ts,
            argMax(created_ts, cdc_lsn) AS committed_ts,
            argMax(is_deleted, cdc_lsn) AS deleted
        FROM s3(
            {silver_files:String}, {s3_access_key_id:String}, {s3_secret_access_key:String}, 'Parquet',
            'telemetry_id UInt64, engine_id String, event_ts DateTime64(6, ''UTC''),
             created_ts DateTime64(6, ''UTC''), cdc_lsn UInt64, is_deleted UInt8'
        )
        -- every version of a pulse carries its event_ts, so this only prunes row groups
        WHERE event_ts >= {window_start:DateTime64(6, 'UTC')}
          AND event_ts < {window_end:DateTime64(6, 'UTC')}
        GROUP BY telemetry_id
    )
    WHERE deleted = 0
      AND committed_ts < {window_end:DateTime64(6, 'UTC')}
    GROUP BY engine, event_hour

    UNION ALL

    -- hf_frames: the window's pulses (as raw holds them) whose HF frame landed
    SELECT
        'hf_frames', engine_id, toStartOfHour(frame_start_ts, 'UTC') AS event_hour,
        uniqExact(telemetry_id), toUInt64(0)
    FROM raw_engine_hf_features
    WHERE frame_start_ts >= {window_start:DateTime64(6, 'UTC')}
      AND frame_start_ts < {window_end:DateTime64(6, 'UTC')}
      AND telemetry_id IN
      (
          SELECT telemetry_id
          FROM raw_engine_telemetry FINAL
          WHERE event_ts >= {window_start:DateTime64(6, 'UTC')}
            AND event_ts < {window_end:DateTime64(6, 'UTC')}
            AND created_ts < {window_end:DateTime64(6, 'UTC')}
            AND is_deleted = 0
      )
    GROUP BY engine_id, event_hour
)
GROUP BY engine_id, event_hour
ORDER BY engine_id, event_hour
"""

# Named explicitly: run as a script, __name__ would be "__main__" in every log line.
logger = logging.getLogger("reconcile_lakehouse")


class Layer(StrEnum):
    """A copy of the source telemetry the audit holds against it."""

    RAW = "raw"
    STAGING = "staging"
    SILVER = "silver"
    HF_FRAMES = "hf_frames"


# Reported, but never a reason to fail the run: the HF path is best effort by design.
INFORMATIONAL_LAYERS = frozenset({Layer.HF_FRAMES})


class CellStatus(StrEnum):
    """How one layer of one engine and hour compares with the source."""

    OK = "OK"
    MISSING = "MISSING"
    EXTRA = "EXTRA"


def classify(source_rows: int, layer_rows: int) -> CellStatus:
    """OK when the layer holds exactly the source's pulses, MISSING below, EXTRA above."""
    if layer_rows == source_rows:
        return CellStatus.OK
    return CellStatus.MISSING if layer_rows < source_rows else CellStatus.EXTRA


@dataclass(frozen=True)
class PostgresSource:
    """Where ClickHouse reaches the source database (an address inside the compose network)."""

    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)

    @property
    def address(self) -> str:
        """``host:port``, as the postgresql() table function takes it."""
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ReconcileSettings:
    """The window of a run (RECONCILE_WINDOW_HOURS, RECONCILE_LAG_SECONDS) and the source (POSTGRES_*)."""

    window_hours: int
    lag_seconds: int
    postgres: PostgresSource


def load_reconcile_settings(env: Mapping[str, str] | None = None) -> ReconcileSettings:
    """Read the audit's settings from ``env`` (defaults to os.environ); ConfigError names a bad one."""
    source = os.environ if env is None else env
    return ReconcileSettings(
        window_hours=_integer(source, "RECONCILE_WINDOW_HOURS", DEFAULT_WINDOW_HOURS, 1, MAX_WINDOW_HOURS),
        lag_seconds=_integer(source, "RECONCILE_LAG_SECONDS", DEFAULT_LAG_SECONDS, 0),
        postgres=PostgresSource(
            host=_text(source, "POSTGRES_HOST", "postgres"),
            port=_integer(source, "POSTGRES_PORT", 5432, 1, 65535),
            database=_text(source, "POSTGRES_DB", "jet_engine_db"),
            user=_text(source, "POSTGRES_USER", "admin"),
            password=source.get("POSTGRES_PASSWORD", "adminpassword"),
        ),
    )


@dataclass(frozen=True)
class ReconcileWindow:
    """Event time [start, end) of one run; ``end`` also bounds when the source committed a pulse."""

    start: datetime
    end: datetime

    @classmethod
    def before(cls, now: datetime, *, hours: int, lag_seconds: int) -> ReconcileWindow:
        """The ``hours`` of event time that end ``lag_seconds`` before ``now`` (UTC, whole seconds)."""
        if now.tzinfo is None:
            raise ValueError("the reconciliation window needs a timezone-aware 'now'")
        end = (now.astimezone(UTC) - timedelta(seconds=lag_seconds)).replace(microsecond=0)
        return cls(end - timedelta(hours=hours), end)

    @property
    def event_dates(self) -> tuple[date, ...]:
        """UTC days the window touches: the silver partitions (event_date=YYYY-MM-DD) it reads."""
        first, last = self.start.date(), (self.end - timedelta(microseconds=1)).date()
        return tuple(first + timedelta(days=offset) for offset in range((last - first).days + 1))

    def describe(self) -> str:
        """``start..end`` in ISO 8601."""
        return f"{_iso(self.start)}..{_iso(self.end)}"


def silver_files_url(s3: S3Settings, event_dates: Sequence[date] | None = None) -> str:
    """s3() glob of the silver telemetry Parquet files: every partition, or those of ``event_dates``."""
    if event_dates is None:
        partitions = "*"
    elif event_dates:
        partitions = "{" + ",".join(day.isoformat() for day in event_dates) + "}"
    else:
        raise ValueError("at least one event date is needed")
    return f"{s3.endpoint_url.rstrip('/')}/{s3.bucket}/{SILVER_DATASET}/event_date={partitions}/*.parquet"


def silver_probe_query(s3: S3Settings) -> tuple[str, dict[str, Any]]:
    """SILVER_PROBE_SQL with its parameters: 1 when the lake has a silver telemetry file, else 0."""
    return SILVER_PROBE_SQL, {"silver_files": silver_files_url(s3), **_s3_parameters(s3)}


def layer_counts_query(
    window: ReconcileWindow, postgres: PostgresSource, s3: S3Settings
) -> tuple[str, dict[str, Any]]:
    """LAYER_COUNTS_SQL with its parameters; the credentials are parameters, never SQL text."""
    parameters = {
        "window_start": window.start,
        "window_end": window.end,
        "pg_address": postgres.address,
        "pg_database": postgres.database,
        "pg_user": postgres.user,
        "pg_password": postgres.password,
        "silver_files": silver_files_url(s3, window.event_dates),
        **_s3_parameters(s3),
    }
    return LAYER_COUNTS_SQL, parameters


@dataclass(frozen=True)
class ReconciliationCell:
    """One layer of one engine and hour, against the source pulses that layer must hold."""

    layer: Layer
    engine_id: str
    event_hour: datetime
    source_rows: int
    layer_rows: int

    @property
    def difference(self) -> int:
        """Pulses the layer holds beyond (positive) or short of (negative) the source."""
        return self.layer_rows - self.source_rows

    @property
    def status(self) -> CellStatus:
        """OK, MISSING or EXTRA."""
        return classify(self.source_rows, self.layer_rows)

    @property
    def informational(self) -> bool:
        """True for a best-effort layer, whose differences are reported but never fail a run."""
        return self.layer in INFORMATIONAL_LAYERS

    @property
    def is_mismatch(self) -> bool:
        """True when this cell fails the run."""
        return self.status is not CellStatus.OK and not self.informational

    def describe(self) -> str:
        """``layer engine hour STATUS difference (layer N, source M)`` for the summary line."""
        return (
            f"{self.layer} {self.engine_id} {_iso(self.event_hour)} {self.status} {abs(self.difference)} "
            f"(layer {self.layer_rows}, source {self.source_rows})"
        )


@dataclass(frozen=True)
class HourlyCounts:
    """Pulses of one engine and hour of event time in every layer: one row of LAYER_COUNTS_SQL."""

    engine_id: str
    event_hour: datetime
    source_rows: int
    source_finite_rows: int
    raw_rows: int
    staging_rows: int
    silver_rows: int
    hf_frame_rows: int

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> HourlyCounts:
        """Build from a result row in COUNT_COLUMNS order."""
        engine_id, event_hour, *counts = row
        return cls(str(engine_id), _utc(event_hour), *(int(count) for count in counts))

    def cells(self) -> tuple[ReconciliationCell, ...]:
        """One cell per layer: staging is held to the finite source pulses, the others to all."""
        held = {
            Layer.RAW: (self.source_rows, self.raw_rows),
            Layer.STAGING: (self.source_finite_rows, self.staging_rows),
            Layer.SILVER: (self.source_rows, self.silver_rows),
            Layer.HF_FRAMES: (self.source_rows, self.hf_frame_rows),
        }
        return tuple(
            ReconciliationCell(layer, self.engine_id, self.event_hour, expected, actual)
            for layer, (expected, actual) in held.items()
        )


COUNT_COLUMNS: tuple[str, ...] = tuple(column.name for column in fields(HourlyCounts))


@dataclass(frozen=True)
class ReconciliationRun:
    """Every cell of one run, as appended to ops_reconciliation."""

    run_id: uuid.UUID
    run_at: datetime
    window: ReconcileWindow
    cells: tuple[ReconciliationCell, ...]

    @property
    def mismatches(self) -> list[ReconciliationCell]:
        """The cells that fail the run, largest difference first."""
        return sorted(
            (cell for cell in self.cells if cell.is_mismatch),
            key=lambda cell: (-abs(cell.difference), cell.layer, cell.engine_id, cell.event_hour),
        )

    @property
    def reconciled(self) -> bool:
        """True when every checked layer matches the source in every engine and hour."""
        return not any(cell.is_mismatch for cell in self.cells)

    def result_rows(self) -> list[tuple[Any, ...]]:
        """The rows of ops_reconciliation, in RESULT_COLUMNS order."""
        return [
            (
                self.run_id,
                self.run_at,
                self.window.start,
                self.window.end,
                cell.layer.value,
                cell.engine_id,
                cell.event_hour,
                cell.source_rows,
                cell.layer_rows,
                cell.difference,
                cell.status.value,
                int(cell.informational),
            )
            for cell in self.cells
        ]

    def summary(self) -> str:
        """One log line: layers checked, hours, engines and source pulses, mismatches (worst first)."""
        checked = ", ".join(layer for layer in Layer if layer not in INFORMATIONAL_LAYERS)
        hours = len({cell.event_hour for cell in self.cells})
        engines = len({cell.engine_id for cell in self.cells})
        pulses = sum(cell.source_rows for cell in self.cells if cell.layer is Layer.RAW)
        compared = sum(1 for cell in self.cells if not cell.informational)
        mismatches = self.mismatches
        if mismatches:
            worst = "; ".join(cell.describe() for cell in mismatches[:WORST_CELLS_LOGGED])
            more = len(mismatches) - WORST_CELLS_LOGGED
            verdict = f"{len(mismatches)} of {compared} cells differ from the source, worst: {worst}"
            verdict += f" (and {more} more)" if more > 0 else ""
        else:
            verdict = f"all {compared} cells match the source"
        return (
            f"reconciliation {self.run_id} of {self.window.describe()}: {checked} against {pulses} source "
            f"pulses of {engines} engine(s) in {hours} hour(s): {verdict}; {self._hf_coverage()}"
        )

    def _hf_coverage(self) -> str:
        frames = [cell for cell in self.cells if cell.layer is Layer.HF_FRAMES]
        expected = sum(cell.source_rows for cell in frames)
        if not expected:
            return "HF frame coverage n/a (informational)"
        covered = sum(cell.layer_rows for cell in frames)
        return f"HF frames cover {100 * covered / expected:.1f}% of the pulses (informational)"


class ReconciliationGateway(Protocol):
    """The ClickHouse capabilities the audit needs (ClickHouseGateway satisfies it)."""

    def table_engines(self, tables: Iterable[str]) -> dict[str, str]:
        """Map existing tables to their engine names."""
        ...

    def query_rows(self, sql: str, parameters: Mapping[str, Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a SELECT and return its rows."""
        ...

    def insert_rows(self, table: str, rows: Sequence[Sequence[Any]], column_names: Sequence[str]) -> None:
        """Insert ``rows`` into ``table`` as one batch."""
        ...


def require_tables(gateway: ReconciliationGateway) -> None:
    """Raise PreconditionsNotMetError naming every table the audit reads or writes that is missing."""
    existing = gateway.table_engines(REQUIRED_TABLES)
    missing = [table for table in REQUIRED_TABLES if table not in existing]
    if missing:
        reasons = "; ".join(dict.fromkeys(REQUIRED_TABLES[table] for table in missing))
        raise PreconditionsNotMetError(f"tables do not exist yet: {', '.join(missing)} ({reasons})")


def require_silver_dataset(gateway: ReconciliationGateway, s3: S3Settings) -> None:
    """Raise PreconditionsNotMetError while the bucket or the silver telemetry dataset does not exist."""
    try:
        rows = gateway.query_rows(*silver_probe_query(s3))
    except DatabaseError as exc:
        if exc.code == _S3_ERROR_CODE and _MISSING_BUCKET_MARKER in str(exc):
            raise PreconditionsNotMetError(f"the lakehouse bucket {s3.bucket!r} does not exist") from exc
        raise
    if not rows or not rows[0][0]:
        raise PreconditionsNotMetError(
            f"no Parquet file under {s3.bucket}/{SILVER_DATASET} yet (the silver lake query writes "
            "the first one after its first micro-batch)"
        )


def run_reconciliation(
    gateway: ReconciliationGateway,
    settings: ReconcileSettings,
    s3: S3Settings,
    *,
    now: datetime,
    stop: StopToken,
) -> ReconciliationRun:
    """Count every layer, classify each cell, append the run to ops_reconciliation and log it.

    Raises PreconditionsNotMetError (exit 3) while a table or the silver dataset is missing,
    or when no layer holds a pulse of the window: there is nothing to reconcile yet.
    """
    window = ReconcileWindow.before(now, hours=settings.window_hours, lag_seconds=settings.lag_seconds)
    require_tables(gateway)
    require_silver_dataset(gateway, s3)
    stop.raise_if_stopped()
    rows = gateway.query_rows(*layer_counts_query(window, settings.postgres, s3))
    if not rows:
        raise PreconditionsNotMetError(
            f"no layer holds a pulse with an event time in {window.describe()} that the source "
            f"committed before {_iso(window.end)}: nothing to reconcile yet"
        )
    cells = tuple(cell for row in rows for cell in HourlyCounts.from_row(row).cells())
    run = ReconciliationRun(uuid.uuid4(), now.astimezone(UTC), window, cells)
    stop.raise_if_stopped()
    gateway.insert_rows(RESULTS_TABLE, run.result_rows(), RESULT_COLUMNS)
    if run.reconciled:
        logger.info("%s", run.summary())
    else:
        logger.warning("%s", run.summary())
    return run


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    # Compose passes unset `${VAR:-}` interpolations as empty strings: treat them as unset.
    return env.get(name, "").strip() or default


def _integer(
    env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int | None = None
) -> int:
    raw = _text(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not _in_range(value, minimum, maximum):
        raise ConfigError(f"{name} must be {_range_text(minimum, maximum)}, got {value}")
    return value


def _in_range(value: int, minimum: int, maximum: int | None) -> bool:
    return value >= minimum and (maximum is None or value <= maximum)


def _range_text(minimum: int, maximum: int | None) -> str:
    return f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"


def _s3_parameters(s3: S3Settings) -> dict[str, str]:
    return {"s3_access_key_id": s3.access_key_id, "s3_secret_access_key": s3.secret_access_key}


def _utc(moment: datetime) -> datetime:
    """``moment`` as an aware UTC datetime (ClickHouse may return naive UTC values)."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bounded_int(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    """argparse type: an integer in [minimum, maximum] (no upper bound when ``maximum`` is None)."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
        if not _in_range(value, minimum, maximum):
            raise argparse.ArgumentTypeError(f"expected a value {_range_text(minimum, maximum)}, got {value}")
        return value

    return parse


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile the source telemetry with the raw, staging and silver layers.",
        epilog="Exit codes: 0 every layer matches the source, 3 skipped (nothing to compare yet), "
        "65 mismatches found, 1 failure.",
    )
    parser.add_argument(
        "--window-hours",
        type=_bounded_int(1, MAX_WINDOW_HOURS),
        default=None,
        help=f"hours of event time compared (default: RECONCILE_WINDOW_HOURS, {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--lag-seconds",
        type=_bounded_int(0),
        default=None,
        help="pulses committed more recently are still in flight and not compared "
        f"(default: RECONCILE_LAG_SECONDS, {DEFAULT_LAG_SECONDS})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``reconcile_lakehouse.py``."""
    args = _parse_args(argv)
    platform = load_settings()
    configure_logging(platform.log_level)
    settings = load_reconcile_settings()
    if args.window_hours is not None:
        settings = replace(settings, window_hours=args.window_hours)
    if args.lag_seconds is not None:
        settings = replace(settings, lag_seconds=args.lag_seconds)
    stop = StopToken().install_signal_handlers()
    with ClickHouseGateway(platform.clickhouse) as gateway:
        run = run_reconciliation(gateway, settings, platform.s3, now=datetime.now(UTC), stop=stop)
    return EXIT_OK if run.reconciled else EXIT_DATA_MISMATCH


if __name__ == "__main__":
    run_cli(main)
