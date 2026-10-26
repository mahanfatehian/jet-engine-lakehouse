"""PostgreSQL persistence of the fleet simulator; every statement here becomes a Debezium CDC event.

    table                    INSERT (op "c")                   UPDATE (op "u")
    engine_status_registry   engine registration               cycle progress, status change, failure, repair
    engine_telemetry         every pulse                       maintenance_id stamped on WARNING..FAILURE rows
    maintenance_log          work order opened at failure      flagged-row count, repair completion

Writes arrive as ordered (operation, payload) lists and each list runs in exactly ONE transaction, so a
failure (telemetry row, work order, flags, registry) or a repair is never half visible downstream.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

import psycopg2
from psycopg2.extras import execute_values

if TYPE_CHECKING:
    from engine_generator import Pulse
    from fleet_runtime import FailureEvent, MaintenanceCompletion, RegistryUpdate
    from psycopg2.extensions import connection, cursor

logger = logging.getLogger("fleet_db")

T = TypeVar("T")
TRANSIENT_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)
CONNECT_ATTEMPTS = 10
RETRY_ATTEMPTS = 5
BACKOFF_BASE_SECONDS, BACKOFF_CAP_SECONDS = 1.0, 30.0
CONNECT_TIMEOUT_SECONDS = 10
INSERT_PAGE_SIZE = 1000
TELEMETRY_COLUMNS = (
    "engine_id",
    "lifecycle_id",
    "cycle_number",
    "event_ts",
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "wear_factor",
    "status",
)

REGISTER_ENGINES_SQL = """
INSERT INTO engine_status_registry
    (engine_id, engine_model, operational_status, current_lifecycle_id, current_cycle, wear_factor)
VALUES %s
ON CONFLICT (engine_id) DO NOTHING"""

INSERT_TELEMETRY_SQL = f"INSERT INTO engine_telemetry ({', '.join(TELEMETRY_COLUMNS)}) VALUES %s"

OPEN_WORK_ORDER_SQL = """
INSERT INTO maintenance_log
    (engine_id, lifecycle_id, failure_cycle, failure_ts, failure_mode, maintenance_status, opened_ts, notes)
VALUES (%(engine_id)s, %(lifecycle_id)s, %(cycle)s, %(ts)s, %(mode)s, 'OPEN', %(ts)s, %(notes)s)
RETURNING maintenance_id"""

FLAG_WARNING_ROWS_SQL = """
UPDATE engine_telemetry SET maintenance_id = %(maintenance_id)s
 WHERE engine_id = %(engine_id)s AND lifecycle_id = %(lifecycle_id)s
   AND status IN ('WARNING', 'CRITICAL', 'FAILURE') AND maintenance_id IS NULL"""

COUNT_FLAGGED_ROWS_SQL = "UPDATE maintenance_log SET warning_rows_flagged = %s WHERE maintenance_id = %s"

REGISTRY_FAILURE_SQL = """
UPDATE engine_status_registry
   SET operational_status = 'FAILURE', status_changed_ts = %(ts)s, current_lifecycle_id = %(lifecycle_id)s,
       current_cycle = %(cycle)s, wear_factor = %(wear)s, cycles_since_new = %(cycles_since_new)s,
       last_telemetry_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

# status NULL = progress only: keep operational_status and status_changed_ts as they are.
REGISTRY_PROGRESS_SQL = """
UPDATE engine_status_registry
   SET operational_status = COALESCE(%(status)s, operational_status),
       status_changed_ts = CASE WHEN %(status)s IS NULL THEN status_changed_ts ELSE %(ts)s END,
       current_lifecycle_id = %(lifecycle_id)s, current_cycle = %(cycle)s, wear_factor = %(wear)s,
       cycles_since_new = %(cycles_since_new)s, last_telemetry_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

CLOSE_WORK_ORDER_SQL = """
UPDATE maintenance_log
   SET maintenance_status = 'COMPLETED', completed_ts = %(ts)s, downtime_seconds = %(downtime)s,
       action_taken = %(action)s
 WHERE engine_id = %(engine_id)s AND lifecycle_id = %(lifecycle_id)s AND maintenance_status = 'OPEN'"""

REGISTRY_RESET_SQL = """
UPDATE engine_status_registry
   SET operational_status = 'MAINTENANCE_RESET', current_lifecycle_id = %(next_lifecycle_id)s,
       current_cycle = 0, wear_factor = 0, lifecycles_completed = lifecycles_completed + 1,
       last_maintenance_ts = %(ts)s, status_changed_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

SNAPSHOT_SQL = """
SELECT r.operational_status, r.current_lifecycle_id, r.current_cycle, r.cycles_since_new,
       p.pulses, p.last_cycle, p.last_wear,
       (SELECT max(t.event_ts) FROM engine_telemetry t WHERE t.engine_id = r.engine_id),
       m.maintenance_status, m.failure_ts, m.failure_mode
  FROM engine_status_registry r
 CROSS JOIN LATERAL (
       SELECT count(*) AS pulses, max(t.cycle_number) AS last_cycle, max(t.wear_factor) AS last_wear
         FROM engine_telemetry t
        WHERE t.engine_id = r.engine_id AND t.lifecycle_id = r.current_lifecycle_id) p
  LEFT JOIN maintenance_log m ON m.engine_id = r.engine_id AND m.lifecycle_id = r.current_lifecycle_id
 WHERE r.engine_id = %s"""


class DatabaseConnectionError(RuntimeError):
    """PostgreSQL stayed unreachable through all bounded retries, or a shutdown interrupted the wait."""


class FleetStateError(RuntimeError):
    """A write did not touch exactly the row it targets: database and simulator state disagree."""


@dataclass(frozen=True)
class DatabaseSettings:
    """POSTGRES_* connection settings; the host defaults to localhost so the simulator also runs on a host."""

    host: str = "localhost"
    port: int = 5432
    dbname: str = "jet_engine_db"
    user: str = "admin"
    password: str = field(default="adminpassword", repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DatabaseSettings:
        """Read POSTGRES_HOST/PORT/DB/USER/PASSWORD; unset or blank variables keep the defaults."""
        env = os.environ if env is None else env
        port = env.get("POSTGRES_PORT", "").strip() or str(cls.port)
        if not port.isdigit():
            raise ValueError(f"POSTGRES_PORT={port!r} is not a valid port")
        return cls(
            host=env.get("POSTGRES_HOST", "").strip() or cls.host,
            port=int(port),
            dbname=env.get("POSTGRES_DB", "").strip() or cls.dbname,
            user=env.get("POSTGRES_USER", "").strip() or cls.user,
            password=env.get("POSTGRES_PASSWORD") or cls.password,
        )


@dataclass(frozen=True)
class WorkOrder:
    """maintenance_log row of an engine's current lifecycle."""

    status: str
    failure_ts: datetime
    failure_mode: str


@dataclass(frozen=True)
class EngineSnapshot:
    """Persisted state needed to resume one engine after a restart."""

    engine_id: str
    operational_status: str
    lifecycle_id: int
    current_cycle: int
    cycles_since_new: int
    pulses_in_lifecycle: int
    last_cycle_number: int | None
    last_wear_factor: float | None
    last_event_ts: datetime | None
    work_order: WorkOrder | None


def backoff_delay(attempt: int) -> float:
    """Exponential backoff before retry ``attempt`` (1-based): 1, 2, 4, ... seconds, capped at 30 s."""
    return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))


def _first_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


def _require_one_row(cur: cursor, what: str) -> None:
    if cur.rowcount != 1:
        raise FleetStateError(f"{what} touched {cur.rowcount} rows instead of 1")


class _Writer:
    """Executes the simulator's write operations on the cursor of one transaction."""

    def __init__(self, cur: cursor) -> None:
        self._cur = cur

    def apply(self, ops: Sequence[tuple[str, Any]]) -> list[Any]:
        """Run ``ops`` in order and return each result (the work-order id for a failure)."""
        handlers: dict[str, Callable[[Any], Any]] = {
            "insert_pulses": self.insert_pulses,
            "record_failure": self.record_failure,
            "update_registry": self.update_registry,
            "complete_maintenance": self.complete_maintenance,
        }
        return [handlers[name](payload) for name, payload in ops]

    def insert_pulses(self, pulses: Sequence[Pulse]) -> int:
        """Batch INSERT of telemetry rows."""
        rows = [tuple(getattr(pulse, column) for column in TELEMETRY_COLUMNS) for pulse in pulses]
        execute_values(self._cur, INSERT_TELEMETRY_SQL, rows, page_size=INSERT_PAGE_SIZE)
        return len(rows)

    def record_failure(self, event: FailureEvent) -> int:
        """FAILURE row, work order, maintenance_id on the lifecycle's warning rows, registry FAILURE."""
        pulse = event.pulse
        params = {
            "engine_id": pulse.engine_id,
            "lifecycle_id": pulse.lifecycle_id,
            "cycle": pulse.cycle_number,
            "ts": pulse.event_ts,
            "mode": event.failure_mode,
            "notes": event.notes,
            "wear": pulse.wear_factor,
            "cycles_since_new": event.cycles_since_new,
        }
        self.insert_pulses([pulse])
        self._cur.execute(OPEN_WORK_ORDER_SQL, params)
        maintenance_id = self._cur.fetchone()[0]
        self._cur.execute(FLAG_WARNING_ROWS_SQL, {**params, "maintenance_id": maintenance_id})
        flagged = self._cur.rowcount
        self._cur.execute(COUNT_FLAGGED_ROWS_SQL, (flagged, maintenance_id))
        self._cur.execute(REGISTRY_FAILURE_SQL, params)
        _require_one_row(self._cur, f"registry FAILURE update of {pulse.engine_id}")
        return int(maintenance_id)

    def update_registry(self, update: RegistryUpdate) -> None:
        """Registry progress (cycle, wear, counters) and, if ``update.status`` is set, the new status."""
        self._cur.execute(
            REGISTRY_PROGRESS_SQL,
            {
                "status": update.status,
                "ts": update.event_ts,
                "lifecycle_id": update.lifecycle_id,
                "cycle": update.current_cycle,
                "wear": update.wear_factor,
                "cycles_since_new": update.cycles_since_new,
                "engine_id": update.engine_id,
            },
        )
        _require_one_row(self._cur, f"registry update of {update.engine_id}")

    def complete_maintenance(self, completion: MaintenanceCompletion) -> None:
        """Close the OPEN work order and reset the registry into the next lifecycle."""
        params = {
            "engine_id": completion.engine_id,
            "lifecycle_id": completion.lifecycle_id,
            "next_lifecycle_id": completion.next_lifecycle_id,
            "ts": completion.completed_ts,
            "downtime": completion.downtime_seconds,
            "action": completion.action_taken,
        }
        self._cur.execute(CLOSE_WORK_ORDER_SQL, params)
        _require_one_row(
            self._cur, f"work order of {completion.engine_id} lifecycle {completion.lifecycle_id}"
        )
        self._cur.execute(REGISTRY_RESET_SQL, params)
        _require_one_row(self._cur, f"registry reset of {completion.engine_id}")


class FleetDatabase:
    """Owns the connection: bounded reconnects with exponential backoff, one transaction per write batch."""

    def __init__(self, settings: DatabaseSettings, stop_event: threading.Event) -> None:
        self._settings = settings
        self._stop = stop_event
        self._conn: connection | None = None

    def connect(self) -> connection:
        """Open the connection, retrying with backoff; DatabaseConnectionError once attempts run out."""
        target = f"{self._settings.host}:{self._settings.port}/{self._settings.dbname}"
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            if self._stop.is_set():
                raise DatabaseConnectionError("shutdown requested while connecting to PostgreSQL")
            try:
                self._conn = psycopg2.connect(
                    host=self._settings.host,
                    port=self._settings.port,
                    dbname=self._settings.dbname,
                    user=self._settings.user,
                    password=self._settings.password,
                    connect_timeout=CONNECT_TIMEOUT_SECONDS,
                    application_name="engine-simulator",
                )
            except psycopg2.OperationalError as exc:
                logger.warning(
                    "PostgreSQL %s unavailable (attempt %d/%d): %s",
                    target,
                    attempt,
                    CONNECT_ATTEMPTS,
                    _first_line(exc),
                )
                if attempt < CONNECT_ATTEMPTS:
                    self._stop.wait(backoff_delay(attempt))
            else:
                logger.info("Connected to PostgreSQL %s", target)
                return self._conn
        raise DatabaseConnectionError(f"PostgreSQL {target} unreachable after {CONNECT_ATTEMPTS} attempts")

    def close(self) -> None:
        """Close the connection if open (safe to call repeatedly)."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def _transaction(self, work: Callable[[cursor], T]) -> T:
        conn = self._conn if self._conn is not None and not self._conn.closed else self.connect()
        # psycopg2's connection context commits on success and rolls back on any exception.
        with conn, conn.cursor() as cur:
            return work(cur)

    def _discard_connection(self, exc: BaseException) -> None:
        logger.warning("Database error, dropping the connection: %s", _first_line(exc))
        self.close()

    def _with_retry(self, work: Callable[[cursor], T], what: str) -> T:
        attempt = 1
        while True:
            try:
                return self._transaction(work)
            except TRANSIENT_ERRORS as exc:
                self._discard_connection(exc)
                if attempt >= RETRY_ATTEMPTS or self._stop.is_set():
                    raise DatabaseConnectionError(f"{what} failed after {attempt} attempts") from exc
                self._stop.wait(backoff_delay(attempt))
                attempt += 1

    def execute(self, ops: Sequence[tuple[str, Any]]) -> list[Any] | None:
        """Run ``ops`` in ONE transaction; None when a transient error rolled it back (not retried)."""
        try:
            return self._transaction(lambda cur: _Writer(cur).apply(ops))
        except TRANSIENT_ERRORS as exc:
            self._discard_connection(exc)
            return None

    def execute_with_retry(self, ops: Sequence[tuple[str, Any]]) -> list[Any]:
        """Run idempotent ``ops`` in ONE transaction, reconnecting and retrying on transient errors."""
        return self._with_retry(lambda cur: _Writer(cur).apply(ops), "write batch")

    def register_engines(self, engine_models: Mapping[str, str]) -> int:
        """INSERT a HEALTHY lifecycle-1 registry row per engine unless present; returns the rows created."""

        def work(cur: cursor) -> int:
            execute_values(
                cur,
                REGISTER_ENGINES_SQL,
                list(engine_models.items()),
                template="(%s, %s, 'HEALTHY', 1, 0, 0)",
                page_size=max(1, len(engine_models)),
            )
            return cur.rowcount

        return self._with_retry(work, "engine registration")

    def telemetry_is_empty(self) -> bool:
        """True when engine_telemetry holds no row at all (first start of the platform)."""

        def work(cur: cursor) -> bool:
            cur.execute("SELECT NOT EXISTS (SELECT 1 FROM engine_telemetry)")
            return bool(cur.fetchone()[0])

        return self._with_retry(work, "telemetry check")

    def load_snapshot(self, engine_id: str) -> EngineSnapshot:
        """Registry row, current-lifecycle progress and work order of ``engine_id``."""

        def work(cur: cursor) -> tuple[Any, ...] | None:
            cur.execute(SNAPSHOT_SQL, (engine_id,))
            return cur.fetchone()

        row = self._with_retry(work, f"snapshot of {engine_id}")
        if row is None:
            raise FleetStateError(f"engine {engine_id} is not registered")
        status, lifecycle_id, cycle, cycles_since_new, pulses, last_cycle, last_wear, last_ts, *order = row
        return EngineSnapshot(
            engine_id=engine_id,
            operational_status=status,
            lifecycle_id=lifecycle_id,
            current_cycle=cycle,
            cycles_since_new=cycles_since_new,
            pulses_in_lifecycle=pulses,
            last_cycle_number=last_cycle,
            last_wear_factor=last_wear,
            last_event_ts=last_ts,
            work_order=WorkOrder(*order) if order[0] is not None else None,
        )
