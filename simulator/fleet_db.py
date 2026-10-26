"""PostgreSQL persistence of the fleet simulator: connection, single-writer lease, transactions, resume reads.

Every connection first takes the single-writer lease (fleet_lease), so a second simulator never writes
next to this one. Writes arrive as ordered (operation, payload) lists that fleet_writer executes in
exactly ONE transaction. The simulator also owns ``simulator_lifecycle_params`` (created here at start-up,
so existing volumes migrate on their own): the SIM_* values each lifecycle was generated with, which
let a restart re-derive an in-progress lifecycle exactly. Debezium does not capture it: the publication
and the connector's table.include.list name the three CDC tables explicitly.
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
from fleet_config import GenerationSettings
from fleet_lease import WriterLease
from fleet_writer import FleetStateError, FleetWriter
from psycopg2.extras import execute_values

if TYPE_CHECKING:
    from psycopg2.extensions import connection, cursor

logger = logging.getLogger("fleet_db")

T = TypeVar("T")
TRANSIENT_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)
CONNECT_ATTEMPTS = 10
RETRY_ATTEMPTS = 5
BACKOFF_BASE_SECONDS, BACKOFF_CAP_SECONDS = 1.0, 30.0
CONNECT_TIMEOUT_SECONDS = 10
# Server-side TCP keepalive: PostgreSQL ends the session of a simulator that vanished without closing
# its socket (and so frees the writer lock) within ~50 s, inside the lease's wait, not after 2 hours.
SERVER_KEEPALIVE_OPTIONS = "-c tcp_keepalives_idle=20 -c tcp_keepalives_interval=10 -c tcp_keepalives_count=3"

LIFECYCLE_PARAMS_DDL = """
CREATE TABLE IF NOT EXISTS simulator_lifecycle_params (
    engine_id         VARCHAR(32)      NOT NULL REFERENCES engine_status_registry (engine_id),
    lifecycle_id      INTEGER          NOT NULL CHECK (lifecycle_id >= 1),
    seed              BIGINT           NOT NULL,
    mean_life_cycles  DOUBLE PRECISION NOT NULL CHECK (mean_life_cycles >= 2),
    pulses_per_cycle  INTEGER          NOT NULL CHECK (pulses_per_cycle >= 1),
    created_ts        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (engine_id, lifecycle_id)
);
COMMENT ON TABLE simulator_lifecycle_params IS
    'Simulator-owned (not captured by CDC): SIM_SEED, SIM_MEAN_LIFE_CYCLES and SIM_PULSES_PER_CYCLE each '
    'engine lifecycle was generated with, so a restart re-derives it exactly after those settings change.'"""

REGISTER_ENGINES_SQL = """
INSERT INTO engine_status_registry
    (engine_id, engine_model, operational_status, current_lifecycle_id, current_cycle, wear_factor)
VALUES %s
ON CONFLICT (engine_id) DO NOTHING"""

SNAPSHOT_SQL = """
SELECT r.operational_status, r.current_lifecycle_id, r.current_cycle, r.cycles_since_new,
       p.pulses, p.last_cycle, p.last_wear,
       (SELECT max(t.event_ts) FROM engine_telemetry t WHERE t.engine_id = r.engine_id),
       s.seed, s.mean_life_cycles, s.pulses_per_cycle,
       m.maintenance_status, m.failure_ts, m.failure_mode
  FROM engine_status_registry r
 CROSS JOIN LATERAL (
       SELECT count(*) AS pulses, max(t.cycle_number) AS last_cycle, max(t.wear_factor) AS last_wear
         FROM engine_telemetry t
        WHERE t.engine_id = r.engine_id AND t.lifecycle_id = r.current_lifecycle_id) p
  LEFT JOIN simulator_lifecycle_params s
         ON s.engine_id = r.engine_id AND s.lifecycle_id = r.current_lifecycle_id
  LEFT JOIN maintenance_log m ON m.engine_id = r.engine_id AND m.lifecycle_id = r.current_lifecycle_id
 WHERE r.engine_id = %s"""


class DatabaseConnectionError(RuntimeError):
    """PostgreSQL stayed unreachable through all bounded retries, or a shutdown interrupted the wait."""


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

    @property
    def target(self) -> str:
        """host:port/dbname for log and error messages."""
        return f"{self.host}:{self.port}/{self.dbname}"


@dataclass(frozen=True)
class WorkOrder:
    """maintenance_log row of an engine's current lifecycle."""

    status: str
    failure_ts: datetime
    failure_mode: str


@dataclass(frozen=True)
class EngineSnapshot:
    """Persisted state needed to resume one engine after a restart.

    ``settings`` are the stored generation settings of the current lifecycle: None for a lifecycle
    without telemetry, and for one written before the simulator recorded them.
    """

    engine_id: str
    operational_status: str
    lifecycle_id: int
    current_cycle: int
    cycles_since_new: int
    pulses_in_lifecycle: int
    last_cycle_number: int | None
    last_wear_factor: float | None
    last_event_ts: datetime | None
    settings: GenerationSettings | None
    work_order: WorkOrder | None


def backoff_delay(attempt: int) -> float:
    """Exponential backoff before retry ``attempt`` (1-based): 1, 2, 4, ... seconds, capped at 30 s."""
    return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))


def _first_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


class FleetDatabase:
    """Owns the connection: bounded reconnects with exponential backoff, one transaction per write batch.

    Every new session takes the single-writer lease before its first statement; WriterLockError
    (from fleet_lease) means another simulator writes to this database and this one must stop.
    """

    def __init__(self, settings: DatabaseSettings, stop_event: threading.Event) -> None:
        self._settings = settings
        self._stop = stop_event
        self._conn: connection | None = None
        self._lease = WriterLease(stop_event)

    def connect(self) -> connection:
        """Open the connection and take the writer lease on it (both with bounded retries).

        Raises DatabaseConnectionError once the connection attempts run out or a shutdown request
        interrupts the wait, and WriterLockError when another simulator holds (or held) the lease.
        """
        conn = self._open()
        try:
            acquired = self._lease.acquire(conn, self._settings.target)
        except BaseException:
            conn.close()
            raise
        if not acquired:
            conn.close()
            raise DatabaseConnectionError("shutdown requested while waiting for the writer lock")
        self._conn = conn
        return conn

    def _open(self) -> connection:
        target = self._settings.target
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            if self._stop.is_set():
                raise DatabaseConnectionError("shutdown requested while connecting to PostgreSQL")
            try:
                conn = psycopg2.connect(
                    host=self._settings.host,
                    port=self._settings.port,
                    dbname=self._settings.dbname,
                    user=self._settings.user,
                    password=self._settings.password,
                    connect_timeout=CONNECT_TIMEOUT_SECONDS,
                    application_name="engine-simulator",
                    options=SERVER_KEEPALIVE_OPTIONS,
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
                return conn
        raise DatabaseConnectionError(f"PostgreSQL {target} unreachable after {CONNECT_ATTEMPTS} attempts")

    def close(self) -> None:
        """Close the connection if open, which also releases the writer lock (safe to call repeatedly)."""
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
            return self._transaction(lambda cur: FleetWriter(cur).apply(ops))
        except TRANSIENT_ERRORS as exc:
            self._discard_connection(exc)
            return None

    def execute_with_retry(self, ops: Sequence[tuple[str, Any]]) -> list[Any]:
        """Run idempotent ``ops`` in ONE transaction, reconnecting and retrying on transient errors."""
        return self._with_retry(lambda cur: FleetWriter(cur).apply(ops), "write batch")

    def ensure_schema(self) -> None:
        """Create the simulator-owned tables that a database initialised by an older release lacks."""

        def work(cur: cursor) -> None:
            cur.execute(LIFECYCLE_PARAMS_DDL)

        self._with_retry(work, "schema check")

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

    def load_snapshot(self, engine_id: str) -> EngineSnapshot:
        """Registry row, current-lifecycle progress, stored generation settings and work order."""

        def work(cur: cursor) -> tuple[Any, ...] | None:
            cur.execute(SNAPSHOT_SQL, (engine_id,))
            return cur.fetchone()

        row = self._with_retry(work, f"snapshot of {engine_id}")
        if row is None:
            raise FleetStateError(f"engine {engine_id} is not registered")
        status, lifecycle_id, cycle, cycles_since_new, pulses, last_cycle, last_wear, last_ts, *rest = row
        seed, mean_life, pulses_per_cycle, *order = rest
        settings = None if seed is None else GenerationSettings(int(seed), float(mean_life), pulses_per_cycle)
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
            settings=settings,
            work_order=WorkOrder(*order) if order[0] is not None else None,
        )
