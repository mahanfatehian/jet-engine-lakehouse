"""Single-writer lease: at most one simulator process writes to the fleet database at any time.

Two writers would emit every pulse index twice, collide on the one-work-order-per-lifecycle key and
inflate the per-lifecycle pulse counts that a restart resumes from. So every session of the simulator
first takes a PostgreSQL session-level advisory lock on a fixed key derived from the name
``jet-engine-simulator`` (advisory locks are per database: simulators of other databases on the same
server do not interfere). The lock lives exactly as long as the session; commits and rollbacks keep
it, and a dead process or a dropped connection releases it.

A process that finds the lock taken retries for LOCK_WAIT_SECONDS, because the session of a container
that has just restarted may linger briefly, and then gives up with WriterLockError. After a reconnect
the new session takes the lock again, and a fencing token in ``simulator_writer_lease`` (a
simulator-owned table that Debezium does not capture) reveals whether another writer held the lock in
between: the in-memory fleet would then be stale, so the process must stop and resume from the
database instead of writing on.

The module never imports psycopg2: connections are duck-typed, so the logic is unit-testable.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg2.extensions import connection

logger = logging.getLogger("fleet_lease")

WRITER_LOCK_NAME = "jet-engine-simulator"
LOCK_WAIT_SECONDS = 60.0
LOCK_RETRY_SECONDS = 2.0


def advisory_lock_key(name: str) -> int:
    """Signed 64-bit advisory-lock key from a SHA-256 of ``name``: the same in every process and host."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)


WRITER_LOCK_KEY = advisory_lock_key(WRITER_LOCK_NAME)

TRY_LOCK_SQL = "SELECT pg_try_advisory_lock(%s)"

# pg_locks shows a bigint advisory key as classid (high 32 bits) and objid (low 32 bits), objsubid 1.
LOCK_HOLDER_SQL = """
SELECT a.pid, a.application_name, host(a.client_addr), a.backend_start
  FROM pg_locks l
  JOIN pg_stat_activity a ON a.pid = l.pid
 WHERE l.locktype = 'advisory' AND l.granted AND l.objsubid = 1
   AND l.classid = %s::oid AND l.objid = %s::oid
   AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())"""

LEASE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS simulator_writer_lease (
    lock_key      BIGINT      PRIMARY KEY,
    writer_token  VARCHAR(32) NOT NULL,
    acquired_ts   TIMESTAMPTZ NOT NULL DEFAULT now()
)"""

CLAIM_LEASE_SQL = """
INSERT INTO simulator_writer_lease (lock_key, writer_token) VALUES (%s, %s)
ON CONFLICT (lock_key) DO UPDATE SET writer_token = EXCLUDED.writer_token, acquired_ts = now()"""

LEASE_TOKEN_SQL = "SELECT writer_token, acquired_ts FROM simulator_writer_lease WHERE lock_key = %s"


class WriterLockError(RuntimeError):
    """Another simulator holds the writer lock, or held it while this process was disconnected."""


def _lock_ids(key: int) -> tuple[int, int]:
    """(classid, objid) under which pg_locks lists the advisory lock on ``key``."""
    return (key >> 32) & 0xFFFFFFFF, key & 0xFFFFFFFF


def describe_holder(row: tuple[Any, ...] | None) -> str:
    """Human-readable session from a LOCK_HOLDER_SQL row (it may have ended in the meantime)."""
    if row is None:
        return "a session that has just ended"
    pid, application, client, started = row
    since = started.isoformat(timespec="seconds") if started is not None else "?"
    return f"pid {pid}, application '{application or '?'}', client {client or 'local socket'}, since {since}"


class WriterLease:
    """Takes the writer lock on every new session of one process and fences off a stale writer."""

    def __init__(
        self,
        stop_event: threading.Event,
        *,
        wait_seconds: float = LOCK_WAIT_SECONDS,
        retry_seconds: float = LOCK_RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stop = stop_event
        self._wait_seconds = wait_seconds
        self._retry_seconds = retry_seconds
        self._clock = clock
        self.token = uuid.uuid4().hex
        self._claimed = False

    def acquire(self, conn: connection, target: str) -> bool:
        """Hold the writer lock on the session of ``conn``; False when a shutdown request ended the wait.

        The first session of the process records its token; a later one (after a reconnect) checks
        that the recorded token is still its own. Raises WriterLockError when the lock stays taken for
        the whole wait, or when another writer held it while this process was disconnected.
        """
        if not self._take_lock(conn, target):
            return False
        with conn, conn.cursor() as cur:
            if self._claimed:
                cur.execute(LEASE_TOKEN_SQL, (WRITER_LOCK_KEY,))
                self._check_token(cur.fetchone(), target)
            else:
                cur.execute(LEASE_TABLE_DDL)
                cur.execute(CLAIM_LEASE_SQL, (WRITER_LOCK_KEY, self.token))
        if not self._claimed:
            logger.info("Single-writer lock '%s' acquired on %s", WRITER_LOCK_NAME, target)
        self._claimed = True
        return True

    def _check_token(self, row: tuple[Any, ...] | None, target: str) -> None:
        if row is not None and row[0] == self.token:
            logger.info("Single-writer lock re-acquired on %s after a reconnect", target)
            return
        since = row[1].isoformat(timespec="seconds") if row is not None else "an unknown time"
        raise WriterLockError(
            f"another simulator took over the writer lock on {target} at {since} while this one was "
            "disconnected, so this process's fleet state is stale: stopping instead of writing on; a "
            "restart resumes every engine from the database"
        )

    def _take_lock(self, conn: connection, target: str) -> bool:
        deadline = self._clock() + self._wait_seconds
        waited = False
        while True:
            with conn, conn.cursor() as cur:
                cur.execute(TRY_LOCK_SQL, (WRITER_LOCK_KEY,))
                if cur.fetchone()[0]:
                    if waited:
                        logger.info("Writer lock on %s released by its holder: acquired", target)
                    return True
                cur.execute(LOCK_HOLDER_SQL, _lock_ids(WRITER_LOCK_KEY))
                holder = describe_holder(cur.fetchone())
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise WriterLockError(
                    f"another simulator is writing to {target} ({holder}): it kept the single-writer "
                    f"lock '{WRITER_LOCK_NAME}' for {self._wait_seconds:g} s. Stop it before starting "
                    "this one; two writers duplicate pulses and corrupt the engine lifecycles"
                )
            if not waited:
                logger.warning(
                    "Writer lock on %s is held by %s: waiting up to %g s for it to be released",
                    target,
                    holder,
                    self._wait_seconds,
                )
                waited = True
            if self._stop.wait(min(self._retry_seconds, remaining)):
                return False
