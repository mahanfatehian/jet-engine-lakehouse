"""Single-writer lease of the simulator: advisory lock, bounded wait, fencing token after a reconnect."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import fleet_lease
import pytest
from fleet_lease import WRITER_LOCK_KEY, WriterLease, WriterLockError, advisory_lock_key

HOLDER = (4242, "engine-simulator", "172.18.0.9", datetime(2026, 10, 26, 12, 0, 5, tzinfo=UTC))
TARGET = "postgres:5432/jet_engine_db"


class FakeTime:
    """Monotonic clock plus a stop event whose ``wait`` advances the clock instead of sleeping."""

    def __init__(self, stop_at: float | None = None) -> None:
        self.now = 0.0
        self.stop_at = stop_at
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now += seconds
        return self.is_set()

    def is_set(self) -> bool:
        return self.stop_at is not None and self.now >= self.stop_at


class FakeCursor:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self._row: Any = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._session.statements.append((sql, params))
        answers = {
            fleet_lease.TRY_LOCK_SQL: lambda: (self._session.lock_results.pop(0),),
            fleet_lease.LOCK_HOLDER_SQL: lambda: self._session.holder,
            fleet_lease.LEASE_TOKEN_SQL: lambda: self._session.token_row,
        }
        self._row = answers.get(sql, lambda: None)()

    def fetchone(self) -> Any:
        return self._row


class FakeSession:
    """Duck-typed psycopg2 connection answering the lease's statements from a script."""

    def __init__(self, lock_results: list[bool], holder: Any = HOLDER, token_row: Any = None) -> None:
        self.lock_results = lock_results
        self.holder = holder
        self.token_row = token_row
        self.statements: list[tuple[str, Any]] = []
        self.commits = 0

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.commits += exc_type is None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def sql(self) -> list[str]:
        return [sql for sql, _ in self.statements]


def lease(time: FakeTime) -> WriterLease:
    return WriterLease(time, wait_seconds=60.0, retry_seconds=2.0, clock=time.clock)  # type: ignore[arg-type]


def test_lock_key_is_a_fixed_signed_64_bit_value_derived_from_the_simulator_name() -> None:
    # Pinned: a new key would let an old and a new simulator version write side by side.
    assert WRITER_LOCK_KEY == advisory_lock_key("jet-engine-simulator") == -5234131028265891191
    assert advisory_lock_key("another-writer") != WRITER_LOCK_KEY
    classid, objid = fleet_lease._lock_ids(WRITER_LOCK_KEY)
    assert (classid, objid) == (3076301199, 3093072521)
    assert ((classid << 32) | objid) - (1 << 64) == WRITER_LOCK_KEY, "pg_locks shows the key's two halves"


def test_free_lock_is_taken_and_the_fencing_token_recorded() -> None:
    time, session = FakeTime(), FakeSession([True])
    writer = lease(time)
    assert writer.acquire(session, TARGET)  # type: ignore[arg-type]
    assert session.sql() == [
        fleet_lease.TRY_LOCK_SQL,
        fleet_lease.LEASE_TABLE_DDL,
        fleet_lease.CLAIM_LEASE_SQL,
    ]
    assert session.statements[0][1] == (WRITER_LOCK_KEY,)
    assert session.statements[2][1] == (WRITER_LOCK_KEY, writer.token)
    assert time.waits == [] and session.commits >= 2


def test_lock_held_by_another_simulator_is_refused_after_the_wait(caplog: pytest.LogCaptureFixture) -> None:
    time, session = FakeTime(), FakeSession([False] * 40)
    with caplog.at_level(logging.WARNING, logger="fleet_lease"), pytest.raises(WriterLockError) as refused:
        lease(time).acquire(session, TARGET)  # type: ignore[arg-type]
    message = str(refused.value)
    assert f"another simulator is writing to {TARGET}" in message
    assert "pid 4242, application 'engine-simulator', client 172.18.0.9" in message
    assert "Stop it before starting this one" in message
    assert time.now == pytest.approx(60.0) and len(time.waits) == 30, "retried every 2 s for 60 s"
    assert session.sql().count(fleet_lease.TRY_LOCK_SQL) == 31
    assert fleet_lease.CLAIM_LEASE_SQL not in session.sql(), "a refused writer records nothing"
    assert caplog.text.count("waiting up to 60 s") == 1


def test_lock_released_during_the_wait_is_taken() -> None:
    time, session = FakeTime(), FakeSession([False, False, True], holder=None)
    assert lease(time).acquire(session, TARGET)  # type: ignore[arg-type]
    assert time.waits == [2.0, 2.0] and fleet_lease.CLAIM_LEASE_SQL in session.sql()


def test_shutdown_request_ends_the_wait_without_an_error() -> None:
    time, session = FakeTime(stop_at=5.0), FakeSession([False] * 10)
    assert not lease(time).acquire(session, TARGET)  # type: ignore[arg-type]
    assert time.now == pytest.approx(6.0) and fleet_lease.CLAIM_LEASE_SQL not in session.sql()


def test_reconnect_reacquires_the_lock_and_checks_the_fencing_token() -> None:
    time = FakeTime()
    writer = lease(time)
    assert writer.acquire(FakeSession([True]), TARGET)  # type: ignore[arg-type]
    acquired_at = datetime(2026, 10, 26, 12, 30, tzinfo=UTC)
    same = FakeSession([True], token_row=(writer.token, acquired_at))
    assert writer.acquire(same, TARGET)  # type: ignore[arg-type]
    assert same.sql() == [fleet_lease.TRY_LOCK_SQL, fleet_lease.LEASE_TOKEN_SQL], "the token is not rewritten"
    for token_row in (("0" * 32, acquired_at), None):
        took_over = FakeSession([True], token_row=token_row)
        with pytest.raises(
            WriterLockError, match=r"took over the writer lock .* while this one was disconnected"
        ):
            writer.acquire(took_over, TARGET)  # type: ignore[arg-type]
        assert took_over.commits == 1, "only the lock attempt commits; the failed token check rolls back"


def test_holder_description_survives_a_vanished_or_local_session() -> None:
    assert fleet_lease.describe_holder(None) == "a session that has just ended"
    local = fleet_lease.describe_holder((7, "", None, None))
    assert local == "pid 7, application '?', client local socket, since ?"
