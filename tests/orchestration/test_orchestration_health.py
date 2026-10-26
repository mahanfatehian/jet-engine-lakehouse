"""Startup wait for ClickHouse and the heartbeat file."""

from __future__ import annotations

import json
import logging
import socket
import tempfile
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from pipeline_health import (
    ClickHouseUnavailableError,
    HeartbeatWriter,
    default_heartbeat_path,
    ping_clickhouse,
    wait_for_clickhouse,
)
from pipeline_signals import ShutdownFlag

URL = "http://clickhouse:8123/ping"


class FlakyProbe:
    """Fails a fixed number of times, then succeeds (or never succeeds)."""

    def __init__(self, failures: float) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, url: str) -> bool:
        self.calls += 1
        return self.calls > self.failures


def _sleeper(clock: Any, sleeps: list[float]) -> Any:
    def sleep(seconds: float) -> bool:
        sleeps.append(seconds)
        clock.advance(seconds)
        return False

    return sleep


def test_waits_with_exponential_backoff_until_clickhouse_answers(clock: Any) -> None:
    sleeps: list[float] = []
    probe = FlakyProbe(failures=3)

    ready = wait_for_clickhouse(URL, ShutdownFlag(), probe=probe, clock=clock, sleep=_sleeper(clock, sleeps))

    assert ready is True
    assert probe.calls == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_with_a_clear_error_after_the_deadline(clock: Any) -> None:
    sleeps: list[float] = []

    with pytest.raises(
        ClickHouseUnavailableError, match=r"did not answer http://clickhouse:8123/ping within 300s"
    ):
        wait_for_clickhouse(
            URL,
            ShutdownFlag(),
            max_wait_seconds=300,
            probe=FlakyProbe(failures=float("inf")),
            clock=clock,
            sleep=_sleeper(clock, sleeps),
        )

    assert sleeps[:6] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert max(sleeps) == 30.0
    assert sum(sleeps) == pytest.approx(300.0)


def test_stop_request_aborts_the_wait(clock: Any) -> None:
    ready = wait_for_clickhouse(
        URL, ShutdownFlag(), probe=FlakyProbe(failures=float("inf")), clock=clock, sleep=lambda seconds: True
    )

    assert ready is False


class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status = 200 if self.path == "/ping" else 503
        body = b"Ok.\n"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def ping_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _PingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_ping_reports_healthy_and_unhealthy_endpoints(ping_server: str) -> None:
    assert ping_clickhouse(f"{ping_server}/ping") is True
    assert ping_clickhouse(f"{ping_server}/unavailable") is False


def test_ping_reports_unreachable_server() -> None:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        free_port = probe_socket.getsockname()[1]

    assert ping_clickhouse(f"http://127.0.0.1:{free_port}/ping", timeout=2.0) is False


def test_heartbeat_is_written_atomically_as_json(clock: Any, tmp_path: Path) -> None:
    path = tmp_path / "pipeline_runner.heartbeat"
    writer = HeartbeatWriter(path, clock=clock, wall_clock=lambda: datetime(2026, 10, 26, 9, 30, tzinfo=UTC))

    writer.publish({"state": "idle", "cycles_completed": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "timestamp": "2026-10-26T09:30:00+00:00",
        "state": "idle",
        "cycles_completed": 3,
    }
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_tick_refreshes_the_timestamp_at_a_bounded_rate(clock: Any, tmp_path: Path) -> None:
    stamps = iter(datetime(2026, 10, 26, 9, minute, tzinfo=UTC) for minute in range(60))
    writer = HeartbeatWriter(
        tmp_path / "hb", min_interval_seconds=10, clock=clock, wall_clock=lambda: next(stamps)
    )
    writer.publish({"state": "running", "current_task": "train"})

    def timestamp() -> str:
        return json.loads(writer.path.read_text(encoding="utf-8"))["timestamp"]

    clock.advance(5)
    writer.tick()
    assert timestamp() == "2026-10-26T09:00:00+00:00"

    clock.advance(6)
    writer.tick()
    assert timestamp() == "2026-10-26T09:01:00+00:00"
    assert json.loads(writer.path.read_text(encoding="utf-8"))["current_task"] == "train"


def test_unwritable_heartbeat_is_reported_once_and_never_raises(
    clock: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pipeline_health")
    writer = HeartbeatWriter(tmp_path / "missing-dir" / "hb", clock=clock)

    writer.publish({"state": "idle"})
    writer.publish({"state": "idle"})

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "cannot write heartbeat file" in warnings[0].getMessage()


def test_default_heartbeat_path_lives_in_the_temp_dir() -> None:
    assert default_heartbeat_path() == Path(tempfile.gettempdir()) / "pipeline_runner.heartbeat"
