"""Startup dependency check (ClickHouse ``/ping``) and the liveness heartbeat file."""

from __future__ import annotations

import http.client
import json
import logging
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pipeline_signals import ShutdownFlag

logger = logging.getLogger(__name__)

CLICKHOUSE_WAIT_SECONDS = 300.0
PING_TIMEOUT_SECONDS = 5.0
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 30.0
HEARTBEAT_FILENAME = "pipeline_runner.heartbeat"
HEARTBEAT_MIN_INTERVAL_SECONDS = 10.0


class ClickHouseUnavailableError(RuntimeError):
    """ClickHouse did not answer its ``/ping`` endpoint within the allowed time."""


def utc_now() -> datetime:
    """Current wall-clock time in UTC."""
    return datetime.now(UTC)


def ping_clickhouse(url: str, timeout: float = PING_TIMEOUT_SECONDS) -> bool:
    """Return True when ``url`` (the ClickHouse ``/ping`` endpoint) answers HTTP 200."""
    # Service names such as `clickhouse` only resolve inside the compose network, so an
    # HTTP(S)_PROXY inherited from the environment must never be used for this probe.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException) as exc:
        logger.debug("ClickHouse ping %s failed: %s", url, exc)
        return False


def wait_for_clickhouse(
    url: str,
    shutdown: ShutdownFlag,
    *,
    max_wait_seconds: float = CLICKHOUSE_WAIT_SECONDS,
    probe: Callable[[str], bool] = ping_clickhouse,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], bool] | None = None,
) -> bool:
    """Block until ClickHouse answers ``/ping``, retrying with capped exponential backoff.

    Returns True once ClickHouse is reachable and False when a stop was requested while
    waiting. Raises :class:`ClickHouseUnavailableError` after ``max_wait_seconds``.
    """
    pause = shutdown.wait if sleep is None else sleep
    deadline = clock() + max_wait_seconds
    delay = INITIAL_RETRY_DELAY_SECONDS
    attempt = 0
    while True:
        attempt += 1
        if probe(url):
            logger.info("ClickHouse is reachable at %s (attempt %d)", url, attempt)
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            raise ClickHouseUnavailableError(
                f"ClickHouse did not answer {url} within {max_wait_seconds:.0f}s ({attempt} attempts): "
                "check that the clickhouse service is running and CLICKHOUSE_HOST/CLICKHOUSE_PORT are correct"
            )
        wait = min(delay, remaining)
        logger.info("waiting for ClickHouse at %s (attempt %d failed, retrying in %.0fs)", url, attempt, wait)
        if pause(wait):
            return False
        delay = min(delay * 2, MAX_RETRY_DELAY_SECONDS)


def default_heartbeat_path() -> Path:
    """Heartbeat location: ``/tmp/pipeline_runner.heartbeat`` in the container (honours TMPDIR)."""
    return Path(tempfile.gettempdir()) / HEARTBEAT_FILENAME


class HeartbeatWriter:
    """Keeps a small JSON status file fresh so probes can tell a live runner from a hung one.

    ``publish`` replaces the status and writes immediately; ``tick`` (called while a task is
    running) only refreshes the timestamp, at most once per ``min_interval_seconds``. Writes
    are atomic (temporary file + rename) and failures are logged, never raised: a full disk
    must not stop the pipeline.
    """

    def __init__(
        self,
        path: Path,
        *,
        min_interval_seconds: float = HEARTBEAT_MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.path = path
        self._min_interval = min_interval_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._status: dict[str, object] = {}
        self._last_write: float | None = None
        self._failing = False

    def publish(self, status: Mapping[str, object]) -> None:
        """Replace the reported status and write the file now."""
        self._status = dict(status)
        self._write()

    def tick(self) -> None:
        """Refresh the timestamp of the current status unless it was written very recently."""
        if self._last_write is not None and self._clock() - self._last_write < self._min_interval:
            return
        self._write()

    def _write(self) -> None:
        payload = {"timestamp": self._wall_clock().isoformat(timespec="seconds"), **self._status}
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_path, self.path)
        except OSError as exc:
            if not self._failing:
                logger.warning("cannot write heartbeat file %s: %s", self.path, exc)
            self._failing = True
            return
        if self._failing:
            logger.info("heartbeat file %s is writable again", self.path)
        self._failing = False
        self._last_write = self._clock()
