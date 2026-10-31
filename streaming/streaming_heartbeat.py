"""Liveness heartbeat of the streaming job: a small JSON status file kept fresh by the supervisor.

The supervisor publishes the state of every query on each round (every two seconds); the
file is rewritten at most every ``min_interval_s`` unless a query changed state, so it is
refreshed every few seconds while queries run, wait for a dependency or back off before a
restart. A container healthcheck therefore only has to test the file's age, for example
``test -n "$(find /tmp/streaming.heartbeat -mmin -2)"``: a supervisor stuck in a hung JVM
call stops refreshing it. The per-query fields tell which query is unhealthy and why:
``active``, ``last_progress_at`` (when the query was last seen completing a trigger, empty
triggers included, so a stale value marks a stalled query), ``consecutive_failures`` and
``last_error``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 5.0


def utc_now_iso() -> str:
    """Current UTC wall-clock time as ISO-8601 with milliseconds and a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class HeartbeatWriter:
    """Writes ``{"timestamp", "pid", **status}`` as JSON, atomically (temporary file + rename).

    Readers never see a partially written file. Write failures are logged once and never
    raised: a full or read-only disk must not stop the queries (the healthcheck reports it).
    """

    def __init__(
        self,
        path: Path,
        *,
        min_interval_s: float = HEARTBEAT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.path = path
        self._temp_path = path.with_name(f".{path.name}.tmp")
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._last_write: float | None = None
        self._first_write: float | None = None
        self._failing = False

    def publish(self, status: Mapping[str, object], *, force: bool = False) -> bool:
        """Write ``status`` unless the file was written less than ``min_interval_s`` ago.

        ``force`` writes immediately (a query changed state). Returns True when written.
        """
        now = self._clock()
        if not force and self._last_write is not None and now - self._last_write < self._min_interval_s:
            return False
        if self._first_write is None:
            self._first_write = now
        elapsed = round(now - self._first_write, 1)
        payload = {"timestamp": self._wall_clock(), "pid": os.getpid(), "uptime_s": elapsed, **status}
        if not self._write(json.dumps(payload, indent=2, default=str) + "\n"):
            return False
        self._last_write = now
        return True

    def remove(self) -> None:
        """Delete the heartbeat file on graceful shutdown (best-effort)."""
        for target in (self.path, self._temp_path):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass

    def _write(self, text: str) -> bool:
        try:
            self._temp_path.write_text(text, encoding="utf-8")
            os.replace(self._temp_path, self.path)
        except OSError as exc:
            if not self._failing:
                logger.warning("cannot write heartbeat file %s: %s", self.path, exc)
            self._failing = True
            return False
        if self._failing:
            logger.info("heartbeat file %s is writable again", self.path)
        self._failing = False
        return True
