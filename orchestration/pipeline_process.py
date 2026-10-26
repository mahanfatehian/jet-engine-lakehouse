"""Execution of one task as a child process: streamed output, timeout, graceful termination."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import IO

from pipeline_signals import ShutdownFlag
from pipeline_tasks import (
    EXIT_SPAWN_FAILED,
    EXIT_TIMEOUT,
    TaskResult,
    TaskSpec,
    TaskStatus,
    classify_exit_code,
    normalize_returncode,
)

logger = logging.getLogger(__name__)

PopenFactory = Callable[..., "subprocess.Popen[str]"]

# Must stay below the orchestrator's docker `stop_grace_period` (30 s) so a forwarded
# SIGTERM can finish before Docker escalates to SIGKILL on the whole container.
TERMINATE_GRACE_SECONDS = 20.0
POLL_INTERVAL_SECONDS = 0.5
OUTPUT_TAIL_LINES = 40
READER_JOIN_SECONDS = 5.0


class SubprocessTaskRunner:
    """Runs a :class:`TaskSpec` as a child process and classifies how it ended.

    The child's stdout and stderr are streamed line by line into the log with a ``[task]``
    prefix, and the most recent lines are kept for the failure report. A child that exceeds
    its timeout, or is running when a stop is requested, receives SIGTERM and is killed if it
    is still alive after the grace period (immediately when the stop is forced).
    """

    def __init__(
        self,
        shutdown: ShutdownFlag,
        *,
        popen: PopenFactory = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        base_env: Mapping[str, str] | None = None,
        on_poll: Callable[[], None] | None = None,
        grace_seconds: float = TERMINATE_GRACE_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        tail_lines: int = OUTPUT_TAIL_LINES,
    ) -> None:
        self._shutdown = shutdown
        self._popen = popen
        self._clock = clock
        self._base_env = base_env
        self._on_poll = on_poll
        self._grace_seconds = grace_seconds
        self._poll_interval = poll_interval
        self._tail_lines = tail_lines

    def run(self, spec: TaskSpec) -> TaskResult:
        """Execute ``spec`` until it exits (or is stopped) and return the classified result."""
        logger.info(
            "task %s starting (timeout %.0fs): %s", spec.name, spec.timeout_seconds, shlex.join(spec.argv)
        )
        started = self._clock()
        try:
            process = self._spawn(spec)
        except OSError as exc:
            message = f"could not start {spec.argv[0]!r}: {exc}"
            return self._finish(spec, started, EXIT_SPAWN_FAILED, tail=(message,))

        tail: deque[str] = deque(maxlen=self._tail_lines)
        reader = threading.Thread(
            target=_stream_output,
            args=(process.stdout, spec.name, tail),
            name=f"{spec.name}-output",
            daemon=True,
        )
        reader.start()
        timed_out, interrupted = self._supervise(process, spec, started)
        # A grandchild that inherited the pipe could keep it open forever: never block on it.
        reader.join(READER_JOIN_SECONDS)
        if reader.is_alive():
            logger.warning(
                "task %s exited but its output pipe is still open (orphaned grandchild?)", spec.name
            )
        exit_code = EXIT_TIMEOUT if timed_out else normalize_returncode(process.returncode)
        # deque.copy() runs entirely in C, so it cannot interleave with a still-running reader.
        output_tail = tuple(tail.copy())
        return self._finish(
            spec, started, exit_code, tail=output_tail, timed_out=timed_out, interrupted=interrupted
        )

    def _spawn(self, spec: TaskSpec) -> subprocess.Popen[str]:
        env = dict(os.environ if self._base_env is None else self._base_env)
        for key, value in spec.env_defaults:
            env.setdefault(key, value)
        # Python block-buffers a piped stdout: without this the log would lag behind the
        # child and a killed child would lose its last (most interesting) lines.
        env["PYTHONUNBUFFERED"] = "1"
        return self._popen(
            list(spec.argv),
            cwd=spec.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _supervise(self, process: subprocess.Popen[str], spec: TaskSpec, started: float) -> tuple[bool, bool]:
        """Wait for the child while enforcing its timeout and forwarding stop requests.

        Returns ``(timed_out, interrupted)``.
        """
        deadline = started + spec.timeout_seconds
        timed_out = interrupted = killed = False
        kill_at: float | None = None
        while not self._exited_within_poll(process):
            if self._on_poll is not None:
                self._on_poll()
            now = self._clock()
            if kill_at is None:
                if self._shutdown.requested:
                    interrupted = True
                    logger.warning(
                        "forwarding SIGTERM to task %s (pid %s): the runner is stopping",
                        spec.name,
                        process.pid,
                    )
                elif now >= deadline:
                    timed_out = True
                    logger.error(
                        "task %s exceeded its %.0fs timeout: sending SIGTERM", spec.name, spec.timeout_seconds
                    )
                else:
                    continue
                process.terminate()
                kill_at = now + self._grace_seconds
            elif not killed and (now >= kill_at or self._shutdown.forced):
                cause = (
                    "stop was forced"
                    if self._shutdown.forced
                    else f"no exit {self._grace_seconds:.0f}s after SIGTERM"
                )
                logger.error("killing task %s (pid %s): %s", spec.name, process.pid, cause)
                process.kill()
                killed = True
        return timed_out, interrupted

    def _exited_within_poll(self, process: subprocess.Popen[str]) -> bool:
        try:
            process.wait(timeout=self._poll_interval)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _finish(
        self,
        spec: TaskSpec,
        started: float,
        exit_code: int,
        *,
        tail: tuple[str, ...],
        timed_out: bool = False,
        interrupted: bool = False,
    ) -> TaskResult:
        result = TaskResult(
            name=spec.name,
            status=classify_exit_code(exit_code),
            exit_code=exit_code,
            duration_seconds=self._clock() - started,
            timed_out=timed_out,
            interrupted=interrupted,
            output_tail=tail,
        )
        _log_result(result)
        return result


def _stream_output(stream: IO[str] | None, task: str, tail: deque[str]) -> None:
    """Log every line the child writes and keep the most recent ones in ``tail``."""
    if stream is None:
        return
    try:
        with stream:
            for raw_line in stream:
                line = raw_line.rstrip()
                tail.append(line)
                logger.info("[%s] %s", task, line)
    except (OSError, ValueError) as exc:
        # The pipe can vanish underneath the reader when a child is killed mid-write.
        logger.debug("[%s] output stream closed: %s", task, exc)


def _log_result(result: TaskResult) -> None:
    duration = result.duration_seconds
    if result.status is TaskStatus.SUCCESS:
        logger.info("task %s SUCCESS in %.1fs", result.name, duration)
        return
    if result.status is TaskStatus.SKIPPED:
        logger.info("task %s SKIPPED in %.1fs (exit code 3: preconditions not met)", result.name, duration)
        return
    cause = "timed out" if result.timed_out else f"exit code {result.exit_code}"
    if result.interrupted:
        cause += ", interrupted by shutdown"
    logger.error(
        "task %s FAILED in %.1fs (%s); last %d output line(s):\n%s",
        result.name,
        duration,
        cause,
        len(result.output_tail),
        "\n".join(result.output_tail) or "<no output>",
    )
