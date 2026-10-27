"""The bounded hand-off between the tick loop and the HF publisher's worker thread.

One FIFO holds the frame requests in commit order (so an engine's frames reach its partition in order), and
one byte budget (SIM_HF_QUEUE_MAX_MB) bounds the memory of the whole path: it covers the live frames waiting
here, at their estimated serialized size, plus every frame handed to Kafka and not yet acknowledged, at its
real size. Kafka down means acknowledgements stop, the in-flight bytes reach the budget, the worker stops
taking frames and new live frames no longer fit: they are dropped and counted, while the tick only ever
takes a lock for a few list operations.

Backfill requests always queue: they are a few hundred bytes each until built, and losing history to a full
buffer would leave whole lifecycles without spectral features. The worker releases them to Kafka only while
the in-flight bytes stay below BACKFILL_SHARE of the budget, so live frames queued behind a replay of history
always keep the other part of the budget.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

BACKFILL_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class QueuedFrame:
    """A frame request waiting for the worker; ``reserved`` is the budget a live frame holds while queued."""

    request: Any
    reserved: int
    backfill: bool


class FrameQueue:
    """Thread-safe FIFO of frame requests under a byte budget (see the module doc)."""

    def __init__(self, budget_bytes: int, frame_bytes: int) -> None:
        if budget_bytes <= 0 or frame_bytes <= 0:
            raise ValueError("the budget and the frame size must be positive")
        self.budget_bytes = budget_bytes
        self.frame_bytes = frame_bytes
        self._lock = threading.Lock()
        self._frames: deque[QueuedFrame] = deque()
        self._queued_live_bytes = 0
        self._in_flight_bytes = 0
        self._backfill = 0

    def offer(self, requests: Iterable[Any]) -> int:
        """Queue live frame requests that fit the budget; returns how many were dropped."""
        dropped = 0
        with self._lock:
            for request in requests:
                if self._queued_live_bytes + self._in_flight_bytes + self.frame_bytes > self.budget_bytes:
                    dropped += 1
                    continue
                self._frames.append(QueuedFrame(request, self.frame_bytes, backfill=False))
                self._queued_live_bytes += self.frame_bytes
        return dropped

    def extend_backfill(self, requests: Iterable[Any]) -> None:
        """Queue backfill frame requests (never dropped)."""
        with self._lock:
            before = len(self._frames)
            self._frames.extend(QueuedFrame(request, 0, backfill=True) for request in requests)
            self._backfill += len(self._frames) - before

    def take(self) -> QueuedFrame | None:
        """The next frame the in-flight budget admits, or None (empty queue, or Kafka has not caught up).

        The head frame needs room for its size within the budget (BACKFILL_SHARE of it for backfill); when
        nothing is in flight any frame is admitted, so a budget below one frame still makes progress.
        """
        with self._lock:
            if not self._frames:
                return None
            head = self._frames[0]
            limit = self.budget_bytes * (BACKFILL_SHARE if head.backfill else 1.0)
            if self._in_flight_bytes and self._in_flight_bytes + self.frame_bytes > limit:
                return None
            self._frames.popleft()
            self._queued_live_bytes -= head.reserved
            self._backfill -= head.backfill
            return head

    def in_flight(self, size: int) -> None:
        """Account a serialized frame of ``size`` bytes handed to Kafka."""
        with self._lock:
            self._in_flight_bytes += size

    def settled(self, size: int) -> None:
        """Release a frame of ``size`` bytes that Kafka acknowledged or failed."""
        with self._lock:
            self._in_flight_bytes = max(0, self._in_flight_bytes - size)

    def reset_in_flight(self) -> None:
        """Forget every in-flight frame (their client was replaced and will never report them)."""
        with self._lock:
            self._in_flight_bytes = 0

    def discard_backfill(self) -> int:
        """Remove every queued backfill frame (shutdown); returns how many were removed."""
        with self._lock:
            removed = self._backfill
            self._frames = deque(frame for frame in self._frames if not frame.backfill)
            self._backfill = 0
            return removed

    def clear(self) -> tuple[int, int]:
        """Discard every queued frame; returns how many (live, backfill) were discarded."""
        with self._lock:
            backfill = self._backfill
            live = len(self._frames) - backfill
            self._frames.clear()
            self._queued_live_bytes = 0
            self._backfill = 0
            return live, backfill

    def depth(self) -> tuple[int, int]:
        """Queued (live, backfill) frames."""
        with self._lock:
            return len(self._frames) - self._backfill, self._backfill

    @property
    def in_flight_bytes(self) -> int:
        """Bytes handed to Kafka and not yet acknowledged."""
        with self._lock:
            return self._in_flight_bytes

    @property
    def used_bytes(self) -> int:
        """Budget in use: queued live frames plus in-flight bytes."""
        with self._lock:
            return self._queued_live_bytes + self._in_flight_bytes
