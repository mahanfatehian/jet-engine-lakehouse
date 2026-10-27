"""The bounded hand-off of the HF path: byte budget, drops, in-flight backpressure and the backfill share."""

from __future__ import annotations

import threading

import pytest
from hf_queue import BACKFILL_SHARE, FrameQueue

FRAME = 100


def drain(queue: FrameQueue) -> list[object]:
    """Take every admitted frame, settling each at once (Kafka keeps up)."""
    taken = []
    while (item := queue.take()) is not None:
        queue.in_flight(FRAME)
        queue.settled(FRAME)
        taken.append(item.request)
    return taken


def test_live_frames_queue_in_order_until_the_budget_is_full_then_drop() -> None:
    queue = FrameQueue(budget_bytes=10 * FRAME, frame_bytes=FRAME)
    assert queue.offer(range(7)) == 0
    assert queue.offer(range(7, 14)) == 4, "three more fit, four are dropped"
    assert queue.depth() == (10, 0) and queue.used_bytes == 10 * FRAME
    assert drain(queue) == list(range(10))
    assert queue.used_bytes == 0 and queue.offer([99]) == 0


def test_unacknowledged_frames_hold_the_budget_until_kafka_settles_them() -> None:
    queue = FrameQueue(budget_bytes=4 * FRAME, frame_bytes=FRAME)
    queue.offer(range(4))
    for _ in range(4):
        item = queue.take()
        assert item is not None
        queue.in_flight(FRAME)
    assert queue.in_flight_bytes == 4 * FRAME and queue.take() is None
    assert queue.offer(["late"]) == 1, "Kafka is not acknowledging: new live frames are dropped"
    queue.settled(FRAME)
    assert queue.offer(["late"]) == 0 and queue.take() is not None, "one acknowledgement frees one frame"
    queue.settled(10 * FRAME)
    assert queue.in_flight_bytes == 0, "settling never goes negative"


def test_backfill_is_never_dropped_and_uses_at_most_its_share_of_the_budget() -> None:
    queue = FrameQueue(budget_bytes=10 * FRAME, frame_bytes=FRAME)
    queue.extend_backfill(range(1000))
    assert queue.depth() == (0, 1000) and queue.used_bytes == 0
    assert queue.offer(["live"]) == 0, "queued backfill holds no budget"
    in_flight = 0
    while queue.take() is not None:
        queue.in_flight(FRAME)
        in_flight += 1
    assert in_flight == int(BACKFILL_SHARE * 10), "backfill stops at its share of the budget"
    assert queue.offer(range(4)) == 0, "live frames keep the rest of the budget"
    queue.settled(in_flight * FRAME)
    rest = drain(queue)
    assert len(rest) == 1000 - in_flight + 5, "nothing was dropped"
    assert rest[-5:] == ["live", 0, 1, 2, 3], "commit order: the backfill before the live frames"


def test_a_frame_larger_than_the_budget_still_moves_when_nothing_is_in_flight() -> None:
    queue = FrameQueue(budget_bytes=FRAME, frame_bytes=FRAME)
    queue.extend_backfill(["big"])
    assert queue.take() is not None


def test_shutdown_helpers_discard_backfill_clear_and_reset() -> None:
    queue = FrameQueue(budget_bytes=10 * FRAME, frame_bytes=FRAME)
    queue.extend_backfill(range(3))
    queue.offer(["a", "b"])
    queue.extend_backfill(range(2))
    assert queue.discard_backfill() == 5 and queue.depth() == (2, 0)
    queue.in_flight(5 * FRAME)
    queue.reset_in_flight()
    assert queue.in_flight_bytes == 0
    assert queue.clear() == (2, 0) and queue.used_bytes == 0 and queue.take() is None


def test_invalid_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        FrameQueue(budget_bytes=0, frame_bytes=FRAME)


def test_concurrent_producers_and_a_consumer_keep_the_books_balanced() -> None:
    queue = FrameQueue(budget_bytes=50 * FRAME, frame_bytes=FRAME)
    dropped, taken, done = [], [], threading.Event()

    def tick_thread(offset: int) -> None:
        for start in range(offset, offset + 2000, 10):
            dropped.append(queue.offer(range(start, start + 10)))

    def worker() -> None:
        while not done.is_set() or queue.depth() != (0, 0):
            item = queue.take()
            if item is not None:
                taken.append(item.request)

    consumer = threading.Thread(target=worker)
    producers = [threading.Thread(target=tick_thread, args=(n * 10_000,)) for n in range(3)]
    consumer.start()
    for thread in producers:
        thread.start()
    for thread in producers:
        thread.join()
    done.set()
    consumer.join(10)
    assert len(taken) + sum(dropped) == 6000 and len(set(taken)) == len(taken)
    assert queue.used_bytes == 0
