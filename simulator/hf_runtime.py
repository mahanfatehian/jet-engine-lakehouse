"""Wiring of the HF frame path into the simulator process (used by ``engine_generator.main``).

``open_hf_path`` returns the commit hooks the fleet calls after each committed transaction (``live`` for
the tick loop, ``backfill`` for each engine's history), ``resume_backfill`` (start-up, after the bootstrap:
queues again the frames earlier backfills still owe) and ``close``. With SIM_HF_ENABLED=false they are
no-ops, ``marks_backfill`` is off, and neither the publisher nor the Kafka client library is loaded. The
frame sequence resumes from the telemetry row count of every engine, read before any backfill, so frame_seq
stays the engine's pulse ordinal; resumed backfill frames carry their original ordinals (hf_backfill).

The resumable backfill (hf_backfill) is bookkeeping on the main thread: the live hook and ``close`` delete
the markers of engines whose backfill frames Kafka has all acknowledged, on the open connection only, and
nothing of it ever stops the 1 Hz path: its errors are logged and leave the markers for the next start.

The live producer never times out a frame (DELIVERY_TIMEOUT_MS): while Kafka is down its frames wait in the
client, and the SIM_HF_QUEUE_MAX_MB budget, not a timer, decides when new frames are dropped (hf_queue).
``close`` still ends within its flush deadline: whatever Kafka has not acknowledged by then is discarded.

Settings that cannot work stop the process at start-up (ValueError, exit code 2): a frame above the topic's
message size limit, or a SIM_HF_QUEUE_MAX_MB that cannot hold one frame. A Kafka client that cannot even be
configured (unreadable schema file, invalid client settings, library missing on a host run) disables the HF
path with an error instead of stopping the lossless 1 Hz path.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from hf_backfill import BackfillStore, clear_paid, resume_backfill
from hf_config import HfConfig
from hf_frames import FrameBuilder, FrameTap, check_frame_size
from hf_kafka import open_kafka_sink
from hf_producer import MIB, FramePublisher, FrameSink

if TYPE_CHECKING:
    from fleet_bootstrap import EngineFactory
    from fleet_runtime import CommitHook, WriteOp

logger = logging.getLogger("hf_runtime")

CLIENT_ID = "engine-simulator-hf"
DELIVERY_TIMEOUT_MS = 0  # message.timeout.ms=0: librdkafka never times out a frame (module doc)


class HfStore(BackfillStore, Protocol):
    """The part of fleet_db.FleetDatabase the HF path needs."""

    def telemetry_counts(self, engine_ids: Sequence[str]) -> dict[str, int]:
        """Telemetry rows per engine."""


def _nothing() -> None:
    """Close of a disabled HF path."""


def _resume_nothing(factory: EngineFactory, stop_event: threading.Event) -> None:
    """Backfill resume of a disabled HF path: the markers stay untouched."""


@dataclass(frozen=True)
class HfPath:
    """Commit hooks for the fleet, the backfill resume and the shutdown of the publisher behind them (all
    no-ops when disabled). ``marks_backfill``: backfilled histories record the HF frames they owe, which is
    SIM_HF_ENABLED, also when the Kafka client cannot be set up (a later start pays them)."""

    live: CommitHook | None = None
    backfill: CommitHook | None = None
    resume_backfill: Callable[[EngineFactory, threading.Event], None] = _resume_nothing
    close: Callable[[], None] = _nothing
    marks_backfill: bool = False


def load_hf_config(options: Any, env: Mapping[str, str] | None = None) -> HfConfig:
    """HF settings from the environment plus the load test's overrides (ValueError naming the problem).

    When frames will be built, one frame must fit the topic's message size limit. When the live path will
    run, SIM_HF_QUEUE_MAX_MB must also hold one frame: a smaller budget drops most frames even while Kafka
    keeps up. The load test only sizes its client queue with it, and that queue is one maximum-size message
    larger anyway.
    """
    config = HfConfig.from_env(env)
    if options.benchmark:
        config = config.for_benchmark(options.benchmark_sample_rate, options.benchmark_vib_channels)
    if config.enabled or options.benchmark:
        frame_bytes = check_frame_size(config)
        if not options.benchmark and config.queue_max_bytes < frame_bytes:
            raise ValueError(
                f"SIM_HF_QUEUE_MAX_MB={config.queue_max_mb} cannot hold one HF frame of {frame_bytes:,} B: "
                f"raise it to at least {math.ceil(frame_bytes / MIB)} or lower SIM_HF_SAMPLE_RATE_HZ, "
                "SIM_HF_VIB_CHANNELS or SIM_HF_FRAME_MS"
            )
    return config


def kafka_sink_factory(config: HfConfig, topic: str, *, statistics: bool = False) -> Callable[[], FrameSink]:
    """Factory of the live path's Kafka sinks for ``topic``, without a delivery timeout (module doc); called
    again when a fatal error replaces the client."""
    return functools.partial(
        open_kafka_sink,
        config,
        topic,
        client_id=CLIENT_ID,
        delivery_timeout_ms=DELIVERY_TIMEOUT_MS,
        statistics=statistics,
    )


def open_hf_path(
    config: HfConfig,
    seed: int,
    store: HfStore,
    engine_models: Mapping[str, str],
    sink_factory: Callable[[], FrameSink] | None = None,
) -> HfPath:
    """Start the publisher and return the fleet's commit hooks (see the module doc).

    Database errors while reading the telemetry counts propagate like any other start-up read.
    """
    if not config.enabled:
        logger.info("HF sensor frames disabled (SIM_HF_ENABLED=false): only the 1 Hz path runs")
        return HfPath()
    counts = store.telemetry_counts(list(engine_models))
    try:
        builder = FrameBuilder(config, seed)
        publisher = FramePublisher(config, builder, sink_factory or kafka_sink_factory(config, config.topic))
    except Exception as exc:
        logger.error("HF sensor frames disabled, the Kafka client cannot be set up: %s", exc)
        return HfPath(marks_backfill=True)
    publisher.start()
    tap = FrameTap(publisher, engine_models, counts)
    logger.info(
        "HF sensor frames enabled: %s; about %.1f KB per frame", config.summary(), builder.frame_bytes / 1e3
    )

    def live(ops: Sequence[WriteOp], results: Sequence[Any]) -> None:
        tap.live(ops, results)
        clear_paid(publisher.ledger, store)

    def resume(factory: EngineFactory, stop_event: threading.Event) -> None:
        try:
            resume_backfill(store, factory, publisher, stop_event)
        except Exception:
            logger.exception("HF backfill resume failed; the remaining markers wait for the next start")

    def close() -> None:
        publisher.close()
        clear_paid(publisher.ledger, store)

    return HfPath(live=live, backfill=tap.backfill, resume_backfill=resume, close=close, marks_backfill=True)
