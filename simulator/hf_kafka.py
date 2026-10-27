"""Kafka side of the HF frame path: the confluent-kafka producer, Schema Registry client and Avro serializer.

The only module that imports confluent_kafka, and only inside :func:`open_kafka_sink`: with
SIM_HF_ENABLED=false (and in the unit tests) the client library is never loaded.

Producer settings (:func:`producer_config`):

    acks=all, enable.idempotence     retries never duplicate or reorder a frame within its partition
    partitioner=murmur2_random       the Java client's key hash: engine_id lands where other clients put it
    compression.type                 SIM_HF_COMPRESSION (zstd) on whole batches of frames
    linger.ms=50, batch.size=1 MiB   batching: the frames of a tick travel in a few requests
    message.max.bytes=8 MiB          the HF topics' max.message.bytes (a full-fidelity frame is ~1.7 MB)
    message.timeout.ms=300000        a frame Kafka has not acknowledged within 5 minutes is reported failed
    allow.auto.create.topics=false   topics are code (infrastructure/kafka/create-topics.sh)
    queue.buffering.max.kbytes       SIM_HF_QUEUE_MAX_MB plus one frame: hf_queue's budget binds first

Values use the Confluent wire format (magic byte 0, 4-byte schema id, Avro body) under the TopicNameStrategy
subject ``<topic>-value``; keys are the engine_id in UTF-8. The message timestamp is the produce time, not
frame_start_ts, so backfilled (backdated) frames stay in the topic as long as live ones.

Client errors reach ``error_cb`` (logged once a minute per error code); librdkafka's own log lines go to the
Python logger ``hf_kafka.librdkafka``, where each facility (FAIL, ...) passes once a minute: while a broker is
down the client repeats the same connect failure every few seconds.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hf_frames import TOPIC_MAX_MESSAGE_BYTES
from hf_producer import first_line

if TYPE_CHECKING:
    from hf_config import HfConfig

logger = logging.getLogger("hf_kafka")
CLIENT_LOGGER_NAME = "hf_kafka.librdkafka"

LINGER_MS = 50
BATCH_BYTES = 1024 * 1024
DELIVERY_TIMEOUT_MS = 300_000
QUEUE_MAX_MESSAGES = 1_000_000  # the byte budget binds first
STATISTICS_INTERVAL_MS = 1000
REGISTRY_TIMEOUT_SECONDS = 5.0
ERROR_LOG_INTERVAL_SECONDS = 60.0

DeliveryCallback = Callable[[Any, Any], None]
"""confluent-kafka delivery report: (KafkaError or None, Message)."""


def producer_config(config: HfConfig, client_id: str, *, statistics: bool = False) -> dict[str, Any]:
    """librdkafka settings of the HF producer (module doc); ``statistics`` adds 1 s client statistics."""
    return {
        "bootstrap.servers": config.bootstrap_servers,
        "client.id": client_id,
        "acks": "all",
        "enable.idempotence": True,
        "max.in.flight.requests.per.connection": 5,
        "partitioner": "murmur2_random",
        "compression.type": config.compression,
        "linger.ms": LINGER_MS,
        "batch.size": BATCH_BYTES,
        "message.max.bytes": TOPIC_MAX_MESSAGE_BYTES,
        "message.timeout.ms": DELIVERY_TIMEOUT_MS,
        "allow.auto.create.topics": False,
        "queue.buffering.max.kbytes": (config.queue_max_bytes + TOPIC_MAX_MESSAGE_BYTES) // 1024,
        "queue.buffering.max.messages": QUEUE_MAX_MESSAGES,
        "statistics.interval.ms": STATISTICS_INTERVAL_MS if statistics else 0,
    }


def read_schema(path: Path) -> str:
    """Text of the Avro schema file; ValueError naming the file when it is missing or not a record schema."""
    try:
        text = path.read_text(encoding="utf-8")
        schema = json.loads(text)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the HF frame schema {path}: {exc}") from exc
    if not isinstance(schema, dict) or schema.get("type") != "record":
        raise ValueError(f"{path} is not an Avro record schema")
    return text


class RepeatFilter(logging.Filter):
    """Passes each librdkafka log facility at most once per ``interval`` seconds.

    confluent-kafka logs ``logger.log(level, "%s [%s] %s", facility, client_name, message)``; a record without
    those arguments is keyed by its message.
    """

    def __init__(
        self, interval: float = ERROR_LOG_INTERVAL_SECONDS, clock: Callable[[], float] = time.monotonic
    ) -> None:
        super().__init__()
        self._interval = interval
        self._clock = clock
        self._passed: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """True when the record's facility has not passed within the interval."""
        args = record.args if isinstance(record.args, tuple) else ()
        key = str(args[0]) if args else record.getMessage()
        now = self._clock()
        if now - self._passed.get(key, -self._interval) < self._interval:
            return False
        self._passed[key] = now
        return True


def client_logger() -> logging.Logger:
    """The logger librdkafka's own log lines go to, with its RepeatFilter attached once."""
    client = logging.getLogger(CLIENT_LOGGER_NAME)
    if not any(isinstance(item, RepeatFilter) for item in client.filters):
        client.addFilter(RepeatFilter())
    return client


class ClientEvents:
    """librdkafka's error and statistics callbacks: throttled logging plus the latest values for the caller.

    ``fatal`` holds the first fatal error (the producer instance is unusable from then on); ``tx_bytes`` the
    bytes the client has sent to the brokers (compressed batches plus protocol framing) when statistics
    are on.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._logged_at: dict[str, float] = {}
        self.errors = 0
        self.fatal: str | None = None
        self.tx_bytes: int | None = None

    def callbacks(self, statistics: bool) -> dict[str, Any]:
        """The callback entries of the producer configuration."""
        hooks: dict[str, Any] = {"error_cb": self.on_error}
        if statistics:
            hooks["stats_cb"] = self.on_stats
        return hooks

    def on_error(self, error: Any) -> None:
        """Client-level error (broker down, authentication, fatal idempotence state); logged once a minute."""
        self.errors += 1
        text = error.str() if hasattr(error, "str") else str(error)
        if getattr(error, "fatal", lambda: False)():
            self.fatal = self.fatal or text
            logger.error("Kafka producer fatal error: %s", text)
            return
        key = str(error.code()) if hasattr(error, "code") else text
        now = self._clock()
        if now - self._logged_at.get(key, -ERROR_LOG_INTERVAL_SECONDS) >= ERROR_LOG_INTERVAL_SECONDS:
            self._logged_at[key] = now
            logger.warning("Kafka client error (repeats are logged once a minute): %s", text)

    def on_stats(self, payload: str) -> None:
        """Statistics JSON: keeps the total bytes sent to the brokers."""
        try:
            self.tx_bytes = int(json.loads(payload)["tx_bytes"])
        except (ValueError, KeyError, TypeError):
            logger.debug("Ignoring unreadable Kafka client statistics")


class KafkaSink:
    """Serializes frames and produces them to one topic (see :func:`open_kafka_sink`)."""

    def __init__(
        self,
        topic: str,
        *,
        producer: Any,
        registry: Any,
        schema: Any,
        serializer: Callable[[Any, Any], bytes],
        context: Any,
        subject: str,
        events: ClientEvents,
        endpoints: tuple[str, str],
    ) -> None:
        self.topic = topic
        self.subject = subject
        self.events = events
        self.schema_id: int | None = None
        self.bootstrap_servers, self.registry_url = endpoints
        self._producer = producer
        self._registry = registry
        self._schema = schema
        self._serializer = serializer
        self._context = context

    def check_ready(self, timeout: float) -> str | None:
        """None once the topic exists and the value schema is registered, else why not, naming the service
        (topic auto-creation is off: a missing topic waits for kafka-init)."""
        return self._topic_problem(timeout) or self._registration_problem()

    def _topic_problem(self, timeout: float) -> str | None:
        try:
            metadata = self._producer.list_topics(topic=self.topic, timeout=timeout)
        except Exception as exc:
            return f"Kafka {self.bootstrap_servers}: {first_line(exc)}"
        topic = metadata.topics.get(self.topic)
        if topic is None or topic.error is not None or not topic.partitions:
            reason = topic.error.str() if topic is not None and topic.error is not None else "no partitions"
            return f"topic {self.topic} does not exist yet ({reason})"
        return None

    def _registration_problem(self) -> str | None:
        try:
            self.schema_id = int(self._registry.register_schema(self.subject, self._schema))
        except Exception as exc:
            return f"Schema Registry {self.registry_url} (subject {self.subject}): {first_line(exc)}"
        return None

    @property
    def fatal_error(self) -> str | None:
        """The producer's fatal error, if one occurred."""
        return self.events.fatal

    @property
    def wire_bytes(self) -> int | None:
        """Bytes sent to the brokers so far (compressed), when the sink was opened with statistics."""
        return self.events.tx_bytes

    def serialize(self, record: Mapping[str, Any]) -> bytes:
        """Confluent wire-format value of one EngineSensorFrame record."""
        return self._serializer(record, self._context)

    def produce(self, key: bytes, value: bytes, on_delivery: DeliveryCallback) -> None:
        """Hand one message to the client (BufferError when its local queue is full)."""
        self._producer.produce(self.topic, value=value, key=key, on_delivery=on_delivery)

    def poll(self, timeout: float) -> int:
        """Serve delivery reports and client callbacks for up to ``timeout`` seconds."""
        return self._producer.poll(timeout)

    def flush(self, timeout: float) -> int:
        """Wait up to ``timeout`` seconds for outstanding messages; returns how many are still pending."""
        return self._producer.flush(timeout)

    def close(self) -> None:
        """Release the registry session (the producer is released with this object)."""
        self._registry.close()


def open_kafka_sink(config: HfConfig, topic: str, *, client_id: str, statistics: bool = False) -> KafkaSink:
    """Producer, Schema Registry client and Avro serializer for ``topic``; no network traffic yet.

    Raises ValueError for an unreadable schema file and the client's exceptions for invalid settings.
    """
    from confluent_kafka import Producer
    from confluent_kafka.schema_registry import Schema, SchemaRegistryClient, topic_subject_name_strategy
    from confluent_kafka.schema_registry.avro import AvroSerializer
    from confluent_kafka.serialization import MessageField, SerializationContext

    schema = Schema(read_schema(config.schema_path), "AVRO")
    # One attempt per call: the publisher retries with its own backoff and honours shutdown requests.
    registry = SchemaRegistryClient(
        {"url": config.schema_registry_url, "timeout": REGISTRY_TIMEOUT_SECONDS, "max.retries": 0}
    )
    # The same Schema object is registered by check_ready, so the serializer finds its id in the client cache.
    serializer = AvroSerializer(
        registry,
        schema,
        conf={"auto.register.schemas": True, "subject.name.strategy": topic_subject_name_strategy},
    )
    context = SerializationContext(topic, MessageField.VALUE)
    events = ClientEvents()
    settings = producer_config(config, client_id, statistics=statistics) | events.callbacks(statistics)
    producer = Producer(settings | {"logger": client_logger()})
    return KafkaSink(
        topic,
        producer=producer,
        registry=registry,
        schema=schema,
        serializer=serializer,
        context=context,
        subject=topic_subject_name_strategy(context, None),
        events=events,
        endpoints=(config.bootstrap_servers, config.schema_registry_url),
    )
