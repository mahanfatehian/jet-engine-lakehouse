"""Startup readiness probes for ClickHouse, MinIO and Kafka.

Starting the queries before their dependencies accept connections would only fail the
first micro-batch and crash-loop the container, so the job waits with a bounded backoff
and gives up with a clear error when a dependency never comes up.
"""

from __future__ import annotations

import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Callable

from streaming_config import StreamingSettings
from streaming_retry import RetryPolicy, WaitFn, call_with_retry

logger = logging.getLogger(__name__)

READINESS_POLICY = RetryPolicy(attempts=12, base_delay_s=1.0, max_delay_s=15.0)  # ~2.4 min of waiting
PROBE_TIMEOUT_S = 5.0

# Probes target in-network endpoints: an HTTP(S)_PROXY from the environment must not apply.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class DependencyNotReadyError(RuntimeError):
    """A dependency did not pass its readiness probe."""


def probe_http(url: str, timeout_s: float = PROBE_TIMEOUT_S) -> None:
    """Succeed when ``url`` answers with any HTTP status below 500."""
    try:
        with _OPENER.open(url, timeout=timeout_s):
            return
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return
        raise DependencyNotReadyError(f"{url} answered HTTP {exc.code}") from exc
    except OSError as exc:  # URLError, refused connections and timeouts are all OSErrors
        raise DependencyNotReadyError(f"{url} is unreachable: {exc}") from exc


def probe_tcp(bootstrap_servers: str, timeout_s: float = PROBE_TIMEOUT_S) -> None:
    """Succeed when any ``host:port`` of a Kafka bootstrap list accepts a TCP connection."""
    failures: list[str] = []
    for address in (part.strip() for part in bootstrap_servers.split(",")):
        host, _, port = address.rpartition(":")
        try:
            with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout_s):
                return
        except (OSError, ValueError) as exc:
            failures.append(f"{address}: {exc}")
    raise DependencyNotReadyError("no Kafka bootstrap server is reachable (" + "; ".join(failures) + ")")


def wait_for_dependencies(
    settings: StreamingSettings, wait: WaitFn, policy: RetryPolicy = READINESS_POLICY
) -> None:
    """Block until ClickHouse, MinIO and Kafka answer, or raise once ``policy`` is exhausted."""
    probes: tuple[tuple[str, Callable[[], None]], ...] = (
        ("ClickHouse", lambda: probe_http(settings.clickhouse_ping_url)),
        ("MinIO", lambda: probe_http(settings.s3_health_url)),
        ("Kafka", lambda: probe_tcp(settings.kafka_bootstrap_servers)),
    )
    for name, probe in probes:
        call_with_retry(
            probe,
            policy=policy,
            is_retryable=lambda exc: isinstance(exc, DependencyNotReadyError),
            description=f"readiness probe of {name}",
            wait=wait,
        )
        logger.info("%s is reachable", name)
