"""Readiness probes of the job's dependencies, checked per query group by the supervisor.

The lake queries need Kafka and MinIO; ``clickhouse_raw`` needs Kafka and ClickHouse (its
tables are checked by ``clickhouse_sink.ClickHouseTarget``). Starting a query before its
dependencies answer would only fail its first micro-batch, so the supervisor runs these
checks before every (re)start. Each check makes one bounded attempt: retrying with backoff
is the supervisor's job, which keeps an unavailable dependency from delaying the queries
that do not need it.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from streaming_config import StreamingSettings

# Short enough that a supervisor round, which may probe three endpoints, stays well inside
# the heartbeat interval even when every endpoint hangs.
PROBE_TIMEOUT_S = 3.0

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


def check_lake_dependencies(settings: StreamingSettings) -> None:
    """Raise :class:`DependencyNotReadyError` unless Kafka and MinIO answer (Path A)."""
    _check(
        (
            ("Kafka", lambda: probe_tcp(settings.kafka_bootstrap_servers)),
            ("MinIO", lambda: probe_http(settings.s3_health_url)),
        )
    )


def check_clickhouse_dependencies(settings: StreamingSettings) -> None:
    """Raise :class:`DependencyNotReadyError` unless Kafka and ClickHouse answer (Path B)."""
    _check(
        (
            ("Kafka", lambda: probe_tcp(settings.kafka_bootstrap_servers)),
            ("ClickHouse", lambda: probe_http(settings.clickhouse_ping_url)),
        )
    )


def _check(probes: Sequence[tuple[str, Callable[[], None]]]) -> None:
    """Run the probes in order; the error names the first dependency that did not answer."""
    for name, probe in probes:
        try:
            probe()
        except DependencyNotReadyError as exc:
            raise DependencyNotReadyError(f"{name}: {exc}") from exc


def check_all_dependencies(settings: StreamingSettings) -> None:
    """Raise :class:`DependencyNotReadyError` unless Kafka, MinIO and ClickHouse all answer.

    Unlike ``check_lake_dependencies`` / ``check_clickhouse_dependencies`` this probes
    every service and collects *all* failures into a single error so operators see every
    unavailable dependency at once.
    """
    failures: list[str] = []
    probes: Sequence[tuple[str, Callable[[], None]]] = (
        ("Kafka", lambda: probe_tcp(settings.kafka_bootstrap_servers)),
        ("MinIO", lambda: probe_http(settings.s3_health_url)),
        ("ClickHouse", lambda: probe_http(settings.clickhouse_ping_url)),
    )
    for name, probe in probes:
        try:
            probe()
        except DependencyNotReadyError as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise DependencyNotReadyError("; ".join(failures))
