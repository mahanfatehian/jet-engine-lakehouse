"""Readiness probes of the job's dependencies, checked per query group by the supervisor.

The lake queries need Kafka and MinIO; ``clickhouse_raw`` needs Kafka and ClickHouse (its
tables are checked by ``clickhouse_sink.ClickHouseTarget``). Starting a query before its
dependencies answer would only fail its first micro-batch, so the supervisor runs these
checks before every (re)start. Each check makes one bounded attempt: retrying with backoff
is the supervisor's job, which keeps an unavailable dependency from delaying the queries
that do not need it.

A group check stops at the first dependency that does not answer: the supervisor only needs to
know that the group must wait. :func:`unavailable_dependencies` instead probes every dependency
and reports each one that is down, for operators (the ``--check-dependencies`` mode of
``spark_kafka_to_lakehouse``).
"""

from __future__ import annotations

import http.client
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence

from streaming_config import StreamingSettings

# Short enough that a supervisor round, which may probe three endpoints, stays well inside
# the heartbeat interval even when every endpoint hangs.
PROBE_TIMEOUT_S = 3.0

# Probes target in-network endpoints: an HTTP(S)_PROXY from the environment must not apply.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

Probe = tuple[str, Callable[[], None]]
"""A dependency's name and its probe: one bounded attempt that raises unless it answers."""


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
    except http.client.HTTPException as exc:  # e.g. a port that speaks another protocol
        raise DependencyNotReadyError(f"{url} did not answer HTTP: {exc!r}") from exc


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


def dependency_probes(settings: StreamingSettings) -> tuple[Probe, Probe, Probe]:
    """The probes of every dependency of the job: Kafka, MinIO and ClickHouse."""
    return (
        ("Kafka", lambda: probe_tcp(settings.kafka_bootstrap_servers)),
        ("MinIO", lambda: probe_http(settings.s3_health_url)),
        ("ClickHouse", lambda: probe_http(settings.clickhouse_ping_url)),
    )


def check_lake_dependencies(settings: StreamingSettings) -> None:
    """Raise :class:`DependencyNotReadyError` unless Kafka and MinIO answer (Path A)."""
    kafka, minio, _ = dependency_probes(settings)
    _check((kafka, minio))


def check_clickhouse_dependencies(settings: StreamingSettings) -> None:
    """Raise :class:`DependencyNotReadyError` unless Kafka and ClickHouse answer (Path B)."""
    kafka, _, clickhouse = dependency_probes(settings)
    _check((kafka, clickhouse))


def _check(probes: Sequence[Probe]) -> None:
    """Run the probes in order; the error names the first dependency that did not answer."""
    for name, probe in probes:
        try:
            probe()
        except DependencyNotReadyError as exc:
            raise DependencyNotReadyError(f"{name}: {exc}") from exc


def unavailable_dependencies(probes: Iterable[Probe]) -> list[str]:
    """Run every probe once, also after a failure; ``"<name>: <reason>"`` per dependency that is down."""
    failures: list[str] = []
    for name, probe in probes:
        try:
            probe()
        except DependencyNotReadyError as exc:
            failures.append(f"{name}: {exc}")
    return failures
