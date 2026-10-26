"""Unit tests of the readiness probes against throwaway local servers."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from streaming_config import StreamingSettings, load_settings
from streaming_readiness import (
    DependencyNotReadyError,
    check_clickhouse_dependencies,
    check_lake_dependencies,
    probe_http,
    probe_tcp,
)

FAST_TIMEOUT_S = 1.0


def _handler_answering(status: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            """Keep the test output clean."""

    return Handler


@pytest.fixture
def http_server() -> Iterator[Callable[[int], str]]:
    servers: list[ThreadingHTTPServer] = []

    def start(status: int) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_answering(status))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def tcp_listener() -> Iterator[int]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]


def _unused_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize("status", [200, 403, 404])
def test_http_probe_accepts_any_non_server_error(http_server: Callable[[int], str], status: int) -> None:
    probe_http(f"{http_server(status)}/ping", timeout_s=FAST_TIMEOUT_S)


def test_http_probe_rejects_server_errors(http_server: Callable[[int], str]) -> None:
    with pytest.raises(DependencyNotReadyError, match="HTTP 503"):
        probe_http(f"{http_server(503)}/ping", timeout_s=FAST_TIMEOUT_S)


def test_http_probe_rejects_unreachable_endpoint() -> None:
    with pytest.raises(DependencyNotReadyError, match="unreachable"):
        probe_http(f"http://127.0.0.1:{_unused_port()}/ping", timeout_s=FAST_TIMEOUT_S)


def test_tcp_probe_accepts_any_reachable_bootstrap_server(tcp_listener: int) -> None:
    probe_tcp(f"127.0.0.1:{_unused_port()}, 127.0.0.1:{tcp_listener}", timeout_s=FAST_TIMEOUT_S)


@pytest.mark.parametrize("servers", ["127.0.0.1:{port}", "no-port-here", "127.0.0.1:not-a-port"])
def test_tcp_probe_rejects_unreachable_or_malformed_servers(servers: str) -> None:
    with pytest.raises(DependencyNotReadyError, match="no Kafka bootstrap server"):
        probe_tcp(servers.format(port=_unused_port()), timeout_s=FAST_TIMEOUT_S)


def _settings(kafka_port: int, minio_url: str, clickhouse_port: int | str) -> StreamingSettings:
    return load_settings(
        {
            "KAFKA_BOOTSTRAP_SERVERS": f"127.0.0.1:{kafka_port}",
            "S3_ENDPOINT_URL": minio_url,
            "CLICKHOUSE_HOST": "127.0.0.1",
            "CLICKHOUSE_PORT": str(clickhouse_port),
        }
    )


def test_lake_needs_kafka_and_minio_but_not_clickhouse(
    http_server: Callable[[int], str], tcp_listener: int
) -> None:
    check_lake_dependencies(_settings(tcp_listener, http_server(200), _unused_port()))


def test_lake_check_names_the_missing_dependency(tcp_listener: int) -> None:
    settings = _settings(tcp_listener, f"http://127.0.0.1:{_unused_port()}", _unused_port())
    with pytest.raises(
        DependencyNotReadyError, match=r"^MinIO: http://127\.0\.0\.1:\d+/minio/health/live is"
    ):
        check_lake_dependencies(settings)


def test_clickhouse_needs_kafka_and_clickhouse_but_not_minio(
    http_server: Callable[[int], str], tcp_listener: int
) -> None:
    clickhouse_port = http_server(200).rsplit(":", 1)[1]
    check_clickhouse_dependencies(
        _settings(tcp_listener, f"http://127.0.0.1:{_unused_port()}", clickhouse_port)
    )


def test_clickhouse_check_names_the_missing_dependency(http_server: Callable[[int], str]) -> None:
    clickhouse_port = http_server(200).rsplit(":", 1)[1]
    settings = _settings(_unused_port(), http_server(200), clickhouse_port)
    with pytest.raises(DependencyNotReadyError, match=r"^Kafka: no Kafka bootstrap server is reachable"):
        check_clickhouse_dependencies(settings)
