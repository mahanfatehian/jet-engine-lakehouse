"""Unit tests of the readiness probes against throwaway local servers."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from streaming_config import load_settings
from streaming_readiness import DependencyNotReadyError, probe_http, probe_tcp, wait_for_dependencies
from streaming_retry import RetryPolicy

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


def test_wait_for_dependencies_passes_when_everything_answers(
    http_server: Callable[[int], str], tcp_listener: int
) -> None:
    url = http_server(200)
    port = url.rsplit(":", 1)[1]
    settings = load_settings(
        {
            "CLICKHOUSE_HOST": "127.0.0.1",
            "CLICKHOUSE_PORT": port,
            "S3_ENDPOINT_URL": url,
            "KAFKA_BOOTSTRAP_SERVERS": f"127.0.0.1:{tcp_listener}",
        }
    )
    waits: list[float] = []
    wait_for_dependencies(settings, lambda seconds: waits.append(seconds) or False)
    assert waits == []


def test_wait_for_dependencies_gives_up_after_the_policy() -> None:
    settings = load_settings({"CLICKHOUSE_HOST": "127.0.0.1", "CLICKHOUSE_PORT": str(_unused_port())})
    waits: list[float] = []
    policy = RetryPolicy(attempts=3, base_delay_s=0.1, max_delay_s=0.2)
    with pytest.raises(DependencyNotReadyError):
        wait_for_dependencies(settings, lambda seconds: waits.append(seconds) or False, policy)
    assert waits == [0.1, 0.2]
