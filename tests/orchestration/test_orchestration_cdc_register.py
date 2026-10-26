"""CDC connector registration (infrastructure/debezium/register_connector.py) against a fake Connect API."""

from __future__ import annotations

import importlib.util
import json
import logging
import socket
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "infrastructure" / "debezium" / "register_connector.py"
NAME = "engine-postgres-cdc"
CONFIG = {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "tasks.max": "1",
    "database.password": "${env:POSTGRES_PASSWORD}",
    "heartbeat.interval.ms": "10000",
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("register_connector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    # No __pycache__ next to the script: compose mounts that directory into the container.
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return module


rc = _load_script()
FAST = rc.Timing(
    reachable_seconds=0.5,
    request_retry_seconds=0.5,
    running_seconds=0.6,
    poll_seconds=0.02,
    restart_grace_seconds=0.15,
)


class FakeConnect:
    """In-memory Kafka Connect REST API of one worker with scripted state transitions."""

    def __init__(self) -> None:
        self.configs: dict[str, dict[str, str]] = {}
        self.connector_state: dict[str, str] = {}
        self.task_states: dict[str, list[str]] = {}
        self.requests: list[tuple[str, str]] = []
        self.bodies: list[Any] = []
        self.state_after_start = "RUNNING"  # task state after create, update, restart or resume
        self.rebalancing_posts = 0  # POST /connectors answers 409 this many times first
        self.reject_create = False

    def add(
        self, config: dict[str, str], connector: str = "RUNNING", tasks: tuple[str, ...] = ("RUNNING",)
    ) -> None:
        self.configs[NAME] = {**config, "name": NAME}
        self.connector_state[NAME] = connector
        self.task_states[NAME] = list(tasks)

    def handle(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        self.requests.append((method, path))
        self.bodies.append(body)
        route, _, query = path.partition("?")
        parts = [part for part in route.split("/") if part]
        if route == "/":
            return 200, {"version": "3.6.1", "commit": "fake"}
        if parts == ["connectors"]:
            return self._create(body) if method == "POST" else (200, sorted(self.configs))
        name = parts[1]
        if name not in self.configs and not (method == "PUT" and parts[2:] == ["config"]):
            return 404, {"error_code": 404, "message": f"Connector {name} not found"}
        action = (method, parts[2] if len(parts) > 2 else "")
        if action == ("GET", "config"):
            return 200, self.configs[name]
        if action == ("PUT", "config"):
            existed = name in self.configs
            self.configs[name] = dict(body)
            self._start(name)
            return (200 if existed else 201), {"name": name, "config": body}
        if action == ("GET", "status"):
            return 200, self._status(name)
        if action == ("POST", "restart"):
            assert query == "includeTasks=true&onlyFailed=true"
            if self.connector_state[name] == "FAILED":
                self.connector_state[name] = "RUNNING"
                self.task_states[name] = self.task_states[name] or ["FAILED"]  # tasks start again too
            self.task_states[name] = [
                self.state_after_start if s == "FAILED" else s for s in self.task_states[name]
            ]
            return 202, self._status(name)
        if action == ("PUT", "resume"):
            self.connector_state[name] = "RUNNING"
            self.task_states[name] = [self.state_after_start] * max(1, len(self.task_states[name]))
            return 202, None
        return 405, {"error_code": 405, "message": "unsupported in the fake"}

    def _create(self, body: Any) -> tuple[int, Any]:
        if self.rebalancing_posts:
            self.rebalancing_posts -= 1
            return 409, {
                "error_code": 409,
                "message": "Request cannot be completed because a rebalance is expected",
            }
        if self.reject_create:
            return 400, {"error_code": 400, "message": "Connector configuration is invalid"}
        name = body["name"]
        if name in self.configs:
            return 409, {"error_code": 409, "message": f"Connector {name} already exists"}
        self.configs[name] = {**body["config"], "name": name}
        self._start(name)
        return 201, {"name": name, "config": body["config"], "tasks": []}

    def _start(self, name: str) -> None:
        self.connector_state[name] = "RUNNING"
        self.task_states[name] = [self.state_after_start]

    def _status(self, name: str) -> dict[str, Any]:
        tasks = [
            {
                "id": index,
                "state": state,
                "worker_id": "debezium:8083",
                **({"trace": "boom"} if state == "FAILED" else {}),
            }
            for index, state in enumerate(self.task_states[name])
        ]
        connector = {"state": self.connector_state[name], "worker_id": "debezium:8083"}
        return {"name": name, "connector": connector, "tasks": tasks, "type": "source"}

    def writes(self) -> list[tuple[str, str]]:
        return [request for request in self.requests if request[0] != "GET"]


class _Handler(BaseHTTPRequestHandler):
    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        status, payload = self.server.fake.handle(self.command, self.path, body)  # type: ignore[attr-defined]
        data = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        self._serve()

    def do_PUT(self) -> None:
        self._serve()

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def connect() -> Iterator[tuple[FakeConnect, str]]:
    fake = FakeConnect()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.fake = fake  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "connector.json"
    path.write_text(json.dumps({"name": NAME, "config": CONFIG}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="register-connector")


def _register(url: str, config_file: Path, *extra: str) -> int:
    return rc.main(["--connect-url", url, "--config", str(config_file), *extra], timing=FAST)


def test_creates_a_missing_connector_and_waits_until_it_runs(connect: Any, config_file: Path) -> None:
    fake, url = connect

    assert _register(url, config_file) == 0
    assert fake.writes() == [("POST", "/connectors")]
    assert fake.bodies[fake.requests.index(("POST", "/connectors"))] == {"name": NAME, "config": CONFIG}
    assert fake.configs[NAME] == {**CONFIG, "name": NAME}


def test_waits_until_the_new_task_is_assigned(connect: Any, config_file: Path) -> None:
    fake, url = connect
    fake.state_after_start = "UNASSIGNED"

    def assign_later() -> None:
        fake.task_states[NAME] = ["RUNNING"]

    timer = threading.Timer(0.1, assign_later)
    timer.start()
    try:
        assert _register(url, config_file) == 0
    finally:
        timer.cancel()


def test_replaces_a_configuration_that_differs_from_the_file(
    connect: Any, config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = connect
    fake.add({**CONFIG, "heartbeat.interval.ms": "5000", "database.password": "old-secret"})

    assert _register(url, config_file) == 0
    assert fake.writes() == [("PUT", f"/connectors/{NAME}/config")]
    assert fake.configs[NAME] == {**CONFIG, "name": NAME}
    assert "database.password, heartbeat.interval.ms" in caplog.text
    assert "old-secret" not in caplog.text and "10000" not in caplog.text


def test_leaves_a_matching_running_connector_alone(connect: Any, config_file: Path) -> None:
    fake, url = connect
    fake.add(CONFIG)

    assert _register(url, config_file) == 0
    assert fake.writes() == []


def test_restarts_failed_tasks_once(connect: Any, config_file: Path) -> None:
    fake, url = connect
    fake.add(CONFIG, tasks=("FAILED",))

    assert _register(url, config_file) == 0
    assert fake.writes() == [("POST", f"/connectors/{NAME}/restart?includeTasks=true&onlyFailed=true")]
    assert fake.task_states[NAME] == ["RUNNING"]


def test_restarts_a_failed_connector(connect: Any, config_file: Path) -> None:
    fake, url = connect
    fake.add(CONFIG, connector="FAILED", tasks=())

    assert _register(url, config_file) == 0
    assert ("POST", f"/connectors/{NAME}/restart?includeTasks=true&onlyFailed=true") in fake.writes()


def test_gives_up_with_the_status_document_when_the_restart_fails_again(
    connect: Any, config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = connect
    fake.add(CONFIG, tasks=("FAILED",))
    fake.state_after_start = "FAILED"

    assert _register(url, config_file) == 1
    restarts = [request for request in fake.writes() if "/restart" in request[1]]
    assert len(restarts) == 1  # one restart, no restart loop
    assert "failed again after a restart" in caplog.text
    assert '"trace": "boom"' in caplog.text


@pytest.mark.parametrize(("connector", "tasks"), [("PAUSED", ("PAUSED",)), ("STOPPED", ())])
def test_resumes_a_paused_or_stopped_connector(
    connect: Any, config_file: Path, connector: str, tasks: tuple[str, ...]
) -> None:
    fake, url = connect
    fake.add(CONFIG, connector=connector, tasks=tasks)

    assert _register(url, config_file) == 0
    assert fake.writes() == [("PUT", f"/connectors/{NAME}/resume")]


def test_times_out_when_the_connector_never_runs(
    connect: Any, config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = connect
    fake.add(CONFIG, tasks=("UNASSIGNED",))

    assert _register(url, config_file) == 1
    assert "did not reach RUNNING within 1s (connector RUNNING, tasks [UNASSIGNED])" in caplog.text
    assert '"state": "UNASSIGNED"' in caplog.text


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (lambda fake: None, "plan: create connector"),
        (lambda fake: fake.add({**CONFIG, "tasks.max": "2"}), "keys that differ: tasks.max"),
        (lambda fake: fake.add(CONFIG, tasks=("FAILED",)), "restart the failed instances"),
        (lambda fake: fake.add(CONFIG, connector="PAUSED", tasks=("PAUSED",)), "resume it"),
        (lambda fake: fake.add(CONFIG), "nothing to restart or resume"),
    ],
)
def test_dry_run_only_reads_and_prints_the_plan(
    connect: Any, config_file: Path, caplog: pytest.LogCaptureFixture, setup: Any, expected: str
) -> None:
    fake, url = connect
    setup(fake)

    assert _register(url, config_file, "--dry-run") == 0
    assert fake.writes() == []
    assert expected in caplog.text


def test_retries_while_a_rebalance_rejects_the_registration(connect: Any, config_file: Path) -> None:
    fake, url = connect
    fake.rebalancing_posts = 2

    assert _register(url, config_file) == 0
    assert fake.writes() == [("POST", "/connectors")] * 3


def test_a_rejected_configuration_fails_at_once(
    connect: Any, config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = connect
    fake.reject_create = True

    assert _register(url, config_file) == 1
    assert fake.writes() == [("POST", "/connectors")]
    assert "HTTP 400: Connector configuration is invalid" in caplog.text


def test_unreachable_connect_fails_after_the_wait(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    assert _register(f"http://127.0.0.1:{free_port}", config_file) == 1
    assert "did not answer within" in caplog.text


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        json.dumps({"name": NAME}),
        json.dumps({"name": "", "config": {}}),
        json.dumps({"name": NAME, "config": {"tasks.max": None}}),
        json.dumps({"name": NAME, "config": {"name": "other"}}),
    ],
)
def test_invalid_connector_files_exit_2(tmp_path: Path, document: str) -> None:
    path = tmp_path / "connector.json"
    path.write_text(document, encoding="utf-8")

    assert rc.main(["--connect-url", "http://127.0.0.1:9", "--config", str(path)], timing=FAST) == 2


def test_scalar_values_are_compared_as_connect_stores_them(connect: Any, tmp_path: Path) -> None:
    fake, url = connect
    fake.add({**CONFIG, "tasks.max": "1", "provide.transaction.metadata": "false"})
    path = tmp_path / "connector.json"
    document = {"name": NAME, "config": {**CONFIG, "tasks.max": 1, "provide.transaction.metadata": False}}
    path.write_text(json.dumps(document), encoding="utf-8")

    assert _register(url, path) == 0
    assert fake.writes() == []


def test_environment_supplies_url_and_file_and_flags_win(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONNECT_URL", "http://connect.example:8083")
    monkeypatch.setenv("CONNECTOR_CONFIG", str(config_file))

    from_env = rc.build_parser().parse_args([])
    assert (from_env.connect_url, from_env.config) == ("http://connect.example:8083", config_file)

    flags = rc.build_parser().parse_args(
        ["--connect-url", "http://other:1", "--config", str(tmp_path / "x.json")]
    )
    assert (flags.connect_url, flags.config) == ("http://other:1", tmp_path / "x.json")


def test_defaults_point_at_debezium_and_the_repository_connector_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONNECT_URL", raising=False)
    monkeypatch.delenv("CONNECTOR_CONFIG", raising=False)

    args = rc.build_parser().parse_args([])

    assert args.connect_url == "http://debezium:8083"
    assert args.config == SCRIPT.with_name("register-postgres-connector.json")
    spec = rc.load_connector_spec(args.config)
    assert spec.name == NAME
    assert spec.config["slot.name"] == "dbz_engine_slot"
