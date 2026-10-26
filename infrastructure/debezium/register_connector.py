#!/usr/bin/env python3
"""Register the Debezium Postgres connector, apply configuration changes and keep it running.

Run by the ``cdc-register`` compose service on every ``make up`` and by ``make register-cdc``.
Standard library only: it runs in a plain ``python:3.12-slim-bookworm`` container.

1. Wait until the Kafka Connect REST API answers.
2. Create the connector when it does not exist (``POST /connectors``). When it exists and the
   stored configuration differs from the file, replace it (``PUT /connectors/<name>/config``).
3. Make sure it runs: restart a FAILED connector or task once
   (``POST /connectors/<name>/restart?includeTasks=true&onlyFailed=true``), resume a PAUSED or
   STOPPED one (``PUT /connectors/<name>/resume``), then wait until the connector and every task
   are RUNNING. Otherwise exit 1 and print the status document, including the failure trace.

``--dry-run`` sends GET requests only and prints the plan. Settings: ``--connect-url`` (env
CONNECT_URL, default http://debezium:8083) and ``--config`` (env CONNECTOR_CONFIG, default the
register-postgres-connector.json next to this script). Exit codes: 0 running, 1 failed or
timed out, 2 invalid connector file.

Adding a table to ``table.include.list`` does not snapshot that table's existing rows:
``snapshot.mode=initial`` only applies before offsets exist; use an incremental snapshot signal.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("register-connector")

DEFAULT_CONNECT_URL = "http://debezium:8083"
DEFAULT_CONFIG_FILE = Path(__file__).resolve().with_name("register-postgres-connector.json")
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_CONFIG = 2
REQUEST_TIMEOUT_SECONDS = 10.0
# 409: Connect rejects requests while its workers rebalance; 5xx: worker busy or restarting.
RETRYABLE_STATUSES = frozenset({409, 500, 502, 503, 504})
ACCEPTED = (200, 202, 204)


class ConnectUnavailableError(RuntimeError):
    """The Kafka Connect REST API was unreachable or kept answering with retryable errors."""


class ConnectRequestError(RuntimeError):
    """Kafka Connect rejected a request with a non-retryable status."""


class ConnectorConfigError(ValueError):
    """The connector file is not a valid ``{"name": ..., "config": {...}}`` document."""


@dataclass(frozen=True)
class Timing:
    """Bounds of every wait, in seconds."""

    reachable_seconds: float = 120.0
    request_retry_seconds: float = 60.0
    running_seconds: float = 90.0
    poll_seconds: float = 2.0
    restart_grace_seconds: float = 15.0


@dataclass(frozen=True)
class ConnectorSpec:
    """Connector name and configuration, with values as Connect stores them (strings)."""

    name: str
    config: dict[str, str]

    @property
    def stored_config(self) -> dict[str, str]:
        """The configuration as ``GET /connectors/<name>/config`` returns it (``name`` included)."""
        return {**self.config, "name": self.name}


@dataclass(frozen=True)
class Response:
    """Status code and decoded JSON body (raw text when the body is not JSON)."""

    status: int
    body: Any


@dataclass(frozen=True)
class ConnectorState:
    """States reported by ``GET /connectors/<name>/status``."""

    connector: str | None
    tasks: tuple[str, ...]

    @classmethod
    def from_status(cls, status: Any) -> ConnectorState:
        """Parse a status document; anything unexpected yields an unknown state."""
        if not isinstance(status, dict):
            return cls(None, ())
        connector = status.get("connector")
        tasks = status.get("tasks")
        return cls(
            connector.get("state") if isinstance(connector, dict) else None,
            tuple(str(task.get("state")) for task in tasks if isinstance(task, dict))
            if isinstance(tasks, list)
            else (),
        )

    @property
    def running(self) -> bool:
        """The connector and at least one task run, and no task is in another state."""
        return self.connector == "RUNNING" and bool(self.tasks) and set(self.tasks) == {"RUNNING"}

    @property
    def failed(self) -> bool:
        """The connector or a task is FAILED."""
        return "FAILED" in (self.connector, *self.tasks)

    @property
    def paused(self) -> bool:
        """The connector was paused or stopped through the REST API."""
        return self.connector in ("PAUSED", "STOPPED")

    def describe(self) -> str:
        """Human-readable summary, e.g. ``connector RUNNING, tasks [FAILED]``."""
        return f"connector {self.connector or 'UNKNOWN'}, tasks [{', '.join(self.tasks) or 'none'}]"


class ConnectClient:
    """Minimal JSON client of the Kafka Connect REST API (never uses an HTTP proxy)."""

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Service names such as `debezium` only resolve inside the compose network.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str, payload: object | None = None) -> Response:
        """Send one request; HTTP error statuses are returned, network failures raised."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return Response(response.status, _decode(response.read()))
        except urllib.error.HTTPError as exc:
            with exc:
                return Response(exc.code, _decode(exc.read()))
        except (OSError, http.client.HTTPException) as exc:
            raise ConnectUnavailableError(f"{method} {self.base_url}{path}: {exc}") from exc


class Registrar:
    """Brings one connector to the state described by its file."""

    def __init__(
        self,
        client: ConnectClient,
        spec: ConnectorSpec,
        *,
        source: Path,
        timing: Timing | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._spec = spec
        self._source = source
        self._timing = timing or Timing()
        self._clock = clock
        self._sleep = sleep
        self._path = f"/connectors/{urllib.parse.quote(spec.name, safe='')}"

    def run(self, *, dry_run: bool) -> int:
        """Register or update the connector, then ensure it runs; returns the exit code."""
        version = self.wait_until_reachable()
        logger.info("Kafka Connect %s at %s", version, self._client.base_url)
        if dry_run:
            logger.info("dry run: only GET requests are sent")
            return self._plan()
        self._upsert()
        return self._ensure_running()

    def wait_until_reachable(self) -> str:
        """Wait until ``GET /connectors`` answers 200; returns the Connect version."""
        deadline = self._clock() + self._timing.reachable_seconds
        while True:
            try:
                response = self._client.request("GET", "/connectors")
                if response.status == 200:
                    root = self._client.request("GET", "/").body
                    return str(root.get("version", "unknown")) if isinstance(root, dict) else "unknown"
                failure = f"HTTP {response.status}"
            except ConnectUnavailableError as exc:
                failure = str(exc)
            if self._clock() >= deadline:
                raise ConnectUnavailableError(
                    f"Kafka Connect at {self._client.base_url} did not answer within "
                    f"{self._timing.reachable_seconds:.0f}s: {failure}"
                )
            logger.info("waiting for Kafka Connect at %s (%s)", self._client.base_url, failure)
            self._sleep(self._timing.poll_seconds)

    def _plan(self) -> int:
        current = self._current_config()
        if current is None:
            logger.info(
                "plan: create connector '%s' from %s (POST /connectors)", self._spec.name, self._source
            )
            logger.info("plan: then wait until the connector and its tasks are RUNNING")
            return EXIT_OK
        changes = config_changes(current, self._spec.stored_config)
        if changes:
            logger.info(
                "plan: replace the configuration (PUT %s/config); keys that differ: %s",
                self._path,
                ", ".join(changes),
            )
        else:
            logger.info("plan: configuration matches %s: no change", self._source)
        status = self._status()
        state = ConnectorState.from_status(status)
        if state.running:
            action = "nothing to restart or resume"
        elif state.failed:
            action = (
                f"restart the failed instances (POST {self._path}/restart?includeTasks=true&onlyFailed=true)"
            )
        else:
            action = f"resume it (PUT {self._path}/resume)" if state.paused else "wait until it is RUNNING"
        logger.info("plan: %s: %s", state.describe(), action)
        if state.failed:
            logger.info("current status:\n%s", _pretty(status))
        return EXIT_OK

    def _upsert(self) -> None:
        deadline = self._clock() + self._timing.request_retry_seconds
        while True:
            current = self._current_config()
            if current is not None:
                self._update(current)
                return
            payload = {"name": self._spec.name, "config": self._spec.config}
            if self._call("POST", "/connectors", payload, expect=(201, 409)).status == 201:
                logger.info("connector '%s' created from %s", self._spec.name, self._source)
                return
            # 409: created concurrently, or the workers are rebalancing: look again.
            if self._clock() >= deadline:
                raise ConnectUnavailableError(
                    f"POST /connectors kept answering HTTP 409 for '{self._spec.name}'"
                )
            self._sleep(self._timing.poll_seconds)

    def _update(self, current: dict[str, str]) -> None:
        changes = config_changes(current, self._spec.stored_config)
        if not changes:
            logger.info("connector '%s' matches %s: no change", self._spec.name, self._source)
            return
        # Key names only: values can hold credentials.
        logger.info("connector '%s': replacing keys that differ: %s", self._spec.name, ", ".join(changes))
        self._call("PUT", f"{self._path}/config", self._spec.stored_config, expect=(200, 201))
        logger.info("connector '%s' configuration replaced", self._spec.name)

    def _ensure_running(self) -> int:
        deadline = self._clock() + self._timing.running_seconds
        restarted_at: float | None = None
        resumed = False
        while True:
            status = self._status()
            state = ConnectorState.from_status(status)
            if state.running:
                logger.info("connector '%s' is healthy: %s", self._spec.name, state.describe())
                return EXIT_OK
            now = self._clock()
            if state.failed and restarted_at is None:
                logger.warning(
                    "connector '%s': %s: restarting the failed instances", self._spec.name, state.describe()
                )
                self._call("POST", f"{self._path}/restart?includeTasks=true&onlyFailed=true", expect=ACCEPTED)
                restarted_at = now
            elif state.failed and now - restarted_at >= self._timing.restart_grace_seconds:
                return self._give_up(f"failed again after a restart ({state.describe()})", status)
            elif state.paused and not resumed:
                logger.warning("connector '%s' is %s: resuming it", self._spec.name, state.connector)
                self._call("PUT", f"{self._path}/resume", expect=ACCEPTED)
                resumed = True
            if now >= deadline:
                return self._give_up(
                    f"did not reach RUNNING within {self._timing.running_seconds:.0f}s ({state.describe()})",
                    status,
                )
            self._sleep(self._timing.poll_seconds)

    def _give_up(self, reason: str, status: Any) -> int:
        logger.error("connector '%s' %s; status document:\n%s", self._spec.name, reason, _pretty(status))
        return EXIT_FAILED

    def _current_config(self) -> dict[str, str] | None:
        response = self._call("GET", f"{self._path}/config", expect=(200, 404))
        if response.status == 404:
            return None
        if not isinstance(response.body, dict):
            raise ConnectRequestError(f"GET {self._path}/config returned no configuration: {response.body!r}")
        return {str(key): str(value) for key, value in response.body.items()}

    def _status(self) -> Any:
        # 404 right after creation: the worker has not published a status yet.
        response = self._call("GET", f"{self._path}/status", expect=(200, 404))
        return response.body if response.status == 200 else None

    def _call(
        self, method: str, path: str, payload: object | None = None, *, expect: Sequence[int]
    ) -> Response:
        """Send a request, retrying network failures and retryable statuses not in ``expect``."""
        deadline = self._clock() + self._timing.request_retry_seconds
        while True:
            try:
                response = self._client.request(method, path, payload)
            except ConnectUnavailableError as exc:
                failure = str(exc)
            else:
                if response.status in expect:
                    return response
                failure = f"HTTP {response.status}: {_message(response.body)}"
                if response.status not in RETRYABLE_STATUSES:
                    raise ConnectRequestError(f"{method} {path} failed with {failure}")
            if self._clock() >= deadline:
                raise ConnectUnavailableError(f"{method} {path} kept failing: {failure}")
            logger.info("%s %s: %s; retrying", method, path, failure)
            self._sleep(self._timing.poll_seconds)


def load_connector_spec(path: Path) -> ConnectorSpec:
    """Read a ``{"name": ..., "config": {...}}`` file; scalar values become Connect strings."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConnectorConfigError(f"not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("config"), dict):
        raise ConnectorConfigError('expected {"name": "<connector>", "config": {...}}')
    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConnectorConfigError('"name" must be a non-empty string')
    config = {str(key): _connect_value(str(key), value) for key, value in document["config"].items()}
    if config.get("name", name) != name:
        raise ConnectorConfigError(f"config.name {config['name']!r} differs from name {name!r}")
    config.pop("name", None)
    return ConnectorSpec(name, config)


def config_changes(current: dict[str, str], desired: dict[str, str]) -> list[str]:
    """Sorted keys whose values differ, or that exist on one side only."""
    return sorted(key for key in current.keys() | desired.keys() if current.get(key) != desired.get(key))


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface; defaults come from CONNECT_URL and CONNECTOR_CONFIG."""
    parser = argparse.ArgumentParser(
        prog="register_connector.py",
        description="Register or update the Debezium connector and make sure it runs.",
    )
    parser.add_argument(
        "--connect-url",
        default=os.environ.get("CONNECT_URL", "").strip() or DEFAULT_CONNECT_URL,
        help="Kafka Connect REST URL (env CONNECT_URL, default %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("CONNECTOR_CONFIG", "").strip() or DEFAULT_CONFIG_FILE),
        help="connector JSON file (env CONNECTOR_CONFIG, default %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="send GET requests only and print the plan")
    return parser


def configure_logging() -> None:
    """Log to stdout with UTC timestamps (no-op when logging is already configured)."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main(argv: Sequence[str] | None = None, *, timing: Timing | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        spec = load_connector_spec(args.config)
    except (OSError, ConnectorConfigError) as exc:
        logger.error("invalid connector file %s: %s", args.config, exc)
        return EXIT_BAD_CONFIG
    registrar = Registrar(ConnectClient(args.connect_url), spec, source=args.config, timing=timing)
    try:
        return registrar.run(dry_run=args.dry_run)
    except (ConnectUnavailableError, ConnectRequestError) as exc:
        logger.error("%s", exc)
        return EXIT_FAILED


def _connect_value(key: str, value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return json.dumps(value)  # true/false and numbers, as Connect's JSON coercion stores them
    raise ConnectorConfigError(f"config value of {key!r} must be a string, number or boolean")


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        return text


def _message(body: Any) -> str:
    if isinstance(body, dict) and "message" in body:
        return str(body["message"])
    return str(body)[:500]


def _pretty(document: Any) -> str:
    return json.dumps(document, indent=2) if document is not None else "<no status reported>"


if __name__ == "__main__":
    sys.exit(main())
