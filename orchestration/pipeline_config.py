"""Settings of the pipeline runner, read from environment variables (contract section 2).

Paths default to locations relative to this file, so the runner works unchanged inside
the data-platform image (repository root = /app) and from a checkout on a host. Two
runner-only variables complement the contract: PIPELINE_LOCK_DIR (where the cross-process
task locks live) and PIPELINE_GATE_ON_DBT_TESTS (whether failed dbt tests block training
and lake snapshots).
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Compose always sets CLICKHOUSE_HOST=clickhouse; the fallback targets the published host
# port so the runner also works from a host shell with no environment configured.
DEFAULT_CLICKHOUSE_HOST = "localhost"
DEFAULT_CLICKHOUSE_PORT = 8123

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """An environment variable holds a value the runner cannot use."""


@dataclass(frozen=True)
class ScheduleIntervals:
    """Loop cadences in seconds (DBT/PREDICT/SNAPSHOT/TRAIN_INTERVAL_SECONDS)."""

    dbt_seconds: float = 120.0
    predict_seconds: float = 60.0
    snapshot_seconds: float = 900.0
    train_seconds: float = 1800.0


def default_lock_dir() -> Path:
    """Lock directory when PIPELINE_LOCK_DIR is unset: the temp dir (``/tmp`` in the container)."""
    return Path(tempfile.gettempdir())


@dataclass(frozen=True)
class RunnerSettings:
    """Everything the runner needs to build task commands and to probe ClickHouse."""

    repo_root: Path
    dbt_project_dir: Path
    dbt_profiles_dir: Path
    clickhouse_host: str
    clickhouse_port: int
    intervals: ScheduleIntervals
    python_executable: str
    dbt_executable: str
    lock_dir: Path = field(default_factory=default_lock_dir)
    gate_on_dbt_tests: bool = True

    @property
    def clickhouse_ping_url(self) -> str:
        """HTTP liveness endpoint of ClickHouse (answers without authentication)."""
        return f"http://{self.clickhouse_host}:{self.clickhouse_port}/ping"


def load_settings(
    env: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
) -> RunnerSettings:
    """Build the runner settings from ``env``, applying the contract defaults.

    Raises :class:`ConfigError` naming the offending variable when a value is invalid.
    """
    defaults = ScheduleIntervals()
    intervals = ScheduleIntervals(
        dbt_seconds=_positive_seconds(env, "DBT_INTERVAL_SECONDS", defaults.dbt_seconds),
        predict_seconds=_positive_seconds(env, "PREDICT_INTERVAL_SECONDS", defaults.predict_seconds),
        snapshot_seconds=_positive_seconds(env, "SNAPSHOT_INTERVAL_SECONDS", defaults.snapshot_seconds),
        train_seconds=_positive_seconds(env, "TRAIN_INTERVAL_SECONDS", defaults.train_seconds),
    )
    analytics_dir = repo_root / "analytics"
    return RunnerSettings(
        repo_root=repo_root,
        dbt_project_dir=_directory(env, "DBT_PROJECT_DIR", repo_root, analytics_dir),
        dbt_profiles_dir=_directory(env, "DBT_PROFILES_DIR", repo_root, analytics_dir),
        clickhouse_host=_value(env, "CLICKHOUSE_HOST") or DEFAULT_CLICKHOUSE_HOST,
        clickhouse_port=_port(env, "CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_PORT),
        intervals=intervals,
        python_executable=python_executable,
        dbt_executable=resolve_dbt_executable(python_executable),
        lock_dir=_directory(env, "PIPELINE_LOCK_DIR", repo_root, default_lock_dir()),
        gate_on_dbt_tests=_flag(env, "PIPELINE_GATE_ON_DBT_TESTS", default=True),
    )


def resolve_dbt_executable(python_executable: str) -> str:
    """Locate the dbt CLI, preferring the one installed beside the running interpreter.

    Looking next to the interpreter first finds dbt inside a virtualenv that was never
    activated. The bare name is the last resort, so a missing installation surfaces as a
    clear task failure instead of a crash at startup.
    """
    beside_interpreter = shutil.which("dbt", path=str(Path(python_executable).parent))
    return beside_interpreter or shutil.which("dbt") or "dbt"


def _value(env: Mapping[str, str], name: str) -> str | None:
    # Compose passes unset `${VAR:-}` interpolations as empty strings: treat them as unset.
    value = env.get(name, "").strip()
    return value or None


def _positive_seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _value(env, name)
    if raw is None:
        return default
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ConfigError(f"{name} must be a positive, finite number of seconds, got {raw!r}")
    return seconds


def _port(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _value(env, name)
    if raw is None:
        return default
    try:
        port = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a TCP port number, got {raw!r}") from exc
    if not 0 < port < 65536:
        raise ConfigError(f"{name} must be between 1 and 65535, got {raw!r}")
    return port


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = _value(env, name)
    if raw is None:
        return default
    if raw.lower() in _TRUE_VALUES:
        return True
    if raw.lower() in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true or false (also 1/0, yes/no, on/off), got {raw!r}")


def _directory(env: Mapping[str, str], name: str, repo_root: Path, default: Path) -> Path:
    raw = _value(env, name)
    if raw is None:
        return default
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path
