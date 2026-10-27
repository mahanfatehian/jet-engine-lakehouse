"""Simulator configuration: the SIM_* variables, the generation settings recorded per lifecycle and the
command line.

A leaf module (standard library only), so every other simulator module can import it at runtime.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_ENGINE_IDS = ("JT9D-001", "JT9D-002", "CFM56-104", "CFM56-105", "GE90-201", "GE90-202")
MAX_ENGINE_ID_LENGTH = 32  # engine_status_registry.engine_id is VARCHAR(32)
SEED_RANGE = (-(2**63), 2**63 - 1)  # simulator_lifecycle_params.seed is a BIGINT


def env_value(env: Mapping[str, str], name: str, cast: Callable[[str], Any], default: Any) -> Any:
    """``cast`` of the variable ``name``: ``default`` when unset or blank, ValueError naming it if invalid."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc


def _exact(value: float) -> str:
    """``value`` without precision loss: 180.0 -> '180', 180.25 -> '180.25', 2**62 -> all digits."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


@dataclass(frozen=True)
class GenerationSettings:
    """The SIM_* values that shape a lifecycle's trajectory (wear path, failure mode, cycle length).

    They are stored per lifecycle in simulator_lifecycle_params, so a restart re-derives an
    in-progress lifecycle exactly even after the variables changed; new values apply from the
    engine's next lifecycle.
    """

    seed: int
    mean_life_cycles: float
    pulses_per_cycle: int

    def describe(self) -> str:
        """The values as SIM_* assignments, for log and error messages."""
        return ", ".join(f"{name}={_exact(value)}" for name, value in self._named_values())

    def changes_to(self, other: GenerationSettings) -> str:
        """``SIM_X old -> new`` for every value that differs in ``other`` (empty when none does)."""
        pairs = zip(self._named_values(), other._named_values(), strict=True)
        return ", ".join(
            f"{name} {_exact(old)} -> {_exact(new)}" for (name, old), (_, new) in pairs if old != new
        )

    def _named_values(self) -> tuple[tuple[str, float], ...]:
        return (
            ("SIM_SEED", self.seed),
            ("SIM_MEAN_LIFE_CYCLES", self.mean_life_cycles),
            ("SIM_PULSES_PER_CYCLE", self.pulses_per_cycle),
        )


@dataclass(frozen=True)
class SimulatorConfig:
    """Simulator settings; the defaults are the contract defaults of the SIM_* variables."""

    engine_ids: tuple[str, ...] = DEFAULT_ENGINE_IDS
    tick_seconds: float = 1.0
    pulses_per_cycle: int = 10
    mean_life_cycles: float = 180.0
    backfill_lifecycles: int = 4
    maintenance_seconds: float = 20.0
    seed: int = 42
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        checks = (
            (bool(self.engine_ids), "SIM_ENGINE_IDS lists no engine"),
            (len(set(self.engine_ids)) == len(self.engine_ids), "SIM_ENGINE_IDS contains duplicates"),
            (
                all(len(engine_id) <= MAX_ENGINE_ID_LENGTH for engine_id in self.engine_ids),
                f"engine ids are limited to {MAX_ENGINE_ID_LENGTH} characters",
            ),
            (0 < self.tick_seconds < math.inf, "SIM_TICK_SECONDS must be > 0"),
            (self.pulses_per_cycle >= 1, "SIM_PULSES_PER_CYCLE must be >= 1"),
            (2 <= self.mean_life_cycles < math.inf, "SIM_MEAN_LIFE_CYCLES must be >= 2"),
            (self.backfill_lifecycles >= 0, "SIM_BACKFILL_LIFECYCLES must be >= 0"),
            (0 <= self.maintenance_seconds < math.inf, "SIM_MAINTENANCE_SECONDS must be >= 0"),
            (SEED_RANGE[0] <= self.seed <= SEED_RANGE[1], "SIM_SEED must fit in a signed 64-bit integer"),
        )
        problems = [message for valid, message in checks if not valid]
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SimulatorConfig:
        """Read SIM_* and LOG_LEVEL; unset or blank variables keep the contract defaults."""
        env = os.environ if env is None else env
        raw_ids = env.get("SIM_ENGINE_IDS", "").strip()
        engine_ids = tuple(part.strip() for part in raw_ids.split(",") if part.strip())
        return cls(
            engine_ids=engine_ids if raw_ids else DEFAULT_ENGINE_IDS,
            tick_seconds=env_value(env, "SIM_TICK_SECONDS", float, cls.tick_seconds),
            pulses_per_cycle=env_value(env, "SIM_PULSES_PER_CYCLE", int, cls.pulses_per_cycle),
            mean_life_cycles=env_value(env, "SIM_MEAN_LIFE_CYCLES", float, cls.mean_life_cycles),
            backfill_lifecycles=env_value(env, "SIM_BACKFILL_LIFECYCLES", int, cls.backfill_lifecycles),
            maintenance_seconds=env_value(env, "SIM_MAINTENANCE_SECONDS", float, cls.maintenance_seconds),
            seed=env_value(env, "SIM_SEED", int, cls.seed),
            log_level=env.get("LOG_LEVEL", "").strip() or cls.log_level,
        )

    @property
    def generation(self) -> GenerationSettings:
        """Settings that every lifecycle starting from now on is generated with."""
        return GenerationSettings(self.seed, self.mean_life_cycles, self.pulses_per_cycle)

    def summary(self) -> str:
        """One-line description for the start-up log."""
        return (
            f"{len(self.engine_ids)} engines ({', '.join(self.engine_ids)}), tick {self.tick_seconds:g} s, "
            f"{self.pulses_per_cycle} pulses/cycle, mean life {self.mean_life_cycles:g} cycles, "
            f"repair {self.maintenance_seconds:g} s, backfill {self.backfill_lifecycles} lifecycles, "
            f"seed {self.seed}"
        )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not 0 < value < math.inf:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


BENCHMARK_DEFAULTS = {"benchmark_seconds": 60.0, "benchmark_engines": 24}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line: ``[--max-ticks N] [--no-backfill]`` for the fleet, or ``--benchmark [--benchmark-*]`` for
    the HF load test (frames only, no PostgreSQL); the --benchmark-* options require --benchmark."""
    parser = argparse.ArgumentParser(description="Jet-engine fleet telemetry simulator (PostgreSQL edge DB).")
    parser.add_argument("--max-ticks", type=_positive_int, help="stop after N live ticks (smoke tests)")
    parser.add_argument("--no-backfill", action="store_true", help="never backfill engines without telemetry")
    bench = parser.add_argument_group("HF load test (no PostgreSQL, no pacing)")
    bench.add_argument("--benchmark", action="store_true", help="measure the HF frame producer and exit")
    bench.add_argument("--benchmark-seconds", type=_positive_float, help="measured duration (default 60)")
    bench.add_argument(
        "--benchmark-engines", type=_positive_int, help="synthetic engines BENCH-0001.. (default 24)"
    )
    bench.add_argument("--benchmark-topic", help="target topic (default HF_BENCHMARK_TOPIC)")
    bench.add_argument(
        "--benchmark-sample-rate", type=_positive_int, help="vibration Hz (default SIM_HF_SAMPLE_RATE_HZ)"
    )
    bench.add_argument(
        "--benchmark-vib-channels", type=_positive_int, help="accelerometers (default SIM_HF_VIB_CHANNELS)"
    )
    args = parser.parse_args(argv)
    given = [name for name in vars(args) if name.startswith("benchmark_") and getattr(args, name) is not None]
    if given and not args.benchmark:
        parser.error(f"--{given[0].replace('_', '-')} requires --benchmark")
    if args.benchmark and (args.max_ticks is not None or args.no_backfill):
        parser.error("--benchmark cannot be combined with --max-ticks or --no-backfill")
    for name, default in BENCHMARK_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args
