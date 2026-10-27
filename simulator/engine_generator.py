"""Edge telemetry simulator: a jet-engine fleet that wears out, fails and is repaired, written to PostgreSQL.

Every SIM_TICK_SECONDS each engine emits one *pulse* (rpm, core temperature, EGT, vibration, fuel flow,
oil pressure) into ``engine_telemetry``. SIM_PULSES_PER_CYCLE pulses form one operating cycle (a flight)
and ``cycle_number`` counts cycles since the last maintenance. Debezium turns the INSERTs and UPDATEs into
CDC events, so the Data Science layer learns Remaining Useful Life against a known ground truth.

Degradation model: one *lifecycle* runs from overhaul (w = 0) to failure (w = 1)
----------------------------------------------------------------------------------------------------------
Lifecycle parameters come from a stable hash of (SIM_SEED, engine_id, lifecycle_id). The SIM_* values a
lifecycle starts with are stored with it, so a restart re-derives exactly the same curve even after they
changed (new values apply from the next lifecycle):
    expected life  L ~ Normal(mu, 0.18 mu) clipped to [0.55 mu, 1.6 mu]          mu = SIM_MEAN_LIFE_CYCLES
    wear curve     c(t) = (t / L)^k                            k ~ U(1.8, 2.8)   polynomial   (p = 0.5)
                   c(t) = (exp(b t / L) - 1) / (exp(b) - 1)    b ~ U(2.5, 4.0)   exponential  (p = 0.5)
    per pulse      dw_i = [c(t_i) - c(t_i-1)] * s_i,   s_i ~ LogNormal(-sigma^2 / 2, sigma),   sigma = 0.25
    wear           w_i = min(1, dw_1 + ... + dw_i),    t_i = i / SIM_PULSES_PER_CYCLE  (fractional cycles)
E[s_i] = 1, so w follows c(t) on average while the shocks make the failure time stochastic; w is monotone
and stays within [0, 1]. Both curve families are convex: damage accrues slowly in a healthy engine and
accelerates towards end of life (crack growth, coating loss), the shape that makes RUL learnable. The
first pulse with w = 1 is the FAILURE event; after SIM_MAINTENANCE_SECONDS of repair the engine restarts
at w = 0 with freshly drawn lifecycle parameters.

Status: HEALTHY w < 0.5 <= WARNING < 0.8 <= CRITICAL < 1.0 = FAILURE.

Each lifecycle draws one failure mode. Its dominant channels get sensitivity m = 1.5 and every other
channel m = 0.8, so all channels keep drifting and RUL stays learnable whatever the mode:
    HPT_BLADE_EROSION  p = 0.35  EGT, core temperature    BEARING_DEGRADATION  p = 0.30  vibration
    OIL_SYSTEM_LEAK    p = 0.20  oil pressure             COMPRESSOR_FOULING   p = 0.15  fuel flow, rpm, EGT

Sensor model: B = model baseline x per-engine manufacturing offset (+-1.5 %), z ~ N(0, 1) per channel
----------------------------------------------------------------------------------------------------------
    theta = 1 + 0.9 (theta_prev - 1) + 0.01 z                 shared operating condition (throttle, ambient)
    rpm   = B theta (1 + 0.012 m w^1.5) + 0.002 B z            core speed creeps up to hold thrust
    egt   = B (1 + 0.8 (theta - 1)) + 110 m w^2.2 + (4 + 6 w) z            hot-section efficiency loss
    core  = B (1 + 0.6 (theta - 1)) + 70 m w^1.8 + (3 + 4 w) z
    vib   = B (1 + 3.5 m w^2.5) exp((0.05 + 0.2 w) z),  x U(1.5, 2.5) with p = 0.002 + 0.03 w^3
    fuel  = B theta^1.5 (1 + 0.05 m w^1.6) + 0.006 B z         specific fuel consumption rises
    oil   = B (1 - 0.02 w - 0.12 m max(0, (w - 0.7) / 0.3)^2) + (0.4 + w) z,  - U(4, 9) with p = 0.04 w^4
Noise widens with wear (looser clearances, unstable combustion), and vibration spikes and oil-pressure
dips become frequent just before failure: the precursors that engine health monitoring looks for.

Usage: ``python engine_generator.py [--max-ticks N] [--no-backfill]``, or ``--benchmark [--benchmark-*]`` for
the HF load test, configured by SIM_*, SIM_HF_*, POSTGRES_*, Kafka and Schema Registry variables. Helpers:
fleet_config (SIM_*, command line), fleet_bootstrap (resume, backfill), fleet_runtime (live loop, work
orders), fleet_db (connection, reads), fleet_writer (writes), fleet_lease (single-writer lock) and hf_* (every
committed pulse also becomes an Avro sensor frame on Kafka; hf_runtime wires it in).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from fleet_bootstrap import ResumeMismatchError, bootstrap_fleet
from fleet_config import SimulatorConfig, parse_args
from fleet_lease import WriterLockError
from fleet_runtime import FleetRunner, configure_logging, install_signal_handlers
from hf_benchmark import run_benchmark
from hf_runtime import HfPath, load_hf_config, open_hf_path

if TYPE_CHECKING:
    from fleet_config import GenerationSettings
    from fleet_runtime import MaintenanceCompletion

logger = logging.getLogger("engine_generator")

CHANNELS = ("rpm", "core_temp_c", "egt_c", "vibration_mm_s", "fuel_flow_kg_h", "oil_pressure_psi")
HEALTHY, WARNING, CRITICAL, FAILURE = "HEALTHY", "WARNING", "CRITICAL", "FAILURE"
MAINTENANCE_RESET = "MAINTENANCE_RESET"
WARNING_WEAR, CRITICAL_WEAR = 0.5, 0.8

# Nominal cruise values per engine type in CHANNELS order; rpm is the core spool speed N2.
ENGINE_TYPES: dict[str, tuple[str, tuple[float, ...]]] = {
    "JT9D": ("JT9D-7R4", (7800.0, 540.0, 610.0, 1.4, 3900.0, 52.0)),
    "CFM56": ("CFM56-7B", (14400.0, 575.0, 650.0, 1.1, 1250.0, 45.0)),
    "GE90": ("GE90-115B", (9300.0, 600.0, 680.0, 1.2, 4200.0, 60.0)),
}
FALLBACK_ENGINE_TYPE = "CFM56"
MANUFACTURING_TOLERANCE = 0.015


class FailureMode(NamedTuple):
    """How a lifecycle ends: draw probability, fastest-degrading channels and the repair performed."""

    probability: float
    dominant_channels: tuple[str, ...]
    maintenance_action: str


FAILURE_MODES: dict[str, FailureMode] = {
    "HPT_BLADE_EROSION": FailureMode(0.35, ("egt_c", "core_temp_c"), "HPT_BLADE_REPLACEMENT"),
    "BEARING_DEGRADATION": FailureMode(0.30, ("vibration_mm_s",), "BEARING_REPLACEMENT"),
    "OIL_SYSTEM_LEAK": FailureMode(0.20, ("oil_pressure_psi",), "OIL_PUMP_AND_SEAL_OVERHAUL"),
    "COMPRESSOR_FOULING": FailureMode(0.15, ("fuel_flow_kg_h", "rpm", "egt_c"), "COMPRESSOR_WASH_AND_BLEND"),
}
DOMINANT_SENSITIVITY, SECONDARY_SENSITIVITY = 1.5, 0.8
LIFE_SPREAD, LIFE_MIN_RATIO, LIFE_MAX_RATIO = 0.18, 0.55, 1.6
POLYNOMIAL_EXPONENT, EXPONENTIAL_RATE = (1.8, 2.8), (2.5, 4.0)
SHOCK_SIGMA = 0.25
THETA_PERSISTENCE, THETA_SIGMA = 0.9, 0.01
WEAR_DECIMALS = 6
MIN_EVENT_GAP = timedelta(milliseconds=1)


def stable_seed(*parts: object) -> int:
    """64-bit seed from a SHA-256 of ``parts``; identical across processes, unlike the salted hash()."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def status_for_wear(wear: float) -> str:
    """Operational status implied by the wear factor (contract thresholds)."""
    if wear >= 1.0:
        return FAILURE
    if wear >= CRITICAL_WEAR:
        return CRITICAL
    return WARNING if wear >= WARNING_WEAR else HEALTHY


@dataclass(frozen=True)
class EngineProfile:
    """Engine type plus the per-channel baselines of this particular unit."""

    engine_id: str
    engine_model: str
    baseline: Mapping[str, float]


@functools.cache
def build_profile(engine_id: str) -> EngineProfile:
    """Profile from the engine-id prefix (JT9D-001 -> JT9D-7R4) with deterministic +-1.5 % unit offsets."""
    engine_type = engine_id.split("-", 1)[0].upper()
    if engine_type not in ENGINE_TYPES:
        engine_type = FALLBACK_ENGINE_TYPE
        logger.warning(
            "Unknown engine type of %s: using %s baselines", engine_id, ENGINE_TYPES[engine_type][0]
        )
    model, nominal = ENGINE_TYPES[engine_type]
    rng = np.random.default_rng(stable_seed("manufacturing", engine_id))
    offsets = rng.uniform(-MANUFACTURING_TOLERANCE, MANUFACTURING_TOLERANCE, len(CHANNELS)).tolist()
    baseline = {ch: base * (1.0 + off) for ch, base, off in zip(CHANNELS, nominal, offsets, strict=True)}
    return EngineProfile(engine_id, model, baseline)


@dataclass(frozen=True)
class LifecycleParams:
    """Hidden parameters of one lifecycle (overhaul to failure)."""

    lifecycle_id: int
    expected_life_cycles: float
    curve: str
    shape: float
    failure_mode: str
    sensitivity: Mapping[str, float]


def derive_lifecycle(seed: int, engine_id: str, lifecycle_id: int, mean_life: float) -> LifecycleParams:
    """Draw expected life (cycles), wear-curve family/shape and failure mode for one lifecycle."""
    rng = np.random.default_rng(stable_seed(seed, engine_id, lifecycle_id, "lifecycle"))
    life = rng.normal(mean_life, LIFE_SPREAD * mean_life)
    life = float(np.clip(life, LIFE_MIN_RATIO * mean_life, LIFE_MAX_RATIO * mean_life))
    if rng.random() < 0.5:
        curve, shape = "polynomial", float(rng.uniform(*POLYNOMIAL_EXPONENT))
    else:
        curve, shape = "exponential", float(rng.uniform(*EXPONENTIAL_RATE))
    names = list(FAILURE_MODES)
    mode = names[int(rng.choice(len(names), p=[FAILURE_MODES[name].probability for name in names]))]
    dominant = FAILURE_MODES[mode].dominant_channels
    sensitivity = {ch: DOMINANT_SENSITIVITY if ch in dominant else SECONDARY_SENSITIVITY for ch in CHANNELS}
    return LifecycleParams(lifecycle_id, life, curve, shape, mode, sensitivity)


def wear_curve(t_cycles: np.ndarray, params: LifecycleParams) -> np.ndarray:
    """Expected wear c(t) after ``t_cycles`` operating cycles: 0 at t = 0 and 1 at t = L."""
    x = t_cycles / params.expected_life_cycles
    if params.curve == "polynomial":
        return np.power(x, params.shape)
    return np.expm1(params.shape * x) / np.expm1(params.shape)


def wear_trajectory(params: LifecycleParams, pulses_per_cycle: int, rng: np.random.Generator) -> np.ndarray:
    """Wear of every pulse of a lifecycle: starts at 0.0 and ends with the failure pulse at exactly 1.0."""
    horizon = max(16, math.ceil(2.0 * params.expected_life_cycles * pulses_per_cycle))
    while True:
        increments = np.diff(wear_curve(np.arange(horizon + 1) / pulses_per_cycle, params))
        shocks = rng.lognormal(-(SHOCK_SIGMA**2) / 2.0, SHOCK_SIGMA, size=horizon)
        wear = np.concatenate(([0.0], np.cumsum(increments * shocks)))
        wear = np.round(np.minimum(wear, 1.0), WEAR_DECIMALS)
        failed = np.flatnonzero(wear >= 1.0)
        if failed.size:
            return wear[: failed[0] + 1]
        # Unlucky shocks kept w below 1 for two expected lives: extend the horizon and redraw.
        horizon *= 2


def next_operating_condition(theta: float, rng: np.random.Generator) -> float:
    """AR(1) step of the shared operating condition theta around 1.0."""
    return 1.0 + THETA_PERSISTENCE * (theta - 1.0) + THETA_SIGMA * float(rng.standard_normal())


def sensor_readings(
    baseline: Mapping[str, float],
    sensitivity: Mapping[str, float],
    wear: float,
    theta: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One noisy reading per channel for wear ``wear`` at operating condition ``theta`` (see module doc)."""
    b, m, w = baseline, sensitivity, wear
    # A fixed number of draws per pulse keeps the random stream aligned whatever events occur.
    z = rng.standard_normal(len(CHANNELS)).tolist()
    u = rng.random(4).tolist()
    rpm = b["rpm"] * theta * (1 + 0.012 * m["rpm"] * w**1.5) + 0.002 * b["rpm"] * z[0]
    egt = b["egt_c"] * (1 + 0.8 * (theta - 1)) + 110 * m["egt_c"] * w**2.2 + (4 + 6 * w) * z[1]
    core = b["core_temp_c"] * (1 + 0.6 * (theta - 1)) + 70 * m["core_temp_c"] * w**1.8 + (3 + 4 * w) * z[2]
    vibration = b["vibration_mm_s"] * (1 + 3.5 * m["vibration_mm_s"] * w**2.5)
    vibration *= math.exp((0.05 + 0.2 * w) * z[3])
    if u[0] < 0.002 + 0.03 * w**3:
        vibration *= 1.5 + u[1]
    fuel = b["fuel_flow_kg_h"] * (theta**1.5 * (1 + 0.05 * m["fuel_flow_kg_h"] * w**1.6) + 0.006 * z[4])
    oil_loss = 0.02 * w + 0.12 * m["oil_pressure_psi"] * max(0.0, (w - 0.7) / 0.3) ** 2
    oil = b["oil_pressure_psi"] * (1 - oil_loss) + (0.4 + w) * z[5]
    if u[2] < 0.04 * w**4:
        oil -= 4 + 5 * u[3]
    return {
        "rpm": round(rpm, 2),
        "core_temp_c": round(core, 2),
        "egt_c": round(egt, 2),
        "vibration_mm_s": round(vibration, 3),
        "fuel_flow_kg_h": round(fuel, 2),
        "oil_pressure_psi": round(oil, 2),
    }


@dataclass(frozen=True, slots=True)
class Pulse:
    """One telemetry row (fields named like the engine_telemetry columns) plus the hidden state its
    high-frequency frame is synthesized from: operating condition theta and the lifecycle's failure mode."""

    engine_id: str
    lifecycle_id: int
    cycle_number: int
    pulse_index: int
    event_ts: datetime
    rpm: float
    core_temp_c: float
    egt_c: float
    vibration_mm_s: float
    fuel_flow_kg_h: float
    oil_pressure_psi: float
    wear_factor: float
    status: str
    theta: float
    failure_mode: str

    @property
    def is_failure(self) -> bool:
        """True for the final pulse of a lifecycle (w = 1.0)."""
        return self.status == FAILURE


class EngineSimulator:
    """Digital twin of one engine: HEALTHY -> WARNING -> CRITICAL -> FAILURE -> MAINTENANCE_RESET -> HEALTHY.

    ``next_pulse`` only proposes a pulse; ``advance`` and ``start_next_lifecycle`` apply changes once the
    database transaction persisting them has committed, so a dropped transaction never desynchronises the
    twin from the database. ``settings`` generate the current lifecycle: the stored ones it started with
    when it is resumed, the current SIM_* values (``config.generation``) for every new lifecycle.
    """

    def __init__(
        self,
        engine_id: str,
        config: SimulatorConfig,
        *,
        lifecycle_id: int = 1,
        pulse_index: int = 0,
        status: str = HEALTHY,
        current_cycle: int = 0,
        cycles_since_new: int = 0,
        wear_factor: float = 0.0,
        last_event_ts: datetime | None = None,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.profile = build_profile(engine_id)
        self._config = config
        self.status = status
        self.current_cycle = current_cycle
        self.cycles_since_new = cycles_since_new
        self.wear_factor = wear_factor
        self.last_event_ts = last_event_ts
        self.maintenance_since: datetime | None = None
        self.maintenance_action = ""
        self._load_lifecycle(lifecycle_id, pulse_index, settings or config.generation)

    @property
    def lifecycle_id(self) -> int:
        """Current lifecycle (1-based)."""
        return self.lifecycle.lifecycle_id

    @property
    def in_maintenance(self) -> bool:
        """True between a committed FAILURE and the committed repair."""
        return self.maintenance_since is not None

    def lifecycle_wear_path(
        self, lifecycle_id: int, settings: GenerationSettings | None = None
    ) -> tuple[LifecycleParams, np.ndarray]:
        """Parameters and per-pulse wear of ``lifecycle_id`` under ``settings`` (default: current SIM_*)."""
        settings = settings or self._config.generation
        params = derive_lifecycle(settings.seed, self.engine_id, lifecycle_id, settings.mean_life_cycles)
        rng = np.random.default_rng(stable_seed(settings.seed, self.engine_id, lifecycle_id, "wear"))
        return params, wear_trajectory(params, settings.pulses_per_cycle, rng)

    def lifecycle_lengths(self, count: int) -> list[int]:
        """Pulse counts (failure pulse included) of the current lifecycle and the ``count - 1`` after it."""
        later = [len(self.lifecycle_wear_path(self.lifecycle_id + offset)[1]) for offset in range(1, count)]
        return [len(self.wear_path), *later][:count]

    def _load_lifecycle(self, lifecycle_id: int, pulse_index: int, settings: GenerationSettings) -> None:
        self.settings = settings
        self.lifecycle, self.wear_path = self.lifecycle_wear_path(lifecycle_id, settings)
        self.pulse_index = pulse_index
        # Seeding noise by the start pulse gives a resumed run its own reproducible stream.
        noise_seed = stable_seed(settings.seed, self.engine_id, lifecycle_id, "noise", pulse_index)
        self._rng = np.random.default_rng(noise_seed)
        self._theta = 1.0

    def next_pulse(self, event_ts: datetime) -> Pulse:
        """Propose the next pulse at ``event_ts``; the twin only changes when ``advance`` is called."""
        if self.in_maintenance:
            raise RuntimeError(f"{self.engine_id} is under maintenance and emits no telemetry")
        if self.last_event_ts is not None and event_ts <= self.last_event_ts:
            # The wall clock stepped back: keep (engine_id, event_ts) unique and increasing.
            event_ts = self.last_event_ts + MIN_EVENT_GAP
        index = min(self.pulse_index, len(self.wear_path) - 1)
        wear = float(self.wear_path[index])
        self._theta = next_operating_condition(self._theta, self._rng)
        readings = sensor_readings(
            self.profile.baseline, self.lifecycle.sensitivity, wear, self._theta, self._rng
        )
        return Pulse(
            engine_id=self.engine_id,
            lifecycle_id=self.lifecycle_id,
            cycle_number=index // self.settings.pulses_per_cycle,
            pulse_index=self.pulse_index,
            event_ts=event_ts,
            wear_factor=wear,
            status=status_for_wear(wear),
            theta=self._theta,
            failure_mode=self.lifecycle.failure_mode,
            **readings,
        )

    def cycles_since_new_at(self, pulse: Pulse) -> int:
        """Completed cycles since new once ``pulse`` is applied (cycles before its own)."""
        return self.cycles_since_new + max(0, pulse.cycle_number - self.current_cycle)

    def advance(self, pulse: Pulse) -> None:
        """Apply a committed ``pulse``; a FAILURE pulse takes the engine into maintenance."""
        if pulse.pulse_index != self.pulse_index or pulse.lifecycle_id != self.lifecycle_id:
            raise ValueError(f"{self.engine_id}: pulse {pulse.pulse_index} is not the pending pulse")
        # The cycle in which the engine failed counts as flown.
        self.cycles_since_new = self.cycles_since_new_at(pulse) + (1 if pulse.is_failure else 0)
        self.pulse_index += 1
        self.current_cycle = pulse.cycle_number
        self.wear_factor = pulse.wear_factor
        self.status = pulse.status
        self.last_event_ts = pulse.event_ts
        if pulse.is_failure:
            self.resume_maintenance(pulse.event_ts, self.lifecycle.failure_mode)

    def resume_maintenance(self, failure_ts: datetime, failure_mode: str) -> None:
        """Put the engine under repair since ``failure_ts`` (also used when resuming after a restart)."""
        self.maintenance_since = failure_ts
        self.maintenance_action = FAILURE_MODES[failure_mode].maintenance_action

    def start_next_lifecycle(self, completion: MaintenanceCompletion) -> None:
        """Apply a committed repair: w = 0 with fresh lifecycle parameters drawn under the current SIM_*
        values, status MAINTENANCE_RESET."""
        if completion.lifecycle_id != self.lifecycle_id:
            raise ValueError(f"{self.engine_id}: repair belongs to lifecycle {completion.lifecycle_id}")
        self._load_lifecycle(completion.next_lifecycle_id, 0, self._config.generation)
        self.status = MAINTENANCE_RESET
        self.current_cycle = 0
        self.wear_factor = 0.0
        self.maintenance_since = None

    def describe(self) -> str:
        """One-line state summary for logs."""
        state = f"L{self.lifecycle_id} c{self.current_cycle} w={self.wear_factor:.3f} {self.status}"
        repair = f" (repair since {self.maintenance_since:%H:%M:%S})" if self.maintenance_since else ""
        return f"{self.engine_id} {state}{repair}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulator, or the HF load test with --benchmark. Exit codes: 0 clean shutdown, 1 fatal error
    (or a load test that lost frames), 2 invalid configuration or refusal to write (another simulator holds
    the writer lock, or stored lifecycles contradict the SIM_* values)."""
    args = parse_args(argv)
    import fleet_db  # deferred so that the physics above stays importable without psycopg2

    try:
        config, hf_config = SimulatorConfig.from_env(), load_hf_config(args)
        db_settings = fleet_db.DatabaseSettings.from_env()
    except ValueError as exc:
        configure_logging("INFO")
        logger.error("Invalid configuration: %s", exc)
        return 2
    configure_logging(config.log_level)
    stop = threading.Event()
    install_signal_handlers(stop)
    if args.benchmark:
        nominal = {model: dict(zip(CHANNELS, values, strict=True)) for model, values in ENGINE_TYPES.values()}
        return run_benchmark(hf_config, args, config.seed, nominal, stop)
    logger.info("Starting fleet simulator: %s", config.summary())
    store, backfill = fleet_db.FleetDatabase(db_settings, stop), not args.no_backfill
    factory = functools.partial(EngineSimulator, config=config)
    models = {engine_id: build_profile(engine_id).engine_model for engine_id in config.engine_ids}
    hf = HfPath()
    try:
        hf = open_hf_path(hf_config, config.seed, store, models)
        engines = bootstrap_fleet(
            store, config, factory, models, allow_backfill=backfill, stop_event=stop, on_commit=hf.backfill
        )
        FleetRunner(engines, store, config, stop, hf.live).run(args.max_ticks)
    except (WriterLockError, ResumeMismatchError) as exc:
        logger.error("Refusing to write: %s", exc)
        return 2
    except fleet_db.DatabaseConnectionError as exc:
        if stop.is_set():
            logger.info("Shutdown requested while PostgreSQL was unavailable")
            return 0
        logger.error("Giving up: %s", exc)
        return 1
    except Exception:
        logger.exception("Simulator crashed")
        return 1
    finally:
        hf.close()
        store.close()
    logger.info("Simulator stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
