"""Physical simulation of the synthetic jet-engine fleet used by the analytics checks.

Engines emit 1 Hz pulses grouped in 10-pulse cycles. Each completed lifecycle degrades along
w(t) = (t / T) ** 1.6 until its last pulse reaches w = 1 (FAILURE); readings follow the engine
model's baseline, a generic wear drift and the lifecycle's failure-mode signature, plus noise.
synthetic_cdc.py turns the pulses into the Debezium change stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

PULSES_PER_CYCLE = 10
MAINTENANCE_SECONDS = 20
RUNNING_PLANNED_CYCLES = 90  # a running lifecycle wears as if it would fail after this many cycles
SIM_START = datetime(2026, 10, 25, 8, 0, 0, tzinfo=UTC)

MODEL_BASELINES: dict[str, dict[str, float]] = {
    "JT9D-7R4": {
        "rpm": 3650.0,
        "core_temp_c": 540.0,
        "egt_c": 600.0,
        "vibration_mm_s": 1.3,
        "fuel_flow_kg_h": 1650.0,
        "oil_pressure_psi": 50.0,
    },
    "CFM56-7B": {
        "rpm": 14460.0,
        "core_temp_c": 575.0,
        "egt_c": 652.0,
        "vibration_mm_s": 1.1,
        "fuel_flow_kg_h": 1250.0,
        "oil_pressure_psi": 45.0,
    },
    "GE90-115B": {
        "rpm": 2355.0,
        "core_temp_c": 610.0,
        "egt_c": 690.0,
        "vibration_mm_s": 0.9,
        "fuel_flow_kg_h": 2900.0,
        "oil_pressure_psi": 55.0,
    },
}
NOISE_SD = {
    "rpm": 12.0,
    "core_temp_c": 1.5,
    "egt_c": 2.0,
    "vibration_mm_s": 0.03,
    "fuel_flow_kg_h": 6.0,
    "oil_pressure_psi": 0.3,
}


@dataclass(frozen=True)
class EnginePlan:
    """Lifecycle plan of one engine: completed lifecycle lengths (cycles) and failure modes."""

    engine_id: str
    engine_model: str
    completed_cycles: tuple[int, ...]
    failure_modes: tuple[str, ...]
    start_offset_s: float


# JT9D-001's third lifecycle fails during the increment phase; the cut-off is derived from it.
FLEET = (
    EnginePlan(
        "JT9D-001",
        "JT9D-7R4",
        (40, 55, 70),
        ("HPT_BLADE_EROSION", "BEARING_DEGRADATION", "OIL_SYSTEM_LEAK"),
        0.113,
    ),
    EnginePlan(
        "CFM56-104",
        "CFM56-7B",
        (50, 45, 40),
        ("COMPRESSOR_FOULING", "OIL_SYSTEM_LEAK", "HPT_BLADE_EROSION"),
        0.372,
    ),
    EnginePlan(
        "GE90-201",
        "GE90-115B",
        (35, 60, 30),
        ("BEARING_DEGRADATION", "HPT_BLADE_EROSION", "COMPRESSOR_FOULING"),
        0.521,
    ),
    EnginePlan(
        "GE90-202",
        "GE90-115B",
        (45, 50, 40),
        ("OIL_SYSTEM_LEAK", "COMPRESSOR_FOULING", "BEARING_DEGRADATION"),
        0.784,
    ),
)
# Registered but never flown: the health board must read its sensor KPIs as unknown (NULL).
SPARE_ENGINE = EnginePlan("CFM56-105", "CFM56-7B", (), (), 0.0)


@dataclass
class Pulse:
    """One telemetry row as first inserted."""

    engine_id: str
    lifecycle_id: int
    index: int
    event_ts: datetime
    readings: dict[str, float]
    wear: float
    status: str
    telemetry_id: int = 0

    @property
    def cycle_number(self) -> int:
        return self.index // PULSES_PER_CYCLE


@dataclass
class Lifecycle:
    """Pulses of one engine lifecycle; failure_mode is None while it is still running."""

    plan: EnginePlan
    lifecycle_id: int
    failure_mode: str | None
    pulses: list[Pulse] = field(default_factory=list)


def wear_to_status(wear: float) -> str:
    """Status thresholds of the platform (CONTRACT.md section 3)."""
    if wear >= 1.0:
        return "FAILURE"
    if wear >= 0.8:
        return "CRITICAL"
    if wear >= 0.5:
        return "WARNING"
    return "HEALTHY"


def sensor_readings(model: str, mode: str | None, wear: float, rng: random.Random) -> dict[str, float]:
    """Readings of one pulse: model baseline, generic wear drift, failure-mode signature, noise."""
    base = MODEL_BASELINES[model]
    w2 = wear * wear
    egt = base["egt_c"] + 12.0 * wear
    core = base["core_temp_c"] + 8.0 * wear
    vib = base["vibration_mm_s"] * (1.0 + 0.15 * wear)
    oil = base["oil_pressure_psi"] - 1.0 * wear
    fuel = base["fuel_flow_kg_h"] * (1.0 + 0.01 * wear)
    rpm = base["rpm"] * (1.0 - 0.003 * wear)
    if mode == "HPT_BLADE_EROSION":
        egt += 70.0 * w2
        core += 45.0 * w2
    elif mode == "BEARING_DEGRADATION":
        vib *= 1.0 + 2.5 * w2
        egt += 8.0 * w2
    elif mode == "OIL_SYSTEM_LEAK":
        oil -= 14.0 * w2
        vib *= 1.0 + 0.3 * w2
    elif mode == "COMPRESSOR_FOULING":
        fuel *= 1.0 + 0.07 * wear
        rpm *= 1.0 - 0.025 * wear
        egt += 35.0 * w2
    values = {
        "rpm": rpm,
        "core_temp_c": core,
        "egt_c": egt,
        "vibration_mm_s": vib,
        "fuel_flow_kg_h": fuel,
        "oil_pressure_psi": oil,
    }
    return {name: round(value + rng.gauss(0.0, NOISE_SD[name]), 4) for name, value in values.items()}


def simulate_engine(plan: EnginePlan, end: datetime, rng: random.Random) -> list[Lifecycle]:
    """Lifecycles of one engine: its completed plan, then a running lifecycle until ``end``."""
    lifecycles: list[Lifecycle] = []
    start = SIM_START + timedelta(seconds=plan.start_offset_s)
    lengths: list[int | None] = [*plan.completed_cycles, None]
    for number, cycles in enumerate(lengths, start=1):
        mode = plan.failure_modes[number - 1] if cycles else None
        lifecycle = Lifecycle(plan, number, mode)
        total = cycles * PULSES_PER_CYCLE if cycles else int((end - start).total_seconds())
        planned = total if cycles else RUNNING_PLANNED_CYCLES * PULSES_PER_CYCLE
        signature = mode or plan.failure_modes[-1]
        for index in range(total):
            wear = 1.0 if cycles and index == total - 1 else min((index / planned) ** 1.6, 0.99)
            # sub-second jitter keeps window boundaries realistic (pulses are not exactly 1 s apart)
            event_ts = start + timedelta(seconds=index, microseconds=rng.randrange(0, 200_000))
            readings = sensor_readings(plan.engine_model, signature, wear, rng)
            lifecycle.pulses.append(
                Pulse(plan.engine_id, number, index, event_ts, readings, round(wear, 4), wear_to_status(wear))
            )
        lifecycles.append(lifecycle)
        if cycles:
            start = lifecycle.pulses[-1].event_ts + timedelta(seconds=MAINTENANCE_SECONDS + 1)
    return lifecycles
