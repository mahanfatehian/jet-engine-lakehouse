"""Physics of the edge simulator: wear curve, lifecycle variability and the sensor signal behind RUL."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import engine_generator as eg
import numpy as np
import pytest
from fleet_bootstrap import SimulatorConfig
from fleet_runtime import completion_at

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TICK = timedelta(seconds=1)


def simulate(engine_id: str, config: SimulatorConfig, lifecycles: int) -> list[eg.Pulse]:
    """Pulses of ``lifecycles`` complete lifecycles of one engine (repairs complete instantly)."""
    engine = eg.EngineSimulator(engine_id, config)
    pulses: list[eg.Pulse] = []
    ts = T0
    for _ in range(lifecycles):
        while True:
            pulse = engine.next_pulse(ts)
            engine.advance(pulse)
            pulses.append(pulse)
            ts += TICK
            if pulse.is_failure:
                engine.start_next_lifecycle(completion_at(engine, ts))
                break
    return pulses


@pytest.fixture(scope="module")
def fleet_sample() -> dict[str, dict[str, np.ndarray]]:
    """Per engine model: wear and sensor arrays from 4 lifecycles of every default engine."""
    config = SimulatorConfig(pulses_per_cycle=5, mean_life_cycles=80)
    by_model: dict[str, list[eg.Pulse]] = {}
    for engine_id in config.engine_ids:
        by_model.setdefault(eg.build_profile(engine_id).engine_model, []).extend(
            simulate(engine_id, config, 4)
        )
    return {
        model: {
            field: np.array([getattr(p, field) for p in pulses]) for field in ("wear_factor", *eg.CHANNELS)
        }
        for model, pulses in by_model.items()
    }


def test_wear_trajectories_are_monotone_bounded_and_end_in_failure() -> None:
    config = SimulatorConfig(pulses_per_cycle=3, mean_life_cycles=40)
    families = set()
    for engine_id in ("JT9D-001", "CFM56-104", "GE90-201"):
        engine = eg.EngineSimulator(engine_id, config)
        for lifecycle_id in range(1, 61):
            params, path = engine.lifecycle_wear_path(lifecycle_id)
            families.add(params.curve)
            assert path[0] == 0.0
            assert path[-1] == 1.0
            assert np.all(np.diff(path) >= 0.0)
            assert np.all((path >= 0.0) & (path <= 1.0))
            assert np.all(path[:-1] < 1.0), "only the final (FAILURE) pulse may reach w = 1"
    assert families == {"polynomial", "exponential"}


@pytest.mark.parametrize(("curve", "shape"), [("polynomial", 1.8), ("polynomial", 2.8), ("exponential", 4.0)])
def test_wear_curve_is_normalised_and_convex(curve: str, shape: float) -> None:
    params = eg.LifecycleParams(1, 50.0, curve, shape, "BEARING_DEGRADATION", {})
    values = eg.wear_curve(np.linspace(0.0, 50.0, 501), params)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert np.all(np.diff(values, 2) >= -1e-12), "wear must accelerate towards end of life"


def test_lifecycle_length_varies_around_the_configured_mean() -> None:
    config = SimulatorConfig(pulses_per_cycle=2, mean_life_cycles=100)
    engine = eg.EngineSimulator("CFM56-104", config)
    failure_cycles = np.array(
        [(len(engine.lifecycle_wear_path(lifecycle_id)[1]) - 1) // 2 for lifecycle_id in range(1, 401)]
    )
    assert failure_cycles.mean() == pytest.approx(100, rel=0.05)
    assert 0.12 < failure_cycles.std() / failure_cycles.mean() < 0.24
    assert failure_cycles.min() >= 50
    assert failure_cycles.max() <= 165
    assert len(np.unique(failure_cycles)) > 40


def test_lifecycle_parameters_derive_from_a_stable_hash() -> None:
    first = eg.derive_lifecycle(42, "JT9D-001", 3, 180.0)
    assert first == eg.derive_lifecycle(42, "JT9D-001", 3, 180.0)
    assert first != eg.derive_lifecycle(43, "JT9D-001", 3, 180.0)
    assert first != eg.derive_lifecycle(42, "JT9D-001", 4, 180.0)
    assert eg.stable_seed(42, "JT9D-001", 3) == eg.stable_seed(42, "JT9D-001", 3)
    assert 0.55 * 180 <= first.expected_life_cycles <= 1.6 * 180


def test_same_seed_reproduces_the_telemetry_and_another_seed_does_not() -> None:
    config = SimulatorConfig(pulses_per_cycle=4, mean_life_cycles=20, seed=7)
    assert simulate("GE90-201", config, 2) == simulate("GE90-201", config, 2)
    other = SimulatorConfig(pulses_per_cycle=4, mean_life_cycles=20, seed=8)
    assert simulate("GE90-201", config, 1)[:50] != simulate("GE90-201", other, 1)[:50]


@pytest.mark.parametrize(
    ("wear", "status"),
    [
        (0.0, "HEALTHY"),
        (0.4999, "HEALTHY"),
        (0.5, "WARNING"),
        (0.7999, "WARNING"),
        (0.8, "CRITICAL"),
        (0.999999, "CRITICAL"),
        (1.0, "FAILURE"),
    ],
)
def test_status_thresholds(wear: float, status: str) -> None:
    assert eg.status_for_wear(wear) == status


def test_engine_profiles_follow_the_engine_id_prefix(caplog: pytest.LogCaptureFixture) -> None:
    expected = {"JT9D-001": "JT9D-7R4", "CFM56-105": "CFM56-7B", "GE90-202": "GE90-115B"}
    for engine_id, model in expected.items():
        profile = eg.build_profile(engine_id)
        nominal = dict(zip(eg.CHANNELS, eg.ENGINE_TYPES[engine_id.split("-")[0]][1], strict=True))
        assert profile.engine_model == model
        for channel, value in profile.baseline.items():
            assert abs(value / nominal[channel] - 1.0) <= eg.MANUFACTURING_TOLERANCE
    assert eg.build_profile("JT9D-001").baseline != eg.build_profile("JT9D-002").baseline
    with caplog.at_level(logging.WARNING, logger="engine_generator"):
        assert eg.build_profile("PW4000-900").engine_model == "CFM56-7B"
    assert "PW4000-900" in caplog.text


def test_failure_modes_follow_the_configured_probabilities() -> None:
    modes = [
        eg.derive_lifecycle(42, "CFM56-104", lifecycle_id, 180.0).failure_mode for lifecycle_id in range(4000)
    ]
    for name, mode in eg.FAILURE_MODES.items():
        assert modes.count(name) / len(modes) == pytest.approx(mode.probability, abs=0.03)


def test_egt_rises_by_at_least_40_c_from_new_to_worn(fleet_sample: dict[str, dict[str, np.ndarray]]) -> None:
    for model, data in fleet_sample.items():
        wear, egt = data["wear_factor"], data["egt_c"]
        assert egt[wear >= 0.8].mean() - egt[wear <= 0.2].mean() >= 40, model


def test_vibration_spread_widens_with_wear(fleet_sample: dict[str, dict[str, np.ndarray]]) -> None:
    for model, data in fleet_sample.items():
        wear, vibration = data["wear_factor"], data["vibration_mm_s"]
        assert vibration[wear >= 0.8].std() > 2 * vibration[wear <= 0.2].std(), model
        assert vibration[wear >= 0.8].mean() > vibration[wear <= 0.2].mean(), model


def test_oil_pressure_drops_before_failure(fleet_sample: dict[str, dict[str, np.ndarray]]) -> None:
    for model, data in fleet_sample.items():
        wear, oil = data["wear_factor"], data["oil_pressure_psi"]
        assert oil[wear >= 0.9].mean() < oil[wear <= 0.5].mean(), model


def test_core_temperature_fuel_flow_and_rpm_drift_upwards(
    fleet_sample: dict[str, dict[str, np.ndarray]],
) -> None:
    for model, data in fleet_sample.items():
        worn, new = data["wear_factor"] >= 0.8, data["wear_factor"] <= 0.2
        for channel in ("core_temp_c", "fuel_flow_kg_h", "rpm"):
            assert data[channel][worn].mean() > data[channel][new].mean(), (model, channel)


def test_readings_stay_physically_plausible(fleet_sample: dict[str, dict[str, np.ndarray]]) -> None:
    for model, data in fleet_sample.items():
        engine_type = next(prefix for prefix, (name, _) in eg.ENGINE_TYPES.items() if name == model)
        base = dict(zip(eg.CHANNELS, eg.ENGINE_TYPES[engine_type][1], strict=True))
        bounds = {
            "rpm": (0.85 * base["rpm"], 1.15 * base["rpm"]),
            "core_temp_c": (base["core_temp_c"] - 100, base["core_temp_c"] + 200),
            "egt_c": (base["egt_c"] - 120, base["egt_c"] + 300),
            "vibration_mm_s": (0.5 * base["vibration_mm_s"], 60.0),
            "fuel_flow_kg_h": (0.75 * base["fuel_flow_kg_h"], 1.35 * base["fuel_flow_kg_h"]),
            "oil_pressure_psi": (15.0, 1.1 * base["oil_pressure_psi"]),
        }
        for channel, (low, high) in bounds.items():
            values = data[channel]
            assert low <= values.min() and values.max() <= high, (model, channel, values.min(), values.max())


def _mean_readings(failure_mode: str, wear: float, draws: int = 3000) -> dict[str, float]:
    baseline = eg.build_profile("CFM56-104").baseline
    dominant = eg.FAILURE_MODES[failure_mode].dominant_channels
    sensitivity = {
        ch: eg.DOMINANT_SENSITIVITY if ch in dominant else eg.SECONDARY_SENSITIVITY for ch in eg.CHANNELS
    }
    rng = np.random.default_rng(11)
    readings = [eg.sensor_readings(baseline, sensitivity, wear, 1.0, rng) for _ in range(draws)]
    return {channel: float(np.mean([r[channel] for r in readings])) for channel in eg.CHANNELS}


def test_failure_mode_multipliers_change_the_dominant_channel() -> None:
    hpt, bearing, oil_leak, fouling = (
        _mean_readings(mode, 0.95)
        for mode in ("HPT_BLADE_EROSION", "BEARING_DEGRADATION", "OIL_SYSTEM_LEAK", "COMPRESSOR_FOULING")
    )
    assert hpt["egt_c"] > bearing["egt_c"] + 40
    assert hpt["core_temp_c"] > bearing["core_temp_c"] + 25
    assert bearing["vibration_mm_s"] > 1.4 * hpt["vibration_mm_s"]
    assert oil_leak["oil_pressure_psi"] < hpt["oil_pressure_psi"] - 1.5
    assert fouling["fuel_flow_kg_h"] > bearing["fuel_flow_kg_h"] + 20
    assert fouling["rpm"] > bearing["rpm"] + 50
    healthy = _mean_readings("BEARING_DEGRADATION", 0.0), _mean_readings("HPT_BLADE_EROSION", 0.0)
    assert healthy[0] == healthy[1], "failure modes must be invisible on a new engine"
