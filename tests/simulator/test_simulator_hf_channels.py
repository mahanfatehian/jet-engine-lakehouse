"""Process-parameter catalog of the HF frames: names, units and A/B pairs, agreement with the 1 Hz summary,
and the oil-system and hot-section behaviour of each failure mode."""

from __future__ import annotations

import re
from types import SimpleNamespace

import hf_channels as hc
import hf_signals as hs
import numpy as np
import pytest

MODES = ("BEARING_DEGRADATION", "OIL_SYSTEM_LEAK", "HPT_BLADE_EROSION", "COMPRESSOR_FOULING")
# The parameters downstream code reads by name (dbt int_vibration_features, the streaming job's egt_ab_split).
FIXED_UNITS = {"EGT": "degC", "OIL_T": "degC", "OIL_QTY": "L", "OIL_DP": "psid", "N1": "rpm", "N2": "rpm"}


def state(mode: str, wear: float, theta: float = 1.0, model: str = "CFM56-7B") -> hs.FrameState:
    summary = SimpleNamespace(
        rpm=14400.0,
        core_temp_c=575.0,
        egt_c=650.0,
        vibration_mm_s=1.5,
        fuel_flow_kg_h=1250.0,
        oil_pressure_psi=45.0,
        wear_factor=wear,
        theta=theta,
        failure_mode=mode,
    )
    return hs.frame_state(summary, model, np.random.default_rng(1))


def means(mode: str, wear: float, theta: float = 1.0) -> dict[str, float]:
    frame_state = state(mode, wear, theta)
    vibration = hs.synthesize_vibration(frame_state, 2048, 2048, 2, np.random.default_rng(2))
    return hc.parameter_means(frame_state, vibration)


def test_forty_station_parameters_each_with_an_a_and_b_sensor() -> None:
    names = [parameter.name for parameter in hc.PROCESS_PARAMETERS]
    assert len(names) == len(set(names)) == 40
    assert len(hc.PROCESS_CHANNELS) == 80
    for index, parameter in enumerate(hc.PROCESS_PARAMETERS):
        pair = hc.PROCESS_CHANNELS[2 * index : 2 * index + 2]
        assert pair == ((f"{parameter.name}_A", parameter.unit), (f"{parameter.name}_B", parameter.unit))
        assert re.fullmatch(r"[A-Z0-9_]+", parameter.name) and parameter.noise > 0 and parameter.bias > 0
    units = dict(hc.PROCESS_CHANNELS)
    for name, unit in FIXED_UNITS.items():
        assert units[f"{name}_A"] == units[f"{name}_B"] == unit
    stations = {"T12", "T25", "T3", "T45", "P0", "P25", "PS3", "P45", "WF", "FMV", "VSV", "VBV", "OIL_P"}
    assert stations | {"FUEL_T", "NAC_T", "BLEED_P", "TCC", "HPTACC"} <= set(names)


def test_frame_means_reproduce_the_one_hz_summary_and_the_synthesized_spools() -> None:
    frame_state = state("HPT_BLADE_EROSION", 0.3)
    vibration = hs.synthesize_vibration(frame_state, 2048, 2048, 2, np.random.default_rng(2))
    values = hc.parameter_means(frame_state, vibration)
    assert set(values) == {parameter.name for parameter in hc.PROCESS_PARAMETERS}
    assert (values["N1"], values["N2"]) == (frame_state.n1_rpm, 14400.0)
    assert (values["EGT"], values["T3"], values["WF"], values["OIL_P"]) == (650.0, 575.0, 1250.0, 45.0)
    assert (values["VIB_N1_TRACKED"], values["VIB_N2_TRACKED"]) == (
        vibration.tracked_n1_mm_s,
        vibration.tracked_n2_mm_s,
    )
    assert all(np.isfinite(value) for value in values.values())


def test_oil_quantity_drops_only_when_the_oil_system_leaks() -> None:
    tank = hc.plant_type("CFM56-7B").oil_tank_l
    for mode in MODES:
        drop = means(mode, 0.0)["OIL_QTY"] - means(mode, 1.0)["OIL_QTY"]
        if mode == "OIL_SYSTEM_LEAK":
            assert drop > 0.4 * tank
        else:
            assert 0 < drop < 0.05 * tank, mode


def test_oil_filter_dp_rises_with_bearing_debris() -> None:
    rise = {mode: means(mode, 1.0)["OIL_DP"] - means(mode, 0.0)["OIL_DP"] for mode in MODES}
    assert rise["BEARING_DEGRADATION"] == pytest.approx(20.0)
    assert rise["BEARING_DEGRADATION"] > 3 * max(rise[mode] for mode in MODES[1:])
    wears = (0.0, 0.25, 0.5, 0.75, 1.0)
    curve = [means("BEARING_DEGRADATION", wear)["OIL_DP"] for wear in wears]
    assert curve == sorted(curve), "the filter only loads up"


def test_oil_temperature_rises_with_wear_in_every_mode() -> None:
    for mode in MODES:
        curve = [means(mode, wear)["OIL_T"] for wear in (0.0, 0.5, 1.0)]
        assert curve == sorted(curve) and curve[-1] - curve[0] >= 20.0, mode
    bearing, erosion = means("BEARING_DEGRADATION", 1.0)["OIL_T"], means("HPT_BLADE_EROSION", 1.0)["OIL_T"]
    assert bearing > erosion, "a failing bearing heats the oil most"


def test_sensor_samples_are_mean_plus_fixed_bias_plus_noise_and_b_reads_the_hot_streak() -> None:
    sensors = hc.ProcessSensors(seed=42)
    frame_state = state("HPT_BLADE_EROSION", 1.0)
    split = hc.egt_split(frame_state)
    assert split == pytest.approx(hc.EGT_SPLIT_C) and hc.egt_split(state("OIL_SYSTEM_LEAK", 1.0)) < split / 2
    values = means("HPT_BLADE_EROSION", 1.0)
    samples = sensors.sample("CFM56-104", values, split, 4000, np.random.default_rng(7))
    assert samples.shape == (80, 4000) and samples.dtype == np.float32
    biases = sensors.biases("CFM56-104")
    expected = np.array([values[parameter.name] for parameter in hc.PROCESS_PARAMETERS])[:, None] + biases
    expected[hc.EGT_INDEX, 1] += split
    noise = np.array([parameter.noise for parameter in hc.PROCESS_PARAMETERS])
    measured = samples.astype(np.float64).reshape(40, 2, 4000)
    assert np.all(np.abs(measured.mean(axis=2) - expected) < 5 * noise[:, None] / np.sqrt(4000) + 0.01)
    assert measured.std(axis=2) == pytest.approx(np.repeat(noise[:, None], 2, axis=1), rel=0.1)


def test_calibration_biases_are_fixed_per_engine_and_seed() -> None:
    first, again, other_seed = hc.ProcessSensors(42), hc.ProcessSensors(42), hc.ProcessSensors(7)
    assert np.array_equal(first.biases("GE90-201"), again.biases("GE90-201"))
    assert not np.array_equal(first.biases("GE90-201"), first.biases("GE90-202"))
    assert not np.array_equal(first.biases("GE90-201"), other_seed.biases("GE90-201"))
    a, b = first.biases("GE90-201").T
    assert not np.array_equal(a, b), "channel B carries an offset of its own"
    assert hc.ProcessSensors(-(2**63)).biases("JT9D-001").shape == (40, 2), "any BIGINT seed works"


def test_unknown_models_and_modes_fall_back_to_neutral_constants() -> None:
    assert hc.plant_type("BENCH-X") == hc.PLANT_TYPES["CFM56-7B"]
    values = means("UNKNOWN_MODE", 1.0)
    assert values["OIL_QTY"] > means("OIL_SYSTEM_LEAK", 1.0)["OIL_QTY"] and np.isfinite(values["OIL_DP"])
