"""Process-parameter catalog of the HF frames: 40 station-named engine parameters, each read by a redundant
A/B sensor pair (80 channels named '<PARAM>_<A|B>'), sampled at SIM_HF_PROCESS_RATE_HZ within the frame.

Station numbers follow SAE ARP755 (0 ambient, 12 fan inlet, 13 fan bypass, 25 HPC inlet, 3 HPC exit, 45 LPT
inlet, 5 LPT exit). Every parameter's frame mean derives from the tick's 1 Hz state (``parameter_means``):
N2, EGT, T3 (= core_temp_c), WF and OIL_P ARE the summary values, the others follow the operating condition
theta, the wear w and the lifecycle's failure mode:

    OIL_T     92 + 15 (theta - 1) + 22 h w^2 degC     rises with wear in every mode (h = 1.5 when the
                                                       bearings or the oil system fail, else 1)
    OIL_QTY   tank (0.85 - 0.04 w - leak w^2) L        drops by 45 % of the tank in OIL_SYSTEM_LEAK
    OIL_DP    6 + debris w^3 psid                      filter loads with bearing debris (20 psid at failure)
    SCAV_T    OIL_T + 12 + 25 bh w^3 degC              bearing sump scavenge oil heats with a failing bearing
    PS3/VSV   HPC pressure ratio loss, vane re-schedule COMPRESSOR_FOULING
    HPTACC    clearance-control valve opens            HPT_BLADE_EROSION (also widens the EGT A/B split)

Sensors: sample = mean + calibration bias + white noise. A sensor's bias is fixed per engine (seeded by
SIM_SEED and the engine id); channel B = channel A + a small bias of its own + independent noise. The EGT
harness of channel B also reads a hot streak that grows with hot-section damage (egt_ab_split).
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from hf_signals import FrameState, VibrationFrame, spool_type

SENSOR_SUFFIXES = ("A", "B")
EGT_SPLIT_C = 12.0  # extra EGT_B reading at w = 1 for full hot-section damage


class ProcessParameter(NamedTuple):
    """One measured parameter: unit, meaning and the 1-sigma sensor noise and calibration bias (in unit)."""

    name: str
    unit: str
    description: str
    noise: float
    bias: float


PROCESS_PARAMETERS: tuple[ProcessParameter, ...] = (
    ProcessParameter("N1", "rpm", "fan (LP) spool speed", 2.0, 1.0),
    ProcessParameter("N2", "rpm", "core (HP) spool speed", 3.0, 2.0),
    ProcessParameter("N1_CMD", "rpm", "FADEC fan speed command", 0.3, 0.2),
    ProcessParameter("TLA", "deg", "throttle lever angle", 0.05, 0.05),
    ProcessParameter("EPR", "ratio", "engine pressure ratio P5/P2", 0.002, 0.001),
    ProcessParameter("T12", "degC", "fan inlet total temperature", 0.3, 0.2),
    ProcessParameter("T25", "degC", "HPC inlet temperature", 0.5, 0.4),
    ProcessParameter("T3", "degC", "HPC discharge temperature", 1.5, 1.0),
    ProcessParameter("T45", "degC", "LPT inlet temperature", 2.0, 1.5),
    ProcessParameter("EGT", "degC", "exhaust gas temperature", 1.5, 1.0),
    ProcessParameter("T5", "degC", "LPT exit temperature", 1.5, 1.0),
    ProcessParameter("P0", "kPa", "ambient static pressure", 0.05, 0.03),
    ProcessParameter("PT2", "kPa", "fan inlet total pressure", 0.08, 0.05),
    ProcessParameter("PS13", "kPa", "fan bypass static pressure", 0.15, 0.1),
    ProcessParameter("P25", "kPa", "HPC inlet total pressure", 0.2, 0.15),
    ProcessParameter("PS3", "kPa", "HPC discharge static pressure", 2.0, 1.5),
    ProcessParameter("P45", "kPa", "LPT inlet total pressure", 0.6, 0.4),
    ProcessParameter("WF", "kg/h", "fuel flow", 6.0, 4.0),
    ProcessParameter("FMV", "pct", "fuel metering valve position", 0.1, 0.08),
    ProcessParameter("FUEL_P", "kPa", "fuel pump discharge pressure", 8.0, 5.0),
    ProcessParameter("FUEL_T", "degC", "fuel temperature after the fuel/oil cooler", 0.3, 0.2),
    ProcessParameter("FUEL_FLT_DP", "kPa", "fuel filter differential pressure", 0.3, 0.2),
    ProcessParameter("VSV", "deg", "variable stator vane angle", 0.05, 0.04),
    ProcessParameter("VBV", "pct", "variable bleed valve position", 0.1, 0.08),
    ProcessParameter("TCC", "pct", "LPT clearance control valve position", 0.2, 0.15),
    ProcessParameter("HPTACC", "pct", "HPT active clearance control valve position", 0.2, 0.15),
    ProcessParameter("BLEED_P", "kPa", "customer bleed air pressure", 1.0, 0.8),
    ProcessParameter("BLEED_T", "degC", "customer bleed air temperature", 0.8, 0.5),
    ProcessParameter("NAC_T", "degC", "nacelle core compartment temperature", 0.4, 0.3),
    ProcessParameter("ECU_T", "degC", "FADEC (ECU) temperature", 0.3, 0.2),
    ProcessParameter("OIL_P", "psi", "oil pressure", 0.4, 0.3),
    ProcessParameter("OIL_T", "degC", "oil temperature", 0.4, 0.3),
    ProcessParameter("OIL_QTY", "L", "oil tank quantity", 0.08, 0.05),
    ProcessParameter("OIL_DP", "psid", "oil filter differential pressure", 0.15, 0.1),
    ProcessParameter("SCAV_T", "degC", "bearing sump scavenge oil temperature", 0.5, 0.4),
    ProcessParameter("IDG_OIL_T", "degC", "integrated drive generator oil temperature", 0.4, 0.3),
    ProcessParameter("GEN_LOAD", "kVA", "generator load", 0.5, 0.3),
    ProcessParameter("VIB_N1_TRACKED", "mm/s", "N1-tracked fan vibration (1xN1 RMS)", 0.01, 0.005),
    ProcessParameter("VIB_N2_TRACKED", "mm/s", "N2-tracked core vibration (1xN2 RMS)", 0.01, 0.005),
    ProcessParameter("FN_EST", "kN", "FADEC thrust estimate", 0.1, 0.05),
)
PROCESS_CHANNELS: tuple[tuple[str, str], ...] = tuple(
    (f"{parameter.name}_{suffix}", parameter.unit)
    for parameter in PROCESS_PARAMETERS
    for suffix in SENSOR_SUFFIXES
)
EGT_INDEX = [parameter.name for parameter in PROCESS_PARAMETERS].index("EGT")


class PlantType(NamedTuple):
    """Per-type constants of the process model."""

    oil_tank_l: float
    pressure_ratio: float
    max_fuel_flow_kg_h: float
    cruise_thrust_kn: float


PLANT_TYPES: dict[str, PlantType] = {
    "JT9D-7R4": PlantType(26.0, 24.0, 9000.0, 50.0),
    "CFM56-7B": PlantType(20.0, 32.0, 3000.0, 24.0),
    "GE90-115B": PlantType(34.0, 42.0, 9500.0, 70.0),
}


class ProcessSignature(NamedTuple):
    """How a failure mode shows in the process channels (see the module doc)."""

    oil_leak: float
    oil_heat: float
    oil_debris: float
    bearing_heat: float
    hot_section: float
    fouling: float


PROCESS_SIGNATURES: dict[str, ProcessSignature] = {
    "BEARING_DEGRADATION": ProcessSignature(0.0, 1.5, 20.0, 1.0, 0.2, 0.0),
    "OIL_SYSTEM_LEAK": ProcessSignature(0.45, 1.5, 5.0, 0.4, 0.2, 0.0),
    "HPT_BLADE_EROSION": ProcessSignature(0.0, 1.0, 1.5, 0.1, 1.0, 0.1),
    "COMPRESSOR_FOULING": ProcessSignature(0.0, 1.0, 1.5, 0.1, 0.4, 1.0),
}
NEUTRAL_PROCESS = ProcessSignature(0.0, 1.0, 1.5, 0.1, 0.2, 0.0)


def plant_type(engine_model: str) -> PlantType:
    """Process constants of ``engine_model`` (CFM56-7B for an unknown model)."""
    return PLANT_TYPES.get(engine_model, PLANT_TYPES["CFM56-7B"])


def parameter_means(state: FrameState, vibration: VibrationFrame) -> dict[str, float]:
    """Frame mean of every catalog parameter (true value, before sensor bias and noise)."""
    s = state.summary
    w, dt, theta = s.wear_factor, s.theta - 1.0, s.theta
    plant = plant_type(state.engine_model)
    sig = PROCESS_SIGNATURES.get(s.failure_mode, NEUTRAL_PROCESS)
    p0 = 24.0 * (1.0 - 1.5 * dt)
    pt2 = 1.52 * p0
    ps3 = pt2 * plant.pressure_ratio * theta**2 * (1.0 - 0.02 * w - 0.08 * sig.fouling * w**1.6)
    oil_t = 92.0 + 15.0 * dt + 22.0 * sig.oil_heat * w**2
    nac_t = 45.0 + 0.04 * (s.egt_c - 650.0) + 6.0 * w
    return {
        "N1": state.n1_rpm,
        "N2": s.rpm,
        "N1_CMD": spool_type(state.engine_model).n1_nominal_rpm * theta,
        "TLA": 40.0 * theta,
        "EPR": 1.0 + 0.45 * theta**2 * (1.0 - 0.03 * w),
        "T12": -30.0 + 25.0 * dt,
        "T25": 95.0 + 120.0 * dt + 6.0 * w,
        "T3": s.core_temp_c,
        "T45": s.egt_c + 175.0 + 25.0 * w,
        "EGT": s.egt_c,
        "T5": s.egt_c - 60.0,
        "P0": p0,
        "PT2": pt2,
        "PS13": pt2 * 1.6 * theta**2,
        "P25": pt2 * 2.4 * theta**2,
        "PS3": ps3,
        "P45": 0.22 * ps3,
        "WF": s.fuel_flow_kg_h,
        "FMV": 100.0 * s.fuel_flow_kg_h / plant.max_fuel_flow_kg_h,
        "FUEL_P": 2800.0 + 900.0 * dt,
        "FUEL_T": 30.0 + 0.25 * (oil_t - 92.0),
        "FUEL_FLT_DP": 25.0 + 10.0 * dt + 3.0 * w,
        "VSV": 4.0 - 30.0 * dt + 3.0 * sig.fouling * w**1.5,
        "VBV": max(0.0, 2.0 - 40.0 * dt + 1.5 * w),
        "TCC": 55.0 + 80.0 * dt + 8.0 * w,
        "HPTACC": 60.0 + 70.0 * dt + 15.0 * sig.hot_section * w**2,
        "BLEED_P": 290.0 + 300.0 * dt,
        "BLEED_T": 200.0 + 150.0 * dt + 10.0 * w,
        "NAC_T": nac_t,
        "ECU_T": 38.0 + 0.3 * (nac_t - 45.0),
        "OIL_P": s.oil_pressure_psi,
        "OIL_T": oil_t,
        "OIL_QTY": plant.oil_tank_l * (0.85 - 0.04 * w - sig.oil_leak * w**2) - 3.0 * dt,
        "OIL_DP": 6.0 + 2.0 * dt + sig.oil_debris * w**3,
        "SCAV_T": oil_t + 12.0 + 25.0 * sig.bearing_heat * w**3,
        "IDG_OIL_T": 85.0 + 10.0 * dt,
        "GEN_LOAD": 55.0 + 20.0 * dt,
        "VIB_N1_TRACKED": vibration.tracked_n1_mm_s,
        "VIB_N2_TRACKED": vibration.tracked_n2_mm_s,
        "FN_EST": plant.cruise_thrust_kn * theta**2 * (1.0 - 0.02 * w),
    }


class ProcessSensors:
    """The A/B sensors of a fleet: fixed per-engine calibration biases plus the per-frame sampling."""

    def __init__(self, seed: int) -> None:
        self._seed = seed % 2**64
        self._noise = np.array([parameter.noise for parameter in PROCESS_PARAMETERS])
        self._bias_sigma = np.array([parameter.bias for parameter in PROCESS_PARAMETERS])
        self._biases: dict[str, np.ndarray] = {}

    def biases(self, engine_id: str) -> np.ndarray:
        """Calibration offsets (parameters x 2) of the engine's A and B sensors; B = A + its own offset."""
        if engine_id not in self._biases:
            rng = np.random.default_rng([self._seed, zlib.crc32(engine_id.encode())])
            a = rng.normal(0.0, self._bias_sigma)
            self._biases[engine_id] = np.stack([a, a + rng.normal(0.0, self._bias_sigma)], axis=1)
        return self._biases[engine_id]

    def sample(
        self,
        engine_id: str,
        means: Mapping[str, float],
        egt_split_c: float,
        samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Samples (80 channels in PROCESS_CHANNELS order x ``samples``, float32) of one frame."""
        mean = np.array([means[parameter.name] for parameter in PROCESS_PARAMETERS])
        values = mean[:, None, None] + self.biases(engine_id)[:, :, None]
        values = values + self._noise[:, None, None] * rng.standard_normal((mean.size, 2, samples))
        values[EGT_INDEX, 1, :] += egt_split_c
        return values.reshape(mean.size * 2, samples).astype(np.float32)


def egt_split(state: FrameState) -> float:
    """Hot-streak offset of the EGT_B harness: EGT_SPLIT_C x hot-section damage x w^2."""
    signature = PROCESS_SIGNATURES.get(state.summary.failure_mode, NEUTRAL_PROCESS)
    return EGT_SPLIT_C * signature.hot_section * state.wear**2
