"""Independent pure-Python reference of the int_sensor_rolling_features semantics.

verify_analytics.py compares the dbt model against this on real data. Semantics (CONTRACT.md
section 7): RANGE frames over epoch seconds that include the current pulse and its timestamp
peers, population standard deviation, lag-1 deltas (0 for the first pulse), rate of change per
minute against the oldest pulse of the window (0 when no time elapsed) and drift against the
expanding mean of the lifecycle's first ``baseline_cycles`` cycles.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

SENSORS = ("rpm", "core_temp_c", "egt_c", "vibration_mm_s", "fuel_flow_kg_h", "oil_pressure_psi")
ROLLED_SENSORS = ("vibration_mm_s", "egt_c", "oil_pressure_psi")
WINDOWS = (("5m", 300), ("15m", 900))


@dataclass(frozen=True)
class Reading:
    """One live pulse of a lifecycle as stored in stg_engine_telemetry."""

    telemetry_id: int
    cycle_number: int
    event_micros: int
    values: dict[str, float]

    @property
    def epoch(self) -> float:
        # the same Float64 the SQL model orders its RANGE frames by
        return self.event_micros / 1_000_000.0


def _window_statistics(values: list[float]) -> tuple[float, float, float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance), min(values), max(values)


def _divide(numerator: float, denominator: float, fallback: float) -> float:
    return numerator / denominator if denominator != 0 else fallback


def _drift(features: dict[str, float], baseline: dict[str, float | None]) -> dict[str, float]:
    def reference(sensor: str, fallback: float) -> float:
        value = baseline[sensor]
        return fallback if value is None else value

    egt = reference("egt_c", features["egt_c_mean_5m"])
    core = reference("core_temp_c", features["core_temp_c"])
    vibration = reference("vibration_mm_s", features["vibration_mm_s_mean_5m"])
    oil = reference("oil_pressure_psi", features["oil_pressure_psi_mean_5m"])
    fuel = reference("fuel_flow_kg_h", features["fuel_flow_kg_h"])
    rpm = reference("rpm", features["rpm"])
    return {
        "egt_c_drift": features["egt_c_mean_5m"] - egt,
        "core_temp_c_drift": features["core_temp_c"] - core,
        "vibration_mm_s_ratio": _divide(features["vibration_mm_s_mean_5m"], vibration, 1.0),
        "oil_pressure_psi_drop": oil - features["oil_pressure_psi_mean_5m"],
        "fuel_flow_kg_h_drift_pct": _divide(100.0 * (features["fuel_flow_kg_h"] - fuel), fuel, 0.0),
        "rpm_drift_pct": _divide(100.0 * (features["rpm"] - rpm), rpm, 0.0),
    }


def expected_features(readings: list[Reading], baseline_cycles: int) -> dict[int, dict[str, float]]:
    """Expected float feature values per telemetry_id for the pulses of ONE lifecycle."""
    rows = sorted(readings, key=lambda reading: (reading.epoch, reading.telemetry_id))
    epochs = [row.epoch for row in rows]
    baseline_end = min(row.cycle_number for row in rows) + baseline_cycles
    baseline_values: dict[str, list[float]] = {sensor: [] for sensor in SENSORS}
    expected: dict[int, dict[str, float]] = {}
    for index, row in enumerate(rows):
        if row.cycle_number < baseline_end:
            for sensor in SENSORS:
                baseline_values[sensor].append(row.values[sensor])
        frame_end = bisect.bisect_right(epochs, row.epoch)
        features = dict(row.values)
        for sensor in ROLLED_SENSORS:
            for label, seconds in WINDOWS:
                frame = rows[bisect.bisect_left(epochs, row.epoch - seconds) : frame_end]
                mean, std, low, high = _window_statistics([pulse.values[sensor] for pulse in frame])
                features.update(
                    {
                        f"{sensor}_mean_{label}": mean,
                        f"{sensor}_std_{label}": std,
                        f"{sensor}_min_{label}": low,
                        f"{sensor}_max_{label}": high,
                    }
                )
                elapsed = row.epoch - frame[0].epoch
                change = row.values[sensor] - frame[0].values[sensor]
                features[f"{sensor}_roc_{label}"] = change / (elapsed / 60.0) if elapsed > 0 else 0.0
            previous = rows[index - 1] if index else row
            features[f"{sensor}_delta_1"] = row.values[sensor] - previous.values[sensor]
        baseline = {
            sensor: math.fsum(values) / len(values) if values else None
            for sensor, values in baseline_values.items()
        }
        features.update(_drift(features, baseline))
        expected[row.telemetry_id] = features
    return expected
