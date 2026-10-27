"""Checks of the spectral feature models (CONTRACT_V2.md section 8) against the HF fixture.

check_hf_staging       stg_engine_hf_features FINAL holds exactly the newest copy of every raw frame
                       with the values of hf_reference.staged_measurements, and every injected
                       case shows its effect (an absent process key reads NULL, never 0; a
                       late-visible frame is picked up; a corrected copy moves its frame)
check_spectral_values  int_vibration_features FINAL equals the pure-Python rolling means of the
                       staged frames, every frame and value; after an incremental run, the frame
                       that moved to another lifecycle left a retraction row behind
check_ml_features      int_ml_features has exactly the rows of int_sensor_rolling_features FINAL,
                       each with the spectral features of its own frame or all NULL without one;
                       the training mart carries the same values for its rows
check_board_spectral   fct_engine_health_status: the KPIs of the newest frame of the current
                       lifecycle, the NULL-safe bearing-defect flag and its part in anomaly_count
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from clickhouse_connect.driver.client import Client
from hf_reference import (
    MEASUREMENTS,
    SPECTRAL_FEATURES,
    VIBRATION_FEATURES,
    SpectralKey,
    StagedFrame,
    board_spectral_kpis,
    pulse_spectral_features,
    rolling_spectral_features,
    staged_frames,
)
from verify_queries import DATABASE, rows, scalar

LOGGER = logging.getLogger("verify_analytics")
PROJECT_FILE = Path(__file__).resolve().parents[2] / "analytics" / "dbt_project.yml"
BOARD_KPIS = ("fan_amp_bpfo_mean_5m", "fan_kurtosis_mean_5m", "oil_quantity_mean_5m", "oil_filter_dp_mean_5m")
RAW_COPY_COLUMNS = (
    "engine_id",
    "lifecycle_id",
    "telemetry_id",
    "frame_micros",
    "ingested_at",
    "kafka_partition",
    "kafka_offset",
    *VIBRATION_FEATURES,
    "egt_ab_split",
    "process_means",
)
EXAMPLES = 3
Keyed = dict[tuple[Any, ...], dict[str, Any]]


def _same(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected


def _mismatches(
    actual: Mapping[Any, Mapping[str, Any]], expected: Mapping[Any, Mapping[str, Any]]
) -> list[str]:
    """Missing and unexpected keys and differing values of two keyed row sets (a few examples each)."""
    issues = [f"missing {key}" for key in sorted(expected.keys() - actual.keys())[:EXAMPLES]]
    issues += [f"unexpected {key}" for key in sorted(actual.keys() - expected.keys())[:EXAMPLES]]
    differing = [
        f"{key}.{column}: {actual[key][column]} != {value}"
        for key in sorted(expected.keys() & actual.keys())
        for column, value in expected[key].items()
        if not _same(actual[key][column], value)
    ]
    return issues + differing[:EXAMPLES] + ([f"{len(differing)} differing values"] if differing else [])


def _keyed(client: Client, sql: str, key_width: int, names: Iterable[str]) -> Keyed:
    """Rows of ``sql`` keyed by their first ``key_width`` columns, the rest named by ``names``."""
    names = tuple(names)
    return {row[:key_width]: dict(zip(names, row[key_width:], strict=True)) for row in rows(client, sql)}


def _nulls(table: Mapping[Any, Mapping[str, Any]], column: str) -> int:
    return sum(1 for row in table.values() if row[column] is None)


def load_staged_reference(client: Client) -> dict[tuple[str, int], StagedFrame]:
    """hf_reference.staged_frames of every raw copy in raw_engine_hf_features."""
    columns = ", ".join(
        "toUnixTimestamp64Micro(frame_start_ts)" if name == "frame_micros" else name
        for name in RAW_COPY_COLUMNS
    )
    copies = [
        dict(zip(RAW_COPY_COLUMNS, row, strict=True))
        for row in rows(client, f"SELECT {columns} FROM {DATABASE}.raw_engine_hf_features")
    ]
    return staged_frames(copies)


def check_hf_staging(
    client: Client, frames: Mapping[tuple[str, int], StagedFrame], hf: dict[str, Any]
) -> list[str]:
    """stg FINAL == the newest copy of every frame with the reference cleaning; cases show their effect."""
    actual = _keyed(
        client,
        f"SELECT engine_id, telemetry_id, lifecycle_id, {', '.join(MEASUREMENTS)} "
        f"FROM {DATABASE}.stg_engine_hf_features FINAL",
        2,
        ("lifecycle_id", *MEASUREMENTS),
    )
    expected = {
        key: {"lifecycle_id": frame.lifecycle_id, **frame.measurements} for key, frame in frames.items()
    }
    failures = [f"stg_engine_hf_features: {issue}" for issue in _mismatches(actual, expected)]
    cases = hf["cases"]
    negative_oil = sum(
        1 for row in actual.values() if row["oil_temp_c"] is not None and row["oil_temp_c"] < 0
    )
    # (effect, observed count, count implied by the injected cases)
    effects = [
        ("frames", len(actual), hf["frames"]),
        (
            "NULL oil_quantity (absent OIL_QTY_A)",
            _nulls(actual, "oil_quantity"),
            len(cases["missing_oil_qty_a"]),
        ),
        ("NULL fan_kurtosis (flat line)", _nulls(actual, "fan_kurtosis"), len(cases["flat_line"])),
        (
            "NULL fan_amp_bpfo (flat line, stopped spool, negative)",
            _nulls(actual, "fan_amp_bpfo"),
            len({*cases["flat_line"], *cases["stopped_spool"], *cases["negative_bpfo"]}),
        ),
        (
            "NULL fan_crest_factor (flat line, infinite)",
            _nulls(actual, "fan_crest_factor"),
            len({*cases["flat_line"], *cases["infinite_crest"]}),
        ),
        ("NULL egt_ab_split (missing EGT_B)", _nulls(actual, "egt_ab_split"), len(cases["missing_egt_b"])),
        ("negative oil_temp_c kept (cold soak)", negative_oil, len(cases["cold_oil"])),
    ]
    for effect, observed, implied in effects:
        LOGGER.info("stg_engine_hf_features %s: %d (expected %d)", effect, observed, implied)
        if observed != implied or not implied:
            failures.append(f"stg_engine_hf_features {effect}: {observed}, expected {implied} (at least 1)")
    return failures + _copy_case_failures(actual, hf)


def _copy_case_failures(staged: Keyed, hf: dict[str, Any]) -> list[str]:
    """Late-visible frames were picked up; the corrected copy moved its frame to its lifecycle."""
    failures = []
    missing = sorted(set(hf["cases"]["late_visible"]) - {telemetry_id for _, telemetry_id in staged})
    if missing:
        failures.append(f"stg_engine_hf_features: late-visible frames {missing[:EXAMPLES]} not published")
    moved = hf["relabelled"]
    lifecycle = staged.get((moved["engine_id"], moved["telemetry_id"]), {}).get("lifecycle_id")
    if lifecycle != moved["to_lifecycle"]:
        failures.append(f"stg_engine_hf_features: corrected frame {moved} is in lifecycle {lifecycle}")
    return failures


def check_spectral_values(
    client: Client, expected: Mapping[SpectralKey, Mapping[str, Any]], hf: dict[str, Any], incremental: bool
) -> list[str]:
    """int_vibration_features FINAL == the reference rolling means of every frame."""
    names = (*SPECTRAL_FEATURES, "frames_5m")
    actual = _keyed(
        client,
        f"SELECT engine_id, lifecycle_id, telemetry_id, {', '.join(names)} "
        f"FROM {DATABASE}.int_vibration_features FINAL",
        3,
        names,
    )
    failures = [f"int_vibration_features: {issue}" for issue in _mismatches(actual, expected)]
    empty_windows = _nulls(expected, "oil_filter_dp_mean_5m")
    LOGGER.info(
        "int_vibration_features: %d frames (expected %d), %d values compared, %d windows without OIL_DP_A",
        len(actual),
        len(expected),
        len(expected) * len(names),
        empty_windows,
    )
    if not empty_windows:
        failures.append("the fixture has no 5-minute window without OIL_DP_A: nothing was tested")
    moved = hf["relabelled"]
    where = f"engine_id = '{moved['engine_id']}' AND telemetry_id = {moved['telemetry_id']}"
    placed = [
        row[0]
        for row in rows(
            client, f"SELECT lifecycle_id FROM {DATABASE}.int_vibration_features FINAL WHERE {where}"
        )
    ]
    if placed != [moved["to_lifecycle"]]:
        failures.append(f"int_vibration_features: corrected frame {moved} sits in lifecycles {placed}")
    retractions = scalar(
        client,
        f"SELECT count() FROM {DATABASE}.int_vibration_features "
        f"WHERE {where} AND lifecycle_id = {moved['from_lifecycle']} AND is_deleted = 1",
    )
    LOGGER.info("int_vibration_features: %d retraction rows of the corrected frame", retractions)
    if incremental and not retractions:
        failures.append("int_vibration_features: the frame that left its lifecycle was not retracted")
    return failures


def check_ml_features(
    client: Client, reference: Mapping[SpectralKey, Mapping[str, Any]], hf: dict[str, Any]
) -> list[str]:
    """int_ml_features: the pulses of int_sensor_rolling_features FINAL with their own frame's features."""
    spectral = ", ".join(SPECTRAL_FEATURES)
    pulses = set(
        rows(
            client,
            f"SELECT engine_id, lifecycle_id, telemetry_id FROM {DATABASE}.int_sensor_rolling_features FINAL",
        )
    )
    view_rows = scalar(client, f"SELECT count() FROM {DATABASE}.int_ml_features")
    actual = _keyed(
        client,
        f"SELECT engine_id, lifecycle_id, telemetry_id, {spectral} FROM {DATABASE}.int_ml_features",
        3,
        SPECTRAL_FEATURES,
    )
    expected = pulse_spectral_features(pulses, reference)
    failures = [f"int_ml_features: {issue}" for issue in _mismatches(actual, expected)]
    if view_rows != len(pulses):
        failures.append(
            f"int_ml_features has {view_rows} rows, int_sensor_rolling_features FINAL {len(pulses)}"
        )
    without_frame = [key for key in pulses if key not in reference]
    silent = [key for key in without_frame if list(key[:2]) == hf["silent_lifecycle"]]
    LOGGER.info(
        "int_ml_features: %d pulses, %d without an HF frame (%d of the HF-silent lifecycle)",
        len(pulses),
        len(without_frame),
        len(silent),
    )
    if not silent or len(without_frame) == len(pulses):
        failures.append("the fixture lacks pulses with or without HF frames: nothing was tested")
    mart = _keyed(
        client,
        f"SELECT engine_id, lifecycle_id, telemetry_id, {spectral} "
        f"FROM {DATABASE}.mart_rul_training_features FINAL",
        3,
        SPECTRAL_FEATURES,
    )
    issues = _mismatches(mart, pulse_spectral_features(mart, reference))
    return failures + [f"mart_rul_training_features: {issue}" for issue in issues]


def bearing_thresholds() -> tuple[float, float]:
    """anomaly_fan_amp_bpfo_mm_s and anomaly_fan_kurtosis as dbt_project.yml defines them."""
    project_vars = yaml.safe_load(PROJECT_FILE.read_text(encoding="utf-8"))["vars"]
    return float(project_vars["anomaly_fan_amp_bpfo_mm_s"]), float(project_vars["anomaly_fan_kurtosis"])


def check_board_spectral(
    client: Client,
    frames: Mapping[tuple[str, int], StagedFrame],
    reference: Mapping[SpectralKey, Mapping[str, Any]],
    hf: dict[str, Any],
) -> list[str]:
    """Board KPIs = the newest frame of the current lifecycle; the flag follows the thresholds."""
    board = _keyed(
        client,
        f"SELECT engine_id, current_lifecycle_id, toUnixTimestamp64Micro(last_hf_frame_ts), "
        f"{', '.join(BOARD_KPIS)}, is_bearing_defect_anomaly, anomaly_count, is_egt_drift_anomaly "
        f"+ is_vibration_growth_anomaly + is_oil_pressure_drop_anomaly + is_telemetry_stale "
        f"FROM {DATABASE}.fct_engine_health_status FINAL",
        1,
        ("lifecycle_id", "frame_micros", *BOARD_KPIS, "flag", "anomaly_count", "other_flags"),
    )
    limits = bearing_thresholds()
    failures = []
    for (engine_id,), row in sorted(board.items()):
        lifecycle = (engine_id, row["lifecycle_id"])
        current = [frame for frame in frames.values() if (frame.engine_id, frame.lifecycle_id) == lifecycle]
        expected = board_spectral_kpis(current, reference, BOARD_KPIS, limits)
        expected["anomaly_count"] = expected["flag"] + row["other_flags"]
        LOGGER.info("fct %s spectral: %s", engine_id, {name: row[name] for name in expected})
        failures += [
            f"fct {engine_id}: {name} {row[name]}, expected {value}"
            for name, value in expected.items()
            if not _same(row[name], value)
        ]
    alarm = board.get((hf["bearing_alarm_engine"],), {}).get("flag")
    silent = board.get((hf["silent_lifecycle"][0],), {})
    if alarm != 1 or silent.get("lifecycle_id") != hf["silent_lifecycle"][1] or silent.get("frame_micros"):
        failures.append(
            f"fct: bearing alarm {alarm}, HF-silent lifecycle {silent}: the cases are not on the board"
        )
    return failures


def check_spectral(client: Client, manifest: dict[str, Any], incremental: bool) -> list[str]:
    """Every spectral check against the current model contents."""
    hf = manifest["hf"]
    frames = load_staged_reference(client)
    reference = rolling_spectral_features(frames.values())
    return [
        *check_hf_staging(client, frames, hf),
        *check_spectral_values(client, reference, hf, incremental),
        *check_ml_features(client, reference, hf),
        *check_board_spectral(client, frames, reference, hf),
    ]
