"""HF sensor frames: the Avro record built from a committed pulse, its round trip through the repository's
.avsc, what the streaming job's feature code reads from it, frame sizes, the telemetry_id mapping of
INSERT ... RETURNING and the commit tap that turns committed pulses into frame requests."""

from __future__ import annotations

import io
import json
import sys
import types
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import engine_generator as eg
import numpy as np
import pytest
from fleet_config import SimulatorConfig
from fleet_runtime import RecordedFailure, committed_pulses, failure_event
from fleet_writer import FleetStateError, FleetWriter, ids_in_pulse_order
from hf_channels import PROCESS_CHANNELS
from hf_config import HfConfig
from hf_frames import (
    HEADER_BYTES,
    TOPIC_MAX_MESSAGE_BYTES,
    FrameBuilder,
    FrameRequest,
    FrameSequencer,
    FrameTap,
    check_frame_size,
    epoch_micros,
    estimated_frame_bytes,
    pack_samples,
)

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "infrastructure" / "schemas" / "engine_sensor_frame.avsc"
EVENT_TS = datetime(2026, 10, 26, 9, 1, 2, 123456, tzinfo=UTC)
CONFIG = HfConfig()


def request(
    *,
    mode: str = "HPT_BLADE_EROSION",
    wear: float = 0.4,
    vibration: float = 2.0,
    engine_id: str = "CFM56-104",
) -> FrameRequest:
    return FrameRequest(
        engine_id=engine_id,
        engine_model="CFM56-7B",
        lifecycle_id=3,
        cycle_number=57,
        telemetry_id=987_654_321_012,
        frame_seq=4321,
        frame_start=EVENT_TS,
        rpm=14_412.5,
        core_temp_c=581.2,
        egt_c=662.4,
        vibration_mm_s=vibration,
        fuel_flow_kg_h=1271.0,
        oil_pressure_psi=44.1,
        wear_factor=wear,
        theta=1.004,
        failure_mode=mode,
    )


def decode(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4")


def test_record_follows_the_schema_field_order_and_carries_the_pulse_identity() -> None:
    record = FrameBuilder(CONFIG, seed=42).build(request())
    fields = [field["name"] for field in json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))["fields"]]
    assert list(record) == fields
    assert (record["schema_version"], record["engine_id"], record["engine_model"]) == (
        1,
        "CFM56-104",
        "CFM56-7B",
    )
    assert (record["lifecycle_id"], record["cycle_number"]) == (3, 57)
    assert (record["telemetry_id"], record["frame_seq"]) == (987_654_321_012, 4321)
    assert record["frame_start_ts"] == epoch_micros(EVENT_TS) == 1_793_005_262_123_456
    assert (
        record["frame_duration_ms"],
        record["vibration_sample_rate_hz"],
        record["process_sample_rate_hz"],
    ) == (
        1000,
        2048,
        8,
    )
    assert record["n2_rpm"] == 14_412.5 and record["bpfo_order"] == 4.81 and record["n1_rpm"] > 5000


def test_waveforms_and_process_channels_are_packed_little_endian_float32() -> None:
    record = FrameBuilder(HfConfig(vibration_channels=4, frame_ms=500), seed=1).build(request())
    assert [item["channel"] for item in record["vibration"]] == [
        "fan_frame_vel",
        "core_trf_vel",
        "fan_frame_vel_b",
        "no1_bearing_vel",
    ]
    assert {item["unit"] for item in record["vibration"]} == {"mm/s"}
    assert all(len(item["samples_f32le"]) == 4 * 1024 for item in record["vibration"]), "2048 Hz x 0.5 s"
    assert [(item["parameter"], item["unit"]) for item in record["process"]] == list(PROCESS_CHANNELS)
    assert all(len(item["samples_f32le"]) == 4 * 4 for item in record["process"]), "8 Hz x 0.5 s"
    fan = decode(record["vibration"][0]["samples_f32le"]).astype(np.float64)
    assert np.sqrt(np.mean(fan**2)) == pytest.approx(2.0, rel=1e-4), "fan RMS = the pulse's vibration_mm_s"
    egt = {item["parameter"]: decode(item["samples_f32le"]).mean() for item in record["process"]}
    assert egt["EGT_A"] == pytest.approx(662.4, abs=8.0) and egt["N2_A"] == pytest.approx(14_412.5, abs=12.0)
    big_endian = np.array([1.5, -2.25], dtype=">f4")
    assert (
        pack_samples(big_endian)
        == np.array([1.5, -2.25], dtype="<f4").tobytes()
        == pack_samples([1.5, -2.25])
    )


def test_frame_start_is_the_exact_event_time_in_epoch_microseconds() -> None:
    assert epoch_micros(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert epoch_micros(datetime(1969, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)) == -1
    india = timezone(timedelta(hours=5, minutes=30))
    assert epoch_micros(EVENT_TS.astimezone(india)) == epoch_micros(EVENT_TS), (
        "the instant, not the wall time"
    )
    with pytest.raises(ValueError, match="naive timestamp"):
        epoch_micros(datetime(2026, 1, 1))


def test_full_frame_round_trips_through_the_repository_avro_schema() -> None:
    fastavro = pytest.importorskip("fastavro")
    schema = fastavro.parse_schema(json.loads(SCHEMA_FILE.read_text(encoding="utf-8")))
    for config in (CONFIG, HfConfig(sample_rate_hz=25_600, vibration_channels=16, process_rate_hz=16)):
        record = FrameBuilder(config, seed=7).build(request(mode="BEARING_DEGRADATION", wear=0.9))
        assert fastavro.validate(record, schema, strict=True)
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, schema, record)
        body = buffer.getvalue()
        decoded = fastavro.schemaless_reader(io.BytesIO(body), schema)
        assert decoded["frame_start_ts"] == EVENT_TS, "timestamp-micros keeps the microseconds"
        for name in (
            "engine_id",
            "engine_model",
            "lifecycle_id",
            "cycle_number",
            "telemetry_id",
            "frame_seq",
        ):
            assert decoded[name] == record[name]
        for name in ("n1_rpm", "n2_rpm", "bpfo_order"):
            assert decoded[name] == np.float32(record[name])
        assert decoded["vibration"] == record["vibration"] and decoded["process"] == record["process"]
        confluent = 5 + len(body)  # magic byte + schema id + Avro body
        assert 0 <= estimated_frame_bytes(config) - confluent <= HEADER_BYTES


def consumer_input(record: dict[str, Any], frame_bytes: int) -> dict[str, Any]:
    """The row hf_decode.feature_input builds for the streaming job's feature workers."""
    waveforms = {item["channel"]: item["samples_f32le"] for item in record["vibration"]}
    return {
        **{name: record[name] for name in ("engine_id", "engine_model", "lifecycle_id", "cycle_number")},
        **{name: record[name] for name in ("telemetry_id", "frame_seq", "frame_duration_ms")},
        "frame_start_us": record["frame_start_ts"],
        **{name: float(np.float32(record[name])) for name in ("n1_rpm", "n2_rpm", "bpfo_order")},
        "vibration_sample_rate_hz": record["vibration_sample_rate_hz"],
        "vibration_channels": len(record["vibration"]),
        "process_channels": len(record["process"]),
        "fan_samples": waveforms["fan_frame_vel"],
        "core_samples": waveforms["core_trf_vel"],
        "process_parameters": [item["parameter"] for item in record["process"]],
        "process_samples": [item["samples_f32le"] for item in record["process"]],
        "frame_bytes": frame_bytes,
        "kafka_partition": 0,
        "kafka_offset": 1,
        "decode_error": None,
    }


def test_the_streaming_feature_code_reads_the_intended_signatures_from_built_frames() -> None:
    pd = pytest.importorskip("pandas")
    hf_features = pytest.importorskip("hf_features")
    builder = FrameBuilder(CONFIG, seed=3)
    cases = {
        "healthy": request(mode="BEARING_DEGRADATION", wear=0.0, vibration=1.1),
        "spalled": request(mode="BEARING_DEGRADATION", wear=1.0, vibration=6.9),
        "eroded": request(mode="HPT_BLADE_EROSION", wear=1.0, vibration=6.9),
    }
    records = {name: builder.build(case) for name, case in cases.items()}
    rows = [consumer_input(record, estimated_frame_bytes(CONFIG)) for record in records.values()]
    table = hf_features.features_frame(pd.DataFrame(rows, columns=list(hf_features.FEATURE_INPUT_COLUMNS)))
    features = dict(zip(cases, table.to_dict("records"), strict=True))
    for name, row in features.items():
        means = row["process_means"]
        assert row["fan_rms"] == pytest.approx(cases[name].vibration_mm_s, rel=0.01)
        assert row["core_rms"] == pytest.approx(0.6 * cases[name].vibration_mm_s, rel=0.01)
        # The job's shaft lines are the tones the edge tracked; in a spalled bearing the broadband impact
        # energy within the line's +-2 bins adds a few percent.
        assert row["fan_amp_1x_n1"] == pytest.approx(np.sqrt(2) * means["VIB_N1_TRACKED_A"], rel=0.1)
        assert row["core_amp_1x_n2"] == pytest.approx(np.sqrt(2) * means["VIB_N2_TRACKED_A"], rel=0.1)
        assert {"OIL_T_A", "OIL_QTY_A", "OIL_DP_A", "EGT_A", "EGT_B"} <= set(means)
        assert row["frame_start_ts"] == pd.Timestamp(EVENT_TS) and row["telemetry_id"] == 987_654_321_012
    spalled, eroded, healthy = features["spalled"], features["eroded"], features["healthy"]
    # Single frames: the margins hold for any seed (BPFO ratios >= 5 and >= 160, kurtosis >= 4.2 vs <= 2.5,
    # high-band share >= 0.27 vs <= 0.06 over seeds 0..5).
    assert (
        spalled["fan_amp_bpfo"] > 3 * eroded["fan_amp_bpfo"]
        and spalled["fan_amp_bpfo"] > 50 * healthy["fan_amp_bpfo"]
    )
    assert spalled["fan_kurtosis"] > 3.5 > 3.0 > eroded["fan_kurtosis"]
    assert spalled["fan_band_energy_hf"] > 0.2 > 0.1 > eroded["fan_band_energy_hf"]
    assert spalled["process_means"]["OIL_DP_A"] > healthy["process_means"]["OIL_DP_A"] + 15
    assert eroded["egt_ab_split"] > healthy["egt_ab_split"] + 5, "the hot streak splits the EGT probes"


def test_frame_size_estimate_and_the_topic_message_limit() -> None:
    assert 15_000 < check_frame_size(CONFIG) < 25_000, "laptop default: about 20 KB per engine-second"
    full = HfConfig(sample_rate_hz=25_600, vibration_channels=16, process_rate_hz=16)
    assert 1_600_000 < check_frame_size(full) < 1_800_000 < TOPIC_MAX_MESSAGE_BYTES
    with pytest.raises(ValueError, match="above the topic limit"):
        check_frame_size(HfConfig(sample_rate_hz=102_400, vibration_channels=16, frame_ms=2000))


def pulse_at(engine_id: str, ts: datetime) -> eg.Pulse:
    return eg.EngineSimulator(engine_id, SimulatorConfig()).next_pulse(ts)


class FakeCursor:
    """Cursor of one transaction: every statement touches one row, RETURNING yields work order 55."""

    rowcount = 1

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append(sql)

    def fetchone(self) -> tuple[int]:
        return (55,)


@pytest.fixture
def fake_psycopg2(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[tuple], bool]]:
    """psycopg2.extras.execute_values that RETURNs (telemetry_id, engine_id, event_ts) in reverse order."""
    calls: list[tuple[str, list[tuple], bool]] = []

    def execute_values(cur: Any, sql: str, rows: list[tuple], page_size: int, fetch: bool) -> list[tuple]:
        calls.append((sql, rows, fetch))
        offset = 100 + 10 * (len(calls) - 1)
        india = timezone(timedelta(hours=5, minutes=30))  # the server's session time zone may differ
        returned = [(offset + i, row[0], row[3].astimezone(india)) for i, row in enumerate(rows)]
        return returned[::-1]  # PostgreSQL does not promise VALUES order

    extras = types.ModuleType("psycopg2.extras")
    extras.execute_values = execute_values  # type: ignore[attr-defined]
    package = types.ModuleType("psycopg2")
    package.extras = extras  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg2", package)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    return calls


def test_insert_returning_maps_every_pulse_to_its_generated_id(fake_psycopg2: list) -> None:
    writer = FleetWriter(FakeCursor())
    pulses = [
        pulse_at("JT9D-001", EVENT_TS),
        pulse_at("CFM56-104", EVENT_TS),
        pulse_at("JT9D-001", EVENT_TS + timedelta(seconds=1)),
    ]
    assert writer.insert_pulses(pulses) == [100, 101, 102]
    sql, rows, fetch = fake_psycopg2[0]
    assert sql.endswith("RETURNING telemetry_id, engine_id, event_ts") and fetch is True and len(rows) == 3
    failing = pulse_at("GE90-201", EVENT_TS)
    engine = eg.EngineSimulator("GE90-201", SimulatorConfig())
    assert writer.record_failure(failure_event(engine, failing, "live")) == RecordedFailure(55, 110)


def test_returned_rows_that_do_not_match_the_pulses_are_a_state_error() -> None:
    first, second = pulse_at("JT9D-001", EVENT_TS), pulse_at("JT9D-002", EVENT_TS)
    with pytest.raises(FleetStateError, match="returned 1 distinct rows"):
        ids_in_pulse_order([first, second], [(1, "JT9D-001", EVENT_TS)])
    with pytest.raises(FleetStateError, match="no row for pulse"):
        ids_in_pulse_order(
            [first, second], [(1, "JT9D-001", EVENT_TS), (2, "JT9D-002", EVENT_TS + timedelta(1))]
        )
    with pytest.raises(FleetStateError, match="distinct rows"):
        ids_in_pulse_order([first, second], [(1, "JT9D-001", EVENT_TS), (2, "JT9D-001", EVENT_TS)])


class RecordingPublisher:
    def __init__(self, fail: bool = False) -> None:
        self.live: list[FrameRequest] = []
        self.backfill: list[FrameRequest] = []
        self.fail = fail

    def submit(self, requests: list[FrameRequest]) -> None:
        if self.fail:
            raise RuntimeError("publisher bug")
        self.live.extend(requests)

    def submit_backfill(self, requests: list[FrameRequest]) -> None:
        self.backfill.extend(requests)


def test_the_tap_numbers_committed_pulses_per_engine_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    regular = [pulse_at("JT9D-001", EVENT_TS), pulse_at("CFM56-104", EVENT_TS)]
    failing = pulse_at("GE90-201", EVENT_TS)
    engine = eg.EngineSimulator("GE90-201", SimulatorConfig())
    ops = [("insert_pulses", regular), ("record_failure", failure_event(engine, failing, "live"))]
    results = [[7, 8], RecordedFailure(3, 9)]
    assert committed_pulses(ops, results) == [(regular[0], 7), (regular[1], 8), (failing, 9)]
    models = {"JT9D-001": "JT9D-7R4", "CFM56-104": "CFM56-7B", "GE90-201": "GE90-115B"}
    publisher = RecordingPublisher()
    tap = FrameTap(publisher, models, {"JT9D-001": 41, "CFM56-104": 0})
    tap.live(ops, results)
    tap.backfill(ops, results)
    assert [(r.engine_id, r.telemetry_id, r.frame_seq) for r in publisher.live] == [
        ("JT9D-001", 7, 42),
        ("CFM56-104", 8, 1),
        ("GE90-201", 9, 1),
    ]
    assert [r.frame_seq for r in publisher.backfill] == [43, 2, 2]
    assert publisher.live[2].engine_model == "GE90-115B" and publisher.live[2].frame_start == EVENT_TS
    broken = FrameTap(RecordingPublisher(fail=True), models, {})
    broken.live(ops, results)
    assert broken.errors == 1 and "HF frames of a live tick were lost" in caplog.text
    sequencer = FrameSequencer({"A": 5})
    assert [sequencer.next("A"), sequencer.next("B"), sequencer.next("A")] == [6, 1, 7]
