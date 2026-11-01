"""Unit tests of the heartbeat file: content, uptime, atomic replacement, rate limit, failures, removal."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pytest
import streaming_heartbeat
from streaming_heartbeat import HeartbeatWriter, utc_now_iso

NOW_TEXT = "2026-10-26T13:00:00.000Z"


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_status_is_written_as_json_with_timestamp_pid_and_uptime(tmp_path: Path) -> None:
    path = tmp_path / "streaming.heartbeat"
    writer = HeartbeatWriter(path, wall_clock=lambda: NOW_TEXT)
    assert writer.publish({"status": "running", "queries": {"clickhouse_raw": {"active": True}}}) is True
    assert read(path) == {
        "timestamp": NOW_TEXT,
        "pid": os.getpid(),
        "uptime_s": 0.0,
        "status": "running",
        "queries": {"clickhouse_raw": {"active": True}},
    }
    assert [entry.name for entry in tmp_path.iterdir()] == ["streaming.heartbeat"]  # no temporary file left


def test_uptime_counts_from_the_first_publish_on_the_monotonic_clock(tmp_path: Path) -> None:
    path, clock = tmp_path / "streaming.heartbeat", FakeClock()
    writer = HeartbeatWriter(path, min_interval_s=5.0, clock=clock, wall_clock=lambda: NOW_TEXT)
    clock.now += 30.0  # creating the writer does not start the count
    writer.publish({"round": 1})
    clock.now += 4.0
    assert not writer.publish({"round": 2})  # rate limited: the file keeps the previous uptime
    assert read(path)["uptime_s"] == 0.0
    clock.now += 57.26
    writer.publish({"round": 3})
    assert read(path)["uptime_s"] == 61.3  # rounded to 0.1 s, whatever the wall clock says


def test_uptime_starts_with_the_first_publish_even_when_its_write_fails(tmp_path: Path) -> None:
    directory, clock = tmp_path / "not-yet", FakeClock()
    writer = HeartbeatWriter(directory / "streaming.heartbeat", clock=clock)
    assert writer.publish({"status": "starting"}) is False
    directory.mkdir()
    clock.now += 12.0
    assert writer.publish({"status": "running"}, force=True) is True
    assert read(directory / "streaming.heartbeat")["uptime_s"] == 12.0


def test_writes_are_rate_limited_unless_forced(tmp_path: Path) -> None:
    path, clock = tmp_path / "streaming.heartbeat", FakeClock()
    writer = HeartbeatWriter(path, min_interval_s=5.0, clock=clock)
    assert writer.publish({"round": 1})
    clock.now += 4.9
    assert not writer.publish({"round": 2})
    assert read(path)["round"] == 1
    assert writer.publish({"round": 3}, force=True)
    clock.now += 5.0
    assert writer.publish({"round": 4})
    assert read(path)["round"] == 4


def test_failed_write_keeps_the_previous_file_and_is_logged_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "streaming.heartbeat"
    writer = HeartbeatWriter(path, min_interval_s=0.0)
    writer.publish({"round": 1})

    def full_disk(_source: object, _target: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(streaming_heartbeat.os, "replace", full_disk)
    with caplog.at_level(logging.INFO):
        assert writer.publish({"round": 2}) is False
        assert writer.publish({"round": 3}) is False
        assert read(path)["round"] == 1  # readers still see the last complete file
        monkeypatch.undo()
        assert writer.publish({"round": 4}) is True
    assert caplog.text.count("cannot write heartbeat file") == 1
    assert "is writable again" in caplog.text
    assert read(path)["round"] == 4


def test_unwritable_location_never_raises(tmp_path: Path) -> None:
    writer = HeartbeatWriter(tmp_path / "missing-directory" / "streaming.heartbeat")
    assert writer.publish({"status": "starting"}) is False


def test_values_json_cannot_encode_are_written_as_text(tmp_path: Path) -> None:
    path = tmp_path / "streaming.heartbeat"
    HeartbeatWriter(path).publish({"status": "running", "detail": ValueError("odd")})
    assert read(path)["detail"] == "odd"


def test_utc_timestamps_are_iso_8601_with_milliseconds() -> None:
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", utc_now_iso())


def test_remove_deletes_the_file_and_a_leftover_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "streaming.heartbeat"
    writer = HeartbeatWriter(path)
    writer.publish({"status": "stopped"})
    (tmp_path / ".streaming.heartbeat.tmp").write_text("{", encoding="utf-8")  # a write cut short
    writer.remove()
    assert list(tmp_path.iterdir()) == []


def test_remove_tolerates_a_missing_file(tmp_path: Path) -> None:
    writer = HeartbeatWriter(tmp_path / "streaming.heartbeat")
    writer.remove()  # never written
    writer.publish({"status": "stopped"})
    writer.remove()
    writer.remove()  # already removed
    assert list(tmp_path.iterdir()) == []


def test_remove_never_raises_and_still_removes_what_it_can(tmp_path: Path) -> None:
    path = tmp_path / "streaming.heartbeat"
    path.mkdir()  # unlink() fails with an OSError (EISDIR or EPERM, depending on the platform)
    (tmp_path / ".streaming.heartbeat.tmp").write_text("{", encoding="utf-8")
    HeartbeatWriter(path).remove()
    assert [entry.name for entry in tmp_path.iterdir()] == ["streaming.heartbeat"]
