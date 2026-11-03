"""make dr-drill (infrastructure/clickhouse/dr-drill.sh) against a fake docker.

The drill deletes raw rows of a running stack, so the checks that decide whether it deletes and
whether it passes are pinned here without one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DRILL = Path(__file__).resolve().parents[2] / "infrastructure" / "clickhouse" / "dr-drill.sh"
DAY = "2026-11-01"
ENDED = f"SELECT toDate('{DAY}') < toDate(now() - INTERVAL 1 HOUR, 'UTC')"
RUN = (
    "compose -f infrastructure/docker-compose.yml run --rm -T -e LOG_LEVEL=INFO orchestrator "
    "python orchestration/pipeline_runner.py task restore-raw --"
)
REPAIR = f'make restore-raw ARGS="--from {DAY} --to {DAY} --tables raw_engine_telemetry"'
BEFORE = f"{DAY}\t167415\t73104244710496"
BOUND = "1762041599123"

# Stands in for docker on PATH. Every SQL statement (whitespace squeezed) and restore-raw run is
# appended to $FAKE/calls. ClickHouse answers the day check from $FAKE/ended, the delete's
# ingested_at bound from $FAKE/bound and the n-th FINAL count from $FAKE/count.n (day, rows,
# version sum), without the sum unless the query asks for it; restore-raw prints $FAKE/plan for a
# dry run and otherwise exits with $FAKE/restore.
FAKE_DOCKER = r"""#!/usr/bin/env bash
set -eu
case " $* " in
    *" exec "*)
        sql="$(tr -s ' \n' ' ' | sed 's/ $//')"
        echo "${sql}" >>"${FAKE}/calls"
        case "${sql}" in
            *"INTERVAL 1 HOUR"*) cat "${FAKE}/ended" ;;
            *"max(ingested_at)"*) cat "${FAKE}/bound" ;;
            *"sum(cdc_lsn)"*) cat "${FAKE}/count.$(grep -c FINAL "${FAKE}/calls")" ;;
            *FINAL*) cut -f 1,2 "${FAKE}/count.$(grep -c FINAL "${FAKE}/calls")" ;;
        esac ;;
    *" ps "*) cat "${FAKE}/simulator" ;;
    *" run "*)
        echo "$*" >>"${FAKE}/calls"
        if [[ "$*" == *--dry-run ]]; then cat "${FAKE}/plan"; else exit "$(cat "${FAKE}/restore")"; fi ;;
esac
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="dr-drill.sh needs bash on PATH")


def plan(lake: int, raw: int, to_restore: int, table: str = "raw_engine_telemetry") -> str:
    """restore-raw --dry-run output as the pipeline runner relays it (test_ds_restore pins the format)."""
    relay = "2026-11-03T07:13:38Z INFO pipeline_process: [restore-raw] 2026-11-03T07:13:38.171Z INFO"
    totals = f"lake {lake}, raw {raw}, to restore {to_restore}"
    return (
        f"{relay} restore_from_lake: plan: {table} {DAY}: {totals}\n"
        f"{relay} restore_from_lake: plan: {table}, 1 day(s) of the lake as of 2026-11-03T07:13:37Z: "
        f"{totals}\n"
    )


def drill(tmp_path: Path, *args: str, counts: list[str], **answers: str) -> tuple[int, str, list[str]]:
    """Run the drill against the fake docker: its exit code, its stderr and the calls it made, in order."""
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    docker = fake / "bin" / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8", newline="\n")
    docker.chmod(0o755)
    files = {"ended": "1", "simulator": "", "plan": plan(167415, 167415, 0), "restore": "0", "calls": ""}
    files["bound"] = BOUND
    files |= answers | {f"count.{n}": answer for n, answer in enumerate(counts, start=1)}
    for name, text in files.items():
        (fake / name).write_text(text, encoding="utf-8", newline="\n")
    env = {**os.environ, "FAKE": str(fake), "PATH": f"{fake / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(DRILL), *args], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    return result.returncode, result.stderr, (fake / "calls").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("table", "column", "version"),
    [
        ("raw_engine_telemetry", "event_ts", ", sum(cdc_lsn)"),
        ("raw_engine_hf_features", "frame_start_ts", ""),
    ],
)
def test_a_drill_that_gets_every_row_back_counts_dry_runs_recounts_deletes_restores_and_passes(
    tmp_path: Path, table: str, column: str, version: str
) -> None:
    code, stderr, calls = drill(
        tmp_path, "--day", DAY, "--table", table, counts=[BEFORE] * 3, plan=plan(167415, 167415, 0, table)
    )

    window = f"toDate({column}) = '{DAY}'"
    count = f"SELECT toDate({column}) AS day, count(){version} FROM {table} FINAL WHERE {window} "
    count += "GROUP BY day ORDER BY day FORMAT TSV"
    restore = f"{RUN} --from {DAY} --to {DAY} --tables {table}"
    # The delete spares the rows ingested after its bound, which is read right before the recount.
    bound = f"SELECT toUnixTimestamp64Milli(max(ingested_at)) FROM {table} WHERE {window}"
    delete = f"ALTER TABLE {table} DELETE WHERE {window} AND toUnixTimestamp64Milli(ingested_at) <= {BOUND} "
    delete += "SETTINGS mutations_sync = 2"
    assert code == 0, stderr
    assert calls == [ENDED, count, f"{restore} --dry-run", bound, count, delete, restore, count]


@pytest.mark.parametrize(
    ("dry_run", "recount"),
    [
        pytest.param(plan(167414, 167415, 0), BEFORE, id="the lake lacks a row of raw"),
        pytest.param(plan(167415, 167416, 0), BEFORE, id="a row reached raw after the count"),
        pytest.param(plan(167415, 167415, 2), BEFORE, id="raw lacks rows of the lake or holds them older"),
        pytest.param("", BEFORE, id="no plan line"),
        pytest.param(plan(167415, 167415, 0), f"{DAY}\t167415\t73104244711496", id="a row changed meanwhile"),
    ],
)
def test_the_drill_deletes_nothing_unless_the_lake_and_raw_hold_the_counted_rows(
    tmp_path: Path, dry_run: str, recount: str
) -> None:
    code, stderr, calls = drill(tmp_path, "--day", DAY, counts=[BEFORE, recount], plan=dry_run)

    assert code == 1
    assert "nothing was deleted" in stderr
    assert not [call for call in calls if call.startswith("ALTER")]


@pytest.mark.parametrize(
    "after",
    [
        pytest.param(f"{DAY}\t167416\t73104244810496", id="a row arrived during the drill"),
        pytest.param(f"{DAY}\t167415\t73104244709496", id="an update came back older"),
    ],
)
def test_the_drill_fails_unless_every_row_comes_back_at_its_version(tmp_path: Path, after: str) -> None:
    code, stderr, calls = drill(tmp_path, "--day", DAY, counts=[BEFORE, BEFORE, after])

    assert code == 1
    assert [call for call in calls if call.startswith("ALTER")]
    assert REPAIR in stderr


def test_a_day_that_ended_less_than_an_hour_ago_is_not_drilled(tmp_path: Path) -> None:
    code, stderr, calls = drill(tmp_path, "--day", DAY, counts=[BEFORE], ended="0")

    assert (code, calls) == (1, [ENDED])
    assert "ended at least an hour ago" in stderr


def test_the_whole_table_is_not_drilled_while_the_simulator_runs(tmp_path: Path) -> None:
    code, stderr, calls = drill(tmp_path, counts=[BEFORE], simulator="4f2a9c1d")

    assert (code, calls) == (1, [])
    assert "(docker compose -f infrastructure/docker-compose.yml stop simulator)" in stderr
