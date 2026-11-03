#!/usr/bin/env bash
# Disaster-recovery drill of the lake restore (make dr-drill): delete the rows of one raw table
# for a settled window, rebuild them with restore-raw and prove that every one came back.
#
#   bash infrastructure/clickhouse/dr-drill.sh [--day YYYY-MM-DD] [--table TABLE]
#
# TABLE is a raw table restore-raw rebuilds (default raw_engine_telemetry). The window is one UTC
# day of its event time that ended at least an hour ago or, without --day, the whole table, which
# is refused while the simulator runs (the nightly CI smoke test stops it and lets streaming drain).
# The drill counts the window's rows per day (FINAL: one row per key) and sums their CDC versions
# (cdc_lsn; the HF frames have none). It deletes them (synchronous ALTER TABLE ... DELETE) only when
# a dry run of the restore counts as many rows in the lake and in raw, none missing from raw or
# older there, and a second count finds nothing changed; the delete spares rows ingested after that
# count began, so a late row stays in raw and fails the drill instead of being lost. Then
# restore-raw must exit 0 (it recounts against the lake and reconciles with the source) and every
# day's rows and versions must be back.
# An update raw held before the lake committed it comes back older and fails the drill (make
# restore-raw brings it back once the lake has it). A dbt run during a raw_maintenance_log drill
# rebuilds stg_maintenance_log without the day's work orders until its next run: drill that table
# on a quiet stack. It runs on the host (bash) and drives the stack through docker compose like the
# make targets: SQL in the clickhouse container with its own credentials, restore-raw through the
# pipeline runner in a one-off orchestrator. Exit 0 when every row is back, 1 on any refusal or
# mismatch.
set -euo pipefail

# Git Bash would rewrite container paths in the docker arguments.
export MSYS_NO_PATHCONV=1
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

fail() { echo "dr-drill: $*" >&2; exit 1; }
compose() { docker compose -f infrastructure/docker-compose.yml "$@"; }
# SQL on stdin, run by the clickhouse container's client with that container's credentials.
ch() {
    compose exec -T clickhouse sh -c \
        'clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" -d "$CLICKHOUSE_DB"'
}
# make restore-raw: the pipeline runner's per-job lock, timeout and exit codes. Always at INFO,
# whatever LOG_LEVEL says, because the safety check reads the plan line the restore logs.
restore() {
    compose run --rm -T -e LOG_LEVEL=INFO orchestrator \
        python orchestration/pipeline_runner.py task restore-raw -- "$@"
}

table=raw_engine_telemetry
day=
while (($# >= 2)); do
    case "$1" in
        --day) day="$2" ;;
        --table) table="$2" ;;
        *) break ;;
    esac
    shift 2
done
(($# == 0)) || fail "usage: dr-drill.sh [--day YYYY-MM-DD] [--table TABLE]"
# The column whose UTC day partitions the table's rows in the lake (restore_from_lake.py), and the
# version the counts sum to notice a row that came back older.
case "${table}" in
    raw_engine_telemetry) column=event_ts version=cdc_lsn ;;
    raw_engine_status_registry) column=updated_ts version=cdc_lsn ;;
    raw_maintenance_log) column=failure_ts version=cdc_lsn ;;
    raw_engine_hf_features) column=frame_start_ts version= ;;
    *) fail "--table: '${table}' is not one of the raw tables restore-raw rebuilds" ;;
esac
if [[ -n "${day}" ]]; then
    [[ "${day}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail "--day takes a UTC day YYYY-MM-DD, got '${day}'"
    # Late rows of a day keep arriving shortly after it ends. ClickHouse's clock decides, which
    # also spares the differences between GNU and BSD date.
    ended="$(ch <<<"SELECT toDate('${day}') < toDate(now() - INTERVAL 1 HOUR, 'UTC')")"
    [[ "${ended}" == 1 ]] || fail "--day ${day}: drill a UTC day that ended at least an hour ago" \
        "(its late rows may still be on their way)"
    window="toDate(${column}) = '${day}'" scope="${table} on ${day}"
else
    simulator="$(compose ps --status running --quiet simulator)"
    [[ -z "${simulator}" ]] || fail "the whole table needs a settled stack: stop the simulator and let" \
        "streaming drain first (docker compose -f infrastructure/docker-compose.yml stop simulator)," \
        "or drill one past day with --day YYYY-MM-DD"
    window=1 scope="${table} (whole table)"
fi
# restore-raw's arguments, word-split where they are used (the day is validated above).
args="${day:+--from ${day} --to ${day} }--tables ${table}"
counts() {
    ch <<<"SELECT toDate(${column}) AS day, count()${version:+, sum(${version})} FROM ${table} FINAL
        WHERE ${window} GROUP BY day ORDER BY day FORMAT TSV"
}

before="$(counts)"
[[ -n "${before}" ]] || fail "${scope}: no rows to drill"
rows="$(awk '{ rows += $2 } END { print rows }' <<<"${before}")"
printf 'dr-drill: %s: %s row(s) (FINAL), per UTC day: rows%s\n%s\n' \
    "${scope}" "${rows}" "${version:+, sum(${version})}" "${before}"
if ! plan="$(restore ${args} --dry-run)"; then
    echo "${plan}"
    fail "the dry run of restore-raw did not succeed (above); nothing was deleted"
fi
echo "${plan}"
totals="s/.* plan: ${table}, .*: lake \([0-9]*\), raw \([0-9]*\), to restore \([0-9]*\)\$/\1 \2 \3/p"
read -r lake raw missing <<<"$(sed -n "${totals}" <<<"${plan}")"
[[ "${lake}" == "${rows}" && "${raw}" == "${rows}" && "${missing}" == 0 ]] || fail "${scope}: counted" \
    "${rows} row(s), the dry run ${lake:-?} in the lake and ${raw:-?} in raw with ${missing:-?} to restore:" \
    "not the same rows, so nothing was deleted (raw lacks rows of the lake: make restore-raw" \
    "ARGS=\"${args}\" first; the lake lacks rows of raw: let streaming drain or pick another day)"
# A row that reached raw after the dry run counted it would be deleted without the lake holding it.
# Rows ingested after this bound (server time, before the recount) are not deleted: the lake may
# not hold them yet, and a row or version that stays changes the after counts.
bound="$(ch <<<"SELECT toUnixTimestamp64Milli(max(ingested_at)) FROM ${table} WHERE ${window}")"
[[ "$(counts)" == "${before}" ]] || fail "${scope}: rows arrived or changed during the dry run;" \
    "nothing was deleted"

echo "dr-drill: deleting ${scope} (if the drill stops before its restore: make restore-raw ARGS=\"${args}\")"
ch <<<"ALTER TABLE ${table} DELETE WHERE ${window} AND toUnixTimestamp64Milli(ingested_at) <= ${bound}
    SETTINGS mutations_sync = 2" \
    || fail "the delete failed; make restore-raw ARGS=\"${args}\" brings back whatever it removed"
restore ${args} \
    || fail "restore-raw exited $? instead of 0 (above); once fixed: make restore-raw ARGS=\"${args}\""
after="$(counts)"
echo "dr-drill: ${scope} before and after the restore, per UTC day: rows${version:+, then sum(${version})}"
LC_ALL=C join -t $'\t' -a 1 -a 2 -e 0 -o "0,1.2,2.2${version:+,1.3,2.3}" \
    <(echo "${before}") <(grep . <<<"${after}")
[[ "${after}" == "${before}" ]] || fail "${scope}: the rows or versions after the restore differ from" \
    "before (above): rows arrived during the drill, or raw held updates the lake had not committed" \
    "yet (make restore-raw ARGS=\"${args}\" brings them back once it has)"
echo "dr-drill: OK, all ${rows} row(s) of ${scope} were deleted and restored from the lake"
