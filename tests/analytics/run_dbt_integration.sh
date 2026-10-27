#!/usr/bin/env bash
# End-to-end check of the dbt analytics project against a throwaway ClickHouse 26.8.
#
#   1. private docker network + clickhouse-server with the platform's init files (no host ports)
#   2. python:3.12-slim-bookworm with the pinned dbt-core / dbt-clickhouse (repo mounted read-only,
#      dbt target and logs kept inside the container)
#   3. seed base -> dbt debug -> dbt build -> seed increment -> dbt build (incremental path);
#      the seed holds synthetic CDC (duplicates, updates, deletes, late rows) and the matching
#      high-frequency frame features (dropped frames, NaN features, absent process keys,
#      redelivered, late and corrected copies, bearing degradation)
#   4. invariant checks, snapshot, dbt build --full-refresh, snapshot, compare (every model,
#      incremental ones included), checks again
#   5. var-driven definitions: a changed baseline_cycles heals on the next incremental run (and
#      equals the full refresh again), the anomaly-threshold guard rejects miscalibrated
#      thresholds (sensor and bearing-defect flags), and the training mart keeps only the
#      newest lifecycles of each engine
#
# Everything it creates is removed on exit: the ${ANALYTICS_IT_PREFIX:-tmp-analytics}-* containers
# and network and ClickHouse's anonymous data volume (ANALYTICS_IT_KEEP=1 keeps them); set the
# prefix to run two copies side by side.
# Usage: bash tests/analytics/run_dbt_integration.sh      (Linux, macOS or Git Bash on Windows)
set -euo pipefail

# Git Bash would rewrite container paths such as /repo into Windows paths.
export MSYS_NO_PATHCONV=1

readonly PREFIX="${ANALYTICS_IT_PREFIX:-tmp-analytics}"
readonly NETWORK="${PREFIX}-net"
readonly CLICKHOUSE="${PREFIX}-ch"
readonly DBT="${PREFIX}-dbt"
readonly CLICKHOUSE_IMAGE=clickhouse/clickhouse-server:26.8
readonly PYTHON_IMAGE=python:3.12-slim-bookworm
readonly DBT_PACKAGES=(dbt-core==1.12.6 dbt-clickhouse==1.10.3)
readonly MANIFEST=/tmp/seed_manifest.json
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && { pwd -W 2>/dev/null || pwd; })"
readonly REPO_ROOT

log() { printf '%s [analytics-it] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

remove_resources() {
    # -v: clickhouse-server declares VOLUME /var/lib/clickhouse, whose anonymous volume would
    # otherwise outlive every run
    docker rm -f -v "${DBT}" "${CLICKHOUSE}" >/dev/null 2>&1 || true
    docker network rm "${NETWORK}" >/dev/null 2>&1 || true
}

cleanup() {
    if [[ "${ANALYTICS_IT_KEEP:-0}" == "1" ]]; then
        log "keeping ${CLICKHOUSE}, ${DBT} and ${NETWORK} (ANALYTICS_IT_KEEP=1)"
        return
    fi
    remove_resources
}

retry() {
    # bounded retries with exponential backoff for network-bound steps (apt, pip);
    # pip's download cache lets each attempt resume where the previous one failed
    local attempts=$1 delay=10 attempt
    shift
    for attempt in $(seq 1 "${attempts}"); do
        if "$@"; then
            return 0
        fi
        if ((attempt < attempts)); then
            log "attempt ${attempt}/${attempts} failed, retrying in ${delay}s: $*"
            sleep "${delay}"
            delay=$((delay * 2))
        fi
    done
    return 1
}

in_dbt() { docker exec "${DBT}" bash -c "cd /repo/analytics && $*"; }
verify() { docker exec "${DBT}" python /repo/tests/analytics/verify_analytics.py "$@"; }
seed() { docker exec "${DBT}" python /repo/tests/analytics/seed_engine_raw.py "$@"; }

expect_test_failure() {
    # $1 = singular test, $2 = dbt vars, $3 = regex one of its failing rows must match. dbt show
    # runs the test query and prints its rows as JSON (untruncated, one key per line: flattened
    # here), so the reason of the failure is checked, not just that the test fails.
    local output
    if ! output="$(in_dbt "dbt show --select $1 --vars '$2' --limit 50 --output json" 2>&1)"; then
        printf '%s\n' "${output}" >&2
        return 1
    fi
    if ! tr -s '[:space:]' ' ' <<<"${output}" | grep -Eq -- "$3"; then
        printf '%s\n' "${output}" >&2
        log "$1 with vars $2 did not report /$3/"
        return 1
    fi
    log "$1 with vars $2 reports /$3/ as expected"
}

wait_for_clickhouse() {
    # Same probe as the compose healthcheck: the first-boot init server only binds loopback,
    # so the container address answers once init-db.sql has been applied.
    local attempt
    for attempt in $(seq 1 60); do
        if docker exec "${CLICKHOUSE}" sh -c 'wget -q --spider "http://$(hostname):8123/ping"' 2>/dev/null; then
            log "ClickHouse ready after ${attempt} probe(s)"
            return 0
        fi
        sleep 2
    done
    log "ClickHouse did not become ready"
    docker logs --tail 50 "${CLICKHOUSE}" >&2
    return 1
}

start_containers() {
    # leftovers of an aborted or kept run carry the same reserved names
    remove_resources
    docker network create "${NETWORK}" >/dev/null
    docker run -d --name "${CLICKHOUSE}" --network "${NETWORK}" \
        -e CLICKHOUSE_USER=clickhouse -e CLICKHOUSE_PASSWORD=clickhouse -e CLICKHOUSE_DB=iot_engine \
        -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 --ulimit nofile=262144:262144 \
        -v "${REPO_ROOT}/infrastructure/clickhouse/init-db.sh:/docker-entrypoint-initdb.d/01-init-db.sh:ro" \
        -v "${REPO_ROOT}/infrastructure/clickhouse/init-db.sql:/opt/clickhouse-init/init-db.sql:ro" \
        "${CLICKHOUSE_IMAGE}" >/dev/null
    docker run -d --name "${DBT}" --network "${NETWORK}" -v "${REPO_ROOT}:/repo:ro" \
        -e CLICKHOUSE_HOST="${CLICKHOUSE}" -e DBT_PROFILES_DIR=/repo/analytics \
        -e DBT_TARGET_PATH=/tmp/dbt-target -e DBT_LOG_PATH=/tmp/dbt-logs \
        -e DBT_SEND_ANONYMOUS_USAGE_STATS=false -e PYTHONDONTWRITEBYTECODE=1 -e ANALYTICS_IT_ALLOW_WRITES=1 \
        "${PYTHON_IMAGE}" sleep infinity >/dev/null
    log "installing ${DBT_PACKAGES[*]} (+ git for dbt debug)"
    retry 3 docker exec "${DBT}" bash -c \
        "apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null"
    retry 4 docker exec "${DBT}" pip install --quiet --disable-pip-version-check --root-user-action=ignore \
        --retries 10 --default-timeout=120 "${DBT_PACKAGES[@]}"
    wait_for_clickhouse
}

check_definition_changes() {
    log "phase 4: baseline_cycles changed in a one-off run, healed by the next incremental run"
    in_dbt "dbt run --select int_sensor_rolling_features --vars '{baseline_cycles: 20}'"
    verify baseline 20
    expect_test_failure assert_features_use_current_baseline_cycles '{}' \
        '"built_with_baseline_cycles": *20.*dbt run --full-refresh -s int_sensor_rolling_features\+'
    in_dbt "dbt build --select int_sensor_rolling_features+"
    verify baseline 10
    verify snapshot healed
    verify compare full_refresh healed

    log "phase 5: the anomaly-threshold guard rejects thresholds that never fire or always fire"
    expect_test_failure assert_anomaly_thresholds_calibrated '{anomaly_oil_pressure_drop_psi: 8.0}' \
        '"flag": *"is_oil_pressure_drop_anomaly", *"issue": *"silent_at_failure"'
    expect_test_failure assert_anomaly_thresholds_calibrated '{anomaly_vibration_ratio: 1.0}' \
        '"flag": *"is_vibration_growth_anomaly", *"issue": *"noisy_while_healthy"'
    expect_test_failure assert_anomaly_thresholds_calibrated \
        '{anomaly_fan_amp_bpfo_mm_s: 9.0, anomaly_fan_kurtosis: 9.0}' \
        '"flag": *"is_bearing_defect_anomaly", *"issue": *"silent_at_failure"'
    expect_test_failure assert_anomaly_thresholds_calibrated '{anomaly_fan_amp_bpfo_mm_s: 0.01}' \
        '"flag": *"is_bearing_defect_anomaly", *"issue": *"noisy_while_healthy"'

    log "phase 6: the training mart keeps the newest completed lifecycles of each engine"
    in_dbt "dbt build --select mart_rul_training_features+ --vars '{mart_max_lifecycles_per_engine: 2}'"
    verify mart-cap 2
}

main() {
    trap cleanup EXIT
    start_containers

    log "phase 1: base history, first build"
    seed --phase base --manifest "${MANIFEST}"
    in_dbt "dbt debug"
    in_dbt "dbt build"

    log "phase 2: late, corrected and deleted rows; incremental build"
    seed --phase increment
    in_dbt "dbt build"
    verify check "${MANIFEST}" --incremental
    verify snapshot incremental

    log "phase 3: full refresh on the same data and comparison"
    in_dbt "dbt build --full-refresh"
    verify snapshot full_refresh
    verify compare incremental full_refresh
    verify check "${MANIFEST}"

    check_definition_changes
    log "all analytics checks passed"
}

main "$@"
