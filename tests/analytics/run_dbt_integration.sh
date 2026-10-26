#!/usr/bin/env bash
# End-to-end check of the dbt analytics project against a throwaway ClickHouse 26.8.
#
#   1. private docker network + clickhouse-server with the platform's init files (no host ports)
#   2. python:3.12-slim-bookworm with the pinned dbt-core / dbt-clickhouse (repo mounted read-only,
#      dbt target and logs kept inside the container)
#   3. seed base -> dbt debug -> dbt build -> seed increment -> dbt build (incremental path)
#   4. invariant checks, snapshot, dbt build --full-refresh, snapshot, compare, checks again
#
# Everything it creates is named tmp-analytics-* and removed on exit (ANALYTICS_IT_KEEP=1 keeps it).
# Usage: bash tests/analytics/run_dbt_integration.sh      (Linux, macOS or Git Bash on Windows)
set -euo pipefail

# Git Bash would rewrite container paths such as /repo into Windows paths.
export MSYS_NO_PATHCONV=1

readonly NETWORK=tmp-analytics-net
readonly CLICKHOUSE=tmp-analytics-ch
readonly DBT=tmp-analytics-dbt
readonly CLICKHOUSE_IMAGE=clickhouse/clickhouse-server:26.8
readonly PYTHON_IMAGE=python:3.12-slim-bookworm
readonly DBT_PACKAGES=(dbt-core==1.12.6 dbt-clickhouse==1.10.3)
readonly MANIFEST=/tmp/seed_manifest.json
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && { pwd -W 2>/dev/null || pwd; })"
readonly REPO_ROOT

log() { printf '%s [analytics-it] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

remove_resources() {
    docker rm -f "${DBT}" "${CLICKHOUSE}" >/dev/null 2>&1 || true
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
    log "all analytics checks passed"
}

main "$@"
