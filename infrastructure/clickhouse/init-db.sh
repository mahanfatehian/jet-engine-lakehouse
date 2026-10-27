#!/bin/bash
# Applies init-db.sql to $CLICKHOUSE_DB on every start of the container.
#
# The clickhouse-server entrypoint runs (or sources) every *.sh file in
# /docker-entrypoint-initdb.d while a temporary server listens on 127.0.0.1:
# on an empty volume, and on every start because docker-compose.yml sets
# CLICKHOUSE_ALWAYS_RUN_INITDB_SCRIPTS. init-db.sql only creates what is
# missing (IF NOT EXISTS), so an existing stack gains the tables of a newer
# version and keeps its data.
# The SQL file is mounted outside that directory on purpose: the entrypoint
# would otherwise execute it without a database context. The work happens in
# a subshell so `set -euo pipefail` never leaks into the sourcing entrypoint.
(
    set -euo pipefail
    database="${CLICKHOUSE_DB:-iot_engine}"
    client=(clickhouse-client --host 127.0.0.1 --port 9000
            --user "${CLICKHOUSE_USER:-default}" --password "${CLICKHOUSE_PASSWORD:-}")

    echo "init-db.sh: applying /opt/clickhouse-init/init-db.sql to database '${database}'"
    "${client[@]}" --query "CREATE DATABASE IF NOT EXISTS \`${database}\`"
    "${client[@]}" --database "${database}" --multiquery < /opt/clickhouse-init/init-db.sql
    "${client[@]}" --query "SELECT name FROM system.tables WHERE database = '${database}' ORDER BY name"
)
