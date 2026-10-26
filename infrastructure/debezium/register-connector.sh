#!/bin/sh
# Registers the Debezium Postgres connector with Kafka Connect (idempotent:
# HTTP 409 means it already exists) and waits until its task is RUNNING.
# Used by the `cdc-register` compose service and by `make register-cdc`.
set -eu

connect_url="${CONNECT_URL:-http://debezium:8083}"
config_file="${CONNECTOR_CONFIG:-/config/register-postgres-connector.json}"
name=$(sed -n 's/^ *"name": *"\([^"]*\)".*/\1/p' "$config_file" | head -n 1)

attempt=0
until curl -fsS "$connect_url/connector-plugins" > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "register-connector: Kafka Connect at $connect_url is not reachable" >&2
        exit 1
    fi
    sleep 2
done

http_status=$(curl -sS -o /tmp/register-response.json -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' --data @"$config_file" "$connect_url/connectors")
case "$http_status" in
    201) echo "register-connector: connector '$name' created" ;;
    409) echo "register-connector: connector '$name' already registered (unchanged)" ;;
    *)   echo "register-connector: registration failed with HTTP $http_status" >&2
         cat /tmp/register-response.json >&2
         exit 1 ;;
esac

# Surface connector/task failures here instead of letting CDC silently stall.
state=""
attempt=0
while [ "$attempt" -lt 45 ]; do
    state=$(curl -fs "$connect_url/connectors/$name/status" || true)
    tasks=${state#*\"tasks\"}
    case "$state" in
        *'"state":"FAILED"'*)
            echo "register-connector: connector failed: $state" >&2
            exit 1 ;;
    esac
    case "$tasks" in
        *'"state":"RUNNING"'*)
            echo "register-connector: $state"
            exit 0 ;;
    esac
    attempt=$((attempt + 1))
    sleep 2
done
echo "register-connector: task did not reach RUNNING in time: $state" >&2
exit 1
