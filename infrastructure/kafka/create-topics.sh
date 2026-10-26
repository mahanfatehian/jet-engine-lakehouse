#!/bin/bash
# Pre-creates the Debezium CDC topics so partition count and retention are
# defined as code instead of whatever broker auto-creation would pick.
# Idempotent: run by the `kafka-init` compose service on every `up`.
set -euo pipefail

bootstrap="${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}"
prefix="${CDC_TOPIC_PREFIX:-engine_cdc}"
partitions="${CDC_TOPIC_PARTITIONS:-3}"
retention_ms="${CDC_TOPIC_RETENTION_MS:-604800000}"

for table in engine_status_registry maintenance_log engine_telemetry; do
    topic="${prefix}.public.${table}"
    kafka-topics --bootstrap-server "${bootstrap}" --create --if-not-exists \
        --topic "${topic}" --partitions "${partitions}" --replication-factor 1 \
        --config "retention.ms=${retention_ms}" --config cleanup.policy=delete
    echo "create-topics.sh: topic ready -> ${topic}"
done

kafka-topics --bootstrap-server "${bootstrap}" --list
