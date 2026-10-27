#!/bin/bash
# Topics as code: partition counts, retention and size limits are defined here
# instead of whatever broker auto-creation would pick. Idempotent: run by the
# `kafka-init` compose service on every `up` (configs are re-applied each time).
set -euo pipefail

bootstrap="${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}"
prefix="${CDC_TOPIC_PREFIX:-engine_cdc}"

# ensure_topic <name> <partitions> <config>...: create if missing, then enforce configs.
ensure_topic() {
    local topic="$1" partitions="$2"
    shift 2
    local create_args=() alter="" config
    for config in "$@"; do
        create_args+=(--config "${config}")
        alter="${alter:+${alter},}${config}"
    done
    kafka-topics --bootstrap-server "${bootstrap}" --create --if-not-exists \
        --topic "${topic}" --partitions "${partitions}" --replication-factor 1 "${create_args[@]}"
    kafka-configs --bootstrap-server "${bootstrap}" --alter --entity-type topics \
        --entity-name "${topic}" --add-config "${alter}" > /dev/null
    echo "create-topics.sh: topic ready -> ${topic} (${alter})"
}

# Debezium CDC topics: lossless operational changes, kept for a week.
for table in engine_status_registry maintenance_log engine_telemetry; do
    ensure_topic "${prefix}.public.${table}" "${CDC_TOPIC_PARTITIONS:-3}" \
        "retention.ms=${CDC_TOPIC_RETENTION_MS:-604800000}" cleanup.policy=delete
done

# High-frequency sensor frames: high volume, best effort, short retention; frames of
# kHz waveforms can be a few MB, so the message size limit is raised for this topic only.
# retention.bytes (per partition, enforced per 256 MiB segment) caps the disk the frames can
# take on the broker the CDC topics share: at the default fidelity two hours of frames stay far
# below it, at full fidelity it binds within minutes instead of the disk filling up.
ensure_topic "${HF_TOPIC:-engine_hf.sensor_frames}" "${HF_TOPIC_PARTITIONS:-6}" \
    "retention.ms=${HF_TOPIC_RETENTION_MS:-7200000}" \
    "retention.bytes=${HF_TOPIC_RETENTION_BYTES:-2147483648}" segment.bytes=268435456 \
    max.message.bytes=8388608 segment.ms=600000 cleanup.policy=delete

# Load-test topic: no consumer, ten minutes and at most 1 GiB per partition of retention.
ensure_topic "${HF_BENCHMARK_TOPIC:-engine_hf.benchmark}" "${HF_TOPIC_PARTITIONS:-6}" \
    retention.ms=600000 retention.bytes=1073741824 segment.bytes=268435456 \
    max.message.bytes=8388608 segment.ms=300000 cleanup.policy=delete

kafka-topics --bootstrap-server "${bootstrap}" --list
