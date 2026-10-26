#!/bin/sh
# Creates the lakehouse bucket, its medallion zones and the lifecycle rules (idempotent).
# S3 has no real directories, so each zone gets a small `.keep` marker object.
set -eu

bucket="${LAKEHOUSE_BUCKET:-lakehouse}"
hf_retention_days="${HF_BRONZE_RETENTION_DAYS:-1}"

mc alias set lake "${S3_ENDPOINT_URL:-http://minio:9000}" \
    "${MINIO_ROOT_USER:-minioadmin}" "${MINIO_ROOT_PASSWORD:-minioadmin}" > /dev/null
mc mb --ignore-existing "lake/${bucket}"

for zone in bronze silver gold ml-artifacts; do
    if ! mc stat "lake/${bucket}/${zone}/.keep" > /dev/null 2>&1; then
        echo "medallion zone '${zone}' (created by minio-init)" \
            | mc pipe "lake/${bucket}/${zone}/.keep" > /dev/null
    fi
    echo "init-buckets: s3://${bucket}/${zone}/ ready"
done

# Tiering: raw high-frequency frames are the bulk of the volume and only needed for
# replay/reprocessing, so they expire; features derived from them live on in silver.
# `mc ilm import` replaces the whole bucket lifecycle, which keeps reruns idempotent.
cat > /tmp/lakehouse-lifecycle.json <<EOF
{
  "Rules": [
    {
      "ID": "expire-raw-hf-sensor-frames",
      "Status": "Enabled",
      "Filter": {"Prefix": "bronze/engine_sensor_frames/"},
      "Expiration": {"Days": ${hf_retention_days}}
    }
  ]
}
EOF
mc ilm import "lake/${bucket}" < /tmp/lakehouse-lifecycle.json > /dev/null
echo "init-buckets: lifecycle -> bronze/engine_sensor_frames/ expires after ${hf_retention_days} day(s)"

mc ls "lake/${bucket}"
