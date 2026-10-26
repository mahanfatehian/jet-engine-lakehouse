#!/bin/sh
# Creates the lakehouse bucket and its medallion zones (idempotent).
# S3 has no real directories, so each zone gets a small `.keep` marker object.
set -eu

bucket="${LAKEHOUSE_BUCKET:-lakehouse}"

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

mc ls "lake/${bucket}"
