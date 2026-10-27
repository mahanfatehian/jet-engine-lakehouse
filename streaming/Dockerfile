# Spark Structured Streaming CDC job: Kafka (Debezium) -> MinIO lakehouse + ClickHouse.
# Build context: streaming/. The pyspark sdist and the connector jars are fetched at BUILD
# time with resumable, SHA-256 verified downloads; the job never resolves packages at runtime.

# ------------------------------------------------------------------ download helper
# curl inside RUN rather than `ADD <url>`: RUN traffic honours the daemon/Docker Desktop
# proxy like any container, and partial downloads resume after a stall instead of
# restarting (pip cannot resume, which breaks the ~320 MB pyspark sdist on slow links).
FROM python:3.11-slim-bookworm AS fetcher
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --chmod=0755 <<'SH' /usr/local/bin/fetch
#!/bin/sh
# fetch <sha256> <url>: download into the working directory, resuming after stalls or
# failures (bounded attempts), then verify the SHA-256 digest.
set -eu
file="${2##*/}"
attempt=1
until curl -fsSL --connect-timeout 20 --speed-limit 1024 --speed-time 60 -C - -o "$file" "$2"; do
    if [ "$attempt" -ge 10 ]; then
        echo "fetch: giving up on $2" >&2
        exit 1
    fi
    sleep $((attempt * 3))
    attempt=$((attempt + 1))
done
echo "$1  $file" | sha256sum -c -
SH
WORKDIR /downloads

# ------------------------------------------------------------------ connector jars
FROM fetcher AS connector-jars
ARG MAVEN=https://repo1.maven.org/maven2
# Kafka source of Spark 3.5.9 plus its runtime dependencies, at the versions of its POM.
RUN fetch fed393dcc3ef70611fe3d4f35ae78a8ea7c55301fd4cdd66c5dc97fcfaa5a00f \
        "${MAVEN}/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.9/spark-sql-kafka-0-10_2.12-3.5.9.jar" \
    && fetch e82612e6070438c5ea48b08c74f367a6d214a390bdf6dcf3ff316dca592ddda5 \
        "${MAVEN}/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.9/spark-token-provider-kafka-0-10_2.12-3.5.9.jar" \
    && fetch 9d0a9058c8ede79db6e54ae378ed16fe8d1bead2894635a4aa6aa0b7981668c6 \
        "${MAVEN}/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar" \
    && fetch ea0505ee7515e58b1ac0e686e4d1a5d9f7d808e251a61bc371aa0595b9963f83 \
        "${MAVEN}/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar"
# Avro (from_avro) for the high-frequency sensor frames. Its runtime dependencies (avro
# 1.11.x, xz) already ship with pyspark; the digest is that of the Maven Central artifact,
# whose published SHA-1 (19e60546b565dd068f907540f6318c05cac2367a) it was checked against.
RUN fetch 9dda7512c1052bcd3774cbe247572cd481d4481eb8a90b7c4865b23a12bbecf8 \
        "${MAVEN}/org/apache/spark/spark-avro_2.12/3.5.9/spark-avro_2.12-3.5.9.jar"
# S3A: hadoop-aws must match the hadoop-client-api bundled with pyspark (3.3.4, asserted below).
RUN fetch 53f9ae03c681a30a50aa17524bd9790ab596b28481858e54efd989a826ed3a4a \
        "${MAVEN}/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" \
    && fetch 873fe7cf495126619997bec21c44de5d992544aea7e632fdc77adb1a0915bae5 \
        "${MAVEN}/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"

# ------------------------------------------------------------------ pyspark sdist
FROM fetcher AS pyspark-dist
# Digest published by PyPI for pyspark-3.5.9.tar.gz (the only distribution of 3.5.9).
RUN fetch ea27adc39ddac9413b8951e45aa748cbed6c785971b81386efc41938f6243d93 \
    https://files.pythonhosted.org/packages/95/ce/81e53e729790556e3983e95de1a7d5df91a34adfcd34b5a5ab0e0c6e9b33/pyspark-3.5.9.tar.gz

# ------------------------------------------------------------------ runtime image
FROM python:3.11-slim-bookworm

ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# openjdk: the Spark driver JVM. procps: `ps` for Spark's launcher scripts.
# tini: PID 1 that forwards SIGTERM to the job and reaps the JVM launcher processes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps tini \
    && rm -rf /var/lib/apt/lists/*

# The verified local sdist satisfies the pyspark==3.5.9 pin; the small wheels come from PyPI.
# Bind mounts keep the ~320 MB sdist (and requirements.txt) out of the image layers.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,from=pyspark-dist,source=/downloads,target=/tmp/dist \
    pip install --no-cache-dir --retries 10 --timeout 120 \
        -r /tmp/requirements.txt /tmp/dist/pyspark-3.5.9.tar.gz

ARG PYSPARK_JARS=/usr/local/lib/python3.11/site-packages/pyspark/jars
RUN test -f "${PYSPARK_JARS}/hadoop-client-api-3.3.4.jar" \
    || { echo "pyspark no longer bundles Hadoop 3.3.4: realign hadoop-aws/aws-java-sdk-bundle" >&2; exit 1; }
COPY --from=connector-jars --chmod=0644 /downloads/ ${PYSPARK_JARS}/

# Build-time smoke test: the baked jars must resolve the Kafka source, from_avro and an S3A
# filesystem (initialising S3AFileSystem also links the AWS SDK bundle), and pandas/pyarrow
# must serve mapInPandas in a Python worker. No network access is needed.
RUN python - <<'PY' && rm -rf /tmp/*
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.avro.functions import from_avro, to_avro

spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.ui.enabled", "false")
    .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.hadoop.fs.s3a.access.key", "smoke")
    .config("spark.hadoop.fs.s3a.secret.key", "smoke")
    .getOrCreate()
)
jvm = spark.sparkContext._jvm
for name in (
    "org.apache.spark.sql.kafka010.KafkaSourceProvider",
    "org.apache.spark.kafka010.KafkaTokenUtil",
    "org.apache.kafka.clients.consumer.KafkaConsumer",
    "org.apache.commons.pool2.impl.GenericKeyedObjectPool",
    "com.amazonaws.services.s3.AmazonS3Client",
):
    jvm.java.lang.Class.forName(name)
fs = jvm.org.apache.hadoop.fs.FileSystem.get(
    jvm.java.net.URI.create("s3a://smoke-bucket/"), spark.sparkContext._jsc.hadoopConfiguration()
)
assert fs.getClass().getName() == "org.apache.hadoop.fs.s3a.S3AFileSystem", fs.getClass().getName()
spark.readStream.format("kafka").option("kafka.bootstrap.servers", "127.0.0.1:9").option("subscribe", "smoke").load()
# Avro round trip of one float32 sample, then its byte count measured in a pandas worker.
schema = '{"type": "record", "name": "Smoke", "fields": [{"name": "samples", "type": "bytes"}]}'
encoded = spark.range(1).select(to_avro(sf.struct(sf.lit(b"\x00\x00\x80\x3f").alias("samples")), schema).alias("v"))
decoded = encoded.select(from_avro("v", schema, {"mode": "PERMISSIVE"})["samples"].alias("samples"))
measured = decoded.mapInPandas(
    lambda frames: (frame.assign(n=frame["samples"].map(len)) for frame in frames), "samples binary, n long"
)
(row,) = measured.collect()
assert row["n"] == 4, row
spark.stop()
print("connector smoke test passed")
PY

RUN groupadd --system --gid 10001 spark \
    && useradd --system --uid 10001 --gid spark --home-dir /home/spark --create-home spark \
    && install -d -o spark -g spark -m 0750 /opt/spark-checkpoints

WORKDIR /app
COPY *.py ./
# PYTHONPATH: Spark's Python workers import the job's modules by name (hf_features in
# mapInPandas). MALLOC_ARENA_MAX: glibc otherwise grows up to 8 malloc arenas per core under
# the JVM's native allocations (Kafka fetches, zstd/snappy codecs); with 2, the job's resident
# memory during a 45k-frame burst of HF frames peaked at 3.6 instead of 4.4 GiB, at equal speed.
ENV PYTHONPATH=/app \
    MALLOC_ARENA_MAX=2

USER spark
EXPOSE 4040
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "spark_kafka_to_lakehouse.py"]
