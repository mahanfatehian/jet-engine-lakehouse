# Edge telemetry simulator: pulses a jet-engine fleet into PostgreSQL (captured by Debezium CDC) and streams
# every committed pulse's high-frequency sensor frame to Kafka (Avro, Schema Registry). The frame schema is
# mounted at /schemas (HF_SCHEMA_PATH), not baked in: infrastructure/schemas is its single source of truth.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# psycopg2-binary, numpy and confluent-kafka (librdkafka bundled) ship manylinux wheels, so no compiler or
# libpq-dev is needed. Retries and a long timeout ride out a flaky network during the build.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin simulator

# Code stays root-owned and read-only for the unprivileged runtime user.
COPY engine_generator.py fleet_*.py hf_*.py ./

USER simulator

# python is PID 1 and handles SIGTERM itself: the running transaction completes, then exit 0.
CMD ["python", "engine_generator.py"]
