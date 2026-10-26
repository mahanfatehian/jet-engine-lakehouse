# Edge telemetry simulator: pulses a jet-engine fleet into PostgreSQL (captured by Debezium CDC).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# psycopg2-binary and numpy ship manylinux wheels, so no compiler or libpq-dev is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin simulator

# Code stays root-owned and read-only for the unprivileged runtime user.
COPY engine_generator.py fleet_bootstrap.py fleet_runtime.py fleet_db.py ./

USER simulator

# python is PID 1 and handles SIGTERM itself: the running transaction completes, then exit 0.
CMD ["python", "engine_generator.py"]
