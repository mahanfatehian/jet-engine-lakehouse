# syntax=docker/dockerfile:1
# Platform job image used by the compose `orchestrator` service:
#   analytics/      dbt project (dbt-core + dbt-clickhouse)
#   data_science/   RUL feature snapshots, training and inference
#   orchestration/  pipeline runner (default command)
# Build context is the repository root:
#   docker build -f data_science/Dockerfile -t iot-engine/data-platform:latest .
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TZ=UTC \
    HOME=/home/app \
    DBT_PROJECT_DIR=/app/analytics \
    DBT_PROFILES_DIR=/app/analytics \
    DBT_SEND_ANONYMOUS_USAGE_STATS=false

# libgomp1 is the OpenMP runtime xgboost links against; git lets `dbt deps`
# resolve git-hosted packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code edits do not invalidate the (large) wheel layer.
COPY data_science/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app \
        --create-home --shell /usr/sbin/nologin app

# The job shares CPUs with Spark, ClickHouse and the simulator. Busy-waiting OpenMP
# workers (xgboost, scikit-learn) stall badly on a contended host, so idle workers
# sleep and model training is capped at two threads; both are overridable at run
# time. Declared after the dependency layers so tuning them never rebuilds those.
ENV OMP_WAIT_POLICY=PASSIVE \
    OMP_NUM_THREADS=2

WORKDIR /app
# Application code stays root-owned (read-only for the runtime user) ...
COPY data_science/ /app/data_science/
COPY orchestration/ /app/orchestration/
# ... except the dbt project: dbt writes target/, logs/, dbt_packages/ and
# package-lock.yml next to dbt_project.yml.
COPY --chown=app:app analytics/ /app/analytics/

USER app

CMD ["python", "orchestration/pipeline_runner.py", "loop"]
