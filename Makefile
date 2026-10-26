# =============================================================================
# IoT Jet Engine Lakehouse & Predictive Maintenance - developer entry points
#
# The host needs only docker (with the compose plugin) and make: every command
# runs inside a container. Run `make` (or `make help`) from the repository root.
# Every knob below can be overridden on the command line, for example
#     make logs SERVICE=simulator
#     make train-ml ARGS="--promotion-policy always"
# =============================================================================

SHELL := /bin/sh
.DEFAULT_GOAL := help

# Git Bash/MSYS rewrites POSIX-looking arguments such as `-w /app/analytics` into
# Windows paths before they reach docker.exe; these switch that off (no-op elsewhere).
export MSYS_NO_PATHCONV := 1
export MSYS2_ARG_CONV_EXCL := *

COMPOSE := docker compose -f infrastructure/docker-compose.yml

SERVICE ?=
TAIL ?= 200
ARGS ?=
CONNECTOR ?= engine-postgres-cdc
LAKEHOUSE_BUCKET ?= lakehouse
ML_ARTIFACT_PREFIX ?= ml-artifacts/rul_model
DATA_PLATFORM_IMAGE ?= iot-engine/data-platform:latest
RUFF_IMAGE ?= ghcr.io/astral-sh/ruff:latest
PYTEST_VERSION ?= 9.1.1
TEST_PATHS ?= tests
LINT_PATHS ?= .
CLEAN_GRACE_SECONDS ?= 5

# Client commands that run INSIDE the containers: `$$VAR` reaches the container
# shell unexpanded, so credentials always come from the container's own env.
PSQL = psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"
CH_CLIENT = clickhouse-client --user "$$CLICKHOUSE_USER" --password "$$CLICKHOUSE_PASSWORD" \
	--database "$$CLICKHOUSE_DB"
MC_VERIFY = export MC_CONFIG_DIR=/tmp/.mc-verify \
	&& mc alias set verify http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null \
	&& mc du --depth 3 verify/$(LAKEHOUSE_BUCKET) \
	&& echo "--- s3://$(LAKEHOUSE_BUCKET)/$(ML_ARTIFACT_PREFIX)/production.json" \
	&& { mc cat verify/$(LAKEHOUSE_BUCKET)/$(ML_ARTIFACT_PREFIX)/production.json 2>/dev/null \
	|| echo "(no production model promoted yet)"; }
PG_VERIFY_SQL = \
	SELECT 'engine_status_registry' AS table_name, count(*) AS row_count FROM engine_status_registry \
	UNION ALL SELECT 'engine_telemetry', count(*) FROM engine_telemetry \
	UNION ALL SELECT 'maintenance_log', count(*) FROM maintenance_log; \
	SELECT engine_id, engine_model, operational_status, current_lifecycle_id, current_cycle, \
	lifecycles_completed, round(wear_factor::numeric, 3) AS wear_factor, last_telemetry_ts \
	FROM engine_status_registry ORDER BY engine_id; \
	SELECT maintenance_status, count(*) AS work_orders FROM maintenance_log \
	GROUP BY maintenance_status ORDER BY maintenance_status; \
	SELECT slot_name, plugin, active, confirmed_flush_lsn, \
	pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)) AS replication_lag \
	FROM pg_replication_slots;
# pytest is installed into a throw-away directory only when the image lacks it.
PYTEST_BOOTSTRAP = python -m pytest --version >/dev/null 2>&1 \
	|| pip install --quiet --disable-pip-version-check --root-user-action=ignore --no-cache-dir --target /tmp/test-deps \
	"pytest==$(PYTEST_VERSION)"

.PHONY: help up down clean ps logs build register-cdc cdc-status stream dbt-run dbt-test \
	snapshot train-ml predict-ml pipeline-once verify verify-postgres verify-minio \
	verify-clickhouse ch-init psql ch-client test lint

help: ## Show this help (default target)
	@echo "IoT Jet Engine Lakehouse - make targets (run from the repository root)"
	@echo
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo "Knobs (override on the command line, e.g. make logs SERVICE=simulator):"
	@printf '  %-32s %s\n' \
		'SERVICE=<name>' 'compose service for logs and build (default: all services)' \
		'TAIL=$(TAIL)' 'log lines shown before following (logs, stream)' \
		'ARGS="..."' 'extra arguments for dbt-run, dbt-test, snapshot, train-ml, predict-ml' \
		'CONNECTOR=$(CONNECTOR)' 'Debezium connector name (cdc-status, verify)' \
		'LAKEHOUSE_BUCKET=$(LAKEHOUSE_BUCKET)' 'bucket listed by verify-minio' \
		'TEST_PATHS=$(TEST_PATHS)' 'pytest targets (test)' \
		'LINT_PATHS=$(LINT_PATHS)' 'ruff targets (lint)' \
		'CLEAN_GRACE_SECONDS=$(CLEAN_GRACE_SECONDS)' 'abort window before clean deletes the volumes'

up: ## Build the images and start the whole stack in the background (CDC registers itself)
	$(COMPOSE) up -d --build
	@echo "Stack starting: 'make ps' shows progress, 'make verify' checks the data flow end to end."

down: ## Stop and remove the containers (data volumes are kept)
	$(COMPOSE) down

clean: ## DANGER: stop the stack and DELETE all data volumes (Postgres, Kafka, MinIO, ClickHouse)
	@echo "WARNING: 'make clean' permanently deletes every data volume of the stack:"
	@echo "         Postgres tables and WAL, Kafka topics, the MinIO lakehouse bucket,"
	@echo "         ClickHouse tables and the Spark checkpoints."
	@echo "         Press Ctrl+C within $(CLEAN_GRACE_SECONDS) seconds to abort."
	@sleep $(CLEAN_GRACE_SECONDS)
	$(COMPOSE) down -v --remove-orphans

ps: ## Show every service, including the finished one-shot init jobs
	$(COMPOSE) ps -a

logs: ## Follow the logs of all services, or of SERVICE=<name>
	$(COMPOSE) logs -f --tail=$(TAIL) $(SERVICE)

build: ## Build the application images (or only SERVICE=<name>)
	$(COMPOSE) build $(SERVICE)

register-cdc: ## (Re-)register the Debezium connector (idempotent) and wait until it runs
	$(COMPOSE) run --rm cdc-register

cdc-status: ## Show the state of the Debezium connector and its task
	@echo "==> Debezium connector $(CONNECTOR)"
	@$(COMPOSE) exec -T debezium curl -fsS http://localhost:8083/connectors/$(CONNECTOR)/status && echo

stream: ## Make sure the Spark streaming job runs, then follow its logs
	$(COMPOSE) up -d streaming
	$(COMPOSE) logs -f --tail=$(TAIL) streaming

dbt-run: ## Build and test every dbt model once (dbt build) in the orchestrator image
	$(COMPOSE) run --rm -w /app/analytics orchestrator dbt build --profiles-dir /app/analytics $(ARGS)

dbt-test: ## Run the dbt tests once
	$(COMPOSE) run --rm -w /app/analytics orchestrator dbt test --profiles-dir /app/analytics $(ARGS)

snapshot: ## Export the silver/gold feature tables from ClickHouse to MinIO parquet
	$(COMPOSE) run --rm orchestrator python data_science/feature_pipeline.py snapshot $(ARGS)

train-ml: ## Train the RUL model now (promotion per MODEL_PROMOTION_POLICY)
	$(COMPOSE) run --rm orchestrator python data_science/train_rul_model.py $(ARGS)

predict-ml: ## Score the latest telemetry of every engine with the production model
	$(COMPOSE) run --rm orchestrator python data_science/predict_rul.py --mode batch $(ARGS)

pipeline-once: ## Run dbt -> snapshot -> train -> predict once (exit 1 if any task failed)
	$(COMPOSE) run --rm orchestrator python orchestration/pipeline_runner.py once

verify: ## End-to-end check: Postgres, CDC connector, MinIO lakehouse, ClickHouse layers
	@$(MAKE) --no-print-directory -k verify-postgres cdc-status verify-minio verify-clickhouse

verify-postgres: ## Source row counts, engine registry and the CDC replication slot
	@echo "==> Postgres: operational tables and the logical replication slot"
	@printf '%s\n' "$(PG_VERIFY_SQL)" | $(COMPOSE) exec -T postgres sh -c '$(PSQL) -X -v ON_ERROR_STOP=1'

verify-minio: ## Object counts and sizes per lakehouse zone/dataset, production model pointer
	@echo "==> MinIO: objects per zone and dataset in s3://$(LAKEHOUSE_BUCKET)"
	@$(COMPOSE) exec -T minio sh -c '$(MC_VERIFY)'

verify-clickhouse: ## Raw CDC tables, dbt models, model runs and latest RUL (verify.sql)
	@echo "==> ClickHouse: infrastructure/clickhouse/verify.sql"
	@$(COMPOSE) exec -T clickhouse sh -c '$(CH_CLIENT) --multiquery --ignore-error' \
		< infrastructure/clickhouse/verify.sql

ch-init: ## Re-apply the idempotent ClickHouse schema (init-db.sql)
	$(COMPOSE) exec -T clickhouse bash /docker-entrypoint-initdb.d/01-init-db.sh

psql: ## Interactive psql session on the operational Postgres database
	$(COMPOSE) exec postgres sh -c '$(PSQL)'

ch-client: ## Interactive clickhouse-client session on the lakehouse database
	$(COMPOSE) exec clickhouse sh -c '$(CH_CLIENT)'

test: ## Run the unit tests inside the data-platform image (repository mounted read-only)
	$(COMPOSE) build orchestrator
	docker run --rm -v "$(CURDIR):/src:ro" -w /src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh \
		$(DATA_PLATFORM_IMAGE) -c '($(PYTEST_BOOTSTRAP)) \
		&& PYTHONPATH=/tmp/test-deps python -m pytest -p no:cacheprovider $(TEST_PATHS)'

lint: ## Lint (ruff check) and format-check (ruff format --check) the Python code
	docker run --rm -v "$(CURDIR):/src:ro" -w /src $(RUFF_IMAGE) check --no-cache $(LINT_PATHS)
	docker run --rm -v "$(CURDIR):/src:ro" -w /src $(RUFF_IMAGE) format --check --no-cache $(LINT_PATHS)
