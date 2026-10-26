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

# infrastructure/.env is optional: compose reads it for the services, and the knobs
# that mirror compose variables (bucket, model prefix, bind address, host ports) read
# it too, with the same precedence: make command line > shell environment > .env >
# built-in default. The file is read with sed rather than `include`d: make would parse
# every line as make syntax, and an included value would also replace a same-named
# shell variable in the environment of every docker compose call (trailing comment
# padding included).
ENV_FILE := infrastructure/.env
HASH := \#
# $(call env_file,NAME): last assignment of NAME in $(ENV_FILE) without its inline
# comment, quotes and surrounding blanks; empty when the file or the key is missing.
env_file = $(strip $(subst ",,$(subst ',,$(shell [ -f $(ENV_FILE) ] && sed -n \
	's/^[[:space:]]*\(export[[:space:]][[:space:]]*\)\{0,1\}$(1)[[:space:]]*=//p' $(ENV_FILE) \
	| tr -d '\r' | sed 's/[[:space:]]$(HASH).*//' | tail -n 1))))
# $(call knob,NAME,DEFAULT): NAME from $(ENV_FILE), else DEFAULT.
knob = $(or $(call env_file,$(1)),$(2))

SERVICE ?=
TAIL ?= 200
ARGS ?=
CONNECTOR ?= engine-postgres-cdc
LAKEHOUSE_BUCKET ?= $(call knob,LAKEHOUSE_BUCKET,lakehouse)
ML_ARTIFACT_PREFIX ?= $(call knob,ML_ARTIFACT_PREFIX,ml-artifacts/rul_model)
BIND_HOST ?= $(call knob,BIND_HOST,127.0.0.1)
MINIO_API_PORT ?= $(call knob,MINIO_API_PORT,9000)
MINIO_CONSOLE_PORT ?= $(call knob,MINIO_CONSOLE_PORT,9001)
CLICKHOUSE_HTTP_PORT ?= $(call knob,CLICKHOUSE_HTTP_PORT,8123)
CLICKHOUSE_TCP_PORT ?= $(call knob,CLICKHOUSE_TCP_PORT,19000)
DEBEZIUM_PORT ?= $(call knob,DEBEZIUM_PORT,8083)
SPARK_UI_PORT ?= $(call knob,SPARK_UI_PORT,4040)
POSTGRES_PORT ?= $(call knob,POSTGRES_PORT,5432)
KAFKA_PORT ?= $(call knob,KAFKA_PORT,9092)
GRAFANA_PORT ?= $(call knob,GRAFANA_PORT,3000)
SCHEMA_REGISTRY_PORT ?= $(call knob,SCHEMA_REGISTRY_PORT,8081)
DATA_PLATFORM_IMAGE ?= iot-engine/data-platform:latest
STREAMING_IMAGE ?= iot-engine/streaming:latest
RUFF_IMAGE ?= ghcr.io/astral-sh/ruff:0.17.1
PYTEST_VERSION ?= 9.1.1
TEST_PATHS ?= tests
STREAMING_TEST_PATHS ?= tests/streaming
LINT_PATHS ?= .
CLEAN_GRACE_SECONDS ?= 5

# Host part of the printed URLs: the address the published ports bind to.
HOST_ADDRESS = $(if $(filter 0.0.0.0 ::,$(BIND_HOST)),localhost,$(BIND_HOST))

# Client commands that run INSIDE the containers: `$$VAR` reaches the container
# shell unexpanded, so credentials always come from the container's own env.
PSQL = psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"
CH_CLIENT = clickhouse-client --user "$$CLICKHOUSE_USER" --password "$$CLICKHOUSE_PASSWORD" \
	--database "$$CLICKHOUSE_DB"
# verify.sql keeps going after a failing statement (--ignore-error), so the client's
# exit status says nothing. awk streams the report through and sets the verdict: a
# statement error (`Code: N.` or `Syntax error` lines), a summary line starting with
# FAIL, or a report that never printed its last line fails the target.
VERIFY_AWK = { print; fflush() } \
	/^Code: [0-9]+\./ || /^Syntax error/ { errors++ } \
	/^FAIL / { fails++ } \
	/^== verify\.sql completed ==/ { done = 1 } \
	END { \
	if (errors) print \"verify-clickhouse: \" errors \" statement(s) failed, see the errors above\"; \
	if (fails) print \"verify-clickhouse: \" fails \" summary check(s) FAIL\"; \
	if (!done) print \"verify-clickhouse: the report stopped before its last statement\"; \
	exit (errors > 0 || fails > 0 || !done) }
CH_VERIFY = $(CH_CLIENT) --multiquery --ignore-error 2>&1 | awk "$(VERIFY_AWK)"
# Prints the status document; fails unless the connector and every task are RUNNING
# (the status endpoint answers HTTP 200 for a FAILED connector too).
CDC_STATUS = status=$$(curl -fsS http://localhost:8083/connectors/$(CONNECTOR)/status) || exit 1; \
	echo "$$status"; \
	states=$$(echo "$$status" | grep -o "\"state\":\"[A-Z_]*\"" | wc -l); \
	running=$$(echo "$$status" | grep -o "\"state\":\"RUNNING\"" | wc -l); \
	[ "$$states" -ge 2 ] && [ "$$states" -eq "$$running" ] \
	|| { echo "cdc-status: not RUNNING; make register-cdc restarts FAILED and resumes PAUSED connectors" >&2; \
	exit 1; }
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
# Manual jobs run through the pipeline runner in a one-off orchestrator container. It
# takes the same per-job locks as the scheduled loop (PIPELINE_LOCK_DIR on a volume both
# share), so a manual run waits for the loop's run of the same job instead of colliding.
RUN_TASK = $(COMPOSE) run --rm orchestrator python orchestration/pipeline_runner.py task
TASK_ARGS = $(if $(strip $(ARGS)),-- $(ARGS))
# pytest is installed into a throw-away directory only when the image lacks it.
PYTEST_BOOTSTRAP = python -m pytest --version >/dev/null 2>&1 \
	|| pip install --quiet --disable-pip-version-check --root-user-action=ignore --no-cache-dir --target /tmp/test-deps \
	"pytest==$(PYTEST_VERSION)"
# The streaming image is the only one with PySpark and a JVM, and its conftest skips the
# Spark tests when no SparkSession can start: any skip in this run is turned into a failure.
JUNIT_SKIPS = import sys, xml.etree.ElementTree as et; \
	print(sum(int(s.get(\"skipped\", 0)) for s in et.parse(sys.argv[1]).getroot().iter(\"testsuite\")))
STREAMING_PYTEST = python -m pytest -p no:cacheprovider -rs --junitxml=/tmp/streaming-tests.xml \
	$(STREAMING_TEST_PATHS) && skipped=$$(python -c "$(JUNIT_SKIPS)" /tmp/streaming-tests.xml) \
	&& if [ "$$skipped" -ne 0 ]; then \
	echo "test-streaming: $$skipped test(s) skipped; every Spark test must run in this image" >&2; exit 1; fi

.PHONY: help up down clean ps logs build urls register-cdc cdc-status stream dbt-run dbt-test \
	snapshot train-ml predict-ml pipeline-once verify verify-postgres verify-minio \
	verify-clickhouse ch-init psql ch-client test test-platform test-streaming lint

help: ## Show this help (default target)
	@echo "IoT Jet Engine Lakehouse - make targets (run from the repository root)"
	@echo
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo "Knobs (override on the command line, e.g. make logs SERVICE=simulator):"
	@printf '  %-44s %s\n' \
		'SERVICE=<name>' 'compose service for logs and build (default: all services)' \
		'TAIL=$(TAIL)' 'log lines shown before following (logs, stream)' \
		'ARGS="..."' 'extra arguments: dbt-run, dbt-test, snapshot, train-ml, predict-ml, register-cdc' \
		'CONNECTOR=$(CONNECTOR)' 'Debezium connector name (cdc-status, urls)' \
		'LAKEHOUSE_BUCKET=$(LAKEHOUSE_BUCKET)' 'bucket listed by verify-minio' \
		'ML_ARTIFACT_PREFIX=$(ML_ARTIFACT_PREFIX)' 'model registry prefix read by verify-minio' \
		'TEST_PATHS=$(TEST_PATHS)' 'pytest targets (test-platform)' \
		'STREAMING_TEST_PATHS=$(STREAMING_TEST_PATHS)' 'pytest targets (test-streaming)' \
		'LINT_PATHS=$(LINT_PATHS)' 'ruff targets (lint)' \
		'CLEAN_GRACE_SECONDS=$(CLEAN_GRACE_SECONDS)' 'abort window before clean deletes the volumes'
	@echo "LAKEHOUSE_BUCKET, ML_ARTIFACT_PREFIX, BIND_HOST and the *_PORT knobs default to the"
	@echo "values in infrastructure/.env, like the compose services."

up: ## Build the images and start the whole stack in the background (CDC registers itself)
	$(COMPOSE) up -d --build
	@echo "Stack starting: 'make ps' shows progress, 'make verify' checks the data flow end to end."
	@$(MAKE) --no-print-directory urls

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

urls: ## Print the host addresses of the web UIs and APIs (ports follow infrastructure/.env)
	@printf '  %-28s %s\n' \
		'Grafana dashboards' 'http://$(HOST_ADDRESS):$(GRAFANA_PORT)' \
		'MinIO console' 'http://$(HOST_ADDRESS):$(MINIO_CONSOLE_PORT)' \
		'Schema Registry' 'http://$(HOST_ADDRESS):$(SCHEMA_REGISTRY_PORT)/subjects' \
		'ClickHouse Play (web SQL)' 'http://$(HOST_ADDRESS):$(CLICKHOUSE_HTTP_PORT)/play' \
		'Spark UI' 'http://$(HOST_ADDRESS):$(SPARK_UI_PORT)' \
		'Debezium connector status' 'http://$(HOST_ADDRESS):$(DEBEZIUM_PORT)/connectors/$(CONNECTOR)/status' \
		'MinIO S3 API' 'http://$(HOST_ADDRESS):$(MINIO_API_PORT)' \
		'ClickHouse native protocol' '$(HOST_ADDRESS):$(CLICKHOUSE_TCP_PORT)' \
		'PostgreSQL' '$(HOST_ADDRESS):$(POSTGRES_PORT)' \
		'Kafka (host listener)' '$(HOST_ADDRESS):$(KAFKA_PORT)'

register-cdc: ## Register or update the CDC connector, restart it if FAILED, resume it if PAUSED, wait for RUNNING
	$(COMPOSE) run --rm cdc-register $(ARGS)

cdc-status: ## Show the state of the Debezium connector and its task (exit 1 unless all RUNNING)
	@echo "==> Debezium connector $(CONNECTOR)"
	@$(COMPOSE) exec -T debezium sh -c '$(CDC_STATUS)'

stream: ## Make sure the Spark streaming job runs, then follow its logs
	$(COMPOSE) up -d streaming
	$(COMPOSE) logs -f --tail=$(TAIL) streaming

dbt-run: ## Build and test every dbt model once (dbt build); waits for a running scheduled dbt job
	$(RUN_TASK) dbt-build $(TASK_ARGS)

dbt-test: ## Run the dbt tests once
	$(RUN_TASK) dbt-test $(TASK_ARGS)

snapshot: ## Export the silver/gold feature tables from ClickHouse to MinIO parquet
	$(RUN_TASK) snapshot $(TASK_ARGS)

train-ml: ## Train the RUL model now (promotion per MODEL_PROMOTION_POLICY)
	$(RUN_TASK) train $(TASK_ARGS)

predict-ml: ## Score the latest telemetry of every engine with the production model
	$(RUN_TASK) predict $(TASK_ARGS)

pipeline-once: ## Run dbt -> snapshot -> train -> predict once (failed dbt tests skip snapshot+train)
	$(COMPOSE) run --rm orchestrator python orchestration/pipeline_runner.py once

verify: ## End-to-end check of Postgres, CDC, MinIO and ClickHouse; exit 1 if any check fails
	@$(MAKE) --no-print-directory -k verify-postgres cdc-status verify-minio verify-clickhouse

verify-postgres: ## Source row counts, engine registry and the CDC replication slot
	@echo "==> Postgres: operational tables and the logical replication slot"
	@printf '%s\n' "$(PG_VERIFY_SQL)" | $(COMPOSE) exec -T postgres sh -c '$(PSQL) -X -v ON_ERROR_STOP=1'

verify-minio: ## Object counts and sizes per lakehouse zone/dataset, production model pointer
	@echo "==> MinIO: objects per zone and dataset in s3://$(LAKEHOUSE_BUCKET)"
	@$(COMPOSE) exec -T minio sh -c '$(MC_VERIFY)'

verify-clickhouse: ## Raw CDC tables, dbt models, model runs, latest RUL; exit 1 on FAIL or errors
	@echo "==> ClickHouse: infrastructure/clickhouse/verify.sql"
	@$(COMPOSE) exec -T clickhouse sh -c '$(CH_VERIFY)' < infrastructure/clickhouse/verify.sql

ch-init: ## Re-apply the idempotent ClickHouse schema (init-db.sql)
	$(COMPOSE) exec -T clickhouse bash /docker-entrypoint-initdb.d/01-init-db.sh

psql: ## Interactive psql session on the operational Postgres database
	$(COMPOSE) exec postgres sh -c '$(PSQL)'

ch-client: ## Interactive clickhouse-client session on the lakehouse database
	$(COMPOSE) exec clickhouse sh -c '$(CH_CLIENT)'

test: test-platform test-streaming ## Run every unit test: data-platform image, then the streaming image

test-platform: ## Unit tests in the data-platform image (the PySpark modules run in test-streaming)
	$(COMPOSE) build orchestrator
	docker run --rm -v "$(CURDIR):/src:ro" -w /src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh \
		$(DATA_PLATFORM_IMAGE) -c '($(PYTEST_BOOTSTRAP)) \
		&& PYTHONPATH=/tmp/test-deps python -m pytest -p no:cacheprovider -rs $(TEST_PATHS)'

test-streaming: ## tests/streaming incl. the PySpark tests in the streaming image; fails on any skip
	$(COMPOSE) build streaming
	docker run --rm -v "$(CURDIR):/src:ro" -w /src -e PYTHONDONTWRITEBYTECODE=1 --entrypoint sh \
		$(STREAMING_IMAGE) -c '$(STREAMING_PYTEST)'

lint: ## Lint (ruff check) and format-check (ruff format --check) the Python code
	docker run --rm -v "$(CURDIR):/src:ro" -w /src $(RUFF_IMAGE) check --no-cache $(LINT_PATHS)
	docker run --rm -v "$(CURDIR):/src:ro" -w /src $(RUFF_IMAGE) format --check --no-cache $(LINT_PATHS)
