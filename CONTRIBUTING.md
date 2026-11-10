# Contributing

Thank you for your interest in improving the Jet Engine Lakehouse. This guide covers the
development setup, the standards a change has to meet and how to submit it.

## Development setup

You need Docker with Compose v2 (give it about 10 GB of memory for the full stack) and GNU
`make`; on Windows, run the commands from Git Bash. Every command runs in a container, so no
local Python is needed.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mahanfatehian/jet-engine-lakehouse.git
   cd jet-engine-lakehouse
   ```

2. **Start the full stack** (every variable has a default, so the `.env` file is optional):

   ```bash
   cp infrastructure/.env.example infrastructure/.env
   make up
   make verify   # a few minutes later: end-to-end check of the data flow
   ```

3. **Run the checks:**

   ```bash
   make test   # unit tests in the data-platform, streaming and simulator images
   make lint   # ruff check + ruff format --check
   ```

## Code style

* **Linter and formatter:** [Ruff](https://docs.astral.sh/ruff/) 0.17.1 (`RUFF_IMAGE` in the
  `Makefile`, pinned in CI as well) with the settings in `pyproject.toml`: line length 110,
  Python 3.11 target. CI fails on anything ruff reports.
* **Type hints** on new functions and methods; every module has
  `from __future__ import annotations` after its docstring.
* **Docstrings** on every module, public class and function: what it does and *why*, not how.
  Comments explain the why as well.

## Branching model

* `main` is the default branch. CI runs on every push to `main` and on every pull request,
  and a nightly full-stack smoke test runs `make up`, `make verify` and, with the simulator
  stopped, `make dr-drill`.
* Branch from `main` and merge back through a pull request. Keep commits atomic and squash
  fixups before merging.

## Pull request checklist

The pull request template (`.github/pull_request_template.md`) asks for:

- [ ] `make lint` passes (ruff check and ruff format --check)
- [ ] `make typecheck` passes (ty)
- [ ] `make test` passes, and new behaviour comes with unit tests
- [ ] dbt models, macros or tests changed: `bash tests/analytics/run_dbt_integration.sh` passes
- [ ] Stack changes (compose, images, CDC, schemas): after `make up`, `make verify` reports every check OK
- [ ] Docs match the change (README, `docs/`, new variables in `infrastructure/.env.example`)

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

Optional longer explanation.
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.

**Scopes** name the component: `simulator`, `cdc`, `streaming`, `dbt`, `analytics` (the dbt test
harness in `tests/analytics/`), `ml`, `orchestration`, `grafana`, `infra`. Leave the scope out for changes across the repository (`docs: ...`).

## Testing

| Suite | Command | Runs in | Covers |
|---|---|---|---|
| Platform unit tests | `make test-platform` | data-platform image | `tests/` except what needs PySpark or fastavro: simulator physics and HF frames, ML pipeline, registry, scheduler |
| Streaming tests | `make test-streaming` | streaming image | CDC parsing, Avro decoding, FFT features, the query supervisor and the ClickHouse sinks with real Spark; fails on any skip |
| Simulator tests | `make test-simulator` | simulator image | the HF frames against the real Avro schema |
| dbt integration | `bash tests/analytics/run_dbt_integration.sh` | throwaway ClickHouse and dbt containers | synthetic CDC and HF data; the incremental models equal a full refresh |
| Lint | `make lint` | ruff image | ruff check + ruff format --check |
| Typecheck | `make typecheck` | ty 0.0.100 | simulator, streaming, data_science, orchestration |


`TEST_PATHS` and `STREAMING_TEST_PATHS` narrow a run, for example
`make test-streaming STREAMING_TEST_PATHS=tests/streaming/test_streaming_supervisor.py`.
Every orchestrated dbt cycle also runs the dbt data tests; a failure skips that cycle's
training and snapshot.

## Reporting issues

Open a GitHub issue; the bug report form asks for:

1. **What you expected** and **what happened instead**.
2. **Steps to reproduce**, ideally a failing test or the output of `make verify`.
3. **Environment:** OS, Docker version and the memory Docker may use.

Do **not** report a security vulnerability in a public issue: see [SECURITY.md](SECURITY.md).

## Adding a dbt model

1. Put the `.sql` file in `analytics/models/silver/` or `analytics/models/gold/`.
2. Document the model and its tests in a YAML file next to it.
3. Run `bash tests/analytics/run_dbt_integration.sh`.
4. Update `docs/architecture.md` if the model changes a data flow.

## Adding a Python module

1. Put it in its service directory: `simulator/`, `streaming/`, `data_science/` or
   `orchestration/`.
2. Add unit tests in the matching `tests/<component>/` directory.
3. Pin a new dependency exactly (`==`) in that image's `requirements.txt` (`simulator/`,
   `streaming/`, or `data_science/`, which also serves `orchestration/`). `simulator/` and
   `streaming/` also pin every transitive dependency: re-run the `uv pip compile` command in the
   file's header; `data_science/requirements.txt` lists only direct dependencies.
4. Wire the module into the existing entry point or add a new one.

## License

The project is licensed under the [MIT License](LICENSE); contributions are accepted under the
same terms.
