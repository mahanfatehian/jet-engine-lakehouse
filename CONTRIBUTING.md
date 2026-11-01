# Contributing

Thank you for your interest in improving the Jet Engine Lakehouse platform. This
document explains how to get started, what standards to follow and how to submit
a change.

## Development setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mahanfatehian/jet-engine-lakehouse.git
   cd jet-engine-lakehouse
   ```

2. **Start the full stack:**

   ```bash
   cp infrastructure/.env.example infrastructure/.env   # optional
   make up
   ```

3. **Run the test suite:**

   ```bash
   make test   # unit tests across all three images
   make lint   # ruff check + ruff format --check
   ```

## Code style

* **Formatter and linter:** [Ruff](https://docs.astral.sh/ruff/) with the
  settings in `pyproject.toml` (line length 110, Python 3.11 target). Run
  `make lint` before pushing; CI rejects anything ruff reports.
* **Type hints:** add them to new functions and methods. Existing code uses
  `from __future__ import annotations` for deferred evaluation.
* **Docstrings:** every public class, function and module gets a docstring that
  explains *what* it does and *why*, not *how*.

## Branching model

* `main` is the default, always-deployable branch. CI runs on every push and
  pull request.
* Feature branches branch from `main` and merge back through a pull request.
  Keep commits atomic; squash fixups before merging.

## Pull request checklist

Use the pull request template (`.github/pull_request_template.md`). The
highlights:

- [ ] `make lint` passes
- [ ] `make test` passes, and new behaviour comes with unit tests
- [ ] dbt models, macros or tests changed → `bash tests/analytics/run_dbt_integration.sh` passes
- [ ] Stack changes → after `make up`, `make verify` reports every check OK
- [ ] Docs match the change (README, `docs/`, `.env.example`)

## Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/)
convention:

```
<type>(<scope>): <short summary>

Optional longer explanation.
```

**Types:** `feat`, `fix`, `docs`, `test`, `build`, `ci`, `refactor`, `perf`.

**Scopes:** `simulator`, `streaming`, `ds`, `orchestration`, `analytics`,
`infra`, `ml`.

## Testing

| Suite | Command | Image | Notes |
|---|---|---|---|
| Platform unit tests | `make test-platform` | data-platform | simulator physics, ML pipeline, registry, scheduler |
| Streaming tests | `make test-streaming` | streaming | CDC parsing, Avro, FFT, Spark (fails on any skip) |
| Simulator tests | `make test-simulator` | simulator | Avro round-trip against the real schema |
| dbt integration | `bash tests/analytics/run_dbt_integration.sh` | — | seeds synthetic data, checks incremental = full refresh |
| Lint | `make lint` | ruff container | ruff check + ruff format --check |

Every dbt cycle also runs 110 data tests; a failure blocks training and
snapshots.

## Reporting issues

Open a GitHub issue with:

1. **What you expected** and **what happened instead**.
2. **Steps to reproduce** (ideally a failing test or a `make verify` output).
3. **Environment:** OS, Docker version, Docker memory limit.

For security vulnerabilities, follow the process in
[SECURITY.md](SECURITY.md): do **not** open a public issue.

## Adding a new dbt model

1. Place the `.sql` file in `analytics/models/silver/` or
   `analytics/models/gold/`.
2. Add a YAML file with the schema contract and tests.
3. Run `bash tests/analytics/run_dbt_integration.sh` to verify.
4. Update `docs/architecture.md` if the model changes a data flow.

## Adding a new Python module

1. Place it in the appropriate service directory (`simulator/`, `streaming/`,
   `data_science/`, `orchestration/`).
2. Add unit tests in the matching `tests/<component>/` directory.
3. Update `requirements.txt` if a new dependency is needed.
4. Wire the module into the existing entry point or add a new one.

## License

This project is licensed under the MIT License. Your contributions will be
licensed under the same terms.
