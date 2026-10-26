## What and why

<!-- The change in one or two sentences and the reason for it. Link the issue, if there is one. -->

## Checklist

- [ ] `make lint` passes (ruff check and ruff format --check)
- [ ] `make test` passes, and new behaviour comes with unit tests
- [ ] dbt models, macros or tests changed: `bash tests/analytics/run_dbt_integration.sh` passes
- [ ] Stack changes (compose, images, CDC, schemas): after `make up`, `make verify` reports every check OK
- [ ] Docs match the change (README, `docs/`, new variables in `infrastructure/.env.example`)
