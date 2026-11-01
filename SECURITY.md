# Security Policy

## Supported versions

There are no releases: fixes land on `main`, and only the latest `main` is supported.

## Reporting a vulnerability

Please do **not** open a public issue. Report it privately through GitHub's private
vulnerability reporting:
<https://github.com/mahanfatehian/jet-engine-lakehouse/security/advisories/new> (the
repository's **Security** tab, **Report a vulnerability**).

Include the affected component (simulator, streaming, dbt, data science, orchestration or
infrastructure), the steps to reproduce and the impact you expect. This is a single-maintainer
project, so reports are handled on a best-effort basis; a confirmed vulnerability is fixed on
`main` and disclosed in a GitHub security advisory.

## Scope

In scope: the code and configuration in this repository, that is the simulator, the Spark
streaming job, the dbt project, the data-science and orchestration jobs, the Dockerfiles and
the compose stack with its service configuration (Postgres, Debezium, Kafka, Schema Registry,
MinIO, ClickHouse, Grafana).

Out of scope:

* The single-node defaults the README lists under *Production hardening notes*, such as the
  default credentials, a single Kafka broker and no TLS: they are documented trade-offs of a
  reference platform.
* Vulnerabilities in third-party images and packages themselves: report them upstream
  (Dependabot proposes updated pins here every week).
* Attacks that need physical access to the host.
