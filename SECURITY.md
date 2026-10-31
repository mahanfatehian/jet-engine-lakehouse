# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in the IoT Jet Engine Lakehouse
platform, please report it responsibly:

1. **Do not** open a public issue.
2. Send a detailed description to the repository owner via a private
   vulnerability report on GitHub (Security → Advisories → "Report a
   vulnerability").
3. Include reproduction steps, the affected component (simulator, streaming,
   data-science, orchestration, infrastructure) and the potential impact.

We aim to acknowledge reports within 48 hours and to issue a fix or mitigation
within 14 days for confirmed vulnerabilities.

## Scope

The following components are in scope:

* **Simulator** – engine telemetry generation and HF signal production.
* **Streaming** – Spark Structured Streaming, Kafka consumers, ClickHouse
  sinks.
* **Data Science** – RUL model training, inference, promotion pipeline.
* **Orchestration** – pipeline scheduler, task runner, inter-process locks.
* **Infrastructure** – Docker Compose stack, ClickHouse/Postgres/MinIO/Kafka
  configuration, Debezium CDC connectors.

## Out of Scope

* Issues that require physical access to the host machine.
* Denial-of-service attacks against third-party dependencies.
* Vulnerabilities in upstream base images that already have a published CVE
  and an available patch.

## Disclosure Timeline

* **Day 0** – report received and acknowledged.
* **Day 7** – initial triage and severity assessment shared with the reporter.
* **Day 14** – fix merged (or a mitigation deployed with a tracked follow-up).
* **Day 30** – public advisory published (coordinated with the reporter).
