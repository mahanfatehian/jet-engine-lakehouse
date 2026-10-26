-- =============================================================================
-- Operational edge database (OLTP) for the jet-engine telemetry simulator.
--
-- Executed once by the postgres image on first boot (empty data directory).
-- Debezium streams every row change of the three tables below out of the WAL
-- (wal_level=logical, pgoutput plugin) into Kafka:
--   * engine_status_registry : one row per engine, UPDATEd on every cycle/status
--   * engine_telemetry       : INSERT per 1 Hz pulse, UPDATE at failure time
--   * maintenance_log        : INSERT on failure, UPDATE when the repair closes
-- =============================================================================

CREATE TABLE IF NOT EXISTS engine_status_registry (
    engine_id             VARCHAR(32)      PRIMARY KEY,
    engine_model          VARCHAR(32)      NOT NULL,
    operational_status    VARCHAR(24)      NOT NULL DEFAULT 'HEALTHY'
        CONSTRAINT chk_registry_status CHECK (operational_status IN
            ('HEALTHY', 'WARNING', 'CRITICAL', 'FAILURE', 'MAINTENANCE_RESET')),
    current_lifecycle_id  INTEGER          NOT NULL DEFAULT 1 CHECK (current_lifecycle_id >= 1),
    current_cycle         INTEGER          NOT NULL DEFAULT 0 CHECK (current_cycle >= 0),
    wear_factor           DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (wear_factor >= 0 AND wear_factor <= 1),
    cycles_since_new      INTEGER          NOT NULL DEFAULT 0 CHECK (cycles_since_new >= 0),
    lifecycles_completed  INTEGER          NOT NULL DEFAULT 0 CHECK (lifecycles_completed >= 0),
    last_maintenance_ts   TIMESTAMPTZ,
    status_changed_ts     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    last_telemetry_ts     TIMESTAMPTZ,
    updated_ts            TIMESTAMPTZ      NOT NULL DEFAULT now(),
    registered_ts         TIMESTAMPTZ      NOT NULL DEFAULT now()
);

COMMENT ON TABLE engine_status_registry IS
    'Current operational state per engine. wear_factor is the simulator ground truth w(t).';

CREATE TABLE IF NOT EXISTS maintenance_log (
    maintenance_id        BIGSERIAL        PRIMARY KEY,
    engine_id             VARCHAR(32)      NOT NULL REFERENCES engine_status_registry (engine_id),
    lifecycle_id          INTEGER          NOT NULL CHECK (lifecycle_id >= 1),
    failure_cycle         INTEGER          NOT NULL CHECK (failure_cycle >= 0),
    failure_ts            TIMESTAMPTZ      NOT NULL,
    failure_mode          VARCHAR(48)      NOT NULL,
    maintenance_status    VARCHAR(16)      NOT NULL DEFAULT 'OPEN'
        CONSTRAINT chk_maintenance_status CHECK (maintenance_status IN ('OPEN', 'COMPLETED')),
    action_taken          VARCHAR(64),
    opened_ts             TIMESTAMPTZ      NOT NULL DEFAULT now(),
    completed_ts          TIMESTAMPTZ,
    downtime_seconds      INTEGER          CHECK (downtime_seconds >= 0),
    warning_rows_flagged  INTEGER          CHECK (warning_rows_flagged >= 0),
    notes                 TEXT,
    -- exactly one failure event per engine lifecycle (guards against replays on restart)
    CONSTRAINT uq_maintenance_engine_lifecycle UNIQUE (engine_id, lifecycle_id)
);

COMMENT ON TABLE maintenance_log IS
    'One work order per engine failure; OPEN on failure, COMPLETED after the repair (MAINTENANCE_RESET).';

CREATE TABLE IF NOT EXISTS engine_telemetry (
    telemetry_id          BIGSERIAL        PRIMARY KEY,
    engine_id             VARCHAR(32)      NOT NULL REFERENCES engine_status_registry (engine_id),
    lifecycle_id          INTEGER          NOT NULL CHECK (lifecycle_id >= 1),
    cycle_number          INTEGER          NOT NULL CHECK (cycle_number >= 0),
    event_ts              TIMESTAMPTZ      NOT NULL,
    rpm                   DOUBLE PRECISION NOT NULL,
    core_temp_c           DOUBLE PRECISION NOT NULL,
    egt_c                 DOUBLE PRECISION NOT NULL,
    vibration_mm_s        DOUBLE PRECISION NOT NULL,
    fuel_flow_kg_h        DOUBLE PRECISION NOT NULL,
    oil_pressure_psi      DOUBLE PRECISION NOT NULL,
    wear_factor           DOUBLE PRECISION NOT NULL CHECK (wear_factor >= 0 AND wear_factor <= 1),
    status                VARCHAR(24)      NOT NULL
        CONSTRAINT chk_telemetry_status CHECK (status IN ('HEALTHY', 'WARNING', 'CRITICAL', 'FAILURE')),
    maintenance_id        BIGINT           REFERENCES maintenance_log (maintenance_id),
    created_ts            TIMESTAMPTZ      NOT NULL DEFAULT now()
);

COMMENT ON TABLE engine_telemetry IS
    '1 Hz sensor pulses. wear_factor is ground truth and must never be used as an ML feature.';
COMMENT ON COLUMN engine_telemetry.maintenance_id IS
    'Back-filled by an UPDATE at failure time on the WARNING/CRITICAL/FAILURE rows of the lifecycle.';

CREATE INDEX IF NOT EXISTS ix_telemetry_engine_lifecycle
    ON engine_telemetry (engine_id, lifecycle_id, cycle_number);
CREATE INDEX IF NOT EXISTS ix_telemetry_engine_event_ts
    ON engine_telemetry (engine_id, event_ts DESC);

-- -----------------------------------------------------------------------------
-- Change data capture: full before-images on UPDATE/DELETE, and an explicit
-- publication so the connector does not need to own the tables.
-- -----------------------------------------------------------------------------
ALTER TABLE engine_status_registry REPLICA IDENTITY FULL;
ALTER TABLE maintenance_log        REPLICA IDENTITY FULL;
ALTER TABLE engine_telemetry       REPLICA IDENTITY FULL;

CREATE PUBLICATION dbz_engine_publication
    FOR TABLE engine_status_registry, maintenance_log, engine_telemetry;
