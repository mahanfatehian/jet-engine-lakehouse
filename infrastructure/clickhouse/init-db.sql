-- =============================================================================
-- ClickHouse hot-path schema (Path B of the dual ingestion).
--
-- Table names are deliberately unqualified: init-db.sh applies this file with
-- `--database $CLICKHOUSE_DB` (default iot_engine). Every statement is
-- idempotent, so the file can be re-applied at any time (`make ch-init`).
--
--   raw_*                       CDC after-images landed by the Spark job
--   raw_cdc_dead_letter         CDC records that could not be parsed
--   gold_engine_rul_predictions RUL predictions written by data_science/predict_rul.py
--   ml_model_runs               training run registry written by train_rul_model.py
--   v_engine_rul_latest         latest prediction per engine
-- dbt builds the silver/gold models on top of the raw_* tables.
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw_engine_telemetry
(
    telemetry_id      UInt64,
    engine_id         LowCardinality(String),
    lifecycle_id      UInt32,
    cycle_number      UInt32,
    event_ts          DateTime64(6, 'UTC'),
    rpm               Float64,
    core_temp_c       Float64,
    egt_c             Float64,
    vibration_mm_s    Float64,
    fuel_flow_kg_h    Float64,
    oil_pressure_psi  Float64,
    wear_factor       Float64,
    status            LowCardinality(String),
    maintenance_id    Nullable(UInt64),
    created_ts        DateTime64(6, 'UTC'),
    cdc_op            LowCardinality(String),
    cdc_lsn           UInt64,
    cdc_source_ts     DateTime64(3, 'UTC'),
    cdc_emitted_ts    DateTime64(3, 'UTC'),
    is_deleted        UInt8 DEFAULT 0,
    kafka_partition   Int32,
    kafka_offset      Int64,
    ingested_at       DateTime64(3, 'UTC') DEFAULT now64(3),
    INDEX ix_event_ts event_ts TYPE minmax GRANULARITY 4
)
ENGINE = ReplacingMergeTree(cdc_lsn)
PARTITION BY toYYYYMM(event_ts)
ORDER BY (engine_id, telemetry_id)
COMMENT 'CDC images of engine_telemetry; the highest cdc_lsn per telemetry_id is the current row version';

CREATE TABLE IF NOT EXISTS raw_engine_status_registry
(
    engine_id             LowCardinality(String),
    engine_model          LowCardinality(String),
    operational_status    LowCardinality(String),
    current_lifecycle_id  UInt32,
    current_cycle         UInt32,
    wear_factor           Float64,
    cycles_since_new      UInt32,
    lifecycles_completed  UInt32,
    last_maintenance_ts   Nullable(DateTime64(6, 'UTC')),
    status_changed_ts     DateTime64(6, 'UTC'),
    last_telemetry_ts     Nullable(DateTime64(6, 'UTC')),
    updated_ts            DateTime64(6, 'UTC'),
    registered_ts         DateTime64(6, 'UTC'),
    cdc_op                LowCardinality(String),
    cdc_lsn               UInt64,
    cdc_source_ts         DateTime64(3, 'UTC'),
    cdc_emitted_ts        DateTime64(3, 'UTC'),
    is_deleted            UInt8 DEFAULT 0,
    kafka_partition       Int32,
    kafka_offset          Int64,
    ingested_at           DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (engine_id, cdc_lsn)
COMMENT 'Full change history of engine_status_registry (one row per CDC event, replays deduplicated)';

CREATE TABLE IF NOT EXISTS raw_maintenance_log
(
    maintenance_id        UInt64,
    engine_id             LowCardinality(String),
    lifecycle_id          UInt32,
    failure_cycle         UInt32,
    failure_ts            DateTime64(6, 'UTC'),
    failure_mode          LowCardinality(String),
    maintenance_status    LowCardinality(String),
    action_taken          Nullable(String),
    opened_ts             DateTime64(6, 'UTC'),
    completed_ts          Nullable(DateTime64(6, 'UTC')),
    downtime_seconds      Nullable(Int32),
    warning_rows_flagged  Nullable(Int32),
    notes                 Nullable(String),
    cdc_op                LowCardinality(String),
    cdc_lsn               UInt64,
    cdc_source_ts         DateTime64(3, 'UTC'),
    cdc_emitted_ts        DateTime64(3, 'UTC'),
    is_deleted            UInt8 DEFAULT 0,
    kafka_partition       Int32,
    kafka_offset          Int64,
    ingested_at           DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(cdc_lsn)
ORDER BY (engine_id, maintenance_id)
COMMENT 'CDC images of maintenance_log; the highest cdc_lsn per maintenance_id is the current version';

CREATE TABLE IF NOT EXISTS raw_cdc_dead_letter
(
    kafka_topic       LowCardinality(String),
    kafka_partition   Int32,
    kafka_offset      Int64,
    kafka_timestamp   DateTime64(3, 'UTC'),
    raw_key           String,
    raw_value         String,
    error             String,
    ingested_at       DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ingested_at)
ORDER BY (kafka_topic, kafka_partition, kafka_offset)
TTL toDateTime(ingested_at) + INTERVAL 30 DAY
COMMENT 'CDC records the streaming job could not parse; inspect and replay from Kafka/bronze';

CREATE TABLE IF NOT EXISTS gold_engine_rul_predictions
(
    predicted_at            DateTime64(3, 'UTC'),
    engine_id               LowCardinality(String),
    engine_model            LowCardinality(String),
    lifecycle_id            UInt32,
    cycle_number            UInt32,
    telemetry_id            UInt64,
    feature_event_ts        DateTime64(6, 'UTC'),
    operational_status      LowCardinality(String),
    rpm                     Float64,
    core_temp_c             Float64,
    egt_c                   Float64,
    vibration_mm_s          Float64,
    fuel_flow_kg_h          Float64,
    oil_pressure_psi        Float64,
    predicted_rul_cycles    Float64,
    predicted_rul_hours     Float64,
    failure_probability     Float64,
    failure_within_horizon  UInt8,
    horizon_cycles          UInt16,
    risk_band               LowCardinality(String),
    model_version           LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (engine_id, predicted_at)
TTL toDateTime(predicted_at) + INTERVAL 180 DAY
COMMENT 'Remaining-useful-life predictions with the sensor readings they were computed from';

CREATE TABLE IF NOT EXISTS ml_model_runs
(
    model_version     String,
    trained_at        DateTime64(3, 'UTC'),
    data_source       LowCardinality(String),
    train_rows        UInt64,
    test_rows         UInt64,
    train_lifecycles  UInt32,
    test_lifecycles   UInt32,
    rmse              Float64,
    mae               Float64,
    r2                Float64,
    nasa_score        Float64,
    clf_f1            Float64,
    clf_precision     Float64,
    clf_recall        Float64,
    clf_roc_auc       Float64,
    promoted          UInt8,
    artifact_uri      String,
    params_json       String
)
ENGINE = MergeTree
ORDER BY (trained_at, model_version)
COMMENT 'One row per training run of the RUL model (metrics, data volume, promotion decision)';

CREATE VIEW IF NOT EXISTS v_engine_rul_latest AS
SELECT *
FROM gold_engine_rul_predictions
ORDER BY predicted_at DESC
LIMIT 1 BY engine_id;

-- -----------------------------------------------------------------------------
-- High-frequency (HF) telemetry path: Spark reduces every Avro sensor frame
-- (kHz vibration waveforms + redundant process channels) to one feature row.
-- The raw frames themselves live only in the bronze lake (ILM-expired).
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw_engine_hf_features
(
    engine_id                 LowCardinality(String),
    engine_model              LowCardinality(String),
    lifecycle_id              UInt32,
    cycle_number              UInt32,
    telemetry_id              UInt64,
    frame_seq                 UInt64,
    frame_start_ts            DateTime64(6, 'UTC'),
    frame_duration_ms         UInt32,
    n1_rpm                    Float32,
    n2_rpm                    Float32,
    bpfo_order                Float32,
    vibration_sample_rate_hz  UInt32,
    vibration_channels        UInt16,
    process_channels          UInt16,
    fan_rms                   Float32,
    fan_peak                  Float32,
    fan_crest_factor          Float32,
    fan_kurtosis              Float32,
    fan_amp_1x_n1             Float32,
    fan_amp_1x_n2             Float32,
    fan_amp_bpfo              Float32,
    fan_band_energy_hf        Float32,
    fan_spectral_centroid_hz  Float32,
    core_rms                  Float32,
    core_peak                 Float32,
    core_crest_factor         Float32,
    core_kurtosis             Float32,
    core_amp_1x_n1            Float32,
    core_amp_1x_n2            Float32,
    core_amp_bpfo             Float32,
    core_band_energy_hf       Float32,
    core_spectral_centroid_hz Float32,
    egt_ab_split              Float32,
    process_means             Map(LowCardinality(String), Float32),
    frame_bytes               UInt32,
    kafka_partition           Int32,
    kafka_offset              Int64,
    ingested_at               DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(frame_start_ts)
ORDER BY (engine_id, telemetry_id)
TTL toDateTime(frame_start_ts) + INTERVAL 90 DAY
COMMENT 'Spectral and statistical features of every HF sensor frame; telemetry_id joins the 1 Hz pulse';

CREATE TABLE IF NOT EXISTS hf_ingest_batches
(
    batch_id          UInt64,
    batch_started_at  DateTime64(3, 'UTC'),
    frames            UInt32,
    bytes             UInt64,
    engines           UInt16,
    min_frame_ts      DateTime64(6, 'UTC'),
    max_frame_ts      DateTime64(6, 'UTC'),
    duration_ms       UInt32,
    ingested_at       DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(batch_started_at)
ORDER BY batch_started_at
TTL toDateTime(batch_started_at) + INTERVAL 30 DAY
COMMENT 'One row per HF micro-batch: frames, bytes and frame-time span, for throughput dashboards';
