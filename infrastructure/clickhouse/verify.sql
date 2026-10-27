-- =============================================================================
-- End-to-end verification of the ClickHouse side of the lakehouse.
--
-- `make verify` (or `make verify-clickhouse`) pipes this file into
--     clickhouse-client --database $CLICKHOUSE_DB --multiquery --ignore-error
-- inside the clickhouse container. --ignore-error keeps going after a failing
-- statement, so every section prints even on a half-built stack (dbt models do
-- not exist before the first dbt run: their UNKNOWN_TABLE errors appear inline).
-- The client's exit status ignores such errors, so the Makefile reads the output
-- instead: the target fails when any statement raised an error, when a summary
-- line starts with FAIL, or when the report never reached its last line. WARN
-- lines are expected while the stack warms up and never fail it.
--
-- Only contract-defined tables and columns are referenced (table names are
-- unqualified, like init-db.sql). dbt models are only count()ed and checked for
-- existence: their columns belong to the dbt project. `SETTINGS final = 1`
-- deduplicates replacing tables and is a no-op for views and plain MergeTree tables.
-- =============================================================================

SET default_format = 'PrettyCompactMonoBlock', output_format_pretty_row_numbers = 0;

SELECT '\n== 1. Tables and views in the current database ==' FORMAT TSVRaw;
SELECT name, engine, total_rows
FROM system.tables
WHERE database = currentDatabase()
ORDER BY name;

SELECT '\n== 2. Raw CDC landing tables: row versions, CDC op mix, engines ==' FORMAT TSVRaw;
SELECT
    table_name,
    count() AS row_versions,
    countIf(cdc_op = 'r') AS op_r_snapshot,
    countIf(cdc_op = 'c') AS op_c_insert,
    countIf(cdc_op = 'u') AS op_u_update,
    countIf(cdc_op = 'd') AS op_d_delete,
    uniqExact(engine_id) AS engines
FROM
(
    SELECT 'raw_engine_telemetry' AS table_name, cdc_op, engine_id FROM raw_engine_telemetry
    UNION ALL
    SELECT 'raw_engine_status_registry', cdc_op, engine_id FROM raw_engine_status_registry
    UNION ALL
    SELECT 'raw_maintenance_log', cdc_op, engine_id FROM raw_maintenance_log
)
GROUP BY table_name
ORDER BY table_name;

SELECT '\n== 3. Freshness and ingestion lag (source commit -> ClickHouse, rows ingested in the last 15 min) ==' FORMAT TSVRaw;
SELECT
    table_name,
    max(cdc_source_ts) AS last_source_commit,
    max(ingested_at) AS last_ingested_at,
    dateDiff('second', max(ingested_at), now64(3)) AS seconds_since_last_ingest,
    countIf(is_recent) AS rows_last_15m,
    round(quantileIf(0.5)(lag_ms, is_recent) / 1000, 3) AS lag_p50_s,
    round(quantileIf(0.95)(lag_ms, is_recent) / 1000, 3) AS lag_p95_s,
    round(maxIf(lag_ms, is_recent) / 1000, 3) AS lag_max_s
FROM
(
    SELECT
        table_name,
        cdc_source_ts,
        ingested_at,
        toUnixTimestamp64Milli(ingested_at) - toUnixTimestamp64Milli(cdc_source_ts) AS lag_ms,
        ingested_at >= now64(3) - INTERVAL 15 MINUTE AS is_recent
    FROM
    (
        SELECT 'raw_engine_telemetry' AS table_name, cdc_source_ts, ingested_at FROM raw_engine_telemetry
        UNION ALL
        SELECT 'raw_engine_status_registry', cdc_source_ts, ingested_at FROM raw_engine_status_registry
        UNION ALL
        SELECT 'raw_maintenance_log', cdc_source_ts, ingested_at FROM raw_maintenance_log
    )
)
GROUP BY table_name
ORDER BY table_name;

SELECT '\n== 4. Telemetry per engine (current row versions) ==' FORMAT TSVRaw;
SELECT
    engine_id,
    count() AS telemetry_rows,
    uniqExact(lifecycle_id) AS lifecycles,
    max(lifecycle_id) AS current_lifecycle_id,
    argMax(cycle_number, event_ts) AS current_cycle,
    countIf(status = 'FAILURE') AS failure_rows,
    min(event_ts) AS first_event_ts,
    max(event_ts) AS latest_event_ts,
    dateDiff('second', max(event_ts), now64(6)) AS seconds_since_latest_event
FROM raw_engine_telemetry FINAL
WHERE is_deleted = 0
GROUP BY engine_id
ORDER BY engine_id;

SELECT '\n== 5. Engine registry: latest CDC state per engine ==' FORMAT TSVRaw;
SELECT
    engine_id,
    argMax(engine_model, cdc_lsn) AS engine_model,
    argMax(operational_status, cdc_lsn) AS operational_status,
    argMax(current_lifecycle_id, cdc_lsn) AS current_lifecycle_id,
    argMax(current_cycle, cdc_lsn) AS current_cycle,
    argMax(lifecycles_completed, cdc_lsn) AS lifecycles_completed,
    round(argMax(wear_factor, cdc_lsn), 3) AS wear_factor,
    argMax(updated_ts, cdc_lsn) AS updated_ts,
    count() AS cdc_events
FROM raw_engine_status_registry
GROUP BY engine_id
ORDER BY engine_id;

SELECT '\n== 6. Maintenance work orders per failure mode (current versions) ==' FORMAT TSVRaw;
SELECT
    failure_mode,
    countIf(maintenance_status = 'OPEN') AS open_orders,
    countIf(maintenance_status = 'COMPLETED') AS completed_orders,
    round(avgIf(downtime_seconds, maintenance_status = 'COMPLETED'), 1) AS avg_downtime_s,
    max(failure_ts) AS latest_failure_ts
FROM raw_maintenance_log FINAL
WHERE is_deleted = 0
GROUP BY failure_mode
ORDER BY failure_mode;

SELECT '\n== 7. Dead-lettered CDC records ==' FORMAT TSVRaw;
SELECT
    count() AS dead_letter_records,
    uniqExact(kafka_topic) AS topics,
    maxOrNull(ingested_at) AS latest_dead_letter_at
FROM raw_cdc_dead_letter;
SELECT
    kafka_topic,
    left(error, 100) AS error_prefix,
    count() AS records,
    max(kafka_timestamp) AS latest_kafka_timestamp
FROM raw_cdc_dead_letter
GROUP BY kafka_topic, error_prefix
ORDER BY records DESC
LIMIT 10;

SELECT '\n== 7b. High-frequency (HF) frame features: landed frames and freshness (best effort) ==' FORMAT TSVRaw;
SELECT
    count() AS frame_copies,
    uniqExact(engine_id, telemetry_id) AS frames,
    uniqExact(engine_id) AS engines,
    maxOrNull(frame_start_ts) AS latest_frame_ts,
    maxOrNull(ingested_at) AS last_ingested_at,
    dateDiff('second', maxOrNull(ingested_at), now64(3)) AS seconds_since_last_ingest
FROM raw_engine_hf_features;
SELECT
    count() AS batches_last_15m,
    sum(frames) AS frames_last_15m,
    round(quantileOrNull(0.5)(duration_ms) / 1000, 3) AS batch_duration_p50_s,
    round(quantileOrNull(0.95)(toUnixTimestamp64Milli(ingested_at) - toUnixTimestamp64Milli(max_frame_ts)) / 1000, 3)
        AS newest_frame_lag_p95_s
FROM hf_ingest_batches
WHERE batch_started_at >= now64(3) - INTERVAL 15 MINUTE;

SELECT '\n== 8. dbt models: row counts (UNKNOWN_TABLE = not built by dbt yet) ==' FORMAT TSVRaw;
SELECT 'stg_engine_telemetry' AS model, count() AS row_count
FROM stg_engine_telemetry SETTINGS final = 1;
SELECT 'int_sensor_rolling_features' AS model, count() AS row_count
FROM int_sensor_rolling_features SETTINGS final = 1;
SELECT 'stg_engine_hf_features' AS model, count() AS row_count
FROM stg_engine_hf_features SETTINGS final = 1;
SELECT 'int_vibration_features' AS model, count() AS row_count
FROM int_vibration_features SETTINGS final = 1;
SELECT 'int_ml_features' AS model, count() AS row_count
FROM int_ml_features SETTINGS final = 1;
SELECT 'mart_rul_training_features' AS model, count() AS row_count
FROM mart_rul_training_features SETTINGS final = 1;
SELECT 'fct_engine_health_status' AS model, count() AS row_count
FROM fct_engine_health_status SETTINGS final = 1;
SELECT 'fct_rul_prediction_backtest' AS model, count() AS row_count
FROM fct_rul_prediction_backtest SETTINGS final = 1;
SELECT 'rpt_model_live_performance' AS model, count() AS row_count
FROM rpt_model_live_performance SETTINGS final = 1;

SELECT '\n== 9. RUL model training runs ==' FORMAT TSVRaw;
SELECT
    count() AS training_runs,
    countIf(promoted = 1) AS promoted_runs,
    maxOrNull(trained_at) AS last_trained_at
FROM ml_model_runs;
SELECT '-- latest training run' FORMAT TSVRaw;
SELECT
    model_version,
    trained_at,
    data_source,
    train_rows,
    test_rows,
    train_lifecycles,
    test_lifecycles,
    rmse,
    mae,
    r2,
    nasa_score,
    clf_f1,
    clf_precision,
    clf_recall,
    clf_roc_auc,
    promoted,
    artifact_uri,
    params_json
FROM ml_model_runs
ORDER BY trained_at DESC
LIMIT 1
FORMAT Vertical;

SELECT '\n== 10. RUL predictions ==' FORMAT TSVRaw;
SELECT
    count() AS prediction_rows,
    uniqExact(engine_id) AS engines,
    uniqExact(model_version) AS model_versions,
    maxOrNull(predicted_at) AS last_predicted_at,
    dateDiff('second', maxOrNull(predicted_at), now64(3)) AS seconds_since_last_prediction
FROM gold_engine_rul_predictions;
SELECT '-- latest prediction per engine (v_engine_rul_latest)' FORMAT TSVRaw;
SELECT
    engine_id,
    engine_model,
    predicted_at,
    lifecycle_id,
    cycle_number,
    operational_status,
    round(predicted_rul_cycles, 1) AS predicted_rul_cycles,
    round(predicted_rul_hours, 1) AS predicted_rul_hours,
    round(failure_probability, 3) AS failure_probability,
    risk_band,
    model_version
FROM v_engine_rul_latest
ORDER BY engine_id;
SELECT '-- sensor readings behind the latest prediction' FORMAT TSVRaw;
SELECT
    engine_id,
    telemetry_id,
    feature_event_ts,
    rpm,
    core_temp_c,
    egt_c,
    vibration_mm_s,
    fuel_flow_kg_h,
    oil_pressure_psi
FROM v_engine_rul_latest
ORDER BY engine_id;

SELECT '\n== 11. Summary (OK / WARN / FAIL; make verify fails on FAIL and on any error above) ==' FORMAT TSVRaw;
-- One line per check, status first, so the Makefile can match `^FAIL ` exactly.
SELECT concat(rightPad(status, 6, ' '), rightPad(check_name, 74, ' '), detail)
FROM
(
    SELECT
        1 AS ord,
        'CDC rows landed in raw_engine_telemetry' AS check_name,
        if(count() > 0, 'OK', 'FAIL') AS status,
        concat(toString(count()), ' row versions') AS detail
    FROM raw_engine_telemetry
    UNION ALL
    SELECT
        2,
        'telemetry ingested within the last 5 minutes',
        if(maxOrNull(ingested_at) >= now64(3) - INTERVAL 5 MINUTE, 'OK', 'WARN'),
        concat('last ingest ', ifNull(toString(maxOrNull(ingested_at)), 'never'))
    FROM raw_engine_telemetry
    UNION ALL
    SELECT
        3,
        'no dead-lettered CDC records',
        if(count() = 0, 'OK', 'WARN'),
        concat(toString(count()), ' records')
    FROM raw_cdc_dead_letter
    UNION ALL
    SELECT
        4,
        'completed lifecycles for training (MIN_TRAINING_LIFECYCLES default 6)',
        if(count() >= 6, 'OK', 'WARN'),
        concat(toString(count()), ' failures recorded')
    FROM raw_maintenance_log FINAL
    WHERE is_deleted = 0
    UNION ALL
    SELECT
        5,
        'a RUL model has been promoted',
        if(countIf(promoted = 1) > 0, 'OK', 'WARN'),
        concat(toString(countIf(promoted = 1)), ' promoted of ', toString(count()), ' runs')
    FROM ml_model_runs
    UNION ALL
    SELECT
        6,
        'RUL predictions within the last 10 minutes',
        if(maxOrNull(predicted_at) >= now64(3) - INTERVAL 10 MINUTE, 'OK', 'WARN'),
        concat('last prediction ', ifNull(toString(maxOrNull(predicted_at)), 'never'))
    FROM gold_engine_rul_predictions
    UNION ALL
    SELECT
        7,
        'dbt silver/gold models built',
        if(empty(missing), 'OK', 'FAIL'),
        if(empty(missing), 'all 7 checked models exist', concat('missing: ', arrayStringConcat(missing, ', ')))
    FROM
    (
        SELECT arrayFilter(
            model -> NOT has(built, model),
            ['stg_engine_telemetry', 'int_sensor_rolling_features', 'stg_engine_hf_features',
             'int_vibration_features', 'int_ml_features', 'mart_rul_training_features',
             'fct_engine_health_status']
        ) AS missing
        FROM (SELECT groupArray(name) AS built FROM system.tables WHERE database = currentDatabase())
    )
    UNION ALL
    SELECT
        8,
        'HF frame features ingested within the last 5 minutes (best effort)',
        if(maxOrNull(ingested_at) >= now64(3) - INTERVAL 5 MINUTE, 'OK', 'WARN'),
        concat('last frame ', ifNull(toString(maxOrNull(ingested_at)), 'never'))
    FROM raw_engine_hf_features
)
ORDER BY ord
FORMAT TSVRaw;

-- Must stay the last statement: the Makefile fails the report when this line is missing.
SELECT '== verify.sql completed ==' FORMAT TSVRaw;
