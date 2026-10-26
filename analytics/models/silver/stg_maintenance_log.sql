{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, maintenance_id)',
    )
}}

{#-
    Current version of every maintenance work order (highest cdc_lsn per maintenance_id),
    deleted work orders removed. Rebuilt on every run: one row per failure keeps it tiny.
-#}

with latest_versions as (

    select
        maintenance_id,
        engine_id,
        lifecycle_id,
        failure_cycle,
        failure_ts,
        failure_mode as source_failure_mode,
        maintenance_status as source_maintenance_status,
        action_taken,
        opened_ts,
        completed_ts,
        downtime_seconds,
        warning_rows_flagged,
        notes,
        cdc_op,
        cdc_lsn,
        ingested_at,
        is_deleted
    from {{ source('engine_raw', 'raw_maintenance_log') }}
    order by engine_id, maintenance_id, cdc_lsn desc, ingested_at desc
    limit 1 by engine_id, maintenance_id

)

select
    maintenance_id,
    engine_id,
    lifecycle_id,
    failure_cycle,
    failure_ts,
    cast(upper(trimBoth(source_failure_mode)) as LowCardinality(String)) as failure_mode,
    cast(upper(trimBoth(source_maintenance_status)) as LowCardinality(String)) as maintenance_status,
    action_taken,
    opened_ts,
    completed_ts,
    downtime_seconds,
    warning_rows_flagged,
    notes,
    cdc_op,
    cdc_lsn,
    ingested_at
from latest_versions
where is_deleted = 0
