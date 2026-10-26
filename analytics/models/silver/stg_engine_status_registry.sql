{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key=['engine_id', 'cdc_lsn'],
        on_schema_change='fail',
        indexes=[{'name': 'idx_ingested_at', 'definition': 'ingested_at TYPE minmax GRANULARITY 1'}],
        engine='ReplacingMergeTree',
        order_by='(engine_id, cdc_lsn)',
    )
}}

{#-
    Status history of every engine: one row per CDC version of engine_status_registry, ordered
    by cdc_lsn. The current state of an engine is its row with the highest cdc_lsn (read with
    FINAL, then argMax / LIMIT 1 BY); is_deleted = 1 on that row means the engine was removed.
    Batches follow the same watermark-with-lookback rule as stg_engine_telemetry and only append
    changes that are not published yet. Rows arrive roughly in ingestion order, so the minmax
    index on ingested_at prunes the watermark scan of the raw table to the newest granules.
    unique_key only makes dbt-clickhouse run its on_schema_change check before appending (it
    skips that check for key-less appends); uniqueness comes from the engine key under FINAL.
-#}

with source_versions as (

    select
        engine_id,
        engine_model,
        operational_status as source_operational_status,
        current_lifecycle_id,
        current_cycle,
        wear_factor,
        cycles_since_new,
        lifecycles_completed,
        last_maintenance_ts,
        status_changed_ts,
        last_telemetry_ts,
        updated_ts,
        registered_ts,
        cdc_op,
        cdc_lsn,
        cdc_source_ts,
        ingested_at,
        is_deleted
    from {{ source('engine_raw', 'raw_engine_status_registry') }}
    {%- if is_incremental() %}
    where ingested_at > {{ lookback_watermark('ingested_at') }}
    {%- endif %}

),

deduplicated as (

    -- the same change can be delivered more than once (at-least-once streaming)
    select *
    from source_versions
    order by engine_id, cdc_lsn, ingested_at desc
    limit 1 by engine_id, cdc_lsn

)

select
    engine_id,
    engine_model,
    cast(upper(trimBoth(source_operational_status)) as LowCardinality(String)) as operational_status,
    current_lifecycle_id,
    current_cycle,
    wear_factor,
    cycles_since_new,
    lifecycles_completed,
    last_maintenance_ts,
    status_changed_ts,
    last_telemetry_ts,
    updated_ts,
    registered_ts,
    cdc_op,
    cdc_lsn,
    cdc_source_ts,
    ingested_at,
    is_deleted
from deduplicated
{%- if is_incremental() %}
-- re-read rows (lookback) and replays of published changes: a primary-key lookup on this model
where (engine_id, cdc_lsn) not in (
    select engine_id, cdc_lsn
    from {{ this }}
    where (engine_id, cdc_lsn) in (select engine_id, cdc_lsn from source_versions)
)
{%- endif %}
