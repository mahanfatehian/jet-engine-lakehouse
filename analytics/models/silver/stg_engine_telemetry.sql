{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key='telemetry_id',
        on_schema_change='fail',
        indexes=[{'name': 'idx_ingested_at', 'definition': 'ingested_at TYPE minmax GRANULARITY 1'}],
        engine='ReplacingMergeTree(cdc_lsn, is_deleted)',
        order_by='(engine_id, telemetry_id)',
        partition_by='toYYYYMM(event_ts)',
    )
}}

{#-
    Cleaned CDC versions of engine_telemetry.

    Each run reads the raw versions ingested after the published watermark minus the lookback
    (late-visible inserts) and appends, per pulse, the newest one if it is newer than the
    version already published; re-read rows and replays are skipped. Late-arriving pulses (old
    event_ts, new ingested_at) are therefore picked up like any other row. The engine keeps the
    highest cdc_lsn per (engine_id, telemetry_id) and FINAL hides rows whose current version is
    retracted (source delete or unusable readings), so readers must use FINAL.
    Rows arrive roughly in ingestion order, so the minmax index on ingested_at prunes the
    watermark scan of the raw table to the newest granules.
    unique_key only makes dbt-clickhouse run its on_schema_change check before appending (it
    skips that check for key-less appends); uniqueness comes from the engine key under FINAL.
-#}

with source_versions as (

    select
        telemetry_id,
        engine_id,
        lifecycle_id,
        cycle_number,
        event_ts,
        rpm,
        core_temp_c,
        egt_c,
        vibration_mm_s,
        fuel_flow_kg_h,
        oil_pressure_psi,
        wear_factor,
        status as source_status,
        maintenance_id,
        created_ts,
        cdc_op,
        cdc_lsn,
        cdc_source_ts,
        ingested_at,
        is_deleted as is_source_delete
    from {{ source('engine_raw', 'raw_engine_telemetry') }}
    {%- if is_incremental() %}
    where ingested_at > {{ lookback_watermark('ingested_at') }}
    {%- endif %}

),

latest_versions as (

    -- replays and successive updates of one pulse can land in the same batch
    select *
    from source_versions
    order by engine_id, telemetry_id, cdc_lsn desc, ingested_at desc
    limit 1 by engine_id, telemetry_id

){% if is_incremental() %},

published_versions as (

    -- newest version already published for the pulses of this batch (primary-key lookup)
    select engine_id, telemetry_id, toNullable(max(cdc_lsn)) as published_lsn
    from {{ this }}
    where (engine_id, telemetry_id) in (select engine_id, telemetry_id from source_versions)
    group by engine_id, telemetry_id

){% endif %}

select
    telemetry_id,
    engine_id,
    lifecycle_id,
    cycle_number,
    event_ts,
    rpm,
    core_temp_c,
    egt_c,
    vibration_mm_s,
    fuel_flow_kg_h,
    oil_pressure_psi,
    wear_factor,
    cast(upper(trimBoth(source_status)) as LowCardinality(String)) as status,
    maintenance_id,
    created_ts,
    cdc_op,
    cdc_lsn,
    cdc_source_ts,
    ingested_at,
    toUInt8(is_source_delete = 1 or not {{ telemetry_readings_are_finite() }}) as is_deleted
from latest_versions
{%- if is_incremental() %}
left join published_versions using (engine_id, telemetry_id)
-- an older or already published version would lose against the published one anyway
where published_lsn is null or cdc_lsn > published_lsn
{%- endif %}
