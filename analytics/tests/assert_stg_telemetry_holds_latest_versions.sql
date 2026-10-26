{#-
    Deduplication check against the raw CDC log. For every pulse whose versions all landed before
    stg's watermark minus the lookback, stg_engine_telemetry FINAL must expose exactly the newest
    version, or nothing when that version is a delete or carries unusable readings. Younger
    pulses are skipped: raw keeps receiving data between `dbt run` and `dbt test`.
-#}

with settled_versions as (

    select
        telemetry_id,
        max(cdc_lsn) as latest_lsn,
        argMax(
            toUInt8(is_deleted = 1 or not {{ telemetry_readings_are_finite() }}),
            (cdc_lsn, ingested_at)
        ) as latest_is_retracted
    from {{ source('engine_raw', 'raw_engine_telemetry') }}
    group by telemetry_id
    having max(ingested_at) <= (
        select max(ingested_at) - toIntervalMinute({{ int_var('ingestion_lookback_minutes') }})
        from {{ ref('stg_engine_telemetry') }}
    )

),

published as (

    select telemetry_id, cdc_lsn as published_lsn, toUInt8(1) as is_published
    from {{ ref('stg_engine_telemetry') }} final

)

-- an unpublished pulse reads as is_published = 0 (ClickHouse join default)
select telemetry_id, latest_lsn, latest_is_retracted, published_lsn, is_published
from settled_versions
left join published using (telemetry_id)
where (latest_is_retracted = 0 and (is_published = 0 or published_lsn != latest_lsn))
   or (latest_is_retracted = 1 and is_published = 1)
