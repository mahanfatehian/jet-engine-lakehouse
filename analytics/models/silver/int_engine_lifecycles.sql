{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, lifecycle_id)',
    )
}}

{#-
    One row per engine lifecycle. A lifecycle is completed (has_failed = 1) once its FAILURE
    pulse has landed; failure_cycle is the cycle of that pulse and stays NULL while the engine
    is still running. The failure mode comes from the lifecycle's maintenance work order, which
    can land after the FAILURE pulse (different CDC topic), hence the LEFT JOIN.
-#}

with lifecycle_telemetry as (

    select
        engine_id,
        lifecycle_id,
        min(event_ts) as lifecycle_start_ts,
        max(event_ts) as lifecycle_end_ts,
        min(cycle_number) as first_cycle,
        max(cycle_number) as last_cycle,
        toUInt8(countIf(status = 'FAILURE') > 0) as has_failed,
        minIf(cycle_number, status = 'FAILURE') as failure_pulse_cycle,
        minIf(event_ts, status = 'FAILURE') as failure_pulse_ts,
        count() as telemetry_rows
    from {{ ref('stg_engine_telemetry') }} final
    group by engine_id, lifecycle_id

),

work_orders as (

    -- the source enforces one work order per engine lifecycle; keep the newest defensively
    select
        engine_id,
        lifecycle_id,
        cast(maintenance_id as Nullable(UInt64)) as work_order_id,
        cast(failure_mode as Nullable(String)) as work_order_failure_mode,
        cast(maintenance_status as Nullable(String)) as work_order_status
    from {{ ref('stg_maintenance_log') }} final
    order by engine_id, lifecycle_id, maintenance_id desc
    limit 1 by engine_id, lifecycle_id

)

select
    engine_id,
    lifecycle_id,
    lifecycle_start_ts,
    lifecycle_end_ts,
    first_cycle,
    last_cycle,
    has_failed,
    cast(if(has_failed = 1, failure_pulse_cycle, null) as Nullable(UInt32)) as failure_cycle,
    cast(if(has_failed = 1, failure_pulse_ts, null) as Nullable(DateTime64(6, 'UTC'))) as failure_ts,
    toUInt32(if(has_failed = 1, failure_pulse_cycle, last_cycle) - first_cycle + 1) as lifecycle_length_cycles,
    telemetry_rows,
    work_order_id as maintenance_id,
    work_order_failure_mode as failure_mode,
    work_order_status as maintenance_status
from lifecycle_telemetry
left join work_orders using (engine_id, lifecycle_id)
