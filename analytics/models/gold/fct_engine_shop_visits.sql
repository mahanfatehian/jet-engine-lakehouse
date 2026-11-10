{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, failed_at)',
    )
}}

{#-
    One row per shop visit: the same identity as fct_engine_failures (engine_id, failed_at),
    plus the operational WARNING/CRITICAL notice period of that lifecycle and the work-order
    clock. This is not the ML HIGH-risk lead (rpt_model_live_performance.median_warning_lead_cycles)
    and not time spent in WARNING (fct_engine_lifecycle_status_time.warning_seconds): it is the
    cycles and seconds from the first registry WARNING of the failed lifecycle to the FAILURE
    pulse, so a sudden death (no WARNING interval) is skipped_warning = 1. Table, rebuilt every
    run; still-open visits keep returned_to_service_at NULL like fct_engine_failures.
-#}

with lifecycle_stages as (

    select
        engine_id,
        lifecycle_id,
        if(
            countIf(operational_status = 'WARNING') = 0,
            null,
            minIf(valid_from, operational_status = 'WARNING')
        ) as first_warning_at,
        if(
            countIf(operational_status = 'WARNING') = 0,
            null,
            argMinIf(cycle_at_entry, valid_from, operational_status = 'WARNING')
        ) as first_warning_cycle,
        if(
            countIf(operational_status = 'CRITICAL') = 0,
            null,
            minIf(valid_from, operational_status = 'CRITICAL')
        ) as first_critical_at,
        if(
            countIf(operational_status = 'CRITICAL') = 0,
            null,
            argMinIf(cycle_at_entry, valid_from, operational_status = 'CRITICAL')
        ) as first_critical_cycle
    from {{ ref('int_engine_status_history') }} final
    group by engine_id, lifecycle_id

),

work_orders as (

    select
        engine_id,
        lifecycle_id,
        toNullable(maintenance_id) as work_order_id,
        toNullable(cast(failure_mode as String)) as work_order_failure_mode,
        toNullable(cast(maintenance_status as String)) as work_order_status,
        opened_ts as work_order_opened_at,
        completed_ts as work_order_completed_at
    from {{ ref('stg_maintenance_log') }} final
    order by engine_id, lifecycle_id, maintenance_id desc
    limit 1 by engine_id, lifecycle_id

)

select
    failures.engine_id as engine_id,
    failures.engine_model as engine_model,
    failures.lifecycle_id as lifecycle_id,
    failures.failure_cycle as failure_cycle,
    failures.failed_at as failed_at,
    stages.first_warning_at as first_warning_at,
    stages.first_warning_cycle as first_warning_cycle,
    stages.first_critical_at as first_critical_at,
    stages.first_critical_cycle as first_critical_cycle,
    if(
        isNull(stages.first_warning_cycle),
        null,
        toInt32(failures.failure_cycle) - toInt32(stages.first_warning_cycle)
    ) as warning_to_failure_cycles,
    {{ micros_between('stages.first_warning_at', 'failures.failed_at') }} / 1000000.0
        as warning_to_failure_seconds,
    if(
        isNull(stages.first_warning_cycle) or isNull(stages.first_critical_cycle),
        null,
        toInt32(stages.first_critical_cycle) - toInt32(stages.first_warning_cycle)
    ) as warning_to_critical_cycles,
    {{ micros_between('stages.first_warning_at', 'stages.first_critical_at') }} / 1000000.0
        as warning_to_critical_seconds,
    if(
        isNull(stages.first_critical_cycle),
        null,
        toInt32(failures.failure_cycle) - toInt32(stages.first_critical_cycle)
    ) as critical_to_failure_cycles,
    {{ micros_between('stages.first_critical_at', 'failures.failed_at') }} / 1000000.0
        as critical_to_failure_seconds,
    toUInt8(isNull(stages.first_warning_at)) as skipped_warning,
    toUInt8(isNull(stages.first_critical_at)) as skipped_critical,
    work_orders.work_order_opened_at as work_order_opened_at,
    work_orders.work_order_completed_at as work_order_completed_at,
    cast(work_orders.work_order_status as Nullable(LowCardinality(String))) as work_order_status,
    {{ micros_between('work_orders.work_order_opened_at', 'work_orders.work_order_completed_at') }}
        / 1000000.0 as shop_visit_seconds,
    failures.returned_to_service_at as returned_to_service_at,
    failures.repair_seconds as repair_seconds,
    work_orders.work_order_id as maintenance_id,
    cast(coalesce(work_orders.work_order_failure_mode, 'UNKNOWN') as LowCardinality(String))
        as failure_mode
from {{ ref('fct_engine_failures') }} as failures final
left join lifecycle_stages as stages
    on failures.engine_id = stages.engine_id
   and failures.lifecycle_id = stages.lifecycle_id
left join work_orders
    on failures.engine_id = work_orders.engine_id
   and failures.lifecycle_id = work_orders.lifecycle_id
