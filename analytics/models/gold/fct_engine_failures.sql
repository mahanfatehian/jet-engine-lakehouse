{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, failed_at)',
    )
}}

{#-
    One row per engine failure, from the SCD2 status history: when the engine failed, how long it
    had operated since its previous return to service (time between failures), how long it took to
    return to service (time to repair) and the failure mode of the work order.

    The history alternates between runs of available intervals (HEALTHY, WARNING, CRITICAL: in
    service) and runs of unavailable ones (FAILURE, MAINTENANCE_RESET). A failure is a run of
    unavailable intervals that contains a FAILURE interval: it fails at that interval's start and
    returns to service when the run ends, i.e. when the next available interval begins. The
    engine's first interval and a deregistration break runs, because what happened before them is
    unknown, unless the interval is a registration (entry_cdc_op = 'c'): an engine enters service
    when it is registered. So the time between failures is only measured for an operating run
    whose start is known (a return to service or a registration), and the time to repair stays
    NULL while the engine is still down or was deregistered before it returned. The operating time
    leaves out the run's time inside platform outages, reported as unobserved (cycles count all).

    The work order is the lifecycle's (maintenance_log is unique per engine lifecycle); its failure
    mode reads UNKNOWN until it has landed. Table, rebuilt on every run.
-#}

with intervals as (

    select
        engine_id,
        engine_model,
        operational_status,
        is_available,
        lifecycle_id,
        cycle_at_entry,
        valid_from,
        valid_to,
        is_current,
        closed_by_delete,
        entry_cdc_op
    from {{ ref('int_engine_status_history') }} final

),

compared as (

    select
        *,
        lagInFrame(toNullable(is_available)) over w_previous as previous_is_available,
        lagInFrame(toNullable(closed_by_delete)) over w_previous as previous_closed_by_delete
    from intervals
    window w_previous as (partition by engine_id order by valid_from rows between 1 preceding and current row)

),

continued as (

    -- 1 when the interval directly follows another one (not the engine's first, not after a delete)
    select
        *,
        toUInt8(ifNull(previous_closed_by_delete = 0, 0)) as follows_interval
    from compared

),

started as (

    -- 1 when the interval's start is a known change: it follows another interval or registers the engine
    select
        *,
        toUInt8(follows_interval = 1 or entry_cdc_op = 'c') as start_observed
    from continued

),

numbered as (

    select
        *,
        sum(toUInt64(follows_interval = 0 or assumeNotNull(previous_is_available) != is_available)) over (
            partition by engine_id order by valid_from rows between unbounded preceding and current row
        ) as run_number
    from started

),

runs as (

    select
        engine_id,
        run_number,
        argMin(engine_model, valid_from) as run_engine_model,
        any(is_available) as run_is_available,
        -- 1 when the run directly follows a run of the other kind
        argMin(follows_interval, valid_from) as run_follows_run,
        -- 1 when its start is known: it follows a run of the other kind or registers the engine
        argMin(start_observed, valid_from) as run_start_observed,
        min(valid_from) as run_start,
        if(max(is_current) = 1, null, max(valid_to)) as run_end,
        max(closed_by_delete) as run_closed_by_delete,
        argMin(lifecycle_id, valid_from) as run_entry_lifecycle_id,
        argMin(cycle_at_entry, valid_from) as run_entry_cycle,
        countIf(operational_status = 'FAILURE') as failure_intervals,
        minIf(valid_from, operational_status = 'FAILURE') as run_failed_at,
        argMinIf(lifecycle_id, valid_from, operational_status = 'FAILURE') as run_failure_lifecycle_id,
        argMinIf(cycle_at_entry, valid_from, operational_status = 'FAILURE') as run_failure_cycle
    from numbered
    group by engine_id, run_number

),

sequenced as (

    -- every run next to the run before it
    select
        *,
        lagInFrame(toNullable(run_start)) over w_previous_run as previous_run_start,
        lagInFrame(toNullable(run_start_observed)) over w_previous_run as previous_run_start_observed,
        lagInFrame(toNullable(run_entry_lifecycle_id)) over w_previous_run as previous_run_lifecycle_id,
        lagInFrame(toNullable(run_entry_cycle)) over w_previous_run as previous_run_entry_cycle
    from runs
    window w_previous_run as (
        partition by engine_id order by run_number rows between 1 preceding and current row
    )

),

failures as (

    select
        engine_id,
        run_engine_model as engine_model,
        run_failure_lifecycle_id as lifecycle_id,
        run_failure_cycle as failure_cycle,
        run_failed_at as failed_at,
        if(run_closed_by_delete = 1, null, run_end) as returned_to_service_at,
        -- the operating run right before the failure, if its start is known
        if(run_follows_run = 1 and previous_run_start_observed = 1, previous_run_start, null)
            as operating_since,
        -- its time inside platform outages (meaningless while its start is unknown: both columns
        -- derived from it below are NULL then)
        {{ outage_overlap_us(
            'toUnixTimestamp64Micro(operating_since)', 'toUnixTimestamp64Micro(failed_at)'
        ) }} as operating_unobserved_us,
        previous_run_lifecycle_id,
        previous_run_entry_cycle
    from sequenced
    where run_is_available = 0
      and failure_intervals > 0

),

work_orders as (

    -- the source enforces one work order per engine lifecycle; keep the newest defensively
    select
        engine_id,
        lifecycle_id,
        toNullable(maintenance_id) as work_order_id,
        toNullable(cast(failure_mode as String)) as work_order_failure_mode,
        downtime_seconds as work_order_downtime_seconds
    from {{ ref('stg_maintenance_log') }} final
    order by engine_id, lifecycle_id, maintenance_id desc
    limit 1 by engine_id, lifecycle_id

)

select
    engine_id,
    cast(engine_model as LowCardinality(String)) as engine_model,
    lifecycle_id,
    failure_cycle,
    failed_at,
    returned_to_service_at,
    {{ micros_between('failed_at', 'returned_to_service_at') }} / 1000000.0 as repair_seconds,
    operating_since,
    ({{ micros_between('operating_since', 'failed_at') }} - operating_unobserved_us) / 1000000.0
        as operating_seconds,
    if(isNull(operating_since), null, operating_unobserved_us / 1000000.0) as operating_unobserved_seconds,
    -- cycles flown in that run, the failure cycle included (like cycles_since_new)
    cast(
        if(
            isNotNull(operating_since)
            and previous_run_lifecycle_id = lifecycle_id
            and failure_cycle >= previous_run_entry_cycle,
            failure_cycle - previous_run_entry_cycle + 1,
            null
        ) as Nullable(UInt32)
    ) as operating_cycles,
    work_order_id as maintenance_id,
    cast(coalesce(work_order_failure_mode, 'UNKNOWN') as LowCardinality(String)) as failure_mode,
    work_order_downtime_seconds
from failures
left join work_orders using (engine_id, lifecycle_id)
