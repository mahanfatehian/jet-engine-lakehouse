{#-
    The status history and the telemetry must count the same failures. A FAILURE pulse and the
    registry's FAILURE update are written in one source transaction, but they travel through two
    CDC topics, so only failures older than ingestion_lookback_minutes before the newest data of
    both streams are compared. Within that window, every failure of fct_engine_failures is a
    lifecycle int_engine_lifecycles has seen fail, at the same cycle and the same instant, and
    every failed lifecycle since the start of the engine's status history is a failure of
    fct_engine_failures; failures before an engine's history began (the CDC connector was
    registered later) are outside it. fct_engine_lifecycle_status_time must count the same
    failures per lifecycle.
-#}

with settled as (

    select
        least(
            {{ status_history_as_of() }},
            (select max(lifecycle_end_ts) from {{ ref('int_engine_lifecycles') }} final)
        ) - toIntervalMinute({{ int_var('ingestion_lookback_minutes') }}) as settled_before

),

history_start as (

    select engine_id, min(valid_from) as history_from
    from {{ ref('int_engine_status_history') }} final
    group by engine_id

),

history_failures as (

    select
        engine_id,
        lifecycle_id,
        failure_cycle as history_failure_cycle,
        failed_at,
        toUInt8(1) as in_history
    from {{ ref('fct_engine_failures') }} final
    where failed_at <= (select settled_before from settled)

),

telemetry_failures as (

    select
        engine_id,
        lifecycle_id,
        assumeNotNull(failure_cycle) as telemetry_failure_cycle,
        assumeNotNull(failure_ts) as failure_ts,
        toUInt8(1) as in_telemetry
    from {{ ref('int_engine_lifecycles') }} final
    inner join history_start using (engine_id)
    where has_failed = 1
      and failure_ts >= history_from
      and failure_ts <= (select settled_before from settled)

),

lifecycle_counts as (

    select engine_id, lifecycle_id, failures as lifecycle_failures
    from {{ ref('fct_engine_lifecycle_status_time') }} final

),

history_counts as (

    select engine_id, lifecycle_id, count() as history_lifecycle_failures
    from {{ ref('fct_engine_failures') }} final
    group by engine_id, lifecycle_id

)

-- a side without the failure reads as 0 (ClickHouse join default)
select
    'failure_not_in_both_streams' as issue,
    engine_id,
    lifecycle_id,
    in_history,
    in_telemetry,
    history_failure_cycle,
    telemetry_failure_cycle,
    toString(failed_at) as history_value,
    toString(failure_ts) as telemetry_value
from history_failures
full outer join telemetry_failures using (engine_id, lifecycle_id)
where in_history != 1
   or in_telemetry != 1
   or history_failure_cycle != telemetry_failure_cycle
   or failed_at != failure_ts

union all

select
    'lifecycle_failure_count_differs' as issue,
    engine_id,
    lifecycle_id,
    toUInt8(1) as in_history,
    toUInt8(1) as in_telemetry,
    toUInt32(0) as history_failure_cycle,
    toUInt32(0) as telemetry_failure_cycle,
    toString(history_lifecycle_failures) as history_value,
    toString(lifecycle_failures) as telemetry_value
from history_counts
full outer join lifecycle_counts using (engine_id, lifecycle_id)
where history_lifecycle_failures != lifecycle_failures
