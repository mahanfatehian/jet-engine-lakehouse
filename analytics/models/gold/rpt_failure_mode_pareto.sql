{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, pareto_rank)',
    )
}}

{#-
    Failure-mode Pareto from the maintenance work orders (stg_maintenance_log: one per failure,
    deleted ones removed): one row per engine and failure mode, plus one engine_id = 'ALL' row per
    mode for the whole fleet. Within each scope the modes are ranked by work orders (ties by
    name), and cumulative_share accumulates the shares in that order up to exactly 1, the curve
    of the Pareto chart. Repair downtime is the work order's own (completed orders only). Table,
    rebuilt on every run; an empty maintenance log yields an empty table.
-#}

{#- the engine-level grouping set rolls engine_id up into the fleet-wide 'ALL' row -#}
{%- set report_grouping = 'grouping sets ((failure_mode), (engine_id, failure_mode))' %}

with work_orders as (

    select
        engine_id,
        failure_mode,
        maintenance_status,
        failure_cycle,
        failure_ts,
        downtime_seconds
    from {{ ref('stg_maintenance_log') }} final

),

modes as (

    select
        if(grouping(engine_id) = 1, 'ALL', engine_id) as engine_scope,
        failure_mode,
        count() as work_orders,
        countIf(maintenance_status = 'COMPLETED') as completed_work_orders,
        uniqExact(engine_id) as engines_affected,
        avg(failure_cycle) as mean_failure_cycle,
        sum(downtime_seconds) as total_downtime_seconds,
        avg(downtime_seconds) as mean_downtime_seconds,
        min(failure_ts) as first_failure_ts,
        max(failure_ts) as last_failure_ts
    from work_orders
    group by {{ report_grouping }}

),

ranked as (

    select
        *,
        row_number() over w_rank as pareto_rank,
        sum(work_orders) over w_rank as cumulative_work_orders,
        sum(work_orders) over (partition by engine_scope) as scope_work_orders
    from modes
    window w_rank as (
        partition by engine_scope
        order by work_orders desc, failure_mode
        rows between unbounded preceding and current row
    )

)

select
    cast(engine_scope as LowCardinality(String)) as engine_id,
    failure_mode,
    pareto_rank,
    work_orders,
    completed_work_orders,
    work_orders / scope_work_orders as share_of_work_orders,
    cumulative_work_orders / scope_work_orders as cumulative_share,
    engines_affected,
    mean_failure_cycle,
    total_downtime_seconds,
    mean_downtime_seconds,
    first_failure_ts,
    last_failure_ts,
    now64(3, 'UTC') as refreshed_at
from ranked
