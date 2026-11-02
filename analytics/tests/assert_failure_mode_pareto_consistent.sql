{#-
    rpt_failure_mode_pareto must be a Pareto of the work orders: in every scope the ranks run
    1..n without gaps, work orders never increase and the cumulative share never decreases with
    the rank, and the last rank reaches a cumulative share of 1; the work orders of each scope
    are those of stg_maintenance_log, and the fleet row of every mode is the sum of its engines.
-#}

with ranked as (

    select
        engine_id,
        failure_mode,
        pareto_rank,
        work_orders,
        cumulative_share,
        lagInFrame(toNullable(work_orders)) over w_previous as previous_work_orders,
        lagInFrame(toNullable(cumulative_share)) over w_previous as previous_cumulative_share,
        count() over (partition by engine_id) as scope_modes,
        sum(work_orders) over (partition by engine_id) as scope_work_orders
    from {{ ref('rpt_failure_mode_pareto') }} final
    window w_previous as (partition by engine_id order by pareto_rank rows between 1 preceding and current row)

),

source_scopes as (

    select
        if(grouping(engine_id) = 1, 'ALL', engine_id) as scope,
        count() as source_work_orders
    from {{ ref('stg_maintenance_log') }} final
    group by grouping sets ((), (engine_id))

),

report_scopes as (

    select toString(engine_id) as scope, any(scope_work_orders) as report_work_orders
    from ranked
    group by engine_id

),

fleet_modes as (

    select failure_mode, work_orders as fleet_work_orders
    from ranked
    where engine_id = 'ALL'

),

engine_modes as (

    select failure_mode, sum(work_orders) as engine_work_orders
    from ranked
    where engine_id != 'ALL'
    group by failure_mode

)

select
    'not_a_pareto' as issue,
    toString(engine_id) as scope,
    toString(failure_mode) as failure_mode,
    pareto_rank as rank_or_count
from ranked
where pareto_rank > scope_modes
   or work_orders > ifNull(previous_work_orders, work_orders)
   or cumulative_share < ifNull(previous_cumulative_share, 0)
   or (pareto_rank = scope_modes and abs(cumulative_share - 1) > 1e-9)

union all

-- a scope missing on one side reads 0 (ClickHouse join default)
select
    'work_orders_differ' as issue,
    scope,
    '' as failure_mode,
    report_work_orders as rank_or_count
from source_scopes
full outer join report_scopes using (scope)
where source_work_orders != report_work_orders

union all

select
    'fleet_row_differs' as issue,
    'ALL' as scope,
    toString(failure_mode) as failure_mode,
    fleet_work_orders as rank_or_count
from fleet_modes
full outer join engine_modes using (failure_mode)
where fleet_work_orders != engine_work_orders
