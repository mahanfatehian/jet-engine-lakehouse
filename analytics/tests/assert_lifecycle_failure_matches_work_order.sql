{{ config(severity='warn') }}

{#-
    The FAILURE pulse and the maintenance work order describe the same event through two CDC
    streams; once both have landed they must agree on the failure cycle.
-#}

with telemetry_failures as (

    select engine_id, lifecycle_id, failure_cycle as telemetry_failure_cycle
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

),

work_orders as (

    select engine_id, lifecycle_id, failure_cycle as work_order_failure_cycle
    from {{ ref('stg_maintenance_log') }} final

)

select engine_id, lifecycle_id, telemetry_failure_cycle, work_order_failure_cycle
from telemetry_failures
inner join work_orders using (engine_id, lifecycle_id)
where telemetry_failure_cycle != work_order_failure_cycle
