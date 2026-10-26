{#-
    Every training row's labels must follow from its lifecycle's FAILURE cycle and the RUL vars:
    rul_cycles = failure_cycle - cycle_number >= 0, the cap and horizon applied as configured,
    and split_group = engine_id#lifecycle_id.
-#}

with training_rows as (

    select
        engine_id,
        lifecycle_id,
        telemetry_id,
        cycle_number,
        failure_cycle,
        rul_cycles,
        rul_cycles_capped,
        failure_within_horizon,
        split_group
    from {{ ref('mart_rul_training_features') }} final

),

completed_lifecycles as (

    select
        engine_id,
        lifecycle_id,
        failure_cycle as lifecycle_failure_cycle
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

)

select
    engine_id,
    lifecycle_id,
    telemetry_id,
    cycle_number,
    failure_cycle,
    lifecycle_failure_cycle,
    rul_cycles,
    rul_cycles_capped,
    failure_within_horizon,
    split_group
from training_rows
left join completed_lifecycles using (engine_id, lifecycle_id)
where isNull(lifecycle_failure_cycle)
   or failure_cycle != lifecycle_failure_cycle
   or rul_cycles < 0
   or rul_cycles != toInt32(failure_cycle) - toInt32(cycle_number)
   or rul_cycles_capped != least(rul_cycles, {{ int_var('rul_cap_cycles') }})
   or failure_within_horizon != toUInt8(rul_cycles <= {{ int_var('rul_failure_horizon_cycles') }})
   or split_group != concat(engine_id, '#', toString(lifecycle_id))
