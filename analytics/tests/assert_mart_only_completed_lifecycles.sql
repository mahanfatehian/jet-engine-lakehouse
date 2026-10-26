{#-
    The training mart must hold exactly the feature rows of the newest
    `mart_max_lifecycles_per_engine` COMPLETED lifecycles of each engine, up to their failure
    cycle: no running or older lifecycle, no missing and no extra pulse per lifecycle.
-#}

with feature_rows as (

    select engine_id, lifecycle_id, cycle_number
    from {{ ref('int_sensor_rolling_features') }} final

),

completed_lifecycles as (

    -- ranked independently of the model: 1 = the engine's newest completed lifecycle
    select
        engine_id,
        lifecycle_id,
        failure_cycle,
        row_number() over (partition by engine_id order by lifecycle_id desc) as recency_rank
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

),

expected as (

    select engine_id, lifecycle_id, count() as expected_rows
    from feature_rows
    inner join completed_lifecycles using (engine_id, lifecycle_id)
    where cycle_number <= failure_cycle
      and recency_rank <= {{ positive_int_var('mart_max_lifecycles_per_engine') }}
    group by engine_id, lifecycle_id

),

actual as (

    select engine_id, lifecycle_id, count() as mart_rows
    from {{ ref('mart_rul_training_features') }} final
    group by engine_id, lifecycle_id

)

-- a side without the lifecycle reads as 0 rows (ClickHouse join default)
select engine_id, lifecycle_id, expected_rows, mart_rows
from expected
full outer join actual using (engine_id, lifecycle_id)
where expected_rows != mart_rows
