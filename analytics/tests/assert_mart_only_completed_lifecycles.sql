{#-
    The training mart must hold exactly the feature rows of COMPLETED lifecycles up to their
    failure cycle: no running lifecycle, no missing and no extra pulse per lifecycle.
-#}

with feature_rows as (

    select engine_id, lifecycle_id, cycle_number
    from {{ ref('int_sensor_rolling_features') }} final

),

completed_lifecycles as (

    select engine_id, lifecycle_id, failure_cycle
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

),

expected as (

    select engine_id, lifecycle_id, count() as expected_rows
    from feature_rows
    inner join completed_lifecycles using (engine_id, lifecycle_id)
    where cycle_number <= failure_cycle
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
