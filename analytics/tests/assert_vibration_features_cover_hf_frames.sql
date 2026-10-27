{#-
    int_vibration_features FINAL must hold exactly the frames of stg_engine_hf_features FINAL,
    each in the lifecycle its newest copy names. Both tables only change during `dbt run`, so any
    difference is an incremental bug or a partial run.
-#}

with live_frames as (

    select engine_id, lifecycle_id, telemetry_id
    from {{ ref('stg_engine_hf_features') }} final

),

feature_frames as (

    select engine_id, lifecycle_id, telemetry_id
    from {{ ref('int_vibration_features') }} final

)

select 'missing_features' as issue, engine_id, lifecycle_id, telemetry_id
from live_frames
where (engine_id, lifecycle_id, telemetry_id) not in (
    select engine_id, lifecycle_id, telemetry_id from feature_frames
)

union all

select 'orphan_features' as issue, engine_id, lifecycle_id, telemetry_id
from feature_frames
where (engine_id, lifecycle_id, telemetry_id) not in (
    select engine_id, lifecycle_id, telemetry_id from live_frames
)
