{#-
    int_sensor_rolling_features FINAL must hold exactly the live pulses of stg_engine_telemetry
    FINAL, each in its current lifecycle and with its engine's current registry model. Both
    tables only change during `dbt run`, so any difference is an incremental bug or a partial run.
-#}

with live_pulses as (

    select engine_id, lifecycle_id, telemetry_id
    from {{ ref('stg_engine_telemetry') }} final

),

feature_pulses as (

    select engine_id, lifecycle_id, telemetry_id, engine_model
    from {{ ref('int_sensor_rolling_features') }} final

),

registry_models as (

    select
        engine_id,
        coalesce(nullIf(argMax(engine_model, cdc_lsn), ''), 'UNKNOWN') as registry_engine_model
    from {{ ref('stg_engine_status_registry') }} final
    group by engine_id

)

select 'missing_features' as issue, engine_id, lifecycle_id, telemetry_id
from live_pulses
where (engine_id, lifecycle_id, telemetry_id) not in (
    select engine_id, lifecycle_id, telemetry_id from feature_pulses
)

union all

select 'orphan_features' as issue, engine_id, lifecycle_id, telemetry_id
from feature_pulses
where (engine_id, lifecycle_id, telemetry_id) not in (select engine_id, lifecycle_id, telemetry_id from live_pulses)

union all

select 'stale_engine_model' as issue, engine_id, lifecycle_id, telemetry_id
from feature_pulses
inner join registry_models using (engine_id)
where engine_model != registry_engine_model
