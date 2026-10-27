{#-
    int_ml_features must hold exactly one row per row of int_sensor_rolling_features FINAL: the
    LEFT JOIN onto the spectral features may neither drop nor duplicate a pulse. A pulse carries
    spectral features exactly when int_vibration_features FINAL holds its frame in the same
    lifecycle (spectral_frames_5m is NULL otherwise, and never NULL for a joined frame).
-#}

with rolling_pulses as (

    select engine_id, lifecycle_id, telemetry_id, toUInt8(1) as is_rolling
    from {{ ref('int_sensor_rolling_features') }} final

),

ml_pulses as (

    select
        engine_id,
        lifecycle_id,
        telemetry_id,
        count() as ml_rows,
        toUInt8(countIf(isNotNull(spectral_frames_5m)) > 0) as has_spectral
    from {{ ref('int_ml_features') }}
    group by engine_id, lifecycle_id, telemetry_id

),

hf_frames as (

    select engine_id, lifecycle_id, telemetry_id, toUInt8(1) as has_frame
    from {{ ref('int_vibration_features') }} final

)

-- a side without the pulse reads as 0 (ClickHouse join default)
select 'row_parity' as issue, engine_id, lifecycle_id, telemetry_id
from rolling_pulses
full outer join ml_pulses using (engine_id, lifecycle_id, telemetry_id)
where is_rolling != 1 or ml_rows != 1

union all

select 'spectral_presence' as issue, engine_id, lifecycle_id, telemetry_id
from ml_pulses
left join hf_frames using (engine_id, lifecycle_id, telemetry_id)
where has_spectral != has_frame
