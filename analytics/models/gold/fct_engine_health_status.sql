{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='engine_id',
    )
}}

{#-
    Fleet health board: one row per engine with its current registry state, latest sensor
    readings and rolling KPIs, anomaly flags, a sensor-driven degradation index and the latest
    RUL prediction for its current lifecycle. Rebuilt on every run; refreshed_at is the build time.
    Right-hand columns are made Nullable before the LEFT JOINs so that "no data" reads as NULL
    instead of ClickHouse's 0 / '' join defaults.
-#}

{%- set latest_columns = raw_sensor_columns()
    + ['vibration_mm_s_mean_5m', 'egt_c_mean_5m', 'oil_pressure_psi_mean_5m']
    + drift_feature_columns() -%}

with registry_latest as (

    select
        engine_id,
        toNullable(engine_model) as registry_engine_model,
        toNullable(operational_status) as registry_status,
        toNullable(status_changed_ts) as registry_status_changed_ts,
        toNullable(current_lifecycle_id) as registry_lifecycle_id,
        toNullable(current_cycle) as registry_cycle,
        toNullable(cycles_since_new) as registry_cycles_since_new,
        toNullable(lifecycles_completed) as registry_lifecycles_completed,
        last_maintenance_ts as registry_last_maintenance_ts,
        toNullable(updated_ts) as registry_updated_ts,
        is_deleted as is_deregistered
    from {{ ref('stg_engine_status_registry') }} final
    order by engine_id, cdc_lsn desc
    limit 1 by engine_id

),

latest_features as (

    select
        engine_id,
        toNullable(engine_model) as feature_engine_model,
        toNullable(lifecycle_id) as feature_lifecycle_id,
        toNullable(cycle_number) as feature_cycle,
        toNullable(telemetry_id) as feature_telemetry_id,
        toNullable(event_ts) as feature_event_ts,
        {%- for column in latest_columns %}
        toNullable({{ column }}) as latest_{{ column }}{{ ',' if not loop.last }}
        {%- endfor %}
    from {{ ref('int_sensor_rolling_features') }} final
    -- an engine's newest pulse is in its newest lifecycle; the sorting-key prefix prunes the rest
    where (engine_id, lifecycle_id) in (
        select engine_id, max(lifecycle_id)
        from {{ ref('int_engine_lifecycles') }} final
        group by engine_id
    )
    order by engine_id, event_ts desc, telemetry_id desc
    limit 1 by engine_id

),

lifecycle_history as (

    select
        engine_id,
        toNullable(countIf(has_failed = 1)) as completed_lifecycles_observed,
        if(
            countIf(has_failed = 1) > 0,
            toNullable(avgIf(toFloat64(lifecycle_length_cycles), has_failed = 1)),
            null
        ) as avg_completed_lifecycle_cycles
    from {{ ref('int_engine_lifecycles') }} final
    group by engine_id

),

fleet as (

    -- engines known to the registry or seen in telemetry, minus deregistered ones
    select engine_id from registry_latest where is_deregistered = 0
    union distinct
    select engine_id from latest_features
    where engine_id not in (select engine_id from registry_latest where is_deregistered = 1)

),

engine_state as (

    select
        engine_id,
        coalesce(
            nullIf(registry_engine_model, ''), nullIf(feature_engine_model, ''), 'UNKNOWN'
        ) as resolved_engine_model,
        coalesce(registry_status, 'UNKNOWN') as resolved_status,
        registry_status_changed_ts,
        coalesce(registry_lifecycle_id, feature_lifecycle_id, 0) as resolved_lifecycle_id,
        coalesce(registry_cycle, feature_cycle, 0) as resolved_cycle,
        coalesce(registry_cycles_since_new, 0) as resolved_cycles_since_new,
        coalesce(registry_lifecycles_completed, completed_lifecycles_observed, 0) as resolved_lifecycles_completed,
        registry_last_maintenance_ts,
        registry_updated_ts,
        feature_telemetry_id,
        feature_event_ts,
        greatest(
            (toUnixTimestamp64Milli(now64(3, 'UTC')) - toUnixTimestamp64Milli(feature_event_ts)) / 1000.0,
            0.0
        ) as staleness_seconds,
        {%- for column in latest_columns %}
        latest_{{ column }},
        {%- endfor %}
        avg_completed_lifecycle_cycles
    from fleet
    left join registry_latest using (engine_id)
    left join latest_features using (engine_id)
    left join lifecycle_history using (engine_id)

),

latest_predictions as (

    -- newest prediction per engine lifecycle; the table may be empty before the first model
    select
        engine_id,
        lifecycle_id as prediction_lifecycle_id,
        toNullable(predicted_rul_cycles) as prediction_rul_cycles,
        toNullable(predicted_rul_hours) as prediction_rul_hours,
        toNullable(failure_probability) as prediction_failure_probability,
        toNullable(cast(risk_band as String)) as prediction_risk_band,
        toNullable(cast(model_version as String)) as prediction_model_version,
        toNullable(predicted_at) as prediction_predicted_at
    from {{ source('engine_raw', 'gold_engine_rul_predictions') }}
    order by engine_id, lifecycle_id, predicted_at desc
    limit 1 by engine_id, lifecycle_id

),

assessed as (

    select
        engine_state.*,
        toUInt8(coalesce(latest_egt_c_drift > {{ number_var('anomaly_egt_drift_c') }}, 0)) as egt_drift_flag,
        toUInt8(coalesce(latest_vibration_mm_s_ratio > {{ number_var('anomaly_vibration_ratio') }}, 0))
            as vibration_growth_flag,
        toUInt8(coalesce(latest_oil_pressure_psi_drop > {{ number_var('anomaly_oil_pressure_drop_psi') }}, 0))
            as oil_pressure_drop_flag,
        -- an engine that never reported telemetry is as unobservable as a silent one
        toUInt8(coalesce(staleness_seconds > {{ number_var('anomaly_stale_telemetry_seconds') }}, 1))
            as stale_telemetry_flag,
        round(
            greatest(
                {{ degradation_component(
                    'latest_egt_c_drift', 'degradation_full_scale_egt_drift_c'
                ) }},
                {{ degradation_component(
                    'latest_vibration_mm_s_ratio', 'degradation_full_scale_vibration_ratio', 1.0
                ) }},
                {{ degradation_component(
                    'latest_oil_pressure_psi_drop', 'degradation_full_scale_oil_pressure_drop_psi'
                ) }}
            ),
            4
        ) as degradation,
        prediction_rul_cycles,
        prediction_rul_hours,
        prediction_failure_probability,
        prediction_risk_band,
        prediction_model_version,
        prediction_predicted_at
    from engine_state
    left join latest_predictions
        on latest_predictions.engine_id = engine_state.engine_id
        and latest_predictions.prediction_lifecycle_id = engine_state.resolved_lifecycle_id

)

select
    engine_id,
    cast(resolved_engine_model as LowCardinality(String)) as engine_model,
    cast(resolved_status as LowCardinality(String)) as operational_status,
    registry_status_changed_ts as status_changed_ts,
    cast(resolved_lifecycle_id as UInt32) as current_lifecycle_id,
    cast(resolved_cycle as UInt32) as current_cycle,
    cast(resolved_cycles_since_new as UInt32) as cycles_since_new,
    cast(resolved_lifecycles_completed as UInt32) as lifecycles_completed,
    registry_last_maintenance_ts as last_maintenance_ts,
    registry_updated_ts as registry_updated_ts,
    feature_telemetry_id as last_telemetry_id,
    feature_event_ts as last_telemetry_ts,
    staleness_seconds as telemetry_staleness_seconds,
    {%- for column in latest_columns %}
    latest_{{ column }} as {{ column }},
    {%- endfor %}
    egt_drift_flag as is_egt_drift_anomaly,
    vibration_growth_flag as is_vibration_growth_anomaly,
    oil_pressure_drop_flag as is_oil_pressure_drop_anomaly,
    stale_telemetry_flag as is_telemetry_stale,
    toUInt8(
        egt_drift_flag + vibration_growth_flag + oil_pressure_drop_flag + stale_telemetry_flag
    ) as anomaly_count,
    degradation as degradation_index,
    round(100.0 * (1.0 - degradation), 1) as health_score,
    avg_completed_lifecycle_cycles,
    prediction_rul_cycles as predicted_rul_cycles,
    prediction_rul_hours as predicted_rul_hours,
    prediction_failure_probability as failure_probability,
    prediction_risk_band as risk_band,
    prediction_model_version as model_version,
    prediction_predicted_at as predicted_at,
    now64(3, 'UTC') as refreshed_at
from assessed
