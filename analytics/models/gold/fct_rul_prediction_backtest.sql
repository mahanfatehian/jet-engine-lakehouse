{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(model_version, engine_id, lifecycle_id, telemetry_id)',
    )
}}

{#-
    Live backtest of the RUL model: once a lifecycle has FAILED, every prediction made during it
    is scored against the truth that the failure revealed, with the labels of
    mart_rul_training_features (RUL = failure_cycle - cycle_number, capped for the regressor).
    Rebuilt on every run, so new failures, new predictions and late work orders are reflected.

    The predictor scores the newest feature row of every engine each round, so the same row is
    scored again until a newer pulse lands (for the whole repair, the FAILURE row stays newest).
    Only the first scoring of a row per model version is kept: the live forecast.

    Scorings of the FAILURE pulse itself (or of anything after it) are excluded: they are made
    after the engine has failed, their truth is RUL 0 by definition and they carry no warning
    value, so they would reward hindsight in the error and horizon metrics. The earlier pulses
    of the failure cycle stay: they were scored before the failure (label 0, like in the mart).
    Warning lead time is a property of a whole lifecycle, so rpt_model_live_performance derives
    it from these rows instead of repeating it on every prediction.
-#}

{%- set rul_cap_cycles = int_var('rul_cap_cycles') -%}

with failed_lifecycles as (

    select
        engine_id,
        lifecycle_id,
        assumeNotNull(failure_cycle) as lifecycle_failure_cycle,
        assumeNotNull(failure_ts) as lifecycle_failure_ts,
        coalesce(failure_mode, 'UNKNOWN') as lifecycle_failure_mode
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

),

first_scorings as (

    select
        predicted_at,
        engine_id,
        engine_model,
        lifecycle_id,
        cycle_number,
        telemetry_id,
        feature_event_ts,
        model_version,
        predicted_rul_cycles,
        failure_probability,
        failure_within_horizon,
        horizon_cycles,
        risk_band
    from {{ source('engine_raw', 'gold_engine_rul_predictions') }}
    where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from failed_lifecycles)
    order by model_version, engine_id, lifecycle_id, telemetry_id, predicted_at
    limit 1 by model_version, engine_id, lifecycle_id, telemetry_id

),

labelled as (

    select
        predicted_at,
        engine_id,
        engine_model,
        lifecycle_id,
        cycle_number,
        telemetry_id,
        feature_event_ts,
        model_version,
        predicted_rul_cycles,
        failure_probability,
        failure_within_horizon,
        horizon_cycles,
        risk_band,
        lifecycle_failure_cycle,
        lifecycle_failure_ts,
        lifecycle_failure_mode,
        toInt32(lifecycle_failure_cycle) - toInt32(cycle_number) as remaining_cycles
    from first_scorings
    inner join failed_lifecycles using (engine_id, lifecycle_id)
    -- only pulses measured before the FAILURE pulse (see above); the cycle bound keeps RUL >= 0
    where feature_event_ts < lifecycle_failure_ts
      and cycle_number <= lifecycle_failure_cycle

),

scored as (

    -- the regressor predicts the RUL capped at rul_cap_cycles, so that is the truth it is held
    -- to; a positive error means the failure came sooner than predicted (late, optimistic)
    select
        *,
        predicted_rul_cycles - least(remaining_cycles, {{ rul_cap_cycles }}) as prediction_error
    from labelled

)

select
    predicted_at,
    engine_id,
    engine_model,
    lifecycle_id,
    cycle_number,
    telemetry_id,
    feature_event_ts,
    model_version,
    predicted_rul_cycles,
    cast(remaining_cycles as Int32) as actual_rul_cycles,
    cast(least(remaining_cycles, {{ rul_cap_cycles }}) as Int32) as actual_rul_cycles_capped,
    prediction_error as error_cycles,
    abs(prediction_error) as abs_error_cycles,
    toUInt8(prediction_error > 0) as is_late_prediction,
    -- asymmetric C-MAPSS score of data_science/rul_metrics.py: late costs more than early
    if(prediction_error < 0, exp(-prediction_error / 13.0), exp(prediction_error / 10.0)) - 1.0 as nasa_score,
    failure_within_horizon as predicted_failure_within_horizon,
    -- judged on the horizon the model was trained for, not on the current var
    toUInt8(remaining_cycles <= horizon_cycles) as actual_failure_within_horizon,
    horizon_cycles,
    failure_probability,
    risk_band,
    cast(lifecycle_failure_cycle as UInt32) as failure_cycle,
    cast(lifecycle_failure_mode as String) as failure_mode,
    lifecycle_failure_ts as failure_ts
from scored
