{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(model_version, engine_model)',
    )
}}

{#-
    Live performance of every RUL model version on the lifecycles that have failed since it
    predicted them (fct_rul_prediction_backtest): one row per model version and engine model,
    plus one engine_model = 'ALL' row per model version for the whole fleet. The metrics follow
    the training holdout (data_science/rul_metrics.py, ml_model_runs), so live and holdout
    numbers of a model compare one to one. A metric without a denominator (no late-life
    prediction, no predicted or actual positive, no warned lifecycle) is NULL instead of NaN;
    an empty backtest yields an empty table. Rebuilt on every run; refreshed_at is the build time.
-#}

{%- set rul_cap_cycles = int_var('rul_cap_cycles') -%}
{#- the model-level grouping set rolls engine_model up into the fleet-wide 'ALL' row -#}
{%- set report_grouping = 'grouping sets ((model_version), (model_version, engine_model))' -%}

with backtest as (

    select
        model_version,
        engine_model,
        engine_id,
        lifecycle_id,
        cycle_number,
        telemetry_id,
        predicted_at,
        actual_rul_cycles,
        error_cycles,
        abs_error_cycles,
        is_late_prediction,
        nasa_score,
        predicted_failure_within_horizon,
        actual_failure_within_horizon,
        risk_band,
        failure_cycle
    from {{ ref('fct_rul_prediction_backtest') }} final

),

prediction_metrics as (

    select
        model_version,
        if(grouping(engine_model) = 1, 'ALL', engine_model) as engine_model_scope,
        count() as predictions_scored,
        sqrt(avg(error_cycles * error_cycles)) as rmse_cycles,
        avg(abs_error_cycles) as mae_cycles,
        avg(error_cycles) as bias_cycles,
        -- late life: the true RUL is within the cap, so the capped target is the exact truth
        countIf(actual_rul_cycles <= {{ rul_cap_cycles }}) as late_life_predictions,
        sumIf(error_cycles * error_cycles, actual_rul_cycles <= {{ rul_cap_cycles }}) as late_life_squared_error,
        avg(nasa_score) as mean_nasa_score,
        avg(is_late_prediction) as late_prediction_rate,
        countIf(predicted_failure_within_horizon = 1 and actual_failure_within_horizon = 1) as true_positives,
        countIf(predicted_failure_within_horizon = 1 and actual_failure_within_horizon = 0) as false_positives,
        countIf(predicted_failure_within_horizon = 0 and actual_failure_within_horizon = 1) as false_negatives,
        countIf(predicted_failure_within_horizon = 0 and actual_failure_within_horizon = 0) as true_negatives,
        min(predicted_at) as first_predicted_at,
        max(predicted_at) as last_predicted_at
    from backtest
    group by {{ report_grouping }}

),

lifecycle_warnings as (

    -- advance warning of each failed lifecycle: cycles from the first HIGH-risk prediction to the
    -- failure; NULL when the model never said HIGH before the engine failed (a missed failure)
    select
        model_version,
        engine_model,
        engine_id,
        lifecycle_id,
        if(
            countIf(risk_band = 'HIGH') > 0,
            toInt32(any(failure_cycle))
                - toInt32(argMinIf(cycle_number, (predicted_at, telemetry_id), risk_band = 'HIGH')),
            null
        ) as warning_lead_cycles
    from backtest
    group by model_version, engine_model, engine_id, lifecycle_id

),

lifecycle_metrics as (

    select
        model_version,
        if(grouping(engine_model) = 1, 'ALL', engine_model) as engine_model_scope,
        count() as lifecycles_covered,
        count(warning_lead_cycles) as lifecycles_warned,
        -- interpolated median (numpy's default); NULLs (missed failures) are skipped
        quantileExactInclusive(0.5)(warning_lead_cycles) as median_warning_lead_cycles
    from lifecycle_warnings
    group by {{ report_grouping }}

)

select
    model_version,
    cast(engine_model_scope as LowCardinality(String)) as engine_model,
    predictions_scored,
    lifecycles_covered,
    lifecycles_warned,
    rmse_cycles,
    mae_cycles,
    bias_cycles,
    sqrt({{ safe_divide('late_life_squared_error', 'late_life_predictions', 'null') }}) as late_life_rmse_cycles,
    mean_nasa_score,
    late_prediction_rate,
    true_positives as clf_true_positives,
    false_positives as clf_false_positives,
    false_negatives as clf_false_negatives,
    true_negatives as clf_true_negatives,
    {{ safe_divide('true_positives', '(true_positives + false_positives)', 'null') }} as clf_precision,
    {{ safe_divide('true_positives', '(true_positives + false_negatives)', 'null') }} as clf_recall,
    {{ safe_divide('2 * true_positives', '(2 * true_positives + false_positives + false_negatives)', 'null') }}
        as clf_f1,
    median_warning_lead_cycles,
    first_predicted_at,
    last_predicted_at,
    now64(3, 'UTC') as refreshed_at
from prediction_metrics
inner join lifecycle_metrics using (model_version, engine_model_scope)
