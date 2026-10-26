{#-
    The backtest must hold exactly the first scoring, per model version, of every pulse of a
    FAILED lifecycle that precedes its FAILURE pulse: no running lifecycle, no scoring of the
    FAILURE pulse or later, no missing and no extra prediction. Its truth must follow from that
    lifecycle's failure, the RUL cap and the model's own horizon. Predictions written while the
    tests run only re-score the newest feature row: either a running lifecycle or a FAILURE
    pulse, both outside the expected set, so the comparison is stable.
-#}

with failed_lifecycles as (

    select
        engine_id,
        lifecycle_id,
        failure_cycle as lifecycle_failure_cycle,
        failure_ts as lifecycle_failure_ts
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1

),

expected as (

    -- recomputed from the raw predictions: earliest scoring per model version and pulse
    select
        model_version,
        engine_id,
        lifecycle_id,
        telemetry_id,
        min(predicted_at) as expected_predicted_at,
        toUInt8(1) as is_expected
    from {{ source('engine_raw', 'gold_engine_rul_predictions') }}
    inner join failed_lifecycles using (engine_id, lifecycle_id)
    where feature_event_ts < lifecycle_failure_ts
      and cycle_number <= lifecycle_failure_cycle
    group by model_version, engine_id, lifecycle_id, telemetry_id

),

backtest as (

    select
        model_version,
        engine_id,
        lifecycle_id,
        telemetry_id,
        predicted_at,
        cycle_number,
        feature_event_ts,
        horizon_cycles,
        actual_rul_cycles,
        actual_rul_cycles_capped,
        actual_failure_within_horizon,
        failure_cycle,
        failure_ts,
        toUInt8(1) as is_published
    from {{ ref('fct_rul_prediction_backtest') }} final

)

-- a side without the prediction reads as 0 (ClickHouse join default)
select 'missing_or_extra_prediction' as issue, model_version, engine_id, lifecycle_id, telemetry_id
from expected
full outer join backtest using (model_version, engine_id, lifecycle_id, telemetry_id)
where is_expected != 1 or is_published != 1 or predicted_at != expected_predicted_at

union all

-- lifecycle_failure_cycle is Nullable, so a lifecycle that has not failed reads as NULL
select 'truth_not_from_failed_lifecycle' as issue, model_version, engine_id, lifecycle_id, telemetry_id
from backtest
left join failed_lifecycles using (engine_id, lifecycle_id)
where isNull(lifecycle_failure_cycle)
   or failure_cycle != lifecycle_failure_cycle
   or failure_ts != lifecycle_failure_ts
   or feature_event_ts >= lifecycle_failure_ts
   or actual_rul_cycles != toInt32(failure_cycle) - toInt32(cycle_number)
   or actual_rul_cycles_capped != least(actual_rul_cycles, {{ int_var('rul_cap_cycles') }})
   or actual_failure_within_horizon != toUInt8(actual_rul_cycles <= horizon_cycles)
