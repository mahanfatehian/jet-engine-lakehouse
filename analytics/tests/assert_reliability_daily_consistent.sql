{#-
    rpt_fleet_reliability_daily must add up: the fleet row of every day is the sum of its engine
    rows (time, engines, failures, repairs and the MTTR / MTBF samples, re-weighted by their
    counts), every engine's failures over all days are its failures in fct_engine_failures, and
    every engine's observed and unobserved time over all days is the length of its status history
    up to the report's horizon (so the days neither lose nor double-count time). Times compare to 1 ms.
-#}

{%- set tolerance_seconds = 0.001 %}

with engine_days as (

    select
        day,
        sum(engines) as engines_sum,
        sum(observed_seconds) as observed_seconds_sum,
        sum(unobserved_seconds) as unobserved_seconds_sum,
        sum(available_seconds) as available_seconds_sum,
        sum(failures) as failures_sum,
        sum(repairs) as repairs_sum,
        sum(ifNull(mttr_seconds, 0) * repairs) as repair_seconds_sum,
        sum(operating_periods) as operating_periods_sum,
        sum(ifNull(mtbf_seconds, 0) * operating_periods) as operating_seconds_sum
    from {{ ref('rpt_fleet_reliability_daily') }} final
    where engine_id != 'ALL'
    group by day

),

fleet_days as (

    select
        day,
        engines as fleet_engines,
        observed_seconds as fleet_observed_seconds,
        unobserved_seconds as fleet_unobserved_seconds,
        available_seconds as fleet_available_seconds,
        failures as fleet_failures,
        repairs as fleet_repairs,
        ifNull(mttr_seconds, 0) * repairs as fleet_repair_seconds,
        operating_periods as fleet_operating_periods,
        ifNull(mtbf_seconds, 0) * operating_periods as fleet_operating_seconds
    from {{ ref('rpt_fleet_reliability_daily') }} final
    where engine_id = 'ALL'

),

engine_totals as (

    select
        engine_id,
        sum(failures) as report_failures,
        sum(observed_seconds + unobserved_seconds) as report_seconds
    from {{ ref('rpt_fleet_reliability_daily') }} final
    where engine_id != 'ALL'
    group by engine_id

),

engine_failures as (

    select engine_id, count() as fact_failures
    from {{ ref('fct_engine_failures') }} final
    group by engine_id

),

engine_history as (

    -- the report's horizon is the same on every row
    select
        engine_id,
        sum(
            toUnixTimestamp64Micro(coalesce(
                valid_to,
                greatest(valid_from, (select max(as_of_ts) from {{ ref('rpt_fleet_reliability_daily') }} final))
            )) - toUnixTimestamp64Micro(valid_from)
        ) / 1000000.0 as history_seconds
    from {{ ref('int_engine_status_history') }} final
    group by engine_id

)

-- a side without the day or engine reads as 0 (ClickHouse join default)
select
    'fleet_row_differs' as issue,
    toString(day) as scope,
    observed_seconds_sum as expected,
    fleet_observed_seconds as actual
from engine_days
full outer join fleet_days using (day)
where engines_sum != fleet_engines
   or failures_sum != fleet_failures
   or repairs_sum != fleet_repairs
   or operating_periods_sum != fleet_operating_periods
   or abs(observed_seconds_sum - fleet_observed_seconds) > {{ tolerance_seconds }}
   or abs(unobserved_seconds_sum - fleet_unobserved_seconds) > {{ tolerance_seconds }}
   or abs(available_seconds_sum - fleet_available_seconds) > {{ tolerance_seconds }}
   or abs(repair_seconds_sum - fleet_repair_seconds) > {{ tolerance_seconds }}
   or abs(operating_seconds_sum - fleet_operating_seconds) > {{ tolerance_seconds }}

union all

select
    'engine_failures_differ' as issue,
    toString(engine_id) as scope,
    toFloat64(fact_failures) as expected,
    toFloat64(report_failures) as actual
from engine_failures
full outer join engine_totals using (engine_id)
where fact_failures != report_failures

union all

select
    'engine_time_differs' as issue,
    toString(engine_id) as scope,
    history_seconds as expected,
    report_seconds as actual
from engine_history
full outer join engine_totals using (engine_id)
where abs(history_seconds - report_seconds) > {{ tolerance_seconds }}
