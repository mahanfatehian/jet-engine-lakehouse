{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(day, engine_id)',
    )
}}

{#-
    Reliability KPIs per UTC day: one row per day and engine, plus one engine_id = 'ALL' row per
    day for the whole fleet (never sum across rows of one day). Time comes from the SCD2 status
    history cut at UTC midnights, the current interval measured up to the history's horizon
    (as_of_ts, see status_history_as_of), so the current day is partial. Failures, repairs and
    their durations come from fct_engine_failures and count on the day of the failure.

      availability = time available (HEALTHY, WARNING, CRITICAL) / time observed
      MTTR         = mean time from a failure to the return to service, over the day's failures
                     that have returned
      MTBF         = mean operating time (and cycles) since the previous return to service (or
                     the registration), over the day's failures whose operating run has a known start

    Time inside platform outages counts to no status but as unobserved, and to no operating time.

    Fleet rows weigh engines by time (engine-seconds) and failures equally. Durations are summed in
    whole microseconds, so the fleet row is the exact sum of its engines. Table, rebuilt on every run.
-#}

{%- set statuses = registry_statuses() %}
{%- set day_us = 86400000000 %}
{#- the engine-level grouping set rolls engine_id up into the fleet-wide 'ALL' row -#}
{%- set report_grouping = 'grouping sets ((day), (day, engine_id))' %}

with horizon as (

    select {{ status_history_as_of() }} as as_of_ts

),

intervals as (

    select
        engine_id,
        operational_status,
        is_available,
        toUnixTimestamp64Micro(valid_from) as from_us,
        toUnixTimestamp64Micro(coalesce(valid_to, greatest(valid_from, as_of_ts))) as to_us
    from {{ ref('int_engine_status_history') }} final
    cross join horizon

),

day_pieces as (

    -- every interval cut at UTC midnights; an interval ending at midnight leaves no empty piece
    select
        engine_id,
        operational_status,
        is_available,
        toDate('1970-01-01') + day_number as day,
        greatest(from_us, day_number * {{ day_us }}) as piece_from_us,
        least(to_us, (day_number + 1) * {{ day_us }}) as piece_to_us,
        {{ outage_overlap_us('piece_from_us', 'piece_to_us') }} as unobserved_us,
        piece_to_us - piece_from_us - unobserved_us as piece_us
    from intervals
    array join range(intDiv(from_us, {{ day_us }}), intDiv(to_us, {{ day_us }}) + 1) as day_number
    where piece_to_us > piece_from_us

),

status_time as (

    select
        day,
        if(grouping(engine_id) = 1, 'ALL', engine_id) as engine_scope,
        uniqExact(engine_id) as engines,
        sum(piece_us) as observed_us,
        sum(unobserved_us) as unobserved_us,
        {%- for entry in statuses %}
        sumIf(piece_us, operational_status = '{{ entry.status }}') as {{ entry.column }}_us,
        {%- endfor %}
        sumIf(piece_us, is_available = 1) as available_us
    from day_pieces
    group by {{ report_grouping }}

),

failure_stats as (

    select
        toDate(failed_at) as day,
        if(grouping(engine_id) = 1, 'ALL', engine_id) as engine_scope,
        count() as failures,
        count(repair_us) as repairs,
        sum(ifNull(repair_us, 0)) as repair_us_total,
        count(operating_us) as operating_periods,
        sum(ifNull(operating_us, 0)) as operating_us_total,
        count(operating_cycles) as operating_cycle_periods,
        sum(ifNull(operating_cycles, 0)) as operating_cycles_total
    from (
        select
            engine_id,
            failed_at,
            {{ micros_between('failed_at', 'returned_to_service_at') }} as repair_us,
            toUnixTimestamp64Micro(operating_since) as since_us,
            toUnixTimestamp64Micro(failed_at) as failed_us,
            -- observed time only, like fct_engine_failures.operating_seconds
            failed_us - since_us - {{ outage_overlap_us('since_us', 'failed_us') }} as operating_us,
            operating_cycles
        from {{ ref('fct_engine_failures') }} final
    )
    group by {{ report_grouping }}

)

-- a day with time but no failure reads 0 failures (ClickHouse join default), and vice versa
select
    day,
    cast(engine_scope as LowCardinality(String)) as engine_id,
    engines,
    observed_us / 1000000.0 as observed_seconds,
    unobserved_us / 1000000.0 as unobserved_seconds,
    {%- for entry in statuses %}
    {{ entry.column }}_us / 1000000.0 as {{ entry.column }}_seconds,
    {%- endfor %}
    available_us / 1000000.0 as available_seconds,
    if(observed_us > 0, available_us / observed_us, null) as availability,
    failures,
    repairs,
    if(repairs > 0, repair_us_total / repairs / 1000000.0, null) as mttr_seconds,
    operating_periods,
    if(operating_periods > 0, operating_us_total / operating_periods / 1000000.0, null) as mtbf_seconds,
    if(operating_cycle_periods > 0, operating_cycles_total / operating_cycle_periods, null) as mtbf_cycles,
    as_of_ts,
    now64(3, 'UTC') as refreshed_at
from status_time
full outer join failure_stats using (day, engine_scope)
cross join horizon
