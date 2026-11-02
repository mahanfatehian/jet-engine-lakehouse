{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, lifecycle_id)',
    )
}}

{#-
    Time in each status per engine lifecycle, from the SCD2 status history: one row per engine
    and lifecycle the history covers (the registry's lifecycle when an interval began, so the
    repair of a failure counts to the lifecycle that failed and MAINTENANCE_RESET to the one it
    starts). The current interval is measured up to the history's horizon (status_history_as_of),
    so a rebuild on the same data yields the same numbers. Durations are summed in whole
    microseconds, so they add up exactly. A lifecycle whose start predates the CDC history (the
    connector's initial snapshot) is covered from its first interval on. Table, rebuilt on every run.
-#}

{%- set statuses = registry_statuses() %}

with intervals as (

    select
        engine_id,
        engine_model,
        lifecycle_id,
        operational_status,
        is_available,
        is_current,
        valid_from,
        coalesce(valid_to, greatest(valid_from, {{ status_history_as_of() }})) as interval_end
    from {{ ref('int_engine_status_history') }} final

),

status_time as (

    select
        engine_id,
        lifecycle_id,
        argMin(engine_model, valid_from) as lifecycle_engine_model,
        min(valid_from) as observed_from,
        max(interval_end) as observed_to,
        max(is_current) as is_current,
        count() as status_intervals,
        {%- for entry in statuses %}
        sumIf({{ micros_between('valid_from', 'interval_end') }}, operational_status = '{{ entry.status }}')
            as {{ entry.column }}_us,
        {%- endfor %}
        sumIf({{ micros_between('valid_from', 'interval_end') }}, is_available = 1) as available_us,
        sumIf({{ micros_between('valid_from', 'interval_end') }}, is_available = 0) as unavailable_us
    from intervals
    group by engine_id, lifecycle_id

),

lifecycle_failures as (

    select engine_id, lifecycle_id, count() as failures
    from {{ ref('fct_engine_failures') }} final
    group by engine_id, lifecycle_id

)

select
    engine_id,
    cast(lifecycle_engine_model as LowCardinality(String)) as engine_model,
    lifecycle_id,
    observed_from,
    observed_to,
    is_current,
    status_intervals,
    -- a lifecycle without failure reads 0 (ClickHouse join default)
    failures,
    {%- for entry in statuses %}
    {{ entry.column }}_us / 1000000.0 as {{ entry.column }}_seconds,
    {%- endfor %}
    available_us / 1000000.0 as available_seconds,
    unavailable_us / 1000000.0 as unavailable_seconds,
    (available_us + unavailable_us) / 1000000.0 as observed_seconds,
    if(available_us + unavailable_us > 0, available_us / (available_us + unavailable_us), null) as availability
from status_time
left join lifecycle_failures using (engine_id, lifecycle_id)
