{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, failure_mode)',
    )
}}

{#-
    Operational notice period of shop visits (fct_engine_shop_visits): median and p90 cycles
    from the first registry WARNING to FAILURE, plus how often WARNING was skipped. One row per
    engine and failure mode, plus engine_id = 'ALL' and failure_mode = 'ALL' rollups. Named
    median_warning_to_failure_cycles so it cannot be confused with the ML metric
    median_warning_lead_cycles. A metric without a denominator is NULL, never NaN. Table,
    rebuilt every run; an empty visit fact yields an empty table.
-#}

{%- set report_grouping = 'grouping sets ((failure_mode), (engine_id, failure_mode), (engine_id), ())' %}

with visits as (

    select
        engine_id,
        failure_mode,
        warning_to_failure_cycles,
        warning_to_failure_seconds,
        skipped_warning,
        shop_visit_seconds,
        returned_to_service_at
    from {{ ref('fct_engine_shop_visits') }} final

),

rolled as (

    select
        if(grouping(engine_id) = 1, 'ALL', engine_id) as engine_scope,
        if(grouping(failure_mode) = 1, 'ALL', failure_mode) as failure_mode_scope,
        count() as shop_visits,
        countIf(skipped_warning = 0) as visits_with_warning,
        countIf(skipped_warning = 1) as visits_skipped_warning,
        countIf(isNotNull(returned_to_service_at)) as visits_returned,
        if(
            countIf(isNotNull(warning_to_failure_cycles)) = 0,
            null,
            quantileExactInclusive(0.5)(warning_to_failure_cycles)
        ) as median_warning_to_failure_cycles,
        if(
            countIf(isNotNull(warning_to_failure_cycles)) = 0,
            null,
            quantileExactInclusive(0.9)(warning_to_failure_cycles)
        ) as p90_warning_to_failure_cycles,
        if(
            countIf(isNotNull(warning_to_failure_seconds)) = 0,
            null,
            quantileExactInclusive(0.5)(warning_to_failure_seconds)
        ) as median_warning_to_failure_seconds,
        if(
            countIf(isNotNull(shop_visit_seconds)) = 0,
            null,
            quantileExactInclusive(0.5)(shop_visit_seconds)
        ) as median_shop_visit_seconds
    from visits
    group by {{ report_grouping }}

)

select
    cast(engine_scope as LowCardinality(String)) as engine_id,
    cast(failure_mode_scope as LowCardinality(String)) as failure_mode,
    shop_visits,
    visits_with_warning,
    visits_skipped_warning,
    visits_returned,
    visits_skipped_warning / shop_visits as share_skipped_warning,
    median_warning_to_failure_cycles,
    p90_warning_to_failure_cycles,
    median_warning_to_failure_seconds,
    median_shop_visit_seconds,
    now64(3, 'UTC') as refreshed_at
from rolled
