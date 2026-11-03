{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='outage_start',
    )
}}

{#-
    Platform outages: every pause of the whole fleet's telemetry longer than
    outage_min_gap_seconds, as [outage_start, outage_end) from the last pulse before it to the
    first pulse after it. A running fleet never pauses that long, so the platform observed
    nothing meanwhile, while the wall-clock status history keeps every engine in its last status:
    the reliability marts count that time as unobserved (outage_overlap_us). Table, rebuilt on
    every run: a late pulse can split or shorten any outage.
-#}

{%- set min_gap_us = positive_int_var('outage_min_gap_seconds') * 1000000 %}

with pauses as (

    -- the engines of one tick share its event time; the first pulse follows itself: no pause
    select
        lagInFrame(event_ts, 1, event_ts) over (order by event_ts rows between 1 preceding and current row)
            as outage_start,
        event_ts as outage_end
    -- no FINAL: a retracted pulse (source delete, unusable readings) still proves the platform ran
    from (select distinct event_ts from {{ ref('stg_engine_telemetry') }})

)

select outage_start, outage_end
from pauses
where {{ micros_between('outage_start', 'outage_end') }} > {{ min_gap_us }}
