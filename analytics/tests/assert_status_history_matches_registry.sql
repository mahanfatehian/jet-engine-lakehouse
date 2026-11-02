{#-
    The open interval of the SCD2 status history is the registry's current state: every engine
    whose newest CDC version (highest cdc_lsn) is not a delete has exactly one open interval, in
    that version's status, since that version's status_changed_ts; a deregistered engine (newest
    version a delete) has none.
-#}

with registry_current as (

    select
        engine_id,
        operational_status as registry_status,
        status_changed_ts as registry_status_changed_ts,
        is_deleted,
        toUInt8(1) as is_registered
    from {{ ref('stg_engine_status_registry') }} final
    order by engine_id, cdc_lsn desc
    limit 1 by engine_id

),

open_intervals as (

    select
        engine_id,
        count() as open_intervals,
        any(operational_status) as history_status,
        any(valid_from) as history_valid_from
    from {{ ref('int_engine_status_history') }} final
    where is_current = 1
    group by engine_id

)

-- a side without the engine reads as 0 / '' (ClickHouse join default)
select
    engine_id,
    is_registered,
    is_deleted,
    registry_status,
    history_status,
    registry_status_changed_ts,
    history_valid_from,
    open_intervals
from registry_current
full outer join open_intervals using (engine_id)
where is_registered = 0
   or (is_deleted = 1 and open_intervals != 0)
   or (
       is_deleted = 0
       and (
           open_intervals != 1
           or history_status != registry_status
           or history_valid_from != registry_status_changed_ts
       )
   )
