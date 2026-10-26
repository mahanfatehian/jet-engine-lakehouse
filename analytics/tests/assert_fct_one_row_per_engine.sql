{#-
    The health board must hold exactly one row per active engine: every engine whose current
    registry version is not a delete, plus every engine seen in telemetry that was not
    deregistered.
-#}

with registry_current as (

    select engine_id, is_deleted
    from {{ ref('stg_engine_status_registry') }} final
    order by engine_id, cdc_lsn desc
    limit 1 by engine_id

),

expected_engines as (

    select engine_id, toUInt8(1) as is_expected
    from (
        select engine_id from registry_current where is_deleted = 0
        union distinct
        select engine_id from {{ ref('stg_engine_telemetry') }} final
        where engine_id not in (select engine_id from registry_current where is_deleted = 1)
    )

),

board as (

    select engine_id, count() as board_rows
    from {{ ref('fct_engine_health_status') }} final
    group by engine_id

)

-- a side without the engine reads as 0 (ClickHouse join default)
select engine_id, is_expected, board_rows
from expected_engines
full outer join board using (engine_id)
where is_expected != 1 or board_rows != 1
