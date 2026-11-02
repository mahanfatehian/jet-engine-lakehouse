{#-
    The SCD2 status history must tile every engine's time: in valid_from order each interval
    ends exactly where the next one begins (no gap, no overlap), except after a deregistration,
    where the next interval may only begin later; only the engine's last interval may be open,
    and it must be open unless the engine was deregistered.
-#}

with ordered as (

    select
        engine_id,
        valid_from,
        valid_to,
        is_current,
        closed_by_delete,
        leadInFrame(toNullable(valid_from)) over w_next as next_valid_from
    from {{ ref('int_engine_status_history') }} final
    window w_next as (partition by engine_id order by valid_from rows between current row and 1 following)

)

select
    engine_id,
    valid_from,
    valid_to,
    next_valid_from,
    is_current,
    closed_by_delete,
    multiIf(
        isNotNull(valid_to) and valid_to <= valid_from, 'empty_interval',
        isNull(next_valid_from) and is_current = 0 and closed_by_delete = 0, 'last_interval_closed',
        isNotNull(next_valid_from) and is_current = 1, 'open_interval_not_last',
        isNotNull(next_valid_from) and closed_by_delete = 0 and valid_to != next_valid_from, 'gap_or_overlap',
        isNotNull(next_valid_from) and closed_by_delete = 1 and valid_to > next_valid_from, 'overlap_after_delete',
        'ok'
    ) as issue
from ordered
where issue != 'ok'
