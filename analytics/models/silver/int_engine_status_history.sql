{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, valid_from)',
    )
}}

{#-
    SCD Type 2 status history of every engine: one row per engine and status interval
    [valid_from, valid_to), valid_to NULL while the interval is open (is_current = 1). No dbt
    snapshot is needed: stg_engine_status_registry keeps every CDC version of the registry, so the
    change log is the history, and rebuilding it from the log always yields the same intervals.

    Versions are read in log order (cdc_lsn) under FINAL: replays collapse, and a version that
    arrived late or out of order sorts where the source wrote it. A version opens an interval when
    its status or status_changed_ts differs from the previous version of the engine (the progress
    updates of one status fold into its interval) or when it follows a delete. A delete (the
    registry row was removed: the engine was deregistered) closes the open interval at its commit
    time; the engine has no interval until a later insert registers it again.

    valid_from is the source's status_changed_ts, i.e. event time: the first-boot backfill commits
    hours of history in one transaction, so commit times would compress it into an instant. That
    history is backdated behind the registration row, so the event times of an engine's changes
    need not grow with the log: every boundary takes the earliest time of itself and all later
    boundaries of the engine (a change backdated before earlier ones supersedes them). Intervals
    therefore never overlap, and the ones left without duration (the registration row the backfill
    wrote its history behind) are dropped.

    Table, rebuilt on every run: every engine writes a registry version per cycle, so an
    incremental model keyed by engine would recompute every engine on every run anyway, and one
    late or backdated version can move intervals anywhere in an engine's history. The rebuild
    reads one registry version per engine and cycle (~52k a day for the default fleet), and it
    equals a full refresh by construction.
-#}

with versions as (

    select
        engine_id,
        engine_model,
        operational_status,
        current_lifecycle_id,
        current_cycle,
        status_changed_ts,
        cdc_op,
        cdc_lsn,
        toDateTime64(cdc_source_ts, 6, 'UTC') as committed_at,
        is_deleted
    from {{ ref('stg_engine_status_registry') }} final

),

compared as (

    -- every version next to the previous version of its engine in log order
    select
        *,
        lagInFrame(toNullable(operational_status)) over w_previous as previous_status,
        lagInFrame(toNullable(status_changed_ts)) over w_previous as previous_status_changed_ts,
        lagInFrame(toNullable(is_deleted)) over w_previous as previous_is_deleted
    from versions
    window w_previous as (partition by engine_id order by cdc_lsn rows between 1 preceding and current row)

),

boundaries as (

    -- versions that open an interval, and deletes, which close one
    select
        engine_id,
        engine_model,
        operational_status,
        current_lifecycle_id,
        current_cycle,
        cdc_op,
        cdc_lsn,
        is_deleted,
        if(is_deleted = 1, committed_at, status_changed_ts) as boundary_ts
    from compared
    where is_deleted = 1
       or isNull(previous_status)
       or previous_is_deleted = 1
       or previous_status != operational_status
       or previous_status_changed_ts != status_changed_ts

),

effective as (

    select
        *,
        -- monotonic event time: the earliest time of this and every later boundary of the engine
        min(boundary_ts) over (
            partition by engine_id order by cdc_lsn rows between current row and unbounded following
        ) as effective_ts,
        sum(is_deleted) over (
            partition by engine_id order by cdc_lsn rows between unbounded preceding and 1 preceding
        ) as deletes_before,
        sum(is_deleted) over (partition by engine_id) as engine_deletes
    from boundaries

),

bounded as (

    select
        *,
        leadInFrame(toNullable(effective_ts)) over (
            partition by engine_id order by cdc_lsn rows between current row and 1 following
        ) as next_boundary_ts
    from effective

),

intervals as (

    -- an opening whose next boundary takes effect at the same time never lasted
    select
        *,
        leadInFrame(toNullable(deletes_before)) over (
            partition by engine_id order by cdc_lsn rows between current row and 1 following
        ) as next_interval_deletes_before
    from bounded
    where is_deleted = 0
      and (isNull(next_boundary_ts) or next_boundary_ts > effective_ts)

)

select
    engine_id,
    engine_model,
    operational_status,
    cast(operational_status in ({{ available_statuses_sql() }}) as UInt8) as is_available,
    current_lifecycle_id as lifecycle_id,
    current_cycle as cycle_at_entry,
    effective_ts as valid_from,
    next_boundary_ts as valid_to,
    toUInt8(isNull(next_boundary_ts)) as is_current,
    {{ micros_between('effective_ts', 'next_boundary_ts') }} / 1000000.0 as duration_seconds,
    -- a delete between this interval and the next one (or after the last one) deregistered the engine
    toUInt8(coalesce(next_interval_deletes_before, engine_deletes) > deletes_before) as closed_by_delete,
    cdc_op as entry_cdc_op,
    cdc_lsn as entry_cdc_lsn
from intervals
