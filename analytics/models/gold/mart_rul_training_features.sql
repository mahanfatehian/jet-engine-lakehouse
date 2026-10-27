{{
    config(
        materialized='table',
        engine='ReplacingMergeTree',
        order_by='(engine_id, lifecycle_id, telemetry_id)',
    )
}}

{#-
    RUL training set (CONTRACT.md section 7): the point-in-time features of every pulse of a
    COMPLETED lifecycle (one that has a FAILURE pulse), labelled with the cycles left until that
    failure. The features come from int_ml_features, the interface predict_rul.py scores too:
    the 46 sensor features plus the 10 spectral features of the HF path (CONTRACT_V2.md
    section 8; NULL where a pulse has no HF frame). Rebuilt on every run so late pulses, late HF
    frames, corrected rows and late work orders (failure mode) are always reflected. Ground
    truth (wear_factor) and status never become columns here.

    Only the newest `mart_max_lifecycles_per_engine` completed lifecycles of each engine are
    kept, so the cost of this rebuild and the size of the training set stay bounded however long
    the fleet runs. Lifecycle ids grow with every repair, so the newest are the highest.
-#}

with completed_lifecycles as (

    select
        engine_id,
        lifecycle_id,
        assumeNotNull(failure_cycle) as lifecycle_failure_cycle,
        coalesce(failure_mode, 'UNKNOWN') as lifecycle_failure_mode
    from {{ ref('int_engine_lifecycles') }} final
    where has_failed = 1
    order by engine_id, lifecycle_id desc
    limit {{ positive_int_var('mart_max_lifecycles_per_engine') }} by engine_id

),

pulse_features as (

    select
        {{ ml_feature_output_columns() | join(',\n        ') }}
    from {{ ref('int_ml_features') }}
    -- pushed down into both FINAL reads of the view, where the sorting-key prefix prunes every
    -- lifecycle outside the training window
    where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from completed_lifecycles)

),

labelled as (

    select
        {{ ml_feature_output_columns() | join(',\n        ') }},
        lifecycle_failure_cycle,
        lifecycle_failure_mode,
        toInt32(lifecycle_failure_cycle) - toInt32(cycle_number) as remaining_cycles
    from pulse_features
    inner join completed_lifecycles using (engine_id, lifecycle_id)
    -- a pulse after the failure pulse would carry a negative RUL
    where cycle_number <= lifecycle_failure_cycle

)

select
    {{ ml_feature_output_columns() | join(',\n    ') }},
    cast(lifecycle_failure_cycle as UInt32) as failure_cycle,
    cast(remaining_cycles as Int32) as rul_cycles,
    cast(least(remaining_cycles, {{ int_var('rul_cap_cycles') }}) as Int32) as rul_cycles_capped,
    toUInt8(remaining_cycles <= {{ int_var('rul_failure_horizon_cycles') }}) as failure_within_horizon,
    cast(lifecycle_failure_mode as String) as failure_mode,
    cast(concat(engine_id, '#', toString(lifecycle_id)) as String) as split_group
from labelled
