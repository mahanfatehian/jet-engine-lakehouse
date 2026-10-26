{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key='telemetry_id',
        on_schema_change='fail',
        engine='ReplacingMergeTree(feature_computed_at, is_deleted)',
        order_by='(engine_id, lifecycle_id, telemetry_id)',
        partition_by='toYYYYMM(event_ts)',
    )
}}

{#-
    Point-in-time sensor features: one row per live telemetry pulse (CONTRACT.md section 7).

    Every feature only looks backwards in time (RANGE windows over epoch seconds, lag-1 deltas,
    expanding lifecycle baseline), so a row's values match what was knowable when it was measured.

    Incremental contract: a lifecycle is recomputed in full whenever any version of one of its
    pulses landed in stg_engine_telemetry after this model's watermark (minus the lookback),
    when its engine_model changed, or when it was computed with another `baseline_cycles` (each
    row records the value it used, so a changed definition never leaves old lifecycles behind).
    A late or corrected pulse shifts the windows, deltas and possibly the baseline of its
    neighbours, so whole-lifecycle recomputation is what keeps the table identical to a
    --full-refresh. Recomputed rows win through the feature_computed_at version; pulses that
    disappeared upstream are retracted with is_deleted = 1. Readers must use FINAL.
    unique_key only makes dbt-clickhouse run its on_schema_change check before appending (it
    skips that check for key-less appends); uniqueness comes from the engine key under FINAL.
-#}

{%- set sensors = raw_sensor_columns() -%}
{%- set rolled_sensors = rolling_sensor_columns() -%}
{%- set windows = rolling_windows() -%}
{%- set baseline_cycles = int_var('baseline_cycles') -%}

with

registry_models as (

    -- an engine without (or with an empty) registry model is published as UNKNOWN
    select
        engine_id,
        coalesce(nullIf(argMax(engine_model, cdc_lsn), ''), 'UNKNOWN') as registry_engine_model
    from {{ ref('stg_engine_status_registry') }} final
    group by engine_id

),
{%- if is_incremental() %}

changed_pulses as (

    -- no FINAL: superseded and deleted versions are changes as well
    select telemetry_id
    from {{ ref('stg_engine_telemetry') }}
    where ingested_at > {{ lookback_watermark('source_ingested_at') }}

),

published_definitions as (

    -- what each lifecycle was last computed with (lifecycles are always recomputed in full)
    select
        engine_id,
        lifecycle_id,
        argMaxIf(engine_model, feature_computed_at, is_deleted = 0) as published_engine_model,
        argMaxIf(baseline_cycles, feature_computed_at, is_deleted = 0) as published_baseline_cycles
    from {{ this }}
    group by engine_id, lifecycle_id

),

touched_lifecycles as (

    -- every lifecycle that any version of a changed pulse belongs, or belonged, to
    select engine_id, lifecycle_id
    from {{ ref('stg_engine_telemetry') }}
    where telemetry_id in (select telemetry_id from changed_pulses)

    union distinct

    -- lifecycles published before their engine's registry row (and model) had landed
    select engine_id, lifecycle_id
    from published_definitions
    inner join registry_models using (engine_id)
    where published_engine_model != registry_engine_model

    union distinct

    -- lifecycles computed with another baseline_cycles (a changed var or a one-off --vars run)
    select engine_id, lifecycle_id
    from published_definitions
    where published_baseline_cycles != {{ baseline_cycles }}

),
{%- endif %}

telemetry as (

    select
        telemetry_id,
        engine_id,
        lifecycle_id,
        cycle_number,
        event_ts,
        {%- for sensor in sensors %}
        {{ sensor }},
        {%- endfor %}
        ingested_at,
        toUnixTimestamp64Micro(event_ts) / 1000000.0 as event_epoch_s
    from {{ ref('stg_engine_telemetry') }} final
    {%- if is_incremental() %}
    where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from touched_lifecycles)
    {%- endif %}

),

baseline_periods as (

    -- healthy reference period of each lifecycle: its first `baseline_cycles` cycles
    select
        engine_id,
        lifecycle_id,
        min(cycle_number) + {{ baseline_cycles }} as baseline_end_cycle
    from telemetry
    group by engine_id, lifecycle_id

),

flagged_telemetry as (

    select
        telemetry_id,
        engine_id,
        lifecycle_id,
        cycle_number,
        event_ts,
        {%- for sensor in sensors %}
        {{ sensor }},
        {%- endfor %}
        ingested_at,
        event_epoch_s,
        cycle_number < baseline_end_cycle as is_baseline_pulse
    from telemetry
    inner join baseline_periods using (engine_id, lifecycle_id)

),

windowed as (

    select
        telemetry_id,
        engine_id,
        lifecycle_id,
        cycle_number,
        event_ts,
        ingested_at,
        {%- for sensor in sensors %}
        {{ sensor }},
        -- expanding mean: until the baseline period is over it only covers the pulses seen so far
        avgIf({{ sensor }}, is_baseline_pulse) over w_to_date as {{ sensor }}_baseline,
        {%- endfor %}
        countIf(is_baseline_pulse) over w_to_date as baseline_pulses,
        {%- for sensor in rolled_sensors %}
        lagInFrame({{ sensor }}, 1, {{ sensor }}) over w_previous as {{ sensor }}_previous,
        {%- for window in windows %}
        {{ window_statistics(sensor, window.label) }},
        {%- endfor %}
        {%- endfor %}
        {%- for window in windows %}
        event_epoch_s - min(event_epoch_s) over w_{{ window.label }} as elapsed_s_{{ window.label }}
        {{- ',' if not loop.last }}
        {%- endfor %}
    from flagged_telemetry
    window
        {%- for window in windows %}
        w_{{ window.label }} as {{ time_range_window(window.seconds) }},
        {%- endfor %}
        w_previous as {{ row_order_window('1 preceding and current row') }},
        w_to_date as {{ row_order_window('unbounded preceding and current row') }}

),

referenced as (

    -- drift references; a pulse that precedes every baseline pulse is its own reference
    select
        *,
        if(baseline_pulses > 0, egt_c_baseline, egt_c_mean_5m) as egt_c_reference,
        if(baseline_pulses > 0, core_temp_c_baseline, core_temp_c) as core_temp_c_reference,
        if(baseline_pulses > 0, vibration_mm_s_baseline, vibration_mm_s_mean_5m) as vibration_mm_s_reference,
        if(baseline_pulses > 0, oil_pressure_psi_baseline, oil_pressure_psi_mean_5m) as oil_pressure_psi_reference,
        if(baseline_pulses > 0, fuel_flow_kg_h_baseline, fuel_flow_kg_h) as fuel_flow_kg_h_reference,
        if(baseline_pulses > 0, rpm_baseline, rpm) as rpm_reference
    from windowed

),

features as (

    select
        telemetry_id,
        engine_id,
        -- the LEFT JOIN yields '' for engines whose registry row has not landed yet
        cast(
            if(registry_engine_model = '', 'UNKNOWN', registry_engine_model) as LowCardinality(String)
        ) as engine_model,
        lifecycle_id,
        cycle_number,
        event_ts,
        {%- for sensor in sensors %}
        {{ sensor }},
        {%- endfor %}
        {%- for column in rolling_feature_columns() %}
        {{ column }},
        {%- endfor %}
        {%- for sensor in rolled_sensors %}
        {{ sensor }} - {{ sensor }}_previous as {{ sensor }}_delta_1,
        {%- for window in windows %}
        {{ rate_of_change(sensor, window.label) }} as {{ sensor }}_roc_{{ window.label }},
        {%- endfor %}
        {%- endfor %}
        egt_c_mean_5m - egt_c_reference as egt_c_drift,
        core_temp_c - core_temp_c_reference as core_temp_c_drift,
        {{ safe_divide('vibration_mm_s_mean_5m', 'vibration_mm_s_reference', '1.0') }} as vibration_mm_s_ratio,
        oil_pressure_psi_reference - oil_pressure_psi_mean_5m as oil_pressure_psi_drop,
        {{ safe_divide('100.0 * (fuel_flow_kg_h - fuel_flow_kg_h_reference)', 'fuel_flow_kg_h_reference', '0.0') }}
            as fuel_flow_kg_h_drift_pct,
        {{ safe_divide('100.0 * (rpm - rpm_reference)', 'rpm_reference', '0.0') }} as rpm_drift_pct,
        toUInt32({{ baseline_cycles }}) as baseline_cycles,
        ingested_at as source_ingested_at,
        now64(6, 'UTC') as feature_computed_at,
        toUInt8(0) as is_deleted
    from referenced
    left join registry_models using (engine_id)

)

select
    {{ feature_output_columns() | join(',\n    ') }},
    baseline_cycles,
    source_ingested_at,
    feature_computed_at,
    is_deleted
from features
{%- if is_incremental() %}

union all

-- retract published pulses of recomputed lifecycles that no longer exist upstream
select
    {{ feature_output_columns() | join(',\n    ') }},
    baseline_cycles,
    source_ingested_at,
    now64(6, 'UTC') as feature_computed_at,
    toUInt8(1) as is_deleted
from {{ this }} final
where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from touched_lifecycles)
  and (engine_id, lifecycle_id, telemetry_id) not in (
      select engine_id, lifecycle_id, telemetry_id from telemetry
  )
{%- endif %}
