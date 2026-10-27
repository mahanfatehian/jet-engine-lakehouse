{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key=['engine_id', 'lifecycle_id', 'telemetry_id'],
        on_schema_change='fail',
        engine='ReplacingMergeTree(feature_computed_at, is_deleted)',
        order_by='(engine_id, lifecycle_id, telemetry_id)',
        partition_by='toYYYYMM(frame_start_ts)',
    )
}}

{#-
    Point-in-time spectral features: one row per HF frame with the 5-minute rolling means of
    its spectral and oil-system features (CONTRACT_V2.md section 8). The RANGE window runs over
    epoch seconds of frame_start_ts (= event_ts of the frame's pulse), partitioned by
    (engine_id, lifecycle_id), current frame included, so a row only depends on frames of its
    own tick or earlier. A feature that is NULL in a frame (unusable or not carried) is left
    out of the mean; a mean without any value in its window is NULL.

    The HF stream is the only input: frames are measurements in their own right, so a frame
    whose 1 Hz pulse is missing, late or deleted still counts in its neighbours' windows;
    int_ml_features attaches a frame's features to its pulse.

    Incremental contract (as int_sensor_rolling_features): a lifecycle is recomputed in full
    whenever any copy of one of its frames landed in stg_engine_hf_features after this model's
    watermark minus the lookback, because a late or corrected frame shifts the windows of its
    neighbours. Recomputed rows win through the feature_computed_at version; a frame that no
    longer belongs to a recomputed lifecycle (a copy corrected its lifecycle_id) is retracted
    with is_deleted = 1. Readers must use FINAL.
-#}

{%- set spectral_features = spectral_feature_sources() -%}
{%- set window = rolling_windows() | selectattr('label', 'equalto', '5m') | first -%}

with
{%- if is_incremental() %}

changed_frames as (

    -- no FINAL: superseded copies are changes as well
    select engine_id, telemetry_id
    from {{ ref('stg_engine_hf_features') }}
    where ingested_at > {{ lookback_watermark('source_ingested_at') }}

),

touched_lifecycles as (

    -- every lifecycle that any copy of a changed frame belongs, or belonged, to
    select distinct engine_id, lifecycle_id
    from {{ ref('stg_engine_hf_features') }}
    where (engine_id, telemetry_id) in (select engine_id, telemetry_id from changed_frames)

),
{%- endif %}

frames as (

    select
        engine_id,
        lifecycle_id,
        telemetry_id,
        cycle_number,
        frame_start_ts,
        {%- for feature in spectral_features %}
        {{ feature.source }},
        {%- endfor %}
        ingested_at,
        toUnixTimestamp64Micro(frame_start_ts) / 1000000.0 as event_epoch_s
    from {{ ref('stg_engine_hf_features') }} final
    {%- if is_incremental() %}
    where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from touched_lifecycles)
    {%- endif %}

),

windowed as (

    select
        engine_id,
        lifecycle_id,
        telemetry_id,
        cycle_number,
        frame_start_ts,
        {%- for feature in spectral_features %}
        avg({{ feature.source }}) over w_{{ window.label }} as {{ feature.name }},
        {%- endfor %}
        toUInt32(count() over w_{{ window.label }}) as frames_{{ window.label }},
        ingested_at
    from frames
    window w_{{ window.label }} as {{ time_range_window(window.seconds) }}

)

select
    engine_id,
    lifecycle_id,
    telemetry_id,
    cycle_number,
    frame_start_ts,
    {%- for feature in spectral_features %}
    {{ feature.name }},
    {%- endfor %}
    frames_{{ window.label }},
    ingested_at as source_ingested_at,
    now64(6, 'UTC') as feature_computed_at,
    toUInt8(0) as is_deleted
from windowed
{%- if is_incremental() %}

union all

-- retract published frames of recomputed lifecycles that no longer belong to them upstream
select
    engine_id,
    lifecycle_id,
    telemetry_id,
    cycle_number,
    frame_start_ts,
    {%- for feature in spectral_features %}
    {{ feature.name }},
    {%- endfor %}
    frames_{{ window.label }},
    source_ingested_at,
    now64(6, 'UTC') as feature_computed_at,
    toUInt8(1) as is_deleted
from {{ this }} final
where (engine_id, lifecycle_id) in (select engine_id, lifecycle_id from touched_lifecycles)
  and (engine_id, lifecycle_id, telemetry_id) not in (
      select engine_id, lifecycle_id, telemetry_id from frames
  )
{%- endif %}
