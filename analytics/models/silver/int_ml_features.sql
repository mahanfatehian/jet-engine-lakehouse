{{ config(materialized='view') }}

{#-
    The ML feature vector of every live telemetry pulse (CONTRACT_V2.md sections 8 and 9): the
    point-in-time features of int_sensor_rolling_features plus the 5-minute spectral features of
    the pulse's own HF frame (int_vibration_features, matched on engine, lifecycle and the
    frame's telemetry_id). The training mart and predict_rul.py both read their features here,
    so training and scoring see the same definition.

    The HF path is best effort and lags the CDC path: the spectral columns are NULL for a pulse
    whose frame was dropped, has not landed yet or was never produced (HF disabled), and the
    models treat NULL as missing. A view over FINAL reads, so a landed frame shows up at the
    next read, and the LEFT JOIN onto a key that is unique under FINAL keeps exactly one row
    per row of int_sensor_rolling_features FINAL.
-#}

select
    {{ ml_feature_output_columns() | join(',\n    ') }},
    baseline_cycles,
    source_ingested_at,
    feature_computed_at,
    spectral_frames_5m
from {{ ref('int_sensor_rolling_features') }} final
left join (

    -- Nullable columns, so a pulse without a frame reads NULL rather than the join default 0
    select
        engine_id,
        lifecycle_id,
        telemetry_id,
        {%- for column in spectral_feature_columns() %}
        {{ column }},
        {%- endfor %}
        toNullable(frames_5m) as spectral_frames_5m
    from {{ ref('int_vibration_features') }} final

) as spectral_features using (engine_id, lifecycle_id, telemetry_id)
