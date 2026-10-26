{#-
    SQL fragments of int_sensor_rolling_features. They expect the columns engine_id,
    lifecycle_id, telemetry_id and event_epoch_s (Float64 epoch seconds of event_ts).
-#}

{% macro time_range_window(seconds) -%}
    (
        partition by engine_id, lifecycle_id
        order by event_epoch_s
        range between {{ seconds }} preceding and current row
    )
{%- endmacro %}


{% macro row_order_window(frame) -%}
    {#- ties on event_ts are broken by telemetry_id so lag/cumulative results are deterministic -#}
    (
        partition by engine_id, lifecycle_id
        order by event_epoch_s, telemetry_id
        rows between {{ frame }}
    )
{%- endmacro %}


{% macro window_statistics(sensor, window_label) -%}
    {#- population std (stddevPopStable) is 0 for a single-row window and never goes negative -#}
    avg({{ sensor }}) over w_{{ window_label }} as {{ sensor }}_mean_{{ window_label }},
    stddevPopStable({{ sensor }}) over w_{{ window_label }} as {{ sensor }}_std_{{ window_label }},
    min({{ sensor }}) over w_{{ window_label }} as {{ sensor }}_min_{{ window_label }},
    max({{ sensor }}) over w_{{ window_label }} as {{ sensor }}_max_{{ window_label }},
    argMin({{ sensor }}, (event_epoch_s, telemetry_id)) over w_{{ window_label }}
        as {{ sensor }}_first_{{ window_label }}
{%- endmacro %}


{% macro rate_of_change(sensor, window_label) -%}
    {#- change per minute since the oldest reading still inside the window -#}
    if(
        elapsed_s_{{ window_label }} > 0,
        ({{ sensor }} - {{ sensor }}_first_{{ window_label }}) / (elapsed_s_{{ window_label }} / 60.0),
        0.0
    )
{%- endmacro %}


{% macro safe_divide(numerator, denominator, fallback) -%}
    if({{ denominator }} != 0, {{ numerator }} / {{ denominator }}, {{ fallback }})
{%- endmacro %}
