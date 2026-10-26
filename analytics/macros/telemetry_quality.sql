{% macro telemetry_readings_are_finite() -%}
    {#-
        True when every sensor reading of a telemetry version is a real number. Postgres double
        precision accepts NaN/Infinity, which would poison every rolling feature of the lifecycle.
    -#}
    (
        {%- for sensor in raw_sensor_columns() %}
        isFinite({{ sensor }}){{ ' and' if not loop.last }}
        {%- endfor %}
    )
{%- endmacro %}
