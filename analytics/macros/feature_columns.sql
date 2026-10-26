{#-
    Single source of truth for the ML feature interface (CONTRACT.md section 7).

    The order returned here is the physical column order of int_sensor_rolling_features and
    mart_rul_training_features, and matches FEATURE_COLUMNS in data_science/feature_pipeline.py
    (cycle_number followed by every float feature).
-#}

{% macro raw_sensor_columns() %}
    {{ return(['rpm', 'core_temp_c', 'egt_c', 'vibration_mm_s', 'fuel_flow_kg_h', 'oil_pressure_psi']) }}
{% endmacro %}


{% macro rolling_sensor_columns() %}
    {{ return(['vibration_mm_s', 'egt_c', 'oil_pressure_psi']) }}
{% endmacro %}


{% macro rolling_windows() %}
    {#- label used in column names -> length of the RANGE frame over epoch seconds -#}
    {{ return([{'label': '5m', 'seconds': 300}, {'label': '15m', 'seconds': 900}]) }}
{% endmacro %}


{% macro rolling_statistics() %}
    {{ return(['mean', 'std', 'min', 'max']) }}
{% endmacro %}


{% macro feature_identifier_columns() %}
    {{ return(['telemetry_id', 'engine_id', 'engine_model', 'lifecycle_id', 'cycle_number', 'event_ts']) }}
{% endmacro %}


{% macro rolling_feature_columns() %}
    {%- set columns = [] -%}
    {%- for sensor in rolling_sensor_columns() -%}
        {%- for window in rolling_windows() -%}
            {%- for statistic in rolling_statistics() -%}
                {%- do columns.append(sensor ~ '_' ~ statistic ~ '_' ~ window.label) -%}
            {%- endfor -%}
        {%- endfor -%}
    {%- endfor -%}
    {{ return(columns) }}
{% endmacro %}


{% macro delta_feature_columns() %}
    {%- set columns = [] -%}
    {%- for sensor in rolling_sensor_columns() -%}
        {%- do columns.append(sensor ~ '_delta_1') -%}
        {%- for window in rolling_windows() -%}
            {%- do columns.append(sensor ~ '_roc_' ~ window.label) -%}
        {%- endfor -%}
    {%- endfor -%}
    {{ return(columns) }}
{% endmacro %}


{% macro drift_feature_columns() %}
    {{ return([
        'egt_c_drift',
        'core_temp_c_drift',
        'vibration_mm_s_ratio',
        'oil_pressure_psi_drop',
        'fuel_flow_kg_h_drift_pct',
        'rpm_drift_pct',
    ]) }}
{% endmacro %}


{% macro float_feature_columns() %}
    {#- the 45 Float64 model inputs; with cycle_number they form the 46 numeric features -#}
    {{ return(
        raw_sensor_columns() + rolling_feature_columns() + delta_feature_columns() + drift_feature_columns()
    ) }}
{% endmacro %}


{% macro feature_output_columns() %}
    {#- identifiers + float features: the column prefix shared by the silver feature table and the gold mart -#}
    {{ return(feature_identifier_columns() + float_feature_columns()) }}
{% endmacro %}
