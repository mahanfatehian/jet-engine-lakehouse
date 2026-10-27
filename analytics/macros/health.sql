{#-
    Health KPI helpers of fct_engine_health_status.

    NULL-preserving clamps. Since ClickHouse 24.12 greatest/least skip NULL arguments
    (least_greatest_legacy_null_behavior = 0): greatest(NULL, 0) is 0 and least(NULL, 1) is 1, so
    a plain clamp turns "no data" into a measured value. These keep NULL as NULL.
-#}

{% macro at_least(expression, floor) -%}
    if(isNull({{ expression }}), NULL, greatest({{ expression }}, {{ floor }}))
{%- endmacro %}


{% macro clamp_unit(expression) -%}
    if(isNull({{ expression }}), NULL, least(greatest({{ expression }}, 0.0), 1.0))
{%- endmacro %}


{% macro degradation_component(kpi, full_scale_var, healthy_value=0.0) -%}
    {#- linear 0..1 score of a drift KPI: 0 at its healthy value, 1 at the configured full scale -#}
    {%- set full_scale = number_var(full_scale_var) -%}
    {%- if full_scale <= healthy_value -%}
        {{ exceptions.raise_compiler_error("var '" ~ full_scale_var ~ "' must be greater than " ~ healthy_value) }}
    {%- endif -%}
    {{ clamp_unit('(' ~ kpi ~ ' - ' ~ healthy_value ~ ') / ' ~ (full_scale - healthy_value)) }}
{%- endmacro %}


{% macro bearing_defect_detected(amp_bpfo, kurtosis) -%}
    {#-
        Bearing-defect rule of fct_engine_health_status (and of its calibration guard): the fan
        bearing's outer-race defect tone (BPFO) or the impulsiveness (kurtosis) of the fan
        vibration exceeds its threshold. SQL three-valued logic: NULL unless a known KPI exceeds
        its threshold or both KPIs are known, so callers decide what missing HF data means.
    -#}
    (
        {{ amp_bpfo }} > {{ number_var('anomaly_fan_amp_bpfo_mm_s') }}
        or {{ kurtosis }} > {{ number_var('anomaly_fan_kurtosis') }}
    )
{%- endmacro %}
