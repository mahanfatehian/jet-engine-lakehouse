{#-
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
