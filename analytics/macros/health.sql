{% macro clamp_unit(expression) -%}
    least(greatest({{ expression }}, 0.0), 1.0)
{%- endmacro %}


{% macro degradation_component(kpi, full_scale_var, healthy_value=0.0) -%}
    {#- linear 0..1 score of a drift KPI: 0 at its healthy value, 1 at the configured full scale -#}
    {%- set full_scale = number_var(full_scale_var) -%}
    {%- if full_scale <= healthy_value -%}
        {{ exceptions.raise_compiler_error("var '" ~ full_scale_var ~ "' must be greater than " ~ healthy_value) }}
    {%- endif -%}
    {{ clamp_unit('(' ~ kpi ~ ' - ' ~ healthy_value ~ ') / ' ~ (full_scale - healthy_value)) }}
{%- endmacro %}
