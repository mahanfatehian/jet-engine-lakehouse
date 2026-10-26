{% test accepted_range(model, column_name, min_value=none, max_value=none, use_final=false) %}
{#- Rows outside [min_value, max_value] (inclusive; either bound optional); NULLs are ignored. -#}
{%- if min_value is none and max_value is none -%}
    {{ exceptions.raise_compiler_error("accepted_range needs min_value and/or max_value") }}
{%- endif -%}

select {{ column_name }}
from {{ model }}{{ ' final' if use_final }}
where {{ column_name }} is not null
  and not (
    {%- if min_value is not none %} {{ column_name }} >= {{ min_value }}{% endif %}
    {%- if min_value is not none and max_value is not none %} and{% endif %}
    {%- if max_value is not none %} {{ column_name }} <= {{ max_value }}{% endif %}
  )

{% endtest %}
