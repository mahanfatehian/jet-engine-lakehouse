{% test expression_is_true(model, expression, use_final=false) %}
{#- Rows for which `expression` is false or NULL. -#}
select *
from {{ model }}{{ ' final' if use_final }}
where not coalesce(({{ expression }}), 0)

{% endtest %}
