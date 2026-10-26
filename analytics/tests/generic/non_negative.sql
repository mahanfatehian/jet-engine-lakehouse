{% test non_negative(model, column_name, use_final=false) %}
{#- Rows with a negative value; NULLs are left to not_null tests. -#}
select {{ column_name }}
from {{ model }}{{ ' final' if use_final }}
where {{ column_name }} < 0

{% endtest %}
