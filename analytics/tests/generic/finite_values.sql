{% test finite_values(model, columns, use_final=false) %}
{#-
    Rows where any of `columns` is NULL, NaN or +/-Inf. ClickHouse float arithmetic yields NaN
    or Inf instead of failing (0/0, sqrt of a negative), so feature tables need this guard.
    `use_final` reads the deduplicated view of a ReplacingMergeTree.
-#}
select *
from {{ model }}{{ ' final' if use_final }}
where not (
    {%- for column in columns %}
    isNotNull({{ column }}) and isFinite({{ column }}){{ ' and' if not loop.last }}
    {%- endfor %}
)

{% endtest %}
