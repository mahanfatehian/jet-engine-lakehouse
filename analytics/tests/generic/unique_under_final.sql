{% test unique_under_final(model, column_name=none, columns=none) %}
{#-
    Uniqueness as consumers see it: ReplacingMergeTree tables keep superseded versions until a
    background merge, so dbt's built-in `unique` would flag them; FINAL collapses them first.
    Attach to a column, or at model level with `columns` for a composite key.
-#}
{%- set key_columns = columns if columns else [column_name] -%}
{%- if not key_columns or key_columns[0] is none -%}
    {{ exceptions.raise_compiler_error("unique_under_final needs a column or a `columns` list") }}
{%- endif -%}

select
    {{ key_columns | join(', ') }},
    count() as row_versions
from {{ model }} final
group by {{ key_columns | join(', ') }}
having row_versions > 1

{% endtest %}
