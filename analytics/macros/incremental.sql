{% macro lookback_watermark(target_column) -%}
    {#-
        Lower bound for the next incremental batch: the newest ingestion time already published
        in this model, minus `ingestion_lookback_minutes`. On an empty target max() returns the
        epoch, so the whole source is read.
    -#}
    (
        select max({{ target_column }}) - toIntervalMinute({{ int_var('ingestion_lookback_minutes') }})
        from {{ this }}
    )
{%- endmacro %}
