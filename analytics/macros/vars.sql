{#-
    Typed accessors for project vars. Vars arrive as strings from env_var() or as numbers from
    --vars; validating them here turns a bad value into a compile error instead of SQL that
    silently compares against garbage.
-#}

{% macro int_var(name) %}
    {%- set raw = var(name) | string | trim -%}
    {%- if not modules.re.fullmatch('[0-9]+', raw) -%}
        {{ exceptions.raise_compiler_error(
            "var '" ~ name ~ "' must be a non-negative integer, got '" ~ raw ~ "'"
        ) }}
    {%- endif -%}
    {{ return(raw | int) }}
{% endmacro %}


{% macro positive_int_var(name) %}
    {%- set value = int_var(name) -%}
    {%- if value < 1 -%}
        {{ exceptions.raise_compiler_error("var '" ~ name ~ "' must be a positive integer, got " ~ value) }}
    {%- endif -%}
    {{ return(value) }}
{% endmacro %}


{% macro number_var(name) %}
    {%- set raw = var(name) | string | trim -%}
    {%- if not modules.re.fullmatch('-?[0-9]+([.][0-9]+)?', raw) -%}
        {{ exceptions.raise_compiler_error("var '" ~ name ~ "' must be a number, got '" ~ raw ~ "'") }}
    {%- endif -%}
    {{ return(raw | float) }}
{% endmacro %}
