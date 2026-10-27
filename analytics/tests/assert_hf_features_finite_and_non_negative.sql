{#-
    HF measurements no consumer could use: a value that is present (not NULL) but NaN, infinite,
    or negative where the quantity cannot be (every vibration feature, the EGT A/B split, oil
    quantity and filter dP; an oil temperature in degC may be negative). Checked in the HF
    staging model and in every model the spectral features flow through, so that the ML feature
    vector and the health board only ever see NULL (unknown) or a usable number. One scan per
    model; each returned row names a column and how many values in it are unusable.
-#}

{%- set non_negative = {} -%}
{%- for column in hf_measurement_columns() -%}
    {%- do non_negative.update({column.name: column.non_negative}) -%}
{%- endfor -%}
{%- set spectral_columns = [] -%}
{%- for feature in spectral_feature_sources() -%}
    {%- do spectral_columns.append({'name': feature.name, 'non_negative': non_negative[feature.source]}) -%}
{%- endfor -%}
{%- set checked_models = [
    {'relation': ref('stg_engine_hf_features'), 'final': true, 'columns': hf_measurement_columns()},
    {'relation': ref('int_vibration_features'), 'final': true, 'columns': spectral_columns},
    {'relation': ref('int_ml_features'), 'final': false, 'columns': spectral_columns},
    {'relation': ref('mart_rul_training_features'), 'final': true, 'columns': spectral_columns},
] -%}

{%- for model in checked_models %}
select
    '{{ model.relation.identifier }}' as model_name,
    column_check.1 as column_name,
    column_check.2 as unusable_values
from (
    select [
        {%- for column in model.columns %}
        (
            '{{ column.name }}',
            countIf(
                isNotNull({{ column.name }})
                and (not isFinite({{ column.name }}){{ ' or ' ~ column.name ~ ' < 0' if column.non_negative }})
            )
        ){{ ',' if not loop.last }}
        {%- endfor %}
    ] as column_checks
    from {{ model.relation }}{{ ' final' if model.final }}
)
array join column_checks as column_check
where unusable_values > 0
{{ 'union all' if not loop.last }}
{%- endfor %}
