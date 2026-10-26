{#-
    Feature rows a model could not consume: a NULL, NaN or infinite value in any of the 45 float
    features, or an empty engine_model, in either the silver feature table or the training mart.
-#}

{%- set feature_models = [ref('int_sensor_rolling_features'), ref('mart_rul_training_features')] %}

{%- for feature_model in feature_models %}
select
    '{{ feature_model.identifier }}' as model_name,
    telemetry_id
from {{ feature_model }} final
where engine_model = ''
    {%- for column in float_feature_columns() %}
    or isNull({{ column }}) or not isFinite({{ column }})
    {%- endfor %}
{{ 'union all' if not loop.last }}
{%- endfor %}
