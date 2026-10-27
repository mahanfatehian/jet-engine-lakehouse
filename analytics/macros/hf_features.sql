{#-
    Column catalogue and cleaning rules of the high-frequency (HF) frame features
    (CONTRACT_V2.md sections 6-8): what stg_engine_hf_features takes from raw_engine_hf_features.
-#}

{% macro hf_vibration_channels() %}
    {#- feature prefixes of the fan (fan_frame_vel) and core (core_trf_vel) vibration channels -#}
    {{ return(['fan', 'core']) }}
{% endmacro %}


{% macro hf_channel_statistics() %}
    {#- per-frame waveform features of every vibration channel, in column order -#}
    {{ return([
        'rms', 'peak', 'crest_factor', 'kurtosis', 'amp_1x_n1', 'amp_1x_n2', 'amp_bpfo',
        'band_energy_hf', 'spectral_centroid_hz',
    ]) }}
{% endmacro %}


{% macro hf_vibration_feature_columns() %}
    {%- set columns = [] -%}
    {%- for channel in hf_vibration_channels() -%}
        {%- for statistic in hf_channel_statistics() -%}
            {%- do columns.append(channel ~ '_' ~ statistic) -%}
        {%- endfor -%}
    {%- endfor -%}
    {{ return(columns) }}
{% endmacro %}


{% macro hf_oil_parameters() %}
    {#-
        Oil-system process parameters extracted from process_means (sensor channel A), by staging
        column. A temperature in degC may legitimately be negative; quantity and filter dP cannot.
    -#}
    {{ return([
        {'name': 'oil_temp_c', 'parameter': 'OIL_T_A', 'non_negative': false},
        {'name': 'oil_quantity', 'parameter': 'OIL_QTY_A', 'non_negative': true},
        {'name': 'oil_filter_dp', 'parameter': 'OIL_DP_A', 'non_negative': true},
    ]) }}
{% endmacro %}


{% macro hf_measurement_columns() %}
    {#- every Nullable measurement column of stg_engine_hf_features with its domain -#}
    {%- set columns = [] -%}
    {%- for column in hf_vibration_feature_columns() + ['egt_ab_split'] -%}
        {%- do columns.append({'name': column, 'non_negative': true}) -%}
    {%- endfor -%}
    {%- for oil in hf_oil_parameters() -%}
        {%- do columns.append({'name': oil.name, 'non_negative': oil.non_negative}) -%}
    {%- endfor -%}
    {{ return(columns) }}
{% endmacro %}


{% macro usable_measurement(expression, non_negative=true) -%}
    {#-
        Nullable(Float64) of a measured value: NULL when it is NaN or infinite (the streaming job
        writes NaN for an undefined feature, e.g. the shaft tone of a stopped spool) or, for a
        non-negative quantity, below 0.
    -#}
    if(
        isFinite({{ expression }}){{ ' and ' ~ expression ~ ' >= 0' if non_negative }},
        toFloat64({{ expression }}),
        NULL
    )
{%- endmacro %}


{% macro channel_has_variance(channel) -%}
    {#-
        A vibration channel whose kurtosis is undefined has no variance: a flat line, i.e. a dead
        or stuck sensor. Its amplitudes still read 0, which would pass for a quiet engine and
        dilute the rolling means. Evaluate it on the raw column, ahead of any alias that shadows it.
    -#}
    isFinite({{ channel }}_kurtosis)
{%- endmacro %}


{% macro channel_measurement(channel, statistic) -%}
    {#- One feature of a vibration channel: NULL for a channel without variance (column <channel>_has_variance) -#}
    if(
        {{ channel }}_has_variance,
        {{ usable_measurement(channel ~ '_' ~ statistic) }},
        NULL
    )
{%- endmacro %}


{% macro process_parameter(parameter, non_negative=true) -%}
    {#-
        One process parameter of process_means. A ClickHouse Map returns 0 for an absent key, so
        the key is tested first: a parameter the frame does not carry is NULL, never 0.
    -#}
    {%- set value = "process_means['" ~ parameter ~ "']" -%}
    if(
        mapContains(process_means, '{{ parameter }}'),
        {{ usable_measurement(value, non_negative) }},
        NULL
    )
{%- endmacro %}
