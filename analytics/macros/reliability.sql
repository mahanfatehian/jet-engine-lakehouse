{#-
    Status model of the reliability analytics (int_engine_status_history and the gold reliability
    models). engine_status_registry moves every engine through HEALTHY -> WARNING -> CRITICAL ->
    FAILURE -> MAINTENANCE_RESET -> HEALTHY: the first three are in service (the engine flies),
    FAILURE (under repair) and MAINTENANCE_RESET (repaired, reset into its next lifecycle, not
    flown yet) are not.
-#}

{% macro registry_statuses() %}
    {#- every operational_status in lifecycle order: column prefix of its time and availability -#}
    {{ return([
        {'status': 'HEALTHY', 'column': 'healthy', 'available': true},
        {'status': 'WARNING', 'column': 'warning', 'available': true},
        {'status': 'CRITICAL', 'column': 'critical', 'available': true},
        {'status': 'FAILURE', 'column': 'failure', 'available': false},
        {'status': 'MAINTENANCE_RESET', 'column': 'maintenance_reset', 'available': false},
    ]) }}
{% endmacro %}


{% macro available_statuses_sql() -%}
    {#- SQL list of the statuses in which an engine is available (in service) -#}
    {%- for entry in registry_statuses() if entry.available -%}
        '{{ entry.status }}'{{ ', ' if not loop.last }}
    {%- endfor -%}
{%- endmacro %}


{% macro status_history_as_of() -%}
    {#-
        Horizon of the status history: the newest event time the registry's CDC log reports (a
        pulse or status change of any engine, or the commit of a delete). An open interval is
        known to have lasted at least until then, so the reliability KPIs measure it up to there:
        the same data always gives the same numbers, and a stopped pipeline adds no time to the
        current intervals.
    -#}
    {#- a scalar subquery reads as Nullable; max() over no rows is the epoch, never NULL -#}
    assumeNotNull((
        select max(if(
            is_deleted = 1,
            toDateTime64(cdc_source_ts, 6, 'UTC'),
            greatest(status_changed_ts, ifNull(last_telemetry_ts, status_changed_ts))
        ))
        from {{ ref('stg_engine_status_registry') }}
    ))
{%- endmacro %}


{% macro micros_between(start_ts, end_ts) -%}
    {#- whole microseconds from start_ts to end_ts (Int64; NULL when either is NULL) -#}
    (toUnixTimestamp64Micro({{ end_ts }}) - toUnixTimestamp64Micro({{ start_ts }}))
{%- endmacro %}


{% macro outage_overlap_us(from_us, to_us) -%}
    {#- microseconds of [from_us, to_us) inside platform outages (unix micros). Both bounds must be
        non-NULL: greatest() and least() skip a NULL, so a NULL bound would count every outage on its
        side; ifNull() only drops the Nullable type that arraySum rejects -#}
    arraySum(
        outage -> ifNull(greatest(least({{ to_us }}, outage.2) - greatest({{ from_us }}, outage.1), 0), 0),
        (
            select groupArray((toUnixTimestamp64Micro(outage_start), toUnixTimestamp64Micro(outage_end)))
            from {{ ref('int_platform_outages') }} final
        )
    )
{%- endmacro %}
