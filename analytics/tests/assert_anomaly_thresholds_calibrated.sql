{#-
    Calibration guard of the fct_engine_health_status anomaly flags (rationale in
    dbt_project.yml): each threshold must separate the healthy noise of its KPI from the failure
    signal. Judged on the training window (mart_rul_training_features: the newest completed
    lifecycles) with the pulse status of stg_engine_telemetry:

    silent_at_failure    the flag is raised at fewer than half of the FAILURE pulses of the
                         lifecycles that failed in the mode the flag exists to detect
    noisy_while_healthy  the flag is raised on more than 5 % of the HEALTHY pulses

    A failure mode without completed lifecycles (fresh stack) has nothing to judge. The KPIs are
    5-minute means of event time, so other SIM_TICK_SECONDS / SIM_PULSES_PER_CYCLE values call
    for a recalibration of the anomaly_* vars from the KPI percentiles per status.
-#}

{%- set flags = [
    {
        'name': 'is_egt_drift_anomaly', 'kpi': 'egt_c_drift',
        'threshold': number_var('anomaly_egt_drift_c'), 'failure_mode': 'HPT_BLADE_EROSION',
    },
    {
        'name': 'is_vibration_growth_anomaly', 'kpi': 'vibration_mm_s_ratio',
        'threshold': number_var('anomaly_vibration_ratio'), 'failure_mode': 'BEARING_DEGRADATION',
    },
    {
        'name': 'is_oil_pressure_drop_anomaly', 'kpi': 'oil_pressure_psi_drop',
        'threshold': number_var('anomaly_oil_pressure_drop_psi'), 'failure_mode': 'OIL_SYSTEM_LEAK',
    },
] %}
{%- set min_detection_rate = 0.5 %}
{%- set max_false_alarm_rate = 0.05 %}

with pulse_status as (

    select engine_id, telemetry_id, status
    from {{ ref('stg_engine_telemetry') }} final
    -- the sorting key (engine_id, telemetry_id) prunes every pulse outside the training window
    where (engine_id, telemetry_id) in (
        select engine_id, telemetry_id from {{ ref('mart_rul_training_features') }}
    )
      and status in ('HEALTHY', 'FAILURE')

),

judged_pulses as (

    select
        status,
        failure_mode,
        {%- for flag in flags %}
        toFloat64({{ flag.kpi }} > {{ flag.threshold }}) as {{ flag.name }}{{ ',' if not loop.last }}
        {%- endfor %}
    from {{ ref('mart_rul_training_features') }} final
    inner join pulse_status using (engine_id, telemetry_id)

),

rates as (

    -- one pass: each flag's detection rate at its mode's failures and false-alarm rate when healthy
    select
        {%- for flag in flags %}
        countIf(status = 'FAILURE' and failure_mode = '{{ flag.failure_mode }}') as {{ flag.name }}_failures,
        avgIf({{ flag.name }}, status = 'FAILURE' and failure_mode = '{{ flag.failure_mode }}')
            as {{ flag.name }}_detection_rate,
        avgIf({{ flag.name }}, status = 'HEALTHY') as {{ flag.name }}_false_alarm_rate,
        {%- endfor %}
        countIf(status = 'HEALTHY') as healthy_pulses
    from judged_pulses

),

checks as (

    select
        calibration.1 as flag,
        calibration.2 as issue,
        calibration.3 as observed_rate,
        calibration.4 as allowed_rate,
        calibration.5 as judged_pulses
    from rates
    array join [
        {%- for flag in flags %}
        (
            '{{ flag.name }}', 'silent_at_failure', {{ flag.name }}_detection_rate,
            {{ min_detection_rate }}, {{ flag.name }}_failures
        ),
        (
            '{{ flag.name }}', 'noisy_while_healthy', {{ flag.name }}_false_alarm_rate,
            {{ max_false_alarm_rate }}, healthy_pulses
        ){{ ',' if not loop.last }}
        {%- endfor %}
    ] as calibration

)

select flag, issue, observed_rate, allowed_rate, judged_pulses
from checks
where judged_pulses > 0
  and if(issue = 'silent_at_failure', observed_rate < allowed_rate, observed_rate > allowed_rate)
