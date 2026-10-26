{#-
    Every live feature row must be computed with the project's current `baseline_cycles`, or the
    training mart mixes two definitions of the six drift features (train/serve skew).

    int_sensor_rolling_features recomputes every lifecycle built with another value on its next
    run, so a failure means it has not run since baseline_cycles changed (or ran with a one-off
    --vars override that a later run did not undo). Remedy, in the returned rows as well:
        dbt run --full-refresh -s int_sensor_rolling_features+
-#}

{%- set baseline_cycles = int_var('baseline_cycles') %}

select
    engine_id,
    lifecycle_id,
    baseline_cycles as built_with_baseline_cycles,
    {{ baseline_cycles }} as project_baseline_cycles,
    count() as feature_rows,
    'run: dbt run --full-refresh -s int_sensor_rolling_features+' as remedy
from {{ ref('int_sensor_rolling_features') }} final
where baseline_cycles != {{ baseline_cycles }}
group by engine_id, lifecycle_id, baseline_cycles
