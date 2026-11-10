{#-
    Every failure is a shop visit at the same engine, instant, lifecycle and cycle, and there are
    no extra visits. The two facts share identity by construction; this test fails if a join
    drops a row or if a visit is built from a different grain.
-#}

select
    'visit_not_in_both_facts' as issue,
    engine_id,
    lifecycle_id,
    in_failures,
    in_visits,
    failure_cycle,
    visit_failure_cycle,
    toString(failed_at) as failure_value,
    toString(visit_failed_at) as visit_value
from (
    select
        engine_id,
        lifecycle_id,
        failure_cycle,
        failed_at,
        toUInt8(1) as in_failures
    from {{ ref('fct_engine_failures') }} final
) as failures
full outer join (
    select
        engine_id,
        lifecycle_id,
        failure_cycle as visit_failure_cycle,
        failed_at as visit_failed_at,
        toUInt8(1) as in_visits
    from {{ ref('fct_engine_shop_visits') }} final
) as visits using (engine_id, lifecycle_id)
where in_failures != 1
   or in_visits != 1
   or failure_cycle != visit_failure_cycle
   or failed_at != visit_failed_at
