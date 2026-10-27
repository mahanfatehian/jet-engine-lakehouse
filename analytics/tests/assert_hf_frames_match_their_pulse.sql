{{ config(severity='warn') }}

{#-
    The two streams of one tick must agree (CONTRACT_V2.md section 1): a frame names the
    telemetry_id of its pulse, and once that pulse has landed both must carry the same lifecycle
    and the frame must start at the pulse's exact event_ts. A disagreement is a producer bug
    and keeps the frame's spectral features away from its pulse in int_ml_features. Warn only:
    the HF path is best effort, and a corrected copy of the frame heals it.
-#}

with frames as (

    select engine_id, telemetry_id, lifecycle_id as frame_lifecycle_id, frame_start_ts
    from {{ ref('stg_engine_hf_features') }} final

),

pulses as (

    select engine_id, telemetry_id, lifecycle_id as pulse_lifecycle_id, event_ts
    from {{ ref('stg_engine_telemetry') }} final

)

select engine_id, telemetry_id, frame_lifecycle_id, pulse_lifecycle_id, frame_start_ts, event_ts
from frames
inner join pulses using (engine_id, telemetry_id)
where frame_lifecycle_id != pulse_lifecycle_id or frame_start_ts != event_ts
