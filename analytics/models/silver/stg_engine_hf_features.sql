{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key=['engine_id', 'telemetry_id'],
        on_schema_change='fail',
        indexes=[
            {'name': 'idx_ingested_at', 'definition': 'ingested_at TYPE minmax GRANULARITY 1'},
            {'name': 'idx_lifecycle_id', 'definition': 'lifecycle_id TYPE minmax GRANULARITY 1'},
        ],
        engine='ReplacingMergeTree(ingested_at)',
        order_by='(engine_id, telemetry_id)',
        partition_by='toYYYYMM(frame_start_ts)',
    )
}}

{#-
    Deduplicated, typed features of the high-frequency (HF) sensor frames: one row per frame,
    keyed like its 1 Hz pulse (the frame's telemetry_id; CONTRACT_V2.md sections 1 and 6).

    Frames land at least once (Kafka redelivery, replayed Spark batches), so a frame can have
    several copies. The copy with the newest ingested_at wins, ties broken by the Kafka position,
    so that every run elects the same copy. That is also the copy raw_engine_hf_features keeps
    when it merges: it is a ReplacingMergeTree(ingested_at), and the server stamps ingested_at. Each run reads the raw rows ingested after the
    published watermark minus the lookback (late-visible inserts) and appends, per frame, the
    batch's winning copy when it beats the published one; re-read and older copies are skipped.
    The engine keeps the newest ingested_at per frame and, on a tie, the copy appended last,
    which is the one with the higher Kafka position. Readers must use FINAL. The raw table
    expires frames after 90 days (TTL) while this model keeps their features, so a
    --full-refresh rebuilds the frames the raw table still holds.

    Typing: the Float32 features become Nullable(Float64). A value no model can use (NaN or
    infinite, e.g. the shaft tone of a stopped spool, or negative where the quantity cannot be)
    is NULL instead of poisoning the rolling means; the source tests report it. Every feature
    of a vibration channel without variance (undefined kurtosis: a flat-line, dead or stuck
    sensor, whose amplitudes still read 0) is NULL too. The oil-system parameters are extracted
    from process_means (sensor channel A), NULL when the frame does not carry them (a Map reads
    0 for an absent key); the full map stays in raw_engine_hf_features and in the silver lake.
    Rows arrive roughly in ingestion order, so the minmax index on ingested_at prunes the
    watermark scan of this model to the newest granules. Within an engine, lifecycle ids grow
    with telemetry ids, so the minmax index on lifecycle_id prunes int_vibration_features'
    reads of whole lifecycles to their granules; ClickHouse applies it under FINAL too
    (use_skip_indexes_if_final with its exact mode, both on by default in ClickHouse 26.8).
-#}

{%- set vibration_features = hf_vibration_feature_columns() -%}

with source_frames as (

    select
        engine_id,
        engine_model,
        lifecycle_id,
        cycle_number,
        telemetry_id,
        frame_seq,
        frame_start_ts,
        frame_duration_ms,
        n1_rpm,
        n2_rpm,
        bpfo_order,
        vibration_sample_rate_hz,
        vibration_channels,
        process_channels,
        {%- for column in vibration_features %}
        {{ column }},
        {%- endfor %}
        egt_ab_split,
        process_means,
        frame_bytes,
        kafka_partition,
        kafka_offset,
        ingested_at
    from {{ source('engine_raw', 'raw_engine_hf_features') }}
    {%- if is_incremental() %}
    where ingested_at > {{ lookback_watermark('ingested_at') }}
    {%- endif %}

),

latest_copies as (

    -- copies of one frame can land in the same batch (the same insert, even)
    select
        *,
        {%- for channel in hf_vibration_channels() %}
        {{ channel_has_variance(channel) }} as {{ channel }}_has_variance{{ ',' if not loop.last }}
        {%- endfor %}
    from source_frames
    order by engine_id, telemetry_id, ingested_at desc, kafka_partition desc, kafka_offset desc
    limit 1 by engine_id, telemetry_id

){% if is_incremental() %},

published_copies as (

    -- newest copy already published for the frames of this batch (primary-key lookup)
    select
        engine_id,
        telemetry_id,
        max((ingested_at, kafka_partition, kafka_offset)) as published_position,
        toUInt8(1) as is_published
    from {{ this }}
    where (engine_id, telemetry_id) in (select engine_id, telemetry_id from source_frames)
    group by engine_id, telemetry_id

){% endif %}

select
    engine_id,
    engine_model,
    lifecycle_id,
    cycle_number,
    telemetry_id,
    frame_seq,
    frame_start_ts,
    frame_duration_ms,
    toFloat64(n1_rpm) as n1_rpm,
    toFloat64(n2_rpm) as n2_rpm,
    toFloat64(bpfo_order) as bpfo_order,
    vibration_sample_rate_hz,
    vibration_channels,
    process_channels,
    {%- for channel in hf_vibration_channels() %}
    {%- for statistic in hf_channel_statistics() %}
    {{ channel_measurement(channel, statistic) }} as {{ channel }}_{{ statistic }},
    {%- endfor %}
    {%- endfor %}
    {{ usable_measurement('egt_ab_split') }} as egt_ab_split,
    {%- for oil in hf_oil_parameters() %}
    {{ process_parameter(oil.parameter, oil.non_negative) }} as {{ oil.name }},
    {%- endfor %}
    frame_bytes,
    kafka_partition,
    kafka_offset,
    ingested_at
from latest_copies
{%- if is_incremental() %}
left join published_copies using (engine_id, telemetry_id)
-- an older or already published copy would lose against the published one anyway
where is_published = 0 or (ingested_at, kafka_partition, kafka_offset) > published_position
{%- endif %}
