"""Declarative description of the three CDC tables handled by the streaming job.

One :class:`TableSpec` per PostgreSQL table drives every stage of the pipeline: the JSON
schema used to parse Debezium row images, the validation that routes unusable records to
the dead-letter table, the silver parquet layout and the ClickHouse insert column lists.
The module is pure Python (no Spark import) so the contract can be unit-tested against the
ClickHouse DDL without a JVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldKind(StrEnum):
    """Logical type of a row-image column, named after its PostgreSQL origin."""

    STRING = "string"  # VARCHAR / TEXT
    INT = "int"  # INTEGER
    LONG = "long"  # BIGINT / BIGSERIAL
    DOUBLE = "double"  # DOUBLE PRECISION
    TIMESTAMP = "timestamp"  # TIMESTAMPTZ, an ISO-8601 UTC string on the wire


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One column of a Debezium row image.

    ``unsigned`` marks columns stored as ClickHouse UInt*: a negative value would make the
    whole insert fail, so such records are dead-lettered during parsing instead.
    """

    name: str
    kind: FieldKind
    nullable: bool = False
    unsigned: bool = False


SUPPORTED_OPS: tuple[str, ...] = ("c", "u", "r", "d")  # create, update, snapshot read, delete
DELETE_OP = "d"

# CDC metadata appended after the image columns, in ClickHouse column order.
CDC_METADATA_COLUMNS: tuple[str, ...] = (
    "cdc_op",
    "cdc_lsn",
    "cdc_source_ts",
    "cdc_emitted_ts",
    "is_deleted",
    "kafka_partition",
    "kafka_offset",
)
CDC_METADATA_TIMESTAMP_COLUMNS: frozenset[str] = frozenset({"cdc_source_ts", "cdc_emitted_ts"})

DEAD_LETTER_TABLE = "raw_cdc_dead_letter"
DEAD_LETTER_COLUMNS: tuple[str, ...] = (
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "raw_key",
    "raw_value",
    "error",
)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Everything the job needs to know about one captured PostgreSQL table."""

    name: str
    clickhouse_table: str
    primary_key: tuple[str, ...]
    event_time_field: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        by_name = {field.name: field for field in self.fields}
        missing = [name for name in (*self.primary_key, self.event_time_field) if name not in by_name]
        if missing:
            raise ValueError(f"{self.name}: unknown key/event-time column(s) {missing}")
        event_time = by_name[self.event_time_field]
        if event_time.kind is not FieldKind.TIMESTAMP or event_time.nullable:
            raise ValueError(f"{self.name}: event-time column must be a non-null TIMESTAMP")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Image column names in ClickHouse column order."""
        return tuple(field.name for field in self.fields)

    @property
    def clickhouse_columns(self) -> tuple[str, ...]:
        """Explicit insert column list of the raw table (``ingested_at`` is server-filled)."""
        return self.field_names + CDC_METADATA_COLUMNS

    @property
    def timestamp_columns(self) -> frozenset[str]:
        """Columns of :attr:`clickhouse_columns` that hold instants (DateTime64)."""
        image = {field.name for field in self.fields if field.kind is FieldKind.TIMESTAMP}
        return frozenset(image) | CDC_METADATA_TIMESTAMP_COLUMNS

    @property
    def dedup_key_columns(self) -> tuple[str, ...]:
        """Identity of one change event: the primary key plus the WAL position."""
        return (*self.primary_key, "cdc_lsn")


ENGINE_TELEMETRY = TableSpec(
    name="engine_telemetry",
    clickhouse_table="raw_engine_telemetry",
    primary_key=("telemetry_id",),
    event_time_field="event_ts",
    fields=(
        FieldSpec("telemetry_id", FieldKind.LONG, unsigned=True),
        FieldSpec("engine_id", FieldKind.STRING),
        FieldSpec("lifecycle_id", FieldKind.INT, unsigned=True),
        FieldSpec("cycle_number", FieldKind.INT, unsigned=True),
        FieldSpec("event_ts", FieldKind.TIMESTAMP),
        FieldSpec("rpm", FieldKind.DOUBLE),
        FieldSpec("core_temp_c", FieldKind.DOUBLE),
        FieldSpec("egt_c", FieldKind.DOUBLE),
        FieldSpec("vibration_mm_s", FieldKind.DOUBLE),
        FieldSpec("fuel_flow_kg_h", FieldKind.DOUBLE),
        FieldSpec("oil_pressure_psi", FieldKind.DOUBLE),
        FieldSpec("wear_factor", FieldKind.DOUBLE),
        FieldSpec("status", FieldKind.STRING),
        FieldSpec("maintenance_id", FieldKind.LONG, nullable=True, unsigned=True),
        FieldSpec("created_ts", FieldKind.TIMESTAMP),
    ),
)

ENGINE_STATUS_REGISTRY = TableSpec(
    name="engine_status_registry",
    clickhouse_table="raw_engine_status_registry",
    primary_key=("engine_id",),
    event_time_field="updated_ts",
    fields=(
        FieldSpec("engine_id", FieldKind.STRING),
        FieldSpec("engine_model", FieldKind.STRING),
        FieldSpec("operational_status", FieldKind.STRING),
        FieldSpec("current_lifecycle_id", FieldKind.INT, unsigned=True),
        FieldSpec("current_cycle", FieldKind.INT, unsigned=True),
        FieldSpec("wear_factor", FieldKind.DOUBLE),
        FieldSpec("cycles_since_new", FieldKind.INT, unsigned=True),
        FieldSpec("lifecycles_completed", FieldKind.INT, unsigned=True),
        FieldSpec("last_maintenance_ts", FieldKind.TIMESTAMP, nullable=True),
        FieldSpec("status_changed_ts", FieldKind.TIMESTAMP),
        FieldSpec("last_telemetry_ts", FieldKind.TIMESTAMP, nullable=True),
        FieldSpec("updated_ts", FieldKind.TIMESTAMP),
        FieldSpec("registered_ts", FieldKind.TIMESTAMP),
    ),
)

MAINTENANCE_LOG = TableSpec(
    name="maintenance_log",
    clickhouse_table="raw_maintenance_log",
    primary_key=("maintenance_id",),
    event_time_field="failure_ts",
    fields=(
        FieldSpec("maintenance_id", FieldKind.LONG, unsigned=True),
        FieldSpec("engine_id", FieldKind.STRING),
        FieldSpec("lifecycle_id", FieldKind.INT, unsigned=True),
        FieldSpec("failure_cycle", FieldKind.INT, unsigned=True),
        FieldSpec("failure_ts", FieldKind.TIMESTAMP),
        FieldSpec("failure_mode", FieldKind.STRING),
        FieldSpec("maintenance_status", FieldKind.STRING),
        FieldSpec("action_taken", FieldKind.STRING, nullable=True),
        FieldSpec("opened_ts", FieldKind.TIMESTAMP),
        FieldSpec("completed_ts", FieldKind.TIMESTAMP, nullable=True),
        FieldSpec("downtime_seconds", FieldKind.INT, nullable=True),
        FieldSpec("warning_rows_flagged", FieldKind.INT, nullable=True),
        FieldSpec("notes", FieldKind.STRING, nullable=True),
    ),
)

TABLES: tuple[TableSpec, ...] = (ENGINE_TELEMETRY, ENGINE_STATUS_REGISTRY, MAINTENANCE_LOG)
