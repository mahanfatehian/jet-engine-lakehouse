"""Write operations of the fleet simulator; every statement on the CDC tables becomes a Debezium event.

    table                       INSERT (op "c")               UPDATE (op "u")
    engine_status_registry      engine registration           cycle progress, status change, failure, repair
    engine_telemetry            every pulse                   maintenance_id stamped on WARNING..FAILURE rows
    maintenance_log             work order opened at failure  flagged-row count, repair completion
    simulator_lifecycle_params  settings of a new lifecycle   (simulator-owned, not captured by CDC)

Writes arrive as ordered (operation, payload) lists and each list runs in exactly ONE transaction, so a
failure (telemetry row, work order, flags, registry) or a repair is never half visible downstream.

Telemetry INSERTs return the generated telemetry_id of every pulse: the high-frequency sensor frame of a
pulse carries it as the exact join key. The driver is imported where it is used, so the row logic of this
module stays importable (and unit-testable) without psycopg2, like fleet_lease.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fleet_runtime import RecordedFailure

if TYPE_CHECKING:
    from engine_generator import Pulse
    from fleet_runtime import FailureEvent, LifecycleSettings, MaintenanceCompletion, RegistryUpdate
    from psycopg2.extensions import cursor

INSERT_PAGE_SIZE = 1000
TELEMETRY_COLUMNS = (
    "engine_id",
    "lifecycle_id",
    "cycle_number",
    "event_ts",
    "rpm",
    "core_temp_c",
    "egt_c",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "wear_factor",
    "status",
)

# RETURNING carries the natural key too: PostgreSQL does not promise to return rows in VALUES order.
INSERT_TELEMETRY_SQL = (
    f"INSERT INTO engine_telemetry ({', '.join(TELEMETRY_COLUMNS)}) VALUES %s "
    "RETURNING telemetry_id, engine_id, event_ts"
)

OPEN_WORK_ORDER_SQL = """
INSERT INTO maintenance_log
    (engine_id, lifecycle_id, failure_cycle, failure_ts, failure_mode, maintenance_status, opened_ts, notes)
VALUES (%(engine_id)s, %(lifecycle_id)s, %(cycle)s, %(ts)s, %(mode)s, 'OPEN', %(ts)s, %(notes)s)
RETURNING maintenance_id"""

FLAG_WARNING_ROWS_SQL = """
UPDATE engine_telemetry SET maintenance_id = %(maintenance_id)s
 WHERE engine_id = %(engine_id)s AND lifecycle_id = %(lifecycle_id)s
   AND status IN ('WARNING', 'CRITICAL', 'FAILURE') AND maintenance_id IS NULL"""

COUNT_FLAGGED_ROWS_SQL = "UPDATE maintenance_log SET warning_rows_flagged = %s WHERE maintenance_id = %s"

REGISTRY_FAILURE_SQL = """
UPDATE engine_status_registry
   SET operational_status = 'FAILURE', status_changed_ts = %(ts)s, current_lifecycle_id = %(lifecycle_id)s,
       current_cycle = %(cycle)s, wear_factor = %(wear)s, cycles_since_new = %(cycles_since_new)s,
       last_telemetry_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

# status NULL = progress only: keep operational_status and status_changed_ts as they are.
REGISTRY_PROGRESS_SQL = """
UPDATE engine_status_registry
   SET operational_status = COALESCE(%(status)s, operational_status),
       status_changed_ts = CASE WHEN %(status)s IS NULL THEN status_changed_ts ELSE %(ts)s END,
       current_lifecycle_id = %(lifecycle_id)s, current_cycle = %(cycle)s, wear_factor = %(wear)s,
       cycles_since_new = %(cycles_since_new)s, last_telemetry_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

CLOSE_WORK_ORDER_SQL = """
UPDATE maintenance_log
   SET maintenance_status = 'COMPLETED', completed_ts = %(ts)s, downtime_seconds = %(downtime)s,
       action_taken = %(action)s
 WHERE engine_id = %(engine_id)s AND lifecycle_id = %(lifecycle_id)s AND maintenance_status = 'OPEN'"""

REGISTRY_RESET_SQL = """
UPDATE engine_status_registry
   SET operational_status = 'MAINTENANCE_RESET', current_lifecycle_id = %(next_lifecycle_id)s,
       current_cycle = 0, wear_factor = 0, lifecycles_completed = lifecycles_completed + 1,
       last_maintenance_ts = %(ts)s, status_changed_ts = %(ts)s, updated_ts = now()
 WHERE engine_id = %(engine_id)s"""

# Written with the first pulse of a lifecycle, so a row exists exactly for lifecycles with telemetry;
# a leftover row of a lifecycle without telemetry generated nothing yet and is simply replaced.
RECORD_SETTINGS_SQL = """
INSERT INTO simulator_lifecycle_params (engine_id, lifecycle_id, seed, mean_life_cycles, pulses_per_cycle)
VALUES (%(engine_id)s, %(lifecycle_id)s, %(seed)s, %(mean_life_cycles)s, %(pulses_per_cycle)s)
ON CONFLICT (engine_id, lifecycle_id) DO UPDATE
   SET seed = EXCLUDED.seed, mean_life_cycles = EXCLUDED.mean_life_cycles,
       pulses_per_cycle = EXCLUDED.pulses_per_cycle, created_ts = now()"""


class FleetStateError(RuntimeError):
    """A write did not touch exactly the row it targets: database and simulator state disagree."""


def _require_one_row(cur: cursor, what: str) -> None:
    if cur.rowcount != 1:
        raise FleetStateError(f"{what} touched {cur.rowcount} rows instead of 1")


def ids_in_pulse_order(pulses: Sequence[Pulse], returned: Sequence[tuple[int, str, datetime]]) -> list[int]:
    """telemetry_id of every pulse, in ``pulses`` order, from (telemetry_id, engine_id, event_ts) rows.

    (engine_id, event_ts) is unique: an engine's event times strictly increase.
    """
    ids = {(engine_id, event_ts): int(telemetry_id) for telemetry_id, engine_id, event_ts in returned}
    if len(ids) != len(returned) or len(returned) != len(pulses):
        raise FleetStateError(f"INSERT of {len(pulses)} pulses returned {len(returned)} distinct rows")
    try:
        return [ids[(pulse.engine_id, pulse.event_ts)] for pulse in pulses]
    except KeyError as exc:
        raise FleetStateError(f"INSERT returned no row for pulse {exc.args[0]}") from exc


class FleetWriter:
    """Executes the simulator's write operations on the cursor of one transaction."""

    def __init__(self, cur: cursor) -> None:
        self._cur = cur

    def apply(self, ops: Sequence[tuple[str, Any]]) -> list[Any]:
        """Run ``ops`` in order and return each result (the work-order id for a failure)."""
        handlers: dict[str, Callable[[Any], Any]] = {
            "record_settings": self.record_settings,
            "insert_pulses": self.insert_pulses,
            "record_failure": self.record_failure,
            "update_registry": self.update_registry,
            "complete_maintenance": self.complete_maintenance,
        }
        return [handlers[name](payload) for name, payload in ops]

    def record_settings(self, record: LifecycleSettings) -> None:
        """Store the generation settings of a lifecycle, so a restart re-derives it exactly."""
        settings = record.settings
        self._cur.execute(
            RECORD_SETTINGS_SQL,
            {
                "engine_id": record.engine_id,
                "lifecycle_id": record.lifecycle_id,
                "seed": settings.seed,
                "mean_life_cycles": float(settings.mean_life_cycles),
                "pulses_per_cycle": settings.pulses_per_cycle,
            },
        )

    def insert_pulses(self, pulses: Sequence[Pulse]) -> list[int]:
        """Batch INSERT of telemetry rows; returns their telemetry_ids in ``pulses`` order."""
        from psycopg2.extras import execute_values

        rows = [tuple(getattr(pulse, column) for column in TELEMETRY_COLUMNS) for pulse in pulses]
        returned = execute_values(
            self._cur, INSERT_TELEMETRY_SQL, rows, page_size=INSERT_PAGE_SIZE, fetch=True
        )
        return ids_in_pulse_order(pulses, returned)

    def record_failure(self, event: FailureEvent) -> RecordedFailure:
        """FAILURE row, work order, maintenance_id on the lifecycle's warning rows, registry FAILURE."""
        pulse = event.pulse
        params = {
            "engine_id": pulse.engine_id,
            "lifecycle_id": pulse.lifecycle_id,
            "cycle": pulse.cycle_number,
            "ts": pulse.event_ts,
            "mode": event.failure_mode,
            "notes": event.notes,
            "wear": pulse.wear_factor,
            "cycles_since_new": event.cycles_since_new,
        }
        (telemetry_id,) = self.insert_pulses([pulse])
        self._cur.execute(OPEN_WORK_ORDER_SQL, params)
        maintenance_id = self._cur.fetchone()[0]
        self._cur.execute(FLAG_WARNING_ROWS_SQL, {**params, "maintenance_id": maintenance_id})
        flagged = self._cur.rowcount
        self._cur.execute(COUNT_FLAGGED_ROWS_SQL, (flagged, maintenance_id))
        self._cur.execute(REGISTRY_FAILURE_SQL, params)
        _require_one_row(self._cur, f"registry FAILURE update of {pulse.engine_id}")
        return RecordedFailure(int(maintenance_id), telemetry_id)

    def update_registry(self, update: RegistryUpdate) -> None:
        """Registry progress (cycle, wear, counters) and, if ``update.status`` is set, the new status."""
        self._cur.execute(
            REGISTRY_PROGRESS_SQL,
            {
                "status": update.status,
                "ts": update.event_ts,
                "lifecycle_id": update.lifecycle_id,
                "cycle": update.current_cycle,
                "wear": update.wear_factor,
                "cycles_since_new": update.cycles_since_new,
                "engine_id": update.engine_id,
            },
        )
        _require_one_row(self._cur, f"registry update of {update.engine_id}")

    def complete_maintenance(self, completion: MaintenanceCompletion) -> None:
        """Close the OPEN work order and reset the registry into the next lifecycle."""
        params = {
            "engine_id": completion.engine_id,
            "lifecycle_id": completion.lifecycle_id,
            "next_lifecycle_id": completion.next_lifecycle_id,
            "ts": completion.completed_ts,
            "downtime": completion.downtime_seconds,
            "action": completion.action_taken,
        }
        self._cur.execute(CLOSE_WORK_ORDER_SQL, params)
        _require_one_row(
            self._cur, f"work order of {completion.engine_id} lifecycle {completion.lifecycle_id}"
        )
        self._cur.execute(REGISTRY_RESET_SQL, params)
        _require_one_row(self._cur, f"registry reset of {completion.engine_id}")
