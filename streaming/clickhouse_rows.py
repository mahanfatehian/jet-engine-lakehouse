"""Pure-Python conversion of collected Spark rows into ClickHouse insert rows.

PySpark's ``collect()`` turns TimestampType values into *naive* datetimes expressed in the
Python process' local timezone (not the Spark session timezone), and clickhouse-connect
serialises naive datetimes as local time too. The Spark projection therefore ships every
instant as epoch microseconds and this module rebuilds timezone-aware UTC datetimes,
which clickhouse-connect converts to the exact DateTime64 tick whatever the process TZ.

Collected row layout (see ``cdc_transforms.clickhouse_projection``)::

    (dead_letter, *clickhouse_values)

``dead_letter`` is ``None`` for a valid record, otherwise a struct ordered like
``DEAD_LETTER_COLUMNS`` with ``kafka_timestamp`` in epoch microseconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from cdc_schemas import DEAD_LETTER_COLUMNS, TableSpec

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class RowWriter(Protocol):
    """Destination of insert-ready rows (``clickhouse_sink.ClickHouseWriter`` in production)."""

    def insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Insert ``rows`` into ``table`` using exactly ``columns``."""


def epoch_micros(instant: datetime) -> int:
    """Epoch microseconds of a timezone-aware datetime (the inverse of :func:`to_utc_datetime`)."""
    return (instant - _EPOCH) // timedelta(microseconds=1)


# ClickHouse DateTime64 range: instants outside it cannot be stored faithfully.
DATETIME64_MIN_MICROS = epoch_micros(datetime(1900, 1, 1, tzinfo=UTC))
DATETIME64_MAX_MICROS = epoch_micros(datetime(2299, 12, 31, 23, 59, 59, 999999, tzinfo=UTC))

_DEAD_LETTER_TIMESTAMP = DEAD_LETTER_COLUMNS.index("kafka_timestamp")
_DEAD_LETTER_TEXT = (DEAD_LETTER_COLUMNS.index("raw_key"), DEAD_LETTER_COLUMNS.index("raw_value"))

Row = tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class TableBatch:
    """Insert-ready content of one table for one micro-batch."""

    table: TableSpec
    rows: list[Row]
    dead_letters: list[Row]
    duplicates: int


def to_utc_datetime(micros: int | None) -> datetime | None:
    """Exact timezone-aware UTC datetime for epoch microseconds (``None`` stays ``None``)."""
    if micros is None:
        return None
    return _EPOCH + timedelta(microseconds=micros)


def convert_row(values: Sequence[Any], timestamp_positions: frozenset[int]) -> Row:
    """Turn epoch-microsecond positions into aware datetimes; other values pass through."""
    return tuple(
        to_utc_datetime(value) if position in timestamp_positions else value
        for position, value in enumerate(values)
    )


def dead_letter_row(record: Sequence[Any]) -> Row:
    """Insert row for ``raw_cdc_dead_letter`` from a collected dead-letter struct."""
    values = list(record)
    values[_DEAD_LETTER_TIMESTAMP] = to_utc_datetime(values[_DEAD_LETTER_TIMESTAMP])
    for position in _DEAD_LETTER_TEXT:
        # Non-nullable String columns: a record without a Kafka key is stored as ''.
        values[position] = values[position] or ""
    return tuple(values)


def deduplicate(rows: Iterable[Row], key_positions: Sequence[int]) -> tuple[list[Row], int]:
    """Keep the first row per key (Kafka order) and report how many were dropped."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[Row] = []
    dropped = 0
    for row in rows:
        key = tuple(row[position] for position in key_positions)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, dropped


def split_collected(table: TableSpec, collected: Iterable[Sequence[Any]]) -> TableBatch:
    """Split collected rows of one table into deduplicated insert rows and dead letters.

    Exact duplicates (same primary key and ``cdc_lsn``, e.g. Debezium re-deliveries after
    a connector restart) are dropped here; replays across micro-batches are absorbed by
    the ReplacingMergeTree tables.
    """
    columns = table.clickhouse_columns
    timestamp_positions = frozenset(i for i, name in enumerate(columns) if name in table.timestamp_columns)
    key_positions = [columns.index(name) for name in table.dedup_key_columns]
    rows: list[Row] = []
    dead_letters: list[Row] = []
    for record in collected:
        if record[0] is not None:
            dead_letters.append(dead_letter_row(record[0]))
        else:
            rows.append(convert_row(record[1:], timestamp_positions))
    unique, duplicates = deduplicate(rows, key_positions)
    return TableBatch(table=table, rows=unique, dead_letters=dead_letters, duplicates=duplicates)
