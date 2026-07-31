"""Data ingestion: sources, canonical schema, validation, caching."""

from __future__ import annotations

from battery_rul.data.base import BatterySource, available_sources, get_source, register_source
from battery_rul.data.loader import CycleDataset, battery_summary_table, load_cycles
from battery_rul.data.schema import CYCLE_SCHEMA, REQUIRED_COLUMNS, coerce_schema, schema_frame
from battery_rul.data.validation import (
    DataValidationError,
    ValidationReport,
    validate_cycles,
)

__all__ = [
    "CYCLE_SCHEMA",
    "REQUIRED_COLUMNS",
    "BatterySource",
    "CycleDataset",
    "DataValidationError",
    "ValidationReport",
    "available_sources",
    "battery_summary_table",
    "coerce_schema",
    "get_source",
    "load_cycles",
    "register_source",
    "schema_frame",
    "validate_cycles",
]
