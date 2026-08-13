"""Fleet intelligence: many cells, one operational picture.

Layering, which is the point of this package existing at all:

``domain``        typed, JSON-serialisable fleet objects — the published shape
``ingestion``     files/frames -> validated per-battery histories, with per-cell
                  failure isolation
``inference``     :class:`FleetInferenceService`, which *calls*
                  :class:`~battery_rul.digital_twin.service.BatteryDigitalTwinService`
                  once per cell and never re-implements a model load or a feature
``maintenance``   the deterministic priority engine (policy, not model)
``replacement``   advisory replacement planning and workload forecasting
``ranking``       the configurable composite priority score
``aggregation``   fleet statistics with explicit denominators
``analytics``     battery- and fleet-level trends
``demo``          a deterministic, clearly-labelled synthetic demo fleet

Nothing in here loads a model artifact, builds a feature or applies a
calibration. Battery-level inference has exactly one entry point and this
package is a consumer of it.
"""

from __future__ import annotations

from battery_rul.fleet.domain import (
    FLEET_SNAPSHOT_SCHEMA_VERSION,
    BatteryIngestionRecord,
    BatteryPriorityRecord,
    FleetBatteryRecord,
    FleetIdentity,
    FleetIngestionResult,
    FleetSnapshot,
    FleetSummary,
    InspectionRecommendation,
    MaintenancePriority,
    ProcessingStatus,
    ReplacementCandidate,
    ReplacementHorizon,
)

__all__ = [
    "FLEET_SNAPSHOT_SCHEMA_VERSION",
    "BatteryIngestionRecord",
    "BatteryPriorityRecord",
    "FleetBatteryRecord",
    "FleetIdentity",
    "FleetIngestionResult",
    "FleetSnapshot",
    "FleetSummary",
    "InspectionRecommendation",
    "MaintenancePriority",
    "ProcessingStatus",
    "ReplacementCandidate",
    "ReplacementHorizon",
]
