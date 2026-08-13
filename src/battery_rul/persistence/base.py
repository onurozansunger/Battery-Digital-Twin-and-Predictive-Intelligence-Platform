"""The repository contract.

A ``Protocol`` rather than an abstract base class: the API layer depends on the
shape, not on an inheritance tree, and a test double satisfies it without
importing anything from here.

Failure is explicit. A write that cannot happen raises; it never returns
``False``, logs a warning and continues. Silent write failure in a monitoring
store is the worst of both worlds — the dashboard shows the last successful run
and nobody knows the newer ones are missing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from battery_rul.fleet.domain import FleetSnapshot
from battery_rul.monitoring.domain import Alert, MonitoringSnapshot
from battery_rul.monitoring.performance import OutcomeLabel, PredictionRecord

__all__ = [
    "PersistenceError",
    "ReadOnlyStoreError",
    "Repository",
    "SCHEMA_VERSION",
]

#: Bumped on any incompatible change to the stored table layout.
SCHEMA_VERSION = "3.0"


class PersistenceError(RuntimeError):
    """A storage operation failed. Never swallowed, never downgraded."""


class ReadOnlyStoreError(PersistenceError):
    """A write was attempted while ``deployment.read_only`` is set."""


@runtime_checkable
class Repository(Protocol):
    """Everything the platform stores."""

    # -- fleet snapshots ---------------------------------------------------
    def save_fleet_snapshot(self, snapshot: FleetSnapshot) -> str: ...

    def get_fleet_snapshot(self, snapshot_id: str) -> FleetSnapshot | None: ...

    def latest_fleet_snapshot(self, fleet_id: str) -> FleetSnapshot | None: ...

    def list_fleet_snapshots(
        self, fleet_id: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    # -- monitoring --------------------------------------------------------
    def save_monitoring_snapshot(self, snapshot: MonitoringSnapshot) -> str: ...

    def latest_monitoring_snapshot(
        self, fleet_id: str | None = None
    ) -> MonitoringSnapshot | None: ...

    def list_monitoring_snapshots(
        self, fleet_id: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    # -- alerts ------------------------------------------------------------
    def save_alerts(self, alerts: list[Alert]) -> int: ...

    def list_alerts(
        self,
        fleet_id: str | None = None,
        *,
        acknowledged: bool | None = None,
        limit: int = 100,
    ) -> list[Alert]: ...

    def acknowledge_alert(self, alert_id: str, *, by: str) -> bool: ...

    # -- predictions and labels -------------------------------------------
    def save_prediction_records(self, records: list[PredictionRecord]) -> int: ...

    def list_prediction_records(
        self, *, model_version: str | None = None, limit: int = 10_000
    ) -> list[PredictionRecord]: ...

    def save_outcome_labels(self, labels: list[OutcomeLabel]) -> int: ...

    def list_outcome_labels(self, *, limit: int = 10_000) -> list[OutcomeLabel]: ...

    # -- batch metadata ----------------------------------------------------
    def save_batch(self, batch_id: str, payload: dict[str, Any]) -> str: ...

    def list_batches(self, *, limit: int = 50) -> list[dict[str, Any]]: ...

    # -- lifecycle ---------------------------------------------------------
    def schema_version(self) -> str: ...

    def close(self) -> None: ...
