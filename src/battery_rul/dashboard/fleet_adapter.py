"""Everything the fleet dashboard needs to fetch, in one testable place.

Same split as the Milestone 2 adapter: Streamlit code is awkward to test, so all
data access lives here as plain functions over plain objects and the dashboard
script does layout only. The tests exercise this module directly.

Two back-ends, one interface — ``service`` (in-process) and ``api`` (HTTP) —
and both go through the same fleet service code path. Neither re-implements
inference, ranking or the priority rules: a dashboard that computed its own
priorities would eventually disagree with the API it is supposed to be a view
of, and the operator would have no way to tell which was right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.demo import DemoFleetSpec, demo_fleet_identity, ingest_demo_fleet
from battery_rul.fleet.domain import FleetSnapshot
from battery_rul.fleet.ingestion import FleetIngestor
from battery_rul.monitoring.domain import MonitoringSnapshot
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "FleetDashboardAdapter",
    "battery_table",
    "score_breakdown_table",
    "workload_table",
]


@dataclass
class FleetDashboardAdapter:
    """Uniform access to fleet data, in-process or over HTTP."""

    cfg: ExperimentConfig
    mode: Literal["service", "api"] = "service"
    base_url: str | None = None
    _service: Any = None
    _repository: Any = None

    @classmethod
    def build(cls, cfg: ExperimentConfig) -> FleetDashboardAdapter:
        if cfg.service.dashboard_mode == "api":
            return cls(cfg=cfg, mode="api", base_url=cfg.service.dashboard_api_url)
        from battery_rul.fleet.inference import FleetInferenceService
        from battery_rul.persistence import build_repository

        repository = None
        try:
            repository = build_repository(cfg)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not raised
            logger.warning("Persistence unavailable: %s", exc)
        return cls(
            cfg=cfg,
            mode="service",
            _service=FleetInferenceService.create(cfg),
            _repository=repository,
        )

    # -- status ------------------------------------------------------------
    def readiness(self) -> dict[str, Any]:
        if self.mode == "service":
            return self._service.readiness() if self._service else {"ready": False, "bundles": {}}
        import httpx

        try:
            return httpx.get(f"{self.base_url}/ready", timeout=10.0).json()
        except Exception as exc:  # noqa: BLE001
            return {"ready": False, "bundles": {}, "errors": {"http": str(exc)}}

    # -- snapshots ---------------------------------------------------------
    def latest_snapshot(self, fleet_id: str) -> FleetSnapshot | None:
        if self.mode == "service":
            return self._repository.latest_fleet_snapshot(fleet_id) if self._repository else None
        import httpx

        try:
            response = httpx.get(
                f"{self.base_url}/v1/fleet/{fleet_id}/latest",
                params={"page_size": self.cfg.fleet.max_page_size},
                timeout=60.0,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            payload["batteries"] = (payload.get("batteries") or {}).get("items", [])
            payload.pop("ingestion_records", None)
            payload["data_fingerprint"] = payload.get("data_fingerprint", "")
            return FleetSnapshot(**payload)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            logger.warning("Could not fetch the latest snapshot over HTTP: %s", exc)
            return None

    def list_snapshots(self, fleet_id: str | None = None, *, limit: int = 25) -> list[dict]:
        if self.mode == "service" and self._repository:
            return self._repository.list_fleet_snapshots(fleet_id, limit=limit)
        return []

    def snapshot_history(self, fleet_id: str, *, limit: int = 25) -> list[FleetSnapshot]:
        """Stored snapshots, oldest first — the input to the trend charts."""
        if self.mode != "service" or not self._repository:
            return []
        out: list[FleetSnapshot] = []
        for row in self._repository.list_fleet_snapshots(fleet_id, limit=limit):
            snapshot = self._repository.get_fleet_snapshot(row["snapshot_id"])
            if snapshot is not None:
                out.append(snapshot)
        return sorted(out, key=lambda s: s.generated_at_utc)

    def run_fleet(
        self, *, source: str, fleet_id: str, path: str | None = None, demo_size: int = 24
    ) -> FleetSnapshot:
        """Score a fleet now. Demo data is generated and labelled as synthetic."""
        if self.mode != "service" or self._service is None:
            raise RuntimeError(
                "Scoring a fleet from the dashboard requires the in-process service "
                "mode. In API mode, run the batch pipeline and read its snapshot."
            )
        if source == "demo":
            from battery_rul.fleet.demo import resolve_construction

            spec = DemoFleetSpec(fleet_id=fleet_id, n_batteries=demo_size)
            ingestion, histories = ingest_demo_fleet(self.cfg, spec)
            identity = demo_fleet_identity(spec, construction=resolve_construction(self.cfg, spec))
        elif source == "processed":
            ingestion, histories = FleetIngestor(cfg=self.cfg).from_processed_cycles(fleet_id)
            identity = None
        elif source == "file":
            if not path:
                raise ValueError("A file path is required for the 'file' source.")
            ingestion, histories = FleetIngestor(cfg=self.cfg).from_file(fleet_id, path)
            identity = None
        else:
            raise ValueError(f"Unknown fleet source {source!r}")

        snapshot = self._service.create_fleet_snapshot(
            fleet_id, histories, ingestion=ingestion, identity=identity
        )
        if self._repository is not None and not self.cfg.deployment.read_only:
            try:
                self._repository.save_fleet_snapshot(snapshot)
            except Exception as exc:  # noqa: BLE001 - shown, never silent
                logger.warning("Snapshot not stored: %s", exc)
        return snapshot

    # -- monitoring --------------------------------------------------------
    def latest_monitoring(self, fleet_id: str | None = None) -> MonitoringSnapshot | None:
        if self.mode == "service":
            return (
                self._repository.latest_monitoring_snapshot(fleet_id) if self._repository else None
            )
        import httpx

        try:
            response = httpx.get(
                f"{self.base_url}/v1/monitoring/latest",
                params={"fleet_id": fleet_id} if fleet_id else None,
                timeout=30.0,
            )
            if response.status_code != 200:
                return None
            return MonitoringSnapshot(**response.json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch monitoring over HTTP: %s", exc)
            return None

    def alerts(self, fleet_id: str | None = None, *, limit: int = 100) -> list[dict]:
        if self.mode == "service" and self._repository:
            return [
                a.model_dump(mode="json")
                for a in self._repository.list_alerts(fleet_id, limit=limit)
            ]
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/v1/fleet/{fleet_id}/alerts", timeout=30.0)
            return response.json().get("alerts", []) if response.status_code == 200 else []
        except Exception:  # noqa: BLE001
            return []

    # -- registry ----------------------------------------------------------
    def models(self) -> dict[str, Any]:
        if self.mode == "service":
            from battery_rul.registry.store import FileModelRegistry, RegistryError

            registry = FileModelRegistry(cfg=self.cfg)
            try:
                production = registry.production_model(task="rul_regression")
                return {
                    "models": [e.to_dict() for e in registry.list_models()],
                    "production": production.to_dict() if production is not None else None,
                    "history": registry.history(limit=25),
                }
            except RegistryError as exc:
                return {"models": [], "production": None, "error": str(exc)}
        import httpx

        try:
            return httpx.get(f"{self.base_url}/v1/models", timeout=30.0).json()
        except Exception as exc:  # noqa: BLE001
            return {"models": [], "production": None, "error": str(exc)}

    def twin_snapshot(self, battery_id: str, history: pd.DataFrame) -> Any:
        """Drill into one cell using the Milestone 2 battery-level service."""
        if self.mode != "service" or self._service is None:
            raise RuntimeError("Battery drill-down requires the in-process service mode.")
        return self._service.twin.create_snapshot(battery_id, history)


# ---------------------------------------------------------------------------
# Presentation helpers (pure functions over the domain objects)
# ---------------------------------------------------------------------------
def battery_table(snapshot: FleetSnapshot) -> pd.DataFrame:
    """The per-battery table the ranking pages render.

    Column names carry their provenance — ``(measured)`` versus ``(predicted)``
    — because a table that prints an 84 % measurement beside a 38-cycle model
    output in the same typeface invites the reader to trust both equally.
    """
    rows = []
    for record in snapshot.batteries:
        inspection = (
            record.priority_record.inspection
            if record.priority_record and record.priority_record.inspection
            else None
        )
        rows.append(
            {
                "battery_id": record.battery_id,
                "status": record.status.value,
                "priority": record.priority.value,
                "priority_score": record.priority_score,
                "risk (predicted)": record.failure_risk,
                "risk_experimental": record.risk_is_experimental,
                "RUL (predicted)": record.predicted_rul,
                "RUL lower": record.rul_lower_bound,
                "RUL upper": record.rul_upper_bound,
                "interval width": record.interval_width,
                "SOH (measured)": record.measured_soh,
                "health class": record.health_class,
                "fade %/10cyc": record.fade_trend_pct_per_10,
                "temp °C/10cyc": record.temperature_trend_c_per_10,
                "data quality": record.data_quality_class,
                "action": record.recommended_action,
                "inspect within (cycles)": inspection.recommended_cycles if inspection else None,
                "latest cycle": record.latest_cycle,
                "model version": record.model_version,
            }
        )
    return pd.DataFrame(rows)


def score_breakdown_table(snapshot: FleetSnapshot, battery_id: str) -> pd.DataFrame:
    """One cell's priority-score components, as rendered on the critical page."""
    record = snapshot.battery(battery_id)
    if record is None or record.priority_record is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "component": component.name,
                "raw value": component.raw_value,
                "normalised": component.normalised,
                "weight": component.weight,
                "contribution": component.contribution,
                "available": component.available,
                "transformation": component.transformation,
            }
            for component in record.priority_record.score_breakdown
        ]
    )


def workload_table(snapshot: FleetSnapshot) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": bucket.label,
                "cycles": bucket.horizon_cycles,
                "batteries": bucket.battery_count,
                "% of evaluated": bucket.percent_of_evaluated,
                "lower (optimistic)": bucket.lower_count,
                "upper (conservative)": bucket.upper_count,
            }
            for bucket in snapshot.workload_forecast.buckets
        ]
    )
