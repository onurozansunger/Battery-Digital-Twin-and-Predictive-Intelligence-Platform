"""Everything the dashboard needs to fetch, in one testable place.

Streamlit code is awkward to test — it needs a script runner and a session. So
all data access lives here as plain functions over plain objects, the dashboard
script does layout only, and the tests exercise this module directly.

Two back-ends, one interface:

``service``
    Calls :class:`BatteryDigitalTwinService` in-process. What the demo mode uses.

``api``
    Calls the running FastAPI service over HTTP. Proves the API contract is
    sufficient to build a real client against.

Both return the same :class:`BatteryTwinSnapshot`, because both are the same
code path — one of them just goes through a socket first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import BatteryTwinSnapshot
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DashboardData", "TwinClient", "load_demo_cycles", "trajectory_frame"]

#: Columns that encode the label; a dashboard rendering a "measurement" must
#: never show one of these, and the demo loader strips them.
_LABEL_COLUMNS = (
    "rul_cycles",
    "rul_raw_cycles",
    "eol_cycle",
    "life_fraction",
    "is_censored",
    "soh_target",
    "soh_health_class",
    "soh_reference_capacity_ah",
)


@dataclass
class DashboardData:
    """The processed cycle table the demo mode renders."""

    cycles: pd.DataFrame
    batteries: list[str]
    source: str

    def history(self, battery_id: str, *, up_to_cycle: int | None = None) -> pd.DataFrame:
        frame = self.cycles.loc[self.cycles["battery_id"] == battery_id]
        if up_to_cycle is not None:
            frame = frame.loc[frame["cycle_index"] <= up_to_cycle]
        return frame.reset_index(drop=True)


def load_demo_cycles(cfg: ExperimentConfig) -> DashboardData | None:
    """Load the processed cycle table produced by the Milestone 1 pipeline.

    Returns ``None`` when it does not exist. The dashboard then says so rather
    than fabricating a fleet — inventing plausible-looking battery data for a
    demo is exactly how a prototype gets mistaken for a working product.
    """
    path = cfg.paths.processed_dir / "cycles.parquet"
    if not path.is_file():
        logger.warning("No processed cycle table at %s; demo mode unavailable", path)
        return None
    frame = pd.read_parquet(path)
    keep = [c for c in frame.columns if c not in _LABEL_COLUMNS]
    frame = frame[keep]
    return DashboardData(
        cycles=frame,
        batteries=sorted(frame["battery_id"].unique().tolist()),
        source=str(path),
    )


@dataclass
class TwinClient:
    """Uniform access to the twin, in-process or over HTTP."""

    cfg: ExperimentConfig
    mode: Literal["service", "api"] = "service"
    service: Any | None = None
    base_url: str | None = None

    @classmethod
    def build(cls, cfg: ExperimentConfig) -> TwinClient:
        mode = cfg.service.dashboard_mode
        if mode == "api":
            return cls(cfg=cfg, mode="api", base_url=cfg.service.dashboard_api_url)
        from battery_rul.digital_twin.service import BatteryDigitalTwinService

        return cls(cfg=cfg, mode="service", service=BatteryDigitalTwinService.create(cfg))

    def readiness(self) -> dict[str, Any]:
        if self.mode == "service":
            assert self.service is not None
            return self.service.readiness()
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/ready", timeout=10.0)
            return response.json()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not raised
            return {"ready": False, "bundles": {}, "errors": {"http": str(exc)}}

    def model_metadata(self) -> dict[str, Any]:
        if self.mode == "service":
            assert self.service is not None
            return self.service.get_model_metadata()
        import httpx

        try:
            return httpx.get(f"{self.base_url}/model-info", timeout=10.0).json()["metadata"]
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def snapshot(self, battery_id: str, history: pd.DataFrame) -> BatteryTwinSnapshot:
        if self.mode == "service":
            assert self.service is not None
            return self.service.create_snapshot(battery_id, history)
        import httpx

        payload = {
            "battery_id": battery_id,
            "history": _records(history),
            "include_explanation": True,
        }
        response = httpx.post(
            f"{self.base_url}/v1/digital-twin/snapshot", json=payload, timeout=60.0
        )
        response.raise_for_status()
        return BatteryTwinSnapshot(**response.json()["snapshot"])


def _records(history: pd.DataFrame) -> list[dict[str, Any]]:
    """Frame to JSON-safe records the API schema accepts."""
    from battery_rul.api.schemas import CycleRecord

    allowed = set(CycleRecord.model_fields)
    frame = history[[c for c in history.columns if c in allowed]].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)
    records = frame.to_dict(orient="records")
    return [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in records]


def trajectory_frame(
    client: TwinClient,
    battery_id: str,
    history: pd.DataFrame,
    *,
    step: int = 5,
    max_points: int = 40,
) -> pd.DataFrame:
    """Replay the cell cycle by cycle, producing a trajectory of twin outputs.

    Each point is a snapshot built from the history **up to that cycle only** —
    which is the whole point: it shows what the twin would have said at the time,
    not what it says now with the benefit of the full record.
    """
    cycles = sorted(history["cycle_index"].unique().tolist())
    if not cycles:
        return pd.DataFrame()
    stride = max(step, len(cycles) // max_points or 1)
    rows: list[dict[str, Any]] = []

    for cycle in cycles[::stride]:
        prefix = history.loc[history["cycle_index"] <= cycle]
        try:
            snapshot = client.snapshot(battery_id, prefix)
        except Exception as exc:  # noqa: BLE001 - a bad point must not kill the chart
            logger.debug("Trajectory point at cycle %s failed: %s", cycle, exc)
            continue
        interval = snapshot.prediction.rul_interval
        rows.append(
            {
                "cycle_index": int(cycle),
                "soh": snapshot.health.soh,
                "soh_measured": snapshot.health.soh_measured,
                "health_class": snapshot.health.health_class,
                "rul": snapshot.prediction.rul_cycles,
                "rul_lower": interval.lower_bound if interval else None,
                "rul_upper": interval.upper_bound if interval else None,
                "risk": snapshot.failure_risk.probability,
                "risk_class": snapshot.failure_risk.risk_class,
                "quality_class": snapshot.data_quality.quality_class,
                "action_code": snapshot.recommendation.action_code,
            }
        )
    return pd.DataFrame(rows)
