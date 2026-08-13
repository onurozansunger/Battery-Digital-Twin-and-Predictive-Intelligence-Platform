"""Request and response schemas for the fleet API.

Requests are their own types (what a client sends is not what the service works
with); responses reuse the fleet domain models, because those *are* the
contract.

Two things this module enforces that the domain model cannot:

*Bounded requests.* A fleet endpoint that accepts an unbounded battery list is a
memory-exhaustion primitive. Every list has a maximum length, checked by
Pydantic before a frame is allocated.

*Pagination.* A 128-cell fleet with full per-battery records is a large
response, and a 10 000-cell one is an unusable one. Battery records are paged,
and the page metadata always states the true total so a client can tell a short
page from a short fleet.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from battery_rul.api.schemas import CycleRecord
from battery_rul.fleet.domain import (
    BatteryIngestionRecord,
    FleetBatteryRecord,
    FleetDataQualitySummary,
    FleetDriftStatus,
    FleetHealthDistribution,
    FleetIdentity,
    FleetMaintenanceSummary,
    FleetModelMetadata,
    FleetReplacementSummary,
    FleetRiskDistribution,
    FleetStatistics,
    FleetSummary,
    FleetWorkloadForecast,
)
from battery_rul.fleet.ranking import RANKING_KEYS

__all__ = [
    "MAX_BATTERIES_PER_REQUEST",
    "MAX_CYCLES_PER_BATTERY",
    "AlertListResponse",
    "BatteryHistoryPayload",
    "CriticalBatteriesResponse",
    "FleetPage",
    "FleetRankRequest",
    "FleetRankResponse",
    "FleetRequest",
    "FleetSnapshotResponse",
    "FleetSummaryResponse",
    "MaintenancePlanResponse",
    "ModelListResponse",
    "MonitoringRunResponse",
    "PageMeta",
    "PromotionRequest",
    "ReplacementPlanResponse",
    "RollbackRequest",
]

#: Hard caps, independent of configuration. Configuration may lower them; it may
#: not raise them past these, because these bound the memory a single unvalidated
#: request can cause the process to allocate.
MAX_BATTERIES_PER_REQUEST = 500
MAX_CYCLES_PER_BATTERY = 20_000


class BatteryHistoryPayload(BaseModel):
    """One cell's history inside a fleet request."""

    model_config = ConfigDict(extra="forbid")

    battery_id: str = Field(min_length=1, max_length=64)
    history: list[CycleRecord] = Field(min_length=1, max_length=MAX_CYCLES_PER_BATTERY)

    @field_validator("battery_id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("battery_id must not be blank")
        if any(ch in cleaned for ch in ("/", "\\", "\0")):
            raise ValueError("battery_id must not contain path separators")
        return cleaned

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.model_dump(exclude_none=False) for record in self.history])


class FleetRequest(BaseModel):
    """A fleet submitted for online scoring."""

    model_config = ConfigDict(extra="forbid")

    fleet_id: str = Field(min_length=1, max_length=64)
    batteries: list[BatteryHistoryPayload] = Field(
        min_length=1, max_length=MAX_BATTERIES_PER_REQUEST
    )
    include_battery_records: bool = Field(
        default=True,
        description="Set false for a summary-only response — the cheapest way to "
        "poll a large fleet.",
    )
    page: int = Field(default=1, ge=1)
    page_size: int | None = Field(default=None, ge=1, le=1000)

    @field_validator("fleet_id")
    @classmethod
    def _clean_fleet_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(ch in cleaned for ch in ("/", "\\", "\0", "..")):
            raise ValueError("fleet_id must be non-blank and free of path separators")
        return cleaned

    @field_validator("batteries")
    @classmethod
    def _unique_ids(cls, value: list[BatteryHistoryPayload]) -> list[BatteryHistoryPayload]:
        identifiers = [b.battery_id.strip() for b in value]
        duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate battery_id(s) in the request: {duplicates}")
        return value


class FleetRankRequest(FleetRequest):
    """A fleet plus how to order it."""

    rank_by: Literal[
        "priority",
        "priority_score",
        "failure_risk",
        "rul",
        "rul_lower_bound",
        "soh",
        "soh_trend",
        "fade_trend",
        "temperature_trend",
        "resistance_trend",
        "data_quality",
        "uncertainty",
    ] = "priority"
    limit: int | None = Field(default=None, ge=1, le=1000)
    include_unevaluated: bool = False

    @field_validator("rank_by")
    @classmethod
    def _supported(cls, value: str) -> str:
        if value not in RANKING_KEYS:
            raise ValueError(f"Unsupported ranking key {value!r}")
        return value


class PageMeta(BaseModel):
    """Where this page sits in the whole result."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool = False

    @classmethod
    def build(cls, *, page: int, page_size: int, total: int) -> PageMeta:
        total_pages = (total + page_size - 1) // page_size if page_size else 0
        return cls(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=total_pages,
            has_next=page < total_pages,
        )


class FleetPage(BaseModel):
    """A page of battery records with its metadata."""

    items: list[FleetBatteryRecord] = Field(default_factory=list)
    pagination: PageMeta


class FleetSnapshotResponse(BaseModel):
    """The fleet snapshot, with battery records paged.

    The aggregate blocks are always complete: paging them would produce a
    summary of a page, which is exactly the misleading aggregate the fleet
    layer exists to avoid.
    """

    model_config = ConfigDict(protected_namespaces=())

    fleet_id: str
    snapshot_id: str
    generated_at_utc: str
    schema_version: str
    identity: FleetIdentity
    battery_count: int
    successfully_processed_count: int
    failed_count: int
    insufficient_data_count: int
    summary: FleetSummary
    health_distribution: FleetHealthDistribution
    risk_distribution: FleetRiskDistribution
    maintenance_summary: FleetMaintenanceSummary
    replacement_summary: FleetReplacementSummary
    workload_forecast: FleetWorkloadForecast
    fleet_statistics: FleetStatistics
    data_quality: FleetDataQualitySummary
    drift_status: FleetDriftStatus
    model_metadata: FleetModelMetadata
    batteries: FleetPage | None = None
    ingestion_records: list[BatteryIngestionRecord] = Field(default_factory=list)
    batch_id: str | None = None
    processing_duration_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class FleetSummaryResponse(BaseModel):
    """Summary only — no per-battery records."""

    model_config = ConfigDict(protected_namespaces=())

    summary: FleetSummary
    health_distribution: FleetHealthDistribution
    risk_distribution: FleetRiskDistribution
    maintenance_summary: FleetMaintenanceSummary
    fleet_statistics: FleetStatistics
    data_quality: FleetDataQualitySummary
    drift_status: FleetDriftStatus
    model_metadata: FleetModelMetadata
    snapshot_id: str
    generated_at_utc: str
    warnings: list[str] = Field(default_factory=list)


class FleetRankResponse(BaseModel):
    """An ordered fleet, with the ordering criterion named."""

    fleet_id: str
    snapshot_id: str
    generated_at_utc: str
    rank_by: str
    ranking: FleetPage
    excluded_unevaluated_count: int = 0
    methodology_note: str = (
        "The composite priority score is a configurable decision-support policy, "
        "not an optimum. Each record carries its own score breakdown."
    )
    warnings: list[str] = Field(default_factory=list)


class MaintenancePlanResponse(BaseModel):
    fleet_id: str
    snapshot_id: str
    generated_at_utc: str
    maintenance_summary: FleetMaintenanceSummary
    workload_forecast: FleetWorkloadForecast
    batteries: FleetPage
    disclaimer: str
    warnings: list[str] = Field(default_factory=list)


class ReplacementPlanResponse(BaseModel):
    fleet_id: str
    snapshot_id: str
    generated_at_utc: str
    replacement_summary: FleetReplacementSummary
    candidates: FleetPage
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CriticalBatteriesResponse(BaseModel):
    fleet_id: str
    snapshot_id: str
    generated_at_utc: str
    critical_priorities: list[str]
    batteries: FleetPage
    total_critical: int = 0


class AlertListResponse(BaseModel):
    fleet_id: str | None = None
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    pagination: PageMeta
    note: str = (
        "Alerts require human review. Nothing in this platform retrains, promotes "
        "or removes an asset from service automatically."
    )


class MonitoringRunResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    snapshot_id: str
    fleet_id: str
    generated_at_utc: str
    overall_status: str
    data_quality_status: str
    feature_drift_status: str
    prediction_drift_status: str
    performance_status: str
    n_alerts: int
    model_version: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]] = Field(default_factory=list)
    production: dict[str, Any] | None = None
    registry_available: bool = True
    note: str = ""


class PromotionRequest(BaseModel):
    """Administrative promotion. Disabled unless explicitly enabled."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    by: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)
    dry_run: bool = True


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, max_length=128)
    by: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)
