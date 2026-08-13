"""The fleet domain model.

Same discipline as :mod:`battery_rul.digital_twin.domain`, one level up: every
value carries a :class:`~battery_rul.digital_twin.domain.Provenance` tag, and the
aggregate objects carry their **denominators**. A fleet page that prints "median
RUL 94 cycles" without saying it was computed over 103 of 128 cells, excluding
21 that could not be scored, is not a summary — it is a way of losing 21 cells.

Every model here is JSON-serialisable by construction (``model_dump(mode="json")``),
because these objects are simultaneously the API response body, the persisted
snapshot and the dashboard's input.

Nothing in this module computes anything. Deriving a priority, a rank or a
statistic lives in the sibling modules; this file only says what the results look
like.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from battery_rul.digital_twin.domain import (
    BatteryTwinSnapshot,
    Provenance,
)

__all__ = [
    "FLEET_SNAPSHOT_SCHEMA_VERSION",
    "BatteryIngestionRecord",
    "BatteryPriorityRecord",
    "FleetBatteryRecord",
    "FleetBatteryReference",
    "FleetDataQualitySummary",
    "FleetDriftStatus",
    "FleetHealthDistribution",
    "FleetIdentity",
    "FleetIngestionResult",
    "FleetMaintenanceSummary",
    "FleetModelMetadata",
    "FleetReplacementSummary",
    "FleetRiskDistribution",
    "FleetSnapshot",
    "FleetStatistics",
    "FleetSummary",
    "FleetTrendPoint",
    "FleetWorkloadForecast",
    "InspectionRecommendation",
    "MaintenancePriority",
    "MonitoringStatus",
    "ProcessingStatus",
    "ReplacementCandidate",
    "ReplacementHorizon",
    "ScoreComponent",
    "WorkloadBucket",
]

#: Bumped on any breaking change to the fleet snapshot wire format.
FLEET_SNAPSHOT_SCHEMA_VERSION = "3.0"


class _Model(BaseModel):
    """Unknown fields are rejected — a typo is an error, not a silent drop."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ProcessingStatus(StrEnum):
    """What happened to one battery during a fleet run."""

    SUCCESS = "success"
    #: Well-formed input, but too thin (or too early) to support a prediction.
    INSUFFICIENT_DATA = "insufficient_data"
    #: Rejected at ingestion, or inference raised. Never silently dropped.
    FAILED = "failed"


class MaintenancePriority(StrEnum):
    """The fleet maintenance priority ladder, most severe first."""

    P0_CRITICAL = "P0_CRITICAL"
    P1_URGENT = "P1_URGENT"
    P2_HIGH = "P2_HIGH"
    P3_MEDIUM = "P3_MEDIUM"
    P4_LOW = "P4_LOW"
    P5_MONITOR = "P5_MONITOR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    @property
    def severity(self) -> int:
        """0 is most severe. ``INSUFFICIENT_DATA`` sorts last: it is not a
        severity at all, it is an absence of evidence, and ranking it beside
        real severities would either hide critical cells or invent urgency."""
        return _PRIORITY_SEVERITY[self]

    @property
    def is_actionable(self) -> bool:
        return self not in (MaintenancePriority.INSUFFICIENT_DATA, MaintenancePriority.P4_LOW)


_PRIORITY_SEVERITY: dict[MaintenancePriority, int] = {
    MaintenancePriority.P0_CRITICAL: 0,
    MaintenancePriority.P1_URGENT: 1,
    MaintenancePriority.P2_HIGH: 2,
    MaintenancePriority.P3_MEDIUM: 3,
    MaintenancePriority.P4_LOW: 4,
    MaintenancePriority.P5_MONITOR: 5,
    MaintenancePriority.INSUFFICIENT_DATA: 6,
}


class ReplacementHorizon(StrEnum):
    """Planning bucket for a replacement candidate."""

    NEAR_TERM = "near_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    NOT_FLAGGED = "not_flagged"
    UNKNOWN = "unknown"


class MonitoringStatus(StrEnum):
    """Three-level status shared by every monitoring surface."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

    @property
    def severity(self) -> int:
        return {"OK": 0, "UNKNOWN": 1, "WARNING": 2, "CRITICAL": 3}[self.value]

    @classmethod
    def worst(cls, statuses: list[MonitoringStatus]) -> MonitoringStatus:
        return max(statuses, key=lambda s: s.severity) if statuses else cls.UNKNOWN


# ---------------------------------------------------------------------------
# Identity and ingestion
# ---------------------------------------------------------------------------
class FleetIdentity(_Model):
    """Which fleet this is, and whether its data is real."""

    fleet_id: str = Field(min_length=1, max_length=64)
    name: str | None = None
    operator: str | None = None
    description: str | None = None
    source: str = Field(
        default="unknown",
        description="Where the histories came from: a processed-cycle table, an "
        "uploaded file, or the synthetic demo generator.",
    )
    is_demo_data: bool = Field(
        default=False,
        description="True when any history is synthetic. Demo fleets are always "
        "labelled: an invented fleet mistaken for a measured one is the single "
        "most damaging thing this platform could do.",
    )
    data_notice: str | None = None

    @field_validator("fleet_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("fleet_id must not be blank")
        if any(ch in cleaned for ch in ("/", "\\", "\0", "..")):
            raise ValueError("fleet_id must not contain path separators or '..'")
        return cleaned


class FleetBatteryReference(_Model):
    """A pointer to one cell's history, without the measurements themselves."""

    battery_id: str = Field(min_length=1, max_length=64)
    n_cycles: int = Field(ge=0)
    first_cycle: int | None = None
    latest_cycle: int | None = None
    source: str | None = None
    is_synthetic: bool = False


class BatteryIngestionRecord(_Model):
    """Per-battery outcome of ingestion. Failures are reported, never dropped."""

    battery_id: str
    status: ProcessingStatus
    n_rows: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: str | None = None

    @model_validator(mode="after")
    def _failed_has_reason(self) -> BatteryIngestionRecord:
        if self.status is ProcessingStatus.FAILED and not self.errors:
            raise ValueError("A failed ingestion record must carry at least one error")
        return self


class FleetIngestionResult(_Model):
    """Everything ingestion produced, including what it refused.

    ``histories`` is deliberately absent: measurement frames do not belong in a
    serialisable result object. The ingestion layer returns them alongside this,
    in memory.
    """

    fleet_id: str
    generated_at_utc: str = Field(default_factory=_utc_now)
    source: str
    accepted: list[FleetBatteryReference] = Field(default_factory=list)
    records: list[BatteryIngestionRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_fingerprint: str = ""
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    is_demo_data: bool = False

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.records if r.status is ProcessingStatus.FAILED)

    def record_for(self, battery_id: str) -> BatteryIngestionRecord | None:
        return next((r for r in self.records if r.battery_id == battery_id), None)


# ---------------------------------------------------------------------------
# Priority, inspection, replacement
# ---------------------------------------------------------------------------
class ScoreComponent(_Model):
    """One term of the composite priority score, with its transformation.

    The transformation string is not documentation-for-its-own-sake: a ranking
    that cannot be explained cannot be argued with, and an engineer told to
    inspect cell B0042 first is entitled to see that its risk term contributed
    27 of its 71 points and how 0.9 became 0.27.
    """

    name: str
    raw_value: float | None = None
    normalised: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    contribution: float
    transformation: str
    available: bool = True


class InspectionRecommendation(_Model):
    """When to look at this cell. Cycle-based; days only when a rate exists."""

    recommended_cycles: int | None = Field(
        default=None, ge=0, description="Inspect within this many cycles. 0 means immediately."
    )
    recommended_label: str
    estimated_days: float | None = Field(
        default=None,
        ge=0.0,
        description="Only populated when a recent cycles-per-day rate could be "
        "estimated from timestamps. A cycle-to-day conversion is never invented.",
    )
    cycles_per_day: float | None = Field(default=None, gt=0.0)
    basis: str
    assumptions: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE_BASED


class BatteryPriorityRecord(_Model):
    """The maintenance-priority verdict for one cell, with its whole argument."""

    battery_id: str
    priority: MaintenancePriority
    priority_score: float = Field(ge=0.0)
    score_breakdown: list[ScoreComponent] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str
    action_title: str = ""
    inspection: InspectionRecommendation | None = None
    critical_override: bool = Field(
        default=False,
        description="A rule forced this priority regardless of the smooth score.",
    )
    disclaimer: str = ""
    provenance: Provenance = Provenance.RULE_BASED


class ReplacementCandidate(_Model):
    """Advisory replacement planning for one cell. Not a scheduler."""

    battery_id: str
    replacement_candidate: bool
    replacement_horizon: ReplacementHorizon
    horizon_cycles: int | None = Field(default=None, ge=0)
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    planning_category: str
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    rul_point: float | None = None
    rul_lower_bound: float | None = None
    rul_upper_bound: float | None = None
    provenance: Provenance = Provenance.RULE_BASED


# ---------------------------------------------------------------------------
# Per-battery fleet record
# ---------------------------------------------------------------------------
class FleetBatteryRecord(_Model):
    """One cell's row in a fleet snapshot.

    A compact projection of the full :class:`BatteryTwinSnapshot` plus the fleet
    layer's own verdicts. The full snapshot is not embedded: a 128-cell fleet
    would produce a response measured in megabytes, and the battery-level
    endpoint already returns it for the one cell a reader drills into.
    """

    battery_id: str
    status: ProcessingStatus
    latest_cycle: int | None = None
    n_cycles: int = Field(default=0, ge=0)

    # -- measured / derived ------------------------------------------------
    measured_soh: float | None = Field(default=None, description="Derived from measured capacity.")
    capacity_fade_percent: float | None = None
    health_class: str = "unknown"

    # -- predicted ---------------------------------------------------------
    predicted_soh_forecast: float | None = None
    soh_forecast_horizon_cycles: int | None = None
    predicted_rul: float | None = None
    rul_lower_bound: float | None = None
    rul_upper_bound: float | None = None
    interval_width: float | None = None
    interval_coverage_target: float | None = None
    failure_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_is_experimental: bool = False
    risk_class: str = "unknown"
    risk_horizon_cycles: int | None = None

    # -- trends (derived, no model involved) --------------------------------
    soh_trend_pct_per_10: float | None = None
    fade_trend_pct_per_10: float | None = None
    temperature_trend_c_per_10: float | None = None
    resistance_trend_pct_per_10: float | None = None

    # -- quality and decisions ---------------------------------------------
    data_quality_class: str = "unknown"
    data_quality_score: float | None = None
    out_of_distribution_feature_count: int = 0
    priority: MaintenancePriority = MaintenancePriority.INSUFFICIENT_DATA
    priority_score: float = 0.0
    priority_record: BatteryPriorityRecord | None = None
    replacement: ReplacementCandidate | None = None
    recommended_action: str = "INSUFFICIENT_DATA"
    twin_action_code: str | None = Field(
        default=None,
        description="The battery-level recommendation from Milestone 2, kept "
        "distinct from the fleet-level action so the two layers stay auditable.",
    )

    # -- provenance --------------------------------------------------------
    model_version: str | None = None
    model_name: str | None = None
    snapshot_generated_at_utc: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    @model_validator(mode="after")
    def _interval_ordered(self) -> FleetBatteryRecord:
        low, point, high = self.rul_lower_bound, self.predicted_rul, self.rul_upper_bound
        if low is not None and point is not None and low > point + 1e-6:
            raise ValueError(f"{self.battery_id}: RUL lower bound {low} exceeds point {point}")
        if high is not None and point is not None and high < point - 1e-6:
            raise ValueError(f"{self.battery_id}: RUL upper bound {high} is below point {point}")
        return self

    @property
    def is_evaluated(self) -> bool:
        """True when this record may enter a predicted-quantity aggregate."""
        return self.status is ProcessingStatus.SUCCESS


# ---------------------------------------------------------------------------
# Distributions and summaries
# ---------------------------------------------------------------------------
class FleetHealthDistribution(_Model):
    """Counts by health class, over the cells that have a measured SOH."""

    counts: dict[str, int] = Field(default_factory=dict)
    denominator: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    provenance: Provenance = Provenance.DERIVED

    def get(self, name: str) -> int:
        return int(self.counts.get(name, 0))


class FleetRiskDistribution(_Model):
    """Counts by risk class plus the probability distribution's shape."""

    counts: dict[str, int] = Field(default_factory=dict)
    denominator: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    quantiles: dict[str, float] = Field(default_factory=dict)
    mean: float | None = None
    experimental_model: bool = Field(
        default=False,
        description="True when the risk model failed its Milestone 2 acceptance "
        "gate. Its probabilities are shown but excluded from the decision rules.",
    )
    above_thresholds: dict[str, int] = Field(default_factory=dict)
    provenance: Provenance = Provenance.PREDICTED


class FleetMaintenanceSummary(_Model):
    """Priority counts and the actions they imply."""

    priority_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    critical_count: int = Field(default=0, ge=0)
    inspection_recommended_count: int = Field(default=0, ge=0)
    insufficient_data_count: int = Field(default=0, ge=0)
    denominator: int = Field(default=0, ge=0)
    high_risk_battery_ids: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE_BASED


class WorkloadBucket(_Model):
    """One horizon of the maintenance-workload forecast."""

    label: str
    horizon_cycles: int | None = None
    battery_count: int = Field(ge=0)
    percent_of_evaluated: float = Field(ge=0.0, le=100.0)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    lower_count: int | None = Field(
        default=None,
        ge=0,
        description="Uncertainty-aware count using the RUL upper bound (fewer cells "
        "fall inside the horizon when each is given its most optimistic life).",
    )
    upper_count: int | None = Field(
        default=None, ge=0, description="Uncertainty-aware count using the RUL lower bound."
    )
    battery_ids: list[str] = Field(default_factory=list)


class FleetWorkloadForecast(_Model):
    """Expected near-term maintenance demand. A forecast, not a schedule."""

    buckets: list[WorkloadBucket] = Field(default_factory=list)
    evaluated_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0)
    basis: str = ""
    caveats: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE_BASED


class FleetReplacementSummary(_Model):
    """Replacement candidates grouped by planning horizon."""

    counts_by_horizon: dict[str, int] = Field(default_factory=dict)
    lower_counts_by_horizon: dict[str, int] = Field(default_factory=dict)
    upper_counts_by_horizon: dict[str, int] = Field(default_factory=dict)
    candidate_count: int = Field(default=0, ge=0)
    denominator: int = Field(default=0, ge=0)
    candidate_battery_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE_BASED


class FleetStatistics(_Model):
    """Fleet-level numeric summaries, each with its own denominator.

    ``*_denominator`` is not decoration. Median RUL over the cells that could be
    scored is a different number from median RUL over the fleet, and only one of
    them is computable. Publishing the first while implying the second is the
    most common way a fleet dashboard misleads.
    """

    soh_median: float | None = None
    soh_mean: float | None = None
    soh_quantiles: dict[str, float] = Field(default_factory=dict)
    soh_denominator: int = Field(default=0, ge=0)
    rul_median: float | None = None
    rul_mean: float | None = None
    rul_quantiles: dict[str, float] = Field(default_factory=dict)
    rul_denominator: int = Field(default=0, ge=0)
    rul_lower_bound_median: float | None = None
    interval_width_median: float | None = None
    risk_median: float | None = None
    risk_mean: float | None = None
    risk_denominator: int = Field(default=0, ge=0)
    below_rul_threshold_counts: dict[str, int] = Field(default_factory=dict)
    above_risk_threshold_counts: dict[str, int] = Field(default_factory=dict)
    missingness: dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of the fleet with no value for each summarised field.",
    )
    provenance: Provenance = Provenance.DERIVED


class FleetDataQualitySummary(_Model):
    """Fleet-level input quality. Not drift — see :class:`FleetDriftStatus`."""

    status: MonitoringStatus = MonitoringStatus.UNKNOWN
    quality_class_counts: dict[str, int] = Field(default_factory=dict)
    mean_quality_score: float | None = None
    poor_or_worse_fraction: float | None = None
    insufficient_fraction: float | None = None
    mean_missing_feature_fraction: float | None = None
    per_feature_missing_rate: dict[str, float] = Field(default_factory=dict)
    batteries_with_schema_mismatch: list[str] = Field(default_factory=list)
    batteries_with_ood_features: list[str] = Field(default_factory=list)
    check_failure_rates: dict[str, float] = Field(default_factory=dict)
    denominator: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.DERIVED


class FleetDriftStatus(_Model):
    """A pointer to the drift verdict; the detail lives in a monitoring snapshot."""

    status: MonitoringStatus = MonitoringStatus.UNKNOWN
    feature_drift_status: MonitoringStatus = MonitoringStatus.UNKNOWN
    prediction_drift_status: MonitoringStatus = MonitoringStatus.UNKNOWN
    n_features_tested: int = Field(default=0, ge=0)
    n_features_drifted: int = Field(default=0, ge=0)
    top_drifted_features: list[str] = Field(default_factory=list)
    reference_id: str | None = None
    monitoring_snapshot_id: str | None = None
    note: str = (
        "Feature or prediction drift indicates that inputs or model behaviour have "
        "changed. It is not by itself evidence that the model has become less "
        "accurate; only labelled outcomes can show that."
    )
    provenance: Provenance = Provenance.DERIVED


class FleetModelMetadata(_Model):
    """Which model produced this snapshot, and under which definitions."""

    active_model_version: str | None = None
    active_model_name: str | None = None
    registry_model_name: str | None = None
    registry_stage: str | None = None
    bundle_versions: dict[str, str] = Field(default_factory=dict)
    feature_pipeline_fingerprint: str | None = None
    data_version: str | None = None
    snapshot_schema_version: str = FLEET_SNAPSHOT_SCHEMA_VERSION
    battery_snapshot_schema_version: str | None = None
    git_revision: str | None = None
    package_version: str | None = None
    risk_horizon_cycles: int | None = None
    end_of_life_definition: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


class FleetSummary(_Model):
    """The executive view: the numbers a fleet page leads with."""

    fleet_id: str
    generated_at_utc: str
    battery_count: int = Field(ge=0)
    successfully_processed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    insufficient_data_count: int = Field(ge=0)
    healthy_count: int = Field(default=0, ge=0)
    slightly_degraded_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    inspection_recommended_count: int = Field(default=0, ge=0)
    replacement_planning_count: int = Field(default=0, ge=0)
    high_priority_battery_ids: list[str] = Field(default_factory=list)
    median_soh: float | None = None
    median_rul: float | None = None
    drift_status: MonitoringStatus = MonitoringStatus.UNKNOWN
    data_quality_status: MonitoringStatus = MonitoringStatus.UNKNOWN
    active_model_version: str | None = None
    is_demo_data: bool = False


class FleetTrendPoint(_Model):
    """One point of a fleet- or battery-level trend series."""

    label: str
    cycle_index: int | None = None
    generated_at_utc: str | None = None
    battery_count: int | None = Field(default=None, ge=0)
    median_soh: float | None = None
    median_rul: float | None = None
    mean_risk: float | None = None
    critical_count: int | None = Field(default=None, ge=0)
    value: float | None = None
    denominator: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------
class FleetSnapshot(_Model):
    """The complete fleet view at one moment.

    Partial success is a first-class outcome: ``batteries`` contains a record for
    every cell that was submitted, including the ones that failed, and the
    aggregate objects carry the denominators that exclude them.
    """

    fleet_id: str
    snapshot_id: str
    generated_at_utc: str = Field(default_factory=_utc_now)
    schema_version: str = FLEET_SNAPSHOT_SCHEMA_VERSION
    identity: FleetIdentity
    battery_count: int = Field(ge=0)
    successfully_processed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    insufficient_data_count: int = Field(ge=0)
    batteries: list[FleetBatteryRecord] = Field(default_factory=list)
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
    data_fingerprint: str = ""
    batch_id: str | None = None
    processing_duration_ms: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Fleet intelligence from a research prototype. Rankings, maintenance "
        "priorities and replacement horizons are configurable engineering policy "
        "applied to model outputs, not validated operational decisions, and not a "
        "substitute for battery-management-system protection or qualified "
        "engineering review."
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    @model_validator(mode="after")
    def _counts_consistent(self) -> FleetSnapshot:
        if self.batteries:
            if len(self.batteries) != self.battery_count:
                raise ValueError(
                    f"battery_count={self.battery_count} but {len(self.batteries)} "
                    "records are present; a fleet snapshot must account for every cell."
                )
            identifiers = [record.battery_id for record in self.batteries]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("A battery appears more than once in the snapshot")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Plain, JSON-safe dictionary — the persisted and served body."""
        return self.model_dump(mode="json")

    def battery(self, battery_id: str) -> FleetBatteryRecord | None:
        return next((r for r in self.batteries if r.battery_id == battery_id), None)

    def evaluated(self) -> list[FleetBatteryRecord]:
        return [r for r in self.batteries if r.is_evaluated]

    def without_batteries(self) -> FleetSnapshot:
        """A copy with the per-battery records dropped, for summary endpoints."""
        return self.model_copy(update={"batteries": [], "battery_count": self.battery_count})


def battery_record_from_snapshot(
    snapshot: BatteryTwinSnapshot,
    *,
    status: ProcessingStatus,
    trends: dict[str, float | None] | None = None,
) -> FleetBatteryRecord:
    """Project a battery-level twin snapshot into a fleet record.

    Lives here rather than in the inference service because it is a pure mapping
    between two published shapes, and because a test can then exercise it without
    a model.
    """
    interval = snapshot.prediction.rul_interval
    trends = trends or {}
    return FleetBatteryRecord(
        battery_id=snapshot.battery_id,
        status=status,
        latest_cycle=snapshot.measurement_summary.latest_cycle,
        n_cycles=snapshot.measurement_summary.n_cycles_supplied,
        measured_soh=snapshot.health.soh_measured,
        capacity_fade_percent=snapshot.health.capacity_fade_percent,
        health_class=snapshot.health.health_class,
        predicted_soh_forecast=snapshot.health.soh_forecast,
        soh_forecast_horizon_cycles=snapshot.health.soh_forecast_horizon_cycles,
        predicted_rul=snapshot.prediction.rul_cycles,
        rul_lower_bound=interval.lower_bound if interval else None,
        rul_upper_bound=interval.upper_bound if interval else None,
        interval_width=(interval.upper_bound - interval.lower_bound) if interval else None,
        interval_coverage_target=interval.interval_coverage_target if interval else None,
        failure_risk=snapshot.failure_risk.probability,
        risk_is_experimental=snapshot.failure_risk.is_experimental,
        risk_class=snapshot.failure_risk.risk_class,
        risk_horizon_cycles=snapshot.failure_risk.horizon_cycles,
        soh_trend_pct_per_10=trends.get("soh_trend_pct_per_10"),
        fade_trend_pct_per_10=trends.get("fade_trend_pct_per_10"),
        temperature_trend_c_per_10=trends.get("temperature_trend_c_per_10"),
        resistance_trend_pct_per_10=trends.get("resistance_trend_pct_per_10"),
        data_quality_class=snapshot.data_quality.quality_class,
        data_quality_score=snapshot.data_quality.quality_score,
        out_of_distribution_feature_count=len(snapshot.data_quality.out_of_distribution_flags),
        twin_action_code=snapshot.recommendation.action_code,
        model_version=snapshot.metadata.model_version,
        model_name=snapshot.metadata.model_name,
        snapshot_generated_at_utc=snapshot.generated_at_utc,
        warnings=list(snapshot.warnings),
    )
