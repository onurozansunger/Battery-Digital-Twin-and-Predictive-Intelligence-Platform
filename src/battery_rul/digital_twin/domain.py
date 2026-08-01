"""The digital-twin domain model.

Every quantity in a snapshot carries a :class:`Provenance` tag saying what kind
of thing it is:

``observed``    measured by the test rig and passed through unchanged
``derived``     computed deterministically from observations (SOH from capacity,
                a temperature trend from a trailing window)
``predicted``   produced by a fitted model — a claim about an unobserved
                quantity, not a fact about the cell
``estimated``   an uncertainty statement about a prediction
``rule_based``  produced by the deterministic recommendation rules, with no
                model involved

The tag is not decoration. A dashboard that renders "SOH 84.6 %" beside "RUL 38
cycles" in the same typeface invites the reader to treat both as measurements,
and one of them is a model output with a 19-cycle interval around it. The
provenance tag is what lets every surface render the difference, and it is why it
is a required field rather than an optional annotation.

These are Pydantic models, deliberately separate from the internal model classes
(``BaseModel``, ``FeaturePipeline``, ``MultiTaskSequenceModel``): the wire format
is a published contract with its own compatibility requirements, and coupling it
to an estimator's attribute names would mean a refactor becomes a breaking API
change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "BatteryExplanation",
    "BatteryHealthState",
    "BatteryIdentity",
    "BatteryMeasurements",
    "BatteryPrediction",
    "BatteryRecommendation",
    "BatteryRiskAssessment",
    "BatteryTwinSnapshot",
    "BatteryUncertainty",
    "DataQualityAssessment",
    "DegradationDriver",
    "Provenance",
    "TwinMetadata",
]

#: Bumped on any breaking change to the snapshot wire format.
SNAPSHOT_SCHEMA_VERSION = "2.0"


class Provenance(StrEnum):
    """Where a value came from, and therefore how much it may be trusted."""

    OBSERVED = "observed"
    DERIVED = "derived"
    PREDICTED = "predicted"
    ESTIMATED = "estimated"
    RULE_BASED = "rule_based"


class _Model(BaseModel):
    """Base: unknown fields are rejected, so a typo is an error not a silent drop."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
class BatteryIdentity(_Model):
    """Which cell this snapshot is about."""

    battery_id: str = Field(min_length=1, max_length=64)
    dataset: str | None = None
    chemistry: str | None = Field(
        default=None,
        description="Cell chemistry where known. The NASA cohort is LCO 18650; "
        "nothing in this platform has been validated on another chemistry.",
    )
    nominal_capacity_ah: float | None = Field(default=None, gt=0.0)
    provenance: Provenance = Provenance.OBSERVED


class BatteryMeasurements(_Model):
    """Summary of what was actually measured. Every field observed or derived."""

    latest_cycle: int = Field(ge=0)
    n_cycles_supplied: int = Field(ge=0)
    first_cycle: int = Field(ge=0)
    measured_capacity_ah: float | None = None
    reference_capacity_ah: float | None = None
    voltage_mean_v: float | None = None
    voltage_min_v: float | None = None
    current_mean_a: float | None = None
    temperature_mean_c: float | None = None
    temperature_max_c: float | None = None
    internal_resistance_ohm: float | None = None
    charge_duration_s: float | None = None
    discharge_duration_s: float | None = None
    energy_throughput_wh: float | None = None
    available_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.OBSERVED


class BatteryHealthState(_Model):
    """Present condition. SOH is a **fraction in [0, 1]**; the percent field is
    a rendering convenience computed from it, never an independent value."""

    soh: float | None = Field(default=None, description="State of health as a fraction.")
    soh_percent: float | None = None
    health_class: Literal["healthy", "slightly_degraded", "warning", "critical", "unknown"] = (
        "unknown"
    )
    soh_measured: float | None = Field(
        default=None,
        description="SOH computed directly from the latest measured capacity. "
        "Derived, not predicted — reported alongside the model estimate so the "
        "two can be compared.",
    )
    capacity_fade_percent: float | None = None
    provenance: Provenance = Provenance.PREDICTED

    @model_validator(mode="after")
    def _fill_percent(self) -> BatteryHealthState:
        if self.soh is not None and self.soh_percent is None:
            object.__setattr__(self, "soh_percent", round(100.0 * self.soh, 2))
        return self


class BatteryUncertainty(_Model):
    """A prediction interval and everything needed to read it correctly."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_coverage_target: float = Field(gt=0.0, lt=1.0)
    uncertainty_method: str
    calibration_sample_size: int = Field(ge=0)
    life_stage: str | None = None
    interval_type: Literal["prediction_interval", "confidence_interval"] = "prediction_interval"
    note: str = ""
    provenance: Provenance = Provenance.ESTIMATED

    @model_validator(mode="after")
    def _ordered(self) -> BatteryUncertainty:
        if not (self.lower_bound <= self.point_estimate <= self.upper_bound):
            raise ValueError(
                f"Interval must bracket the point estimate: {self.lower_bound} <= "
                f"{self.point_estimate} <= {self.upper_bound}"
            )
        return self


class BatteryPrediction(_Model):
    """Remaining useful life, in discharge cycles."""

    rul_cycles: float | None = Field(default=None, ge=0.0)
    rul_interval: BatteryUncertainty | None = None
    model_name: str | None = None
    is_scoreable: bool = True
    unscoreable_reason: str | None = None
    provenance: Provenance = Provenance.PREDICTED


class BatteryRiskAssessment(_Model):
    """Probability of reaching end of life within the configured horizon."""

    horizon_cycles: int = Field(gt=0)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    is_calibrated: bool = False
    calibration_method: str | None = None
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    exceeds_threshold: bool | None = None
    risk_class: Literal["low", "medium", "high", "very_high", "unknown"] = "unknown"
    label_definition: str = (
        "Derived label: the cell's smoothed capacity is projected to cross the "
        "configured end-of-life threshold within the horizon. Not an observed "
        "safety failure — the source dataset records none."
    )
    provenance: Provenance = Provenance.PREDICTED


class DegradationDriver(_Model):
    """One feature's contribution to a prediction, in careful language."""

    feature_name: str
    display_name: str
    contribution_direction: Literal[
        "increases_risk", "decreases_risk", "increases_rul", "decreases_rul", "neutral"
    ]
    contribution_magnitude: float = Field(ge=0.0)
    current_value: float | None = None
    reference_value: float | None = None
    explanation_text: str
    provenance: Provenance = Provenance.DERIVED


class BatteryExplanation(_Model):
    """Attribution for the prediction, with its method and its limits."""

    method: str
    drivers: list[DegradationDriver] = Field(default_factory=list)
    baseline_value: float | None = None
    caveat: str = (
        "Feature attributions describe how the model's output responds to its "
        "inputs. They are not causal claims about the cell, and attention "
        "weights in particular are a diagnostic, not an explanation."
    )
    provenance: Provenance = Provenance.DERIVED


class DataQualityAssessment(_Model):
    """Whether the supplied history can support a prediction at all."""

    quality_score: float = Field(ge=0.0, le=1.0)
    quality_class: Literal["GOOD", "ACCEPTABLE", "POOR", "INSUFFICIENT"]
    n_cycles: int = Field(ge=0)
    missing_feature_fraction: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    out_of_distribution_flags: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Provenance.DERIVED


class BatteryRecommendation(_Model):
    """A deterministic, rule-derived engineering suggestion."""

    action_code: str
    priority: Literal["none", "low", "medium", "high", "urgent"]
    title: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    suggested_window_cycles: tuple[int, int] | None = None
    disclaimer: str
    provenance: Provenance = Provenance.RULE_BASED


class TwinMetadata(_Model):
    """Provenance of the machinery that produced the snapshot."""

    snapshot_schema_version: str = SNAPSHOT_SCHEMA_VERSION
    model_version: str | None = None
    model_name: str | None = None
    feature_pipeline_fingerprint: str | None = None
    data_version: str | None = None
    git_revision: str | None = None
    end_of_life_definition: dict[str, Any] = Field(default_factory=dict)
    risk_horizon_cycles: int | None = None
    uncertainty_method: str | None = None
    calibration_method: str | None = None
    soh_definition: dict[str, Any] = Field(default_factory=dict)
    input_cycle_range: tuple[int, int] | None = None
    first_scoreable_cycle: int | None = None
    warmup_policy: dict[str, Any] = Field(default_factory=dict)


class BatteryTwinSnapshot(_Model):
    """The complete digital-twin view of one cell at one moment.

    JSON-serialisable by construction (``model_dump(mode="json")``), which is
    what the API returns and what the dashboard renders.
    """

    battery_id: str
    generated_at_utc: str = Field(default_factory=_utc_now)
    identity: BatteryIdentity
    measurement_summary: BatteryMeasurements
    health: BatteryHealthState
    prediction: BatteryPrediction
    failure_risk: BatteryRiskAssessment
    explanation: BatteryExplanation | None = None
    recommendation: BatteryRecommendation
    data_quality: DataQualityAssessment
    metadata: TwinMetadata
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Research prototype for engineering decision support. Not validated for "
        "production deployment, not a substitute for battery-management-system "
        "protection, and not suitable for autonomous safety-critical control."
    )

    @field_validator("battery_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("battery_id must not be blank")
        return value

    def to_json_dict(self) -> dict[str, Any]:
        """Plain, JSON-safe dictionary — the exact API response body."""
        return self.model_dump(mode="json")
