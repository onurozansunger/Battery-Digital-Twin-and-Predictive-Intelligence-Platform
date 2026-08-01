"""API request and response schemas.

Kept separate from :mod:`battery_rul.digital_twin.domain` and from every internal
model class. The wire format is a published contract with its own compatibility
rules; binding it to an estimator's attribute names would make an internal
refactor a breaking API change, and binding it to the domain model would mean
request validation rules leak into the domain.

Responses reuse the domain models directly — those *are* the contract — but
requests are their own types, because what a client sends (a list of cycle
records) is not what the service works with (a validated, health-derived,
feature-engineered frame).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from battery_rul.digital_twin.domain import (
    BatteryExplanation,
    BatteryHealthState,
    BatteryPrediction,
    BatteryRiskAssessment,
    BatteryTwinSnapshot,
)

__all__ = [
    "CycleRecord",
    "ErrorResponse",
    "ExplainResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionRequest",
    "ReadyResponse",
    "RiskResponse",
    "RulResponse",
    "SnapshotResponse",
    "SohResponse",
    "VersionResponse",
]

#: Hard cap on request size, independent of configuration, so a malformed or
#: hostile request cannot allocate an unbounded frame before validation runs.
MAX_CYCLES_PER_REQUEST = 20_000


class CycleRecord(BaseModel):
    """One discharge cycle as supplied by a client.

    Only ``cycle_index`` and ``capacity_ah`` are required. Every other sensor is
    optional: a real fleet does not have impedance sweeps on every cell, and
    demanding them would make the API unusable rather than making the data
    better. Absent signals are reported in the snapshot's ``missing_signals`` and
    counted by the data-quality assessment.
    """

    model_config = ConfigDict(extra="forbid")

    cycle_index: int = Field(ge=0, le=1_000_000)
    capacity_ah: float = Field(gt=0.0, le=1000.0)
    timestamp: str | None = None
    ambient_temperature_c: float | None = Field(default=None, ge=-100.0, le=200.0)
    voltage_mean_v: float | None = Field(default=None, ge=0.0, le=1000.0)
    voltage_min_v: float | None = Field(default=None, ge=0.0, le=1000.0)
    voltage_max_v: float | None = Field(default=None, ge=0.0, le=1000.0)
    voltage_std_v: float | None = Field(default=None, ge=0.0)
    voltage_slope_v_per_s: float | None = None
    voltage_knee_v: float | None = Field(default=None, ge=0.0)
    voltage_drop_v: float | None = None
    current_mean_a: float | None = None
    current_std_a: float | None = Field(default=None, ge=0.0)
    temperature_mean_c: float | None = Field(default=None, ge=-100.0, le=500.0)
    temperature_max_c: float | None = Field(default=None, ge=-100.0, le=500.0)
    temperature_rise_c: float | None = None
    discharge_duration_s: float | None = Field(default=None, ge=0.0)
    charge_duration_s: float | None = Field(default=None, ge=0.0)
    charge_cc_duration_s: float | None = Field(default=None, ge=0.0)
    charge_cv_duration_s: float | None = Field(default=None, ge=0.0)
    cc_ct_ratio: float | None = None
    energy_throughput_wh: float | None = None
    internal_resistance_ohm: float | None = Field(default=None, ge=0.0)
    charge_transfer_resistance_ohm: float | None = Field(default=None, ge=0.0)
    time_to_min_voltage_s: float | None = Field(default=None, ge=0.0)
    dvdt_mean_v_per_s: float | None = None
    dvdt_min_v_per_s: float | None = None
    n_samples_discharge: int | None = Field(default=None, ge=0)
    charge_voltage_mean_v: float | None = None
    charge_current_mean_a: float | None = None
    charge_temperature_max_c: float | None = None
    charge_energy_wh: float | None = None
    coulombic_efficiency: float | None = None


class PredictionRequest(BaseModel):
    """A cell's history, from which a twin view is produced."""

    model_config = ConfigDict(extra="forbid")

    battery_id: str = Field(min_length=1, max_length=64)
    history: list[CycleRecord] = Field(min_length=1, max_length=MAX_CYCLES_PER_REQUEST)
    include_explanation: bool = True

    @field_validator("battery_id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("battery_id must not be blank")
        # No path separators: the id is echoed into logs and responses and must
        # never be mistakable for a filesystem path.
        if any(ch in cleaned for ch in ("/", "\\", "\0")):
            raise ValueError("battery_id must not contain path separators")
        return cleaned

    def to_frame(self) -> pd.DataFrame:
        """The history as a canonical-schema DataFrame."""
        return pd.DataFrame([record.model_dump(exclude_none=False) for record in self.history])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    bundles: dict[str, bool]
    errors: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None


class VersionResponse(BaseModel):
    api_version: str
    package_version: str
    snapshot_schema_version: str
    git_revision: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    metadata: dict[str, Any]


class RulResponse(BaseModel):
    battery_id: str
    generated_at_utc: str
    prediction: BatteryPrediction
    data_quality_class: str
    warnings: list[str] = Field(default_factory=list)


class SohResponse(BaseModel):
    battery_id: str
    generated_at_utc: str
    health: BatteryHealthState
    data_quality_class: str
    warnings: list[str] = Field(default_factory=list)


class RiskResponse(BaseModel):
    battery_id: str
    generated_at_utc: str
    failure_risk: BatteryRiskAssessment
    data_quality_class: str
    warnings: list[str] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    battery_id: str
    generated_at_utc: str
    explanation: BatteryExplanation


class SnapshotResponse(BaseModel):
    """The full digital-twin snapshot — the union of everything above."""

    snapshot: BatteryTwinSnapshot


class ErrorResponse(BaseModel):
    """Structured error body. Every non-2xx response has this shape."""

    error: str
    detail: str
    request_id: str | None = None
    hint: str | None = None
