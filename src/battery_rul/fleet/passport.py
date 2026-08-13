"""Battery Passport — an optional demonstration layer.

**This is not regulatory compliance.** The EU Battery Regulation (2023/1542)
defines a digital battery passport with specific mandatory fields, a conformity
process, an assigned unique identifier and a registered data carrier. None of
that exists here. This module demonstrates *what a passport export could look
like* when a digital twin already knows a cell's health and remaining life, and
it says so on every document it produces.

The honest part of a passport is the part this platform can actually source:

| Field group | Source | Trustworthy? |
| --- | --- | --- |
| current SOH, predicted RUL, lifecycle status, data quality, model version | the digital twin | yes, with the twin's own caveats |
| chemistry, manufacturer, manufacturing date, nominal capacity | **supplied by the operator** | only as far as whoever supplied it |
| carbon footprint, recycled content | **not available** | absent, and marked absent |

Every field carries where it came from. A passport that renders a supplied
manufacturer name in the same typeface as a measured state of health invites the
reader to trust both equally, and only one of them was measured here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import FleetBatteryRecord, MaintenancePriority

__all__ = [
    "PASSPORT_SCHEMA_VERSION",
    "BatteryPassport",
    "PassportFieldSource",
    "SuppliedBatteryMetadata",
    "build_passport",
]

PASSPORT_SCHEMA_VERSION = "0.1-demo"

NOT_COMPLIANCE_NOTICE = (
    "DEMONSTRATION ONLY. This document is not a regulatory battery passport, is "
    "not compliant with EU Regulation 2023/1542 or any other regime, carries no "
    "assigned unique identifier and no conformity assessment. It illustrates how "
    "digital-twin outputs could populate such a document."
)


class PassportFieldSource(BaseModel):
    """Where one group of fields came from."""

    model_config = ConfigDict(extra="forbid")

    field_group: str
    source: Literal["measured", "derived", "predicted", "supplied", "unavailable"]
    detail: str


class SuppliedBatteryMetadata(BaseModel):
    """Manufacturing facts this platform cannot measure and does not invent.

    Every field defaults to ``None``. A passport built without them says
    "not supplied" rather than filling in a plausible manufacturer, and no code
    path here generates a value for any of them.
    """

    model_config = ConfigDict(extra="forbid")

    chemistry: str | None = None
    manufacturer: str | None = None
    manufacturing_date: str | None = None
    nominal_capacity_ah: float | None = Field(default=None, gt=0.0)
    nominal_voltage_v: float | None = Field(default=None, gt=0.0)
    mass_kg: float | None = Field(default=None, gt=0.0)
    serial_number: str | None = None
    #: Carbon footprint in kg CO2-equivalent, if an operator has an assessment.
    #: This platform never estimates one.
    carbon_footprint_kg_co2e: float | None = Field(default=None, ge=0.0)
    carbon_footprint_methodology: str | None = None
    recycled_content_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    maintenance_history: list[dict[str, Any]] = Field(default_factory=list)
    provenance_note: str = "Supplied by the operator; not verified by this platform."


class BatteryPassport(BaseModel):
    """A demonstration passport for one cell."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    battery_id: str
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = PASSPORT_SCHEMA_VERSION

    # -- supplied ----------------------------------------------------------
    chemistry: str | None = None
    manufacturer: str | None = None
    manufacturing_date: str | None = None
    nominal_capacity_ah: float | None = None
    serial_number: str | None = None

    # -- measured / derived -------------------------------------------------
    current_soh: float | None = None
    current_soh_percent: float | None = None
    capacity_fade_percent: float | None = None
    latest_cycle: int | None = None
    cycles_recorded: int = 0
    lifecycle_status: Literal[
        "in_service", "in_service_degraded", "end_of_life_approaching", "unknown"
    ] = "unknown"

    # -- predicted ----------------------------------------------------------
    predicted_rul_cycles: float | None = None
    predicted_rul_lower_bound: float | None = None
    predicted_rul_upper_bound: float | None = None
    failure_risk: float | None = None
    failure_risk_is_experimental: bool = False

    # -- platform assessments ------------------------------------------------
    data_quality_status: str = "unknown"
    maintenance_priority: str = MaintenancePriority.INSUFFICIENT_DATA.value
    recommended_action: str | None = None
    recycling_readiness: Literal[
        "not_ready", "assess_within_planning_horizon", "assess_now", "unknown"
    ] = "unknown"

    # -- absent --------------------------------------------------------------
    carbon_footprint_kg_co2e: float | None = None
    carbon_footprint_methodology: str | None = None
    recycled_content_fraction: float | None = None
    maintenance_history: list[dict[str, Any]] = Field(default_factory=list)

    # -- provenance ----------------------------------------------------------
    model_version: str | None = None
    field_sources: list[PassportFieldSource] = Field(default_factory=list)
    compliance_notice: str = NOT_COMPLIANCE_NOTICE
    caveats: list[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _lifecycle_status(record: FleetBatteryRecord, cfg: ExperimentConfig) -> str:
    if not record.is_evaluated or record.measured_soh is None:
        return "unknown"
    if record.priority in (MaintenancePriority.P0_CRITICAL, MaintenancePriority.P1_URGENT):
        return "end_of_life_approaching"
    if record.measured_soh < cfg.soh.healthy_min:
        return "in_service_degraded"
    return "in_service"


def _recycling_readiness(record: FleetBatteryRecord) -> str:
    """A *planning* category, not a disposal instruction.

    Whether a cell is actually fit for second life or recycling depends on a
    physical inspection, the chemistry's recovery route and local regulation —
    none of which this platform knows. This only says how soon that assessment
    is worth booking.
    """
    if not record.is_evaluated:
        return "unknown"
    if record.priority is MaintenancePriority.P0_CRITICAL:
        return "assess_now"
    if record.replacement is not None and record.replacement.replacement_candidate:
        return "assess_within_planning_horizon"
    return "not_ready"


def build_passport(
    record: FleetBatteryRecord,
    cfg: ExperimentConfig,
    supplied: SuppliedBatteryMetadata | None = None,
) -> BatteryPassport:
    """Assemble a demonstration passport from a fleet record.

    Nothing is invented: fields the platform cannot source stay ``None`` and are
    listed as ``unavailable`` in ``field_sources``.
    """
    supplied = supplied or SuppliedBatteryMetadata()

    sources = [
        PassportFieldSource(
            field_group="identity_and_manufacturing",
            source="supplied" if supplied.manufacturer else "unavailable",
            detail=(
                supplied.provenance_note
                if supplied.manufacturer
                else "Not supplied. This platform has no manufacturing record and does "
                "not generate one."
            ),
        ),
        PassportFieldSource(
            field_group="current_health",
            source="derived",
            detail=(
                "State of health computed from measured capacity under the configured "
                f"reference strategy ({cfg.soh.reference_strategy})."
            ),
        ),
        PassportFieldSource(
            field_group="remaining_useful_life",
            source="predicted",
            detail=(
                "Model output with a conformal prediction interval at "
                f"{cfg.uncertainty.coverage:.0%} target coverage. Not a measurement."
            ),
        ),
        PassportFieldSource(
            field_group="failure_risk",
            source="predicted",
            detail=(
                "Calibrated probability of crossing the end-of-life threshold within "
                f"{cfg.risk.horizon_cycles} cycles. A derived label, not an observed "
                "safety event."
            ),
        ),
        PassportFieldSource(
            field_group="carbon_footprint",
            source="supplied" if supplied.carbon_footprint_kg_co2e is not None else "unavailable",
            detail=(
                supplied.carbon_footprint_methodology
                or "No life-cycle assessment is available. This platform never estimates "
                "a carbon footprint."
            ),
        ),
        PassportFieldSource(
            field_group="recycling_readiness",
            source="derived",
            detail=(
                "A planning category derived from the maintenance priority and "
                "replacement assessment. Not a disposal instruction: fitness for second "
                "life or recycling requires physical inspection."
            ),
        ),
    ]

    caveats = [
        NOT_COMPLIANCE_NOTICE,
        "Remaining life and failure risk are model outputs from a research "
        "prototype validated only on a small laboratory cohort.",
    ]
    if record.risk_is_experimental:
        caveats.append(
            "The failure-risk model failed its out-of-fold acceptance gate; its "
            "probability is reported but was excluded from every decision."
        )
    if not record.is_evaluated:
        caveats.append(
            f"This cell was not successfully evaluated (status {record.status.value}); "
            "predicted fields are absent."
        )

    return BatteryPassport(
        battery_id=record.battery_id,
        chemistry=supplied.chemistry,
        manufacturer=supplied.manufacturer,
        manufacturing_date=supplied.manufacturing_date,
        nominal_capacity_ah=supplied.nominal_capacity_ah,
        serial_number=supplied.serial_number,
        current_soh=record.measured_soh,
        current_soh_percent=(
            None if record.measured_soh is None else round(100 * record.measured_soh, 2)
        ),
        capacity_fade_percent=record.capacity_fade_percent,
        latest_cycle=record.latest_cycle,
        cycles_recorded=record.n_cycles,
        lifecycle_status=_lifecycle_status(record, cfg),  # type: ignore[arg-type]
        predicted_rul_cycles=record.predicted_rul,
        predicted_rul_lower_bound=record.rul_lower_bound,
        predicted_rul_upper_bound=record.rul_upper_bound,
        failure_risk=record.failure_risk,
        failure_risk_is_experimental=record.risk_is_experimental,
        data_quality_status=record.data_quality_class,
        maintenance_priority=record.priority.value,
        recommended_action=record.recommended_action,
        recycling_readiness=_recycling_readiness(record),  # type: ignore[arg-type]
        carbon_footprint_kg_co2e=supplied.carbon_footprint_kg_co2e,
        carbon_footprint_methodology=supplied.carbon_footprint_methodology,
        recycled_content_fraction=supplied.recycled_content_fraction,
        maintenance_history=list(supplied.maintenance_history),
        model_version=record.model_version,
        field_sources=sources,
        caveats=caveats,
    )
