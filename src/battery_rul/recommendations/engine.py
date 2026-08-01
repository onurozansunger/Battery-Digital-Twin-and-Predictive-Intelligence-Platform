"""Deterministic recommendation rules.

Why this is not part of the model
---------------------------------
Three reasons, and all three are practical rather than stylistic.

*Auditability.* An engineer asked to act on "schedule inspection within 10–20
cycles" is entitled to know exactly why. A rule set answers that in one line; a
learned policy answers it with an attribution plot.

*Changeability.* Maintenance thresholds are a business decision and they move —
a fleet operator tightening its inspection policy should edit configuration, not
retrain a network.

*Separation of failure modes.* When the model degrades, the rules keep behaving
predictably; when the policy changes, the model is untouched. Fusing them means
every policy change invalidates the model's evaluation.

The rules read model *outputs* (RUL, its lower bound, calibrated risk, health
class) plus deterministic trend statistics, and emit exactly one primary action.
Thresholds all live in ``recommendations`` configuration.

The conservative bound
----------------------
Rules fire on the **lower bound** of the RUL interval, not the point estimate.
A point estimate of 45 cycles with a lower bound of 12 is not a 45-cycle
situation, and planning against the middle of a wide interval is how prognostics
gets someone stranded.

What this layer is not
----------------------
It is decision *support*. It does not schedule work, does not command a battery
management system, and does not replace a qualified engineer's judgement. Every
recommendation carries that disclaimer in its payload, not merely in the docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import BatteryRecommendation, Provenance
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ActionCode", "RecommendationEngine", "RecommendationInputs"]


class ActionCode(StrEnum):
    """The closed set of actions this layer can suggest."""

    NORMAL_OPERATION = "NORMAL_OPERATION"
    MONITOR_MORE_FREQUENTLY = "MONITOR_MORE_FREQUENTLY"
    REDUCE_HIGH_TEMPERATURE_OPERATION = "REDUCE_HIGH_TEMPERATURE_OPERATION"
    REDUCE_AGGRESSIVE_CHARGING = "REDUCE_AGGRESSIVE_CHARGING"
    SCHEDULE_INSPECTION = "SCHEDULE_INSPECTION"
    PLAN_REPLACEMENT = "PLAN_REPLACEMENT"
    IMMEDIATE_ENGINEERING_REVIEW = "IMMEDIATE_ENGINEERING_REVIEW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


_PRIORITY: dict[ActionCode, str] = {
    ActionCode.NORMAL_OPERATION: "none",
    ActionCode.MONITOR_MORE_FREQUENTLY: "low",
    ActionCode.REDUCE_HIGH_TEMPERATURE_OPERATION: "medium",
    ActionCode.REDUCE_AGGRESSIVE_CHARGING: "medium",
    ActionCode.SCHEDULE_INSPECTION: "medium",
    ActionCode.PLAN_REPLACEMENT: "high",
    ActionCode.IMMEDIATE_ENGINEERING_REVIEW: "urgent",
    ActionCode.INSUFFICIENT_DATA: "low",
}

_TITLE: dict[ActionCode, str] = {
    ActionCode.NORMAL_OPERATION: "Continue normal operation",
    ActionCode.MONITOR_MORE_FREQUENTLY: "Increase monitoring frequency",
    ActionCode.REDUCE_HIGH_TEMPERATURE_OPERATION: "Reduce high-temperature operation",
    ActionCode.REDUCE_AGGRESSIVE_CHARGING: "Reduce aggressive charging",
    ActionCode.SCHEDULE_INSPECTION: "Schedule an inspection",
    ActionCode.PLAN_REPLACEMENT: "Plan replacement",
    ActionCode.IMMEDIATE_ENGINEERING_REVIEW: "Immediate engineering review",
    ActionCode.INSUFFICIENT_DATA: "Insufficient data for a recommendation",
}


@dataclass(slots=True)
class RecommendationInputs:
    """Everything the rules read. Assembled by the service, never by a model."""

    rul_point: float | None = None
    rul_lower_bound: float | None = None
    soh: float | None = None
    health_class: str = "unknown"
    risk_probability: float | None = None
    risk_class: str = "unknown"
    quality_class: str = "GOOD"
    #: Trailing trends, per 10 cycles, computed deterministically from history.
    temperature_trend_c_per_10: float | None = None
    resistance_trend_pct_per_10: float | None = None
    fade_trend_pct_per_10: float | None = None
    is_scoreable: bool = True

    def evidence(self) -> list[str]:
        """Human-readable statements of the facts the rules used."""
        items: list[str] = []
        if self.soh is not None and np.isfinite(self.soh):
            items.append(f"Estimated state of health {100 * self.soh:.1f} % ({self.health_class}).")
        if self.rul_point is not None and np.isfinite(self.rul_point):
            if self.rul_lower_bound is not None and np.isfinite(self.rul_lower_bound):
                items.append(
                    f"Estimated remaining useful life {self.rul_point:.0f} cycles "
                    f"(lower bound of the prediction interval: {self.rul_lower_bound:.0f})."
                )
            else:
                items.append(f"Estimated remaining useful life {self.rul_point:.0f} cycles.")
        if self.risk_probability is not None and np.isfinite(self.risk_probability):
            items.append(
                f"Calibrated probability of reaching end of life within the horizon: "
                f"{100 * self.risk_probability:.0f} % ({self.risk_class})."
            )
        if self.temperature_trend_c_per_10 is not None:
            items.append(
                f"Operating temperature trend {self.temperature_trend_c_per_10:+.2f} °C "
                "per 10 cycles."
            )
        if self.resistance_trend_pct_per_10 is not None:
            items.append(
                f"Internal resistance trend {self.resistance_trend_pct_per_10:+.2f} % "
                "per 10 cycles."
            )
        if self.fade_trend_pct_per_10 is not None:
            items.append(f"Capacity fade trend {self.fade_trend_pct_per_10:+.2f} % per 10 cycles.")
        items.append(f"Input data quality: {self.quality_class}.")
        return items


@dataclass
class RecommendationEngine:
    """Pure, deterministic rules over model outputs and trends.

    Stateless and side-effect free: the same inputs always produce the same
    recommendation, which is what makes the rules testable one at a time.
    """

    cfg: ExperimentConfig
    secondary_actions: list[ActionCode] = field(default_factory=list)

    def recommend(self, inputs: RecommendationInputs) -> BatteryRecommendation:
        """Return the single primary recommendation for this state."""
        rules = self.cfg.recommendations

        # --- hard stop: the input cannot support a maintenance decision -------
        if inputs.quality_class == "INSUFFICIENT" or not inputs.is_scoreable:
            reason = (
                "the supplied history is insufficient"
                if inputs.quality_class == "INSUFFICIENT"
                else "the cell has not reached the first scoreable cycle under the "
                "training warm-up policy"
            )
            return self._build(
                ActionCode.INSUFFICIENT_DATA,
                explanation=(
                    f"No maintenance recommendation is issued because {reason}. "
                    "Supply more cycles of history, or verify the sensor coverage, "
                    "before relying on any prediction for this cell."
                ),
                evidence=inputs.evidence(),
                window=None,
            )

        # --- the conservative planning quantity -------------------------------
        effective_rul = _first_finite(inputs.rul_lower_bound, inputs.rul_point)
        risk = inputs.risk_probability if inputs.risk_probability is not None else float("nan")

        secondary = self._secondary_actions(inputs)
        self.secondary_actions = secondary

        # --- severity ladder, most severe first -------------------------------
        if _at_or_below(effective_rul, rules.urgent_rul_cycles) or _at_or_above(
            risk, rules.urgent_risk
        ):
            return self._build(
                ActionCode.IMMEDIATE_ENGINEERING_REVIEW,
                explanation=(
                    "The model places this cell within a few cycles of the configured "
                    "end-of-life threshold, or assigns it a very high probability of "
                    "crossing it within the horizon. Remove it from planned duty and "
                    "have a qualified engineer review it before further cycling."
                ),
                evidence=inputs.evidence(),
                window=(0, max(rules.urgent_rul_cycles, 1)),
                secondary=secondary,
            )

        if (
            _at_or_below(effective_rul, rules.replacement_rul_cycles)
            or _at_or_above(risk, rules.replacement_risk)
            or inputs.health_class == "critical"
        ):
            return self._build(
                ActionCode.PLAN_REPLACEMENT,
                explanation=(
                    "The estimated remaining life, the calibrated risk, or the measured "
                    "state of health indicates the cell is approaching the configured "
                    "end-of-life threshold. Begin replacement planning; the estimate is "
                    "decision support and should be confirmed against measured capacity."
                ),
                evidence=inputs.evidence(),
                window=(0, rules.replacement_rul_cycles),
                secondary=secondary,
            )

        if (
            _at_or_below(effective_rul, rules.inspection_rul_cycles)
            or _at_or_above(risk, rules.inspection_risk)
            or inputs.health_class == "warning"
        ):
            return self._build(
                ActionCode.SCHEDULE_INSPECTION,
                explanation=(
                    "Degradation has progressed far enough that a physical check is "
                    "warranted before the cell reaches the end-of-life threshold. "
                    "The suggested window is derived from the configured inspection "
                    "policy, not from the model."
                ),
                evidence=inputs.evidence(),
                window=rules.inspection_window_cycles,
                secondary=secondary,
            )

        if secondary:
            primary = secondary[0]
            return self._build(
                primary,
                explanation=_secondary_explanation(primary),
                evidence=inputs.evidence(),
                window=None,
                secondary=secondary[1:],
            )

        if inputs.health_class == "slightly_degraded":
            return self._build(
                ActionCode.MONITOR_MORE_FREQUENTLY,
                explanation=(
                    "The cell is degrading but remains well clear of the end-of-life "
                    "threshold. Increasing the monitoring cadence keeps the estimate "
                    "fresh as the trajectory develops."
                ),
                evidence=inputs.evidence(),
                window=None,
            )

        return self._build(
            ActionCode.NORMAL_OPERATION,
            explanation=(
                "No rule threshold is met: estimated remaining life, calibrated risk "
                "and measured state of health are all within the configured normal "
                "operating bands."
            ),
            evidence=inputs.evidence(),
            window=None,
        )

    # -- trend-driven advisories ------------------------------------------
    def _secondary_actions(self, inputs: RecommendationInputs) -> list[ActionCode]:
        """Advisories that describe *how* the cell is being operated."""
        rules = self.cfg.recommendations
        out: list[ActionCode] = []
        if _at_or_above(inputs.temperature_trend_c_per_10, rules.temperature_trend_c_per_10_cycles):
            out.append(ActionCode.REDUCE_HIGH_TEMPERATURE_OPERATION)
        if _at_or_above(
            inputs.resistance_trend_pct_per_10, rules.resistance_trend_pct_per_10_cycles
        ) or _at_or_above(inputs.fade_trend_pct_per_10, rules.fade_trend_pct_per_10_cycles):
            out.append(ActionCode.REDUCE_AGGRESSIVE_CHARGING)
        return out

    def _build(
        self,
        action: ActionCode,
        *,
        explanation: str,
        evidence: list[str],
        window: tuple[int, int] | None,
        secondary: list[ActionCode] | None = None,
    ) -> BatteryRecommendation:
        extra = list(secondary or [])
        if extra:
            evidence = [
                *evidence,
                "Additional advisories: " + ", ".join(a.value for a in extra) + ".",
            ]
        return BatteryRecommendation(
            action_code=action.value,
            priority=_PRIORITY[action],  # type: ignore[arg-type]
            title=_TITLE[action],
            explanation=explanation,
            evidence=evidence,
            suggested_window_cycles=window,
            disclaimer=self.cfg.recommendations.disclaimer,
            provenance=Provenance.RULE_BASED,
        )


def _secondary_explanation(action: ActionCode) -> str:
    if action is ActionCode.REDUCE_HIGH_TEMPERATURE_OPERATION:
        return (
            "Recent operating temperature is trending upward faster than the "
            "configured limit. Elevated temperature accelerates capacity fade in "
            "lithium-ion cells; reducing high-temperature duty where operationally "
            "possible is a low-cost mitigation."
        )
    return (
        "Internal resistance or capacity fade is trending faster than the configured "
        "limit. Less aggressive charging is a low-cost mitigation while the trend is "
        "investigated."
    )


def _first_finite(*values: float | None) -> float | None:
    for value in values:
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _at_or_below(value: float | None, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and value <= threshold


def _at_or_above(value: float | None, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and value >= threshold
