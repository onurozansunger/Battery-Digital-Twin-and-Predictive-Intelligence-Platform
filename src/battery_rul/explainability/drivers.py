"""Per-prediction degradation drivers for the digital twin.

Milestone 1's explainability module produces global, dataset-level attributions
for a report. The twin needs something different: for *this* cell at *this*
cycle, which inputs pushed the model's estimate, in which direction, and by how
much — packaged so an API can return it and a dashboard can render it.

Method selection
----------------
``shap_tree``
    SHAP's exact TreeExplainer where the model is a tree ensemble. Local, signed,
    additive, and fast enough to run per request.

``feature_ablation``
    Model-agnostic fallback, used for linear, neural and multi-task models. Each
    feature is replaced in turn by its training-reference value and the change in
    output is recorded. It is a genuine local sensitivity measure, and unlike
    attention weights it is defined in terms of the quantity anyone actually
    cares about — the prediction.

Attention is deliberately not used as an explanation. The multi-task encoder
exposes its pooling weights for diagnostics, and they are informative about where
the encoder looked, but "looked at" is not "was influenced by", and a
high-attention timestep can have no effect on the output at all. Presenting them
as explanations would be an unsupported claim dressed as transparency.

Language
--------
Driver text never asserts causation about the cell. "Recent operating temperature
is above the training reference range and contributed to the model's elevated
risk estimate" is supportable. "High temperature caused this battery to fail" is
not, and no phrasing in this module produces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.digital_twin.domain import (
    BatteryExplanation,
    DegradationDriver,
    Provenance,
)
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DriverExplainer", "display_name_for", "explain_row"]


#: Human-readable names for the signal families the engineered columns come from.
_SIGNAL_LABELS: dict[str, str] = {
    "capacity_ah": "measured discharge capacity",
    "capacity_smooth_ah": "smoothed discharge capacity",
    "soh": "state of health",
    "discharge_duration_s": "discharge duration",
    "charge_duration_s": "charge duration",
    "voltage_mean_v": "mean discharge voltage",
    "voltage_min_v": "minimum discharge voltage",
    "voltage_std_v": "discharge-voltage dispersion",
    "voltage_slope_v_per_s": "discharge-curve slope",
    "current_mean_a": "mean load current",
    "temperature_mean_c": "mean cell temperature",
    "temperature_max_c": "peak cell temperature",
    "internal_resistance_ohm": "internal resistance",
    "energy_throughput_wh": "energy throughput",
    "cc_ct_ratio": "constant-current charge fraction",
    "cycle_index": "cycle count",
}

#: Suffix patterns describing the transformation applied to a base signal.
_TRANSFORM_LABELS: list[tuple[str, str]] = [
    ("_rmean_", "{n}-cycle rolling mean of {signal}"),
    ("_rstd_", "{n}-cycle rolling variability of {signal}"),
    ("_rmin_", "{n}-cycle rolling minimum of {signal}"),
    ("_rmax_", "{n}-cycle rolling maximum of {signal}"),
    ("_rrange_", "{n}-cycle range of {signal}"),
    ("_rdev_", "deviation of {signal} from its {n}-cycle mean"),
    ("_ewm_", "exponentially weighted {signal} (half-life {n})"),
    ("_lag_", "{signal} {n} cycles ago"),
    ("_diff_", "change in {signal} over {n} cycles"),
    ("_pct_", "percentage change in {signal} over {n} cycles"),
    ("_slope_", "{n}-cycle trend in {signal}"),
]


def display_name_for(feature_name: str) -> str:
    """A readable label for an engineered column.

    Returns the raw name when the pattern is unrecognised: an honest identifier
    beats a confidently wrong paraphrase.
    """
    if feature_name in _SIGNAL_LABELS:
        return _SIGNAL_LABELS[feature_name].capitalize()

    for suffix, template in _TRANSFORM_LABELS:
        if suffix in feature_name:
            base, _, tail = feature_name.partition(suffix)
            signal = _SIGNAL_LABELS.get(base, base.replace("_", " "))
            if tail.isdigit():
                return template.format(signal=signal, n=tail).capitalize()

    for tail, template in (
        ("_ratio_to_initial", "{signal} relative to its beginning-of-life value"),
        ("_delta_from_initial", "change in {signal} since beginning of life"),
        ("_cummean", "running mean of {signal}"),
        ("_cummin", "running minimum of {signal}"),
        ("_cummax", "running maximum of {signal}"),
        ("_cumstd", "running variability of {signal}"),
        ("_is_missing", "{signal} reading unavailable"),
    ):
        if feature_name.endswith(tail):
            base = feature_name[: -len(tail)]
            signal = _SIGNAL_LABELS.get(base, base.replace("_", " "))
            return template.format(signal=signal).capitalize()

    return feature_name.replace("_", " ").capitalize()


def _direction(contribution: float, task: str) -> str:
    """Signed contribution to a direction label appropriate to the task."""
    if abs(contribution) < 1e-12:
        return "neutral"
    if task == "risk":
        return "increases_risk" if contribution > 0 else "decreases_risk"
    # For RUL, a positive contribution raises predicted remaining life.
    return "increases_rul" if contribution > 0 else "decreases_rul"


def _explanation_text(
    display: str,
    contribution: float,
    value: float | None,
    reference: float | None,
    task: str,
) -> str:
    direction = "raised" if contribution > 0 else "lowered"
    quantity = {
        "risk": "the model's estimated failure risk",
        "rul": "the model's remaining-life estimate",
        "soh": "the model's state-of-health estimate",
    }.get(task, "the model's output")

    comparison = ""
    if (
        value is not None
        and reference is not None
        and np.isfinite(value)
        and np.isfinite(reference)
    ):
        if value > reference:
            comparison = " Its current value is above the training reference level."
        elif value < reference:
            comparison = " Its current value is below the training reference level."
    return f"{display} {direction} {quantity} for this cycle.{comparison}"


@dataclass
class DriverExplainer:
    """Produces local attributions for one row of one cell.

    ``reference_values`` are the training-partition medians the fitted pipeline
    already persists, so the ablation baseline is the training-time typical cell
    rather than zero — ablating to zero in scaled space is a different and much
    less interpretable counterfactual.
    """

    feature_names: list[str]
    reference_values: dict[str, float]
    top_k: int = 5

    def explain(
        self,
        *,
        predict_fn: Any,
        scaled_row: np.ndarray,
        raw_values: dict[str, float] | None = None,
        task: str = "rul",
        model: Any = None,
    ) -> BatteryExplanation:
        """Attribute one prediction, preferring exact SHAP where it applies."""
        if model is not None and getattr(model, "is_tree", False):
            explanation = self._shap_tree(model, scaled_row, raw_values or {}, task)
            if explanation is not None:
                return explanation
        return self._ablation(predict_fn, scaled_row, raw_values or {}, task)

    # -- SHAP ---------------------------------------------------------------
    def _shap_tree(
        self, model: Any, scaled_row: np.ndarray, raw_values: dict[str, float], task: str
    ) -> BatteryExplanation | None:
        try:
            import shap

            estimator = getattr(model, "estimator", model)
            explainer = shap.TreeExplainer(estimator)
            values = np.asarray(explainer.shap_values(scaled_row.reshape(1, -1)))
            contributions = values.reshape(-1)
            baseline = float(np.ravel(explainer.expected_value)[0])
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail a request
            logger.info("TreeExplainer unavailable (%s); falling back to ablation", exc)
            return None

        return self._package(contributions, raw_values, task, method="shap_tree", baseline=baseline)

    # -- ablation ------------------------------------------------------------
    def _ablation(
        self, predict_fn: Any, scaled_row: np.ndarray, raw_values: dict[str, float], task: str
    ) -> BatteryExplanation:
        base = float(predict_fn(scaled_row.reshape(1, -1))[0])
        contributions = np.zeros(len(self.feature_names), dtype=float)

        # One batched forward pass over all single-feature ablations: n+1 rows
        # rather than n+1 separate calls, which matters for a per-request path.
        matrix = np.repeat(scaled_row.reshape(1, -1), len(self.feature_names), axis=0)
        for i in range(len(self.feature_names)):
            matrix[i, i] = 0.0  # scaled space: 0 is the training-median cell
        try:
            ablated = np.asarray(predict_fn(matrix), dtype=float).reshape(-1)
            contributions = base - ablated
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature ablation failed (%s); returning no drivers", exc)
            return BatteryExplanation(method="feature_ablation_failed", drivers=[])

        return self._package(
            contributions, raw_values, task, method="feature_ablation", baseline=base
        )

    # -- packaging -----------------------------------------------------------
    def _package(
        self,
        contributions: np.ndarray,
        raw_values: dict[str, float],
        task: str,
        *,
        method: str,
        baseline: float | None,
    ) -> BatteryExplanation:
        contributions = np.nan_to_num(np.asarray(contributions, dtype=float))
        n = min(len(self.feature_names), contributions.size)
        order = np.argsort(np.abs(contributions[:n]))[::-1][: self.top_k]

        drivers: list[DegradationDriver] = []
        for i in order:
            name = self.feature_names[int(i)]
            contribution = float(contributions[int(i)])
            value = raw_values.get(name)
            reference = self.reference_values.get(name)
            drivers.append(
                DegradationDriver(
                    feature_name=name,
                    display_name=display_name_for(name),
                    contribution_direction=_direction(contribution, task),  # type: ignore[arg-type]
                    contribution_magnitude=round(abs(contribution), 6),
                    current_value=None if value is None else round(float(value), 6),
                    reference_value=None if reference is None else round(float(reference), 6),
                    explanation_text=_explanation_text(
                        display_name_for(name), contribution, value, reference, task
                    ),
                    provenance=Provenance.DERIVED,
                )
            )
        return BatteryExplanation(
            method=method,
            drivers=drivers,
            baseline_value=None if baseline is None else round(float(baseline), 6),
        )


def explain_row(
    frame: pd.DataFrame,
    row_index: int,
    *,
    explainer: DriverExplainer,
    predict_fn: Any,
    scaled: np.ndarray,
    task: str = "rul",
    model: Any = None,
) -> BatteryExplanation:
    """Convenience wrapper: attribute the prediction at ``row_index``."""
    raw_values = {
        name: float(frame.iloc[row_index][name])
        for name in explainer.feature_names
        if name in frame.columns and np.isfinite(frame.iloc[row_index][name])
    }
    return explainer.explain(
        predict_fn=predict_fn,
        scaled_row=scaled[row_index],
        raw_values=raw_values,
        task=task,
        model=model,
    )
