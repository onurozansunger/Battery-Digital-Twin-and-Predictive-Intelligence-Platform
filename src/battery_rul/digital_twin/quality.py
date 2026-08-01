"""Input data-quality assessment.

The model will return a number for any input whose columns type-check. Whether
that number means anything is a separate question, and this module is where it
gets asked — before inference, not after.

Scoring
-------
A weighted score in [0, 1] over independent checks, each contributing a penalty:

======================  ============================================
history length          fewer cycles than the warm-up + window
                        requirement makes prediction impossible;
                        below ``min_cycles_good`` makes it thin
missing features        fraction of required feature columns absent
                        or all-NaN in the supplied history
cycle ordering          duplicates, non-monotonic indices
cycle gaps              a jump larger than ``max_cycle_gap`` means
                        the ageing clock skipped, and trailing
                        windows span a discontinuity
value plausibility      capacity, voltage, temperature outside
                        physically sensible ranges
distribution            features far outside the training reference
                        range, by the training scaler's own units
======================  ============================================

``INSUFFICIENT`` is a hard stop: the service returns a limited snapshot with the
``INSUFFICIENT_DATA`` recommendation and no prediction. That is the whole point
of the class — a maintenance action derived from four cycles of a cell is worse
than no answer, because it looks like an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import DataQualityAssessment, Provenance
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["QualityCheck", "assess_data_quality"]


@dataclass(slots=True)
class QualityCheck:
    """One check's outcome."""

    name: str
    passed: bool
    penalty: float
    detail: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "penalty": round(self.penalty, 4),
            "detail": self.detail,
            "value": self.value,
        }


def _classify(score: float, insufficient: bool, cfg: ExperimentConfig) -> str:
    if insufficient:
        return "INSUFFICIENT"
    q = cfg.quality
    if score >= q.good_min_score:
        return "GOOD"
    if score >= q.acceptable_min_score:
        return "ACCEPTABLE"
    if score >= q.poor_min_score:
        return "POOR"
    return "INSUFFICIENT"


def assess_data_quality(
    history: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    required_features: list[str] | None = None,
    feature_frame: pd.DataFrame | None = None,
    reference_ranges: dict[str, tuple[float, float]] | None = None,
    min_history_cycles: int | None = None,
) -> DataQualityAssessment:
    """Assess whether ``history`` can support a prediction.

    Parameters
    ----------
    history:
        Raw canonical cycle rows for one cell, as supplied by the caller.
    required_features:
        Feature columns the fitted pipeline needs.
    feature_frame:
        The engineered frame, when already computed — lets the missing-feature
        check look at what the model will actually receive rather than guessing
        from the raw columns.
    reference_ranges:
        Per-feature ``(low, high)`` from the training partition, for the
        out-of-distribution flag. Absent ranges simply skip that check rather
        than inventing one.
    min_history_cycles:
        Warm-up + window requirement from :mod:`battery_rul.features.warmup`.
    """
    q = cfg.quality
    checks: list[QualityCheck] = []
    warnings: list[str] = []
    missing_features: list[str] = []
    ood_flags: list[str] = []
    insufficient = False

    n_cycles = int(len(history))
    minimum = int(min_history_cycles or q.min_cycles_for_prediction)

    # --- history length ---------------------------------------------------
    if n_cycles < minimum:
        insufficient = True
        checks.append(
            QualityCheck(
                "history_length",
                False,
                1.0,
                f"{n_cycles} cycles supplied; at least {minimum} are needed before "
                "the first scoreable cycle under the training warm-up policy.",
                n_cycles,
            )
        )
        warnings.append(
            f"Insufficient history: {n_cycles} cycles, {minimum} required. No "
            "prediction is produced."
        )
    elif n_cycles < q.min_cycles_good:
        checks.append(
            QualityCheck(
                "history_length",
                True,
                0.15,
                f"{n_cycles} cycles is enough to score but below the "
                f"{q.min_cycles_good}-cycle comfort threshold; trailing windows "
                "are short and early-life predictions are the least reliable.",
                n_cycles,
            )
        )
        warnings.append(f"Short history ({n_cycles} cycles): treat the estimate as indicative.")
    else:
        checks.append(QualityCheck("history_length", True, 0.0, f"{n_cycles} cycles.", n_cycles))

    # --- duplicate / ordering ---------------------------------------------
    if "cycle_index" in history.columns:
        cycles = history["cycle_index"].to_numpy()
        n_duplicates = int(len(cycles) - len(np.unique(cycles)))
        monotonic = bool(np.all(np.diff(cycles) > 0)) if len(cycles) > 1 else True
        if n_duplicates:
            checks.append(
                QualityCheck(
                    "duplicate_cycles",
                    False,
                    0.25,
                    f"{n_duplicates} duplicate cycle_index value(s).",
                    n_duplicates,
                )
            )
            warnings.append(f"{n_duplicates} duplicate cycle(s) in the supplied history.")
        else:
            checks.append(QualityCheck("duplicate_cycles", True, 0.0, "No duplicates.", 0))

        if not monotonic:
            checks.append(
                QualityCheck(
                    "cycle_ordering",
                    False,
                    0.20,
                    "cycle_index is not strictly increasing; rows were reordered "
                    "before feature generation.",
                )
            )
            warnings.append("Cycle indices were not in order and have been sorted.")
        else:
            checks.append(QualityCheck("cycle_ordering", True, 0.0, "Strictly increasing."))

        gaps = np.diff(np.sort(np.unique(cycles))) if len(cycles) > 1 else np.array([1])
        largest = int(gaps.max()) if gaps.size else 1
        if largest > q.max_cycle_gap:
            checks.append(
                QualityCheck(
                    "cycle_gaps",
                    False,
                    0.15,
                    f"Largest gap is {largest} cycles (limit {q.max_cycle_gap}); "
                    "trailing windows span a discontinuity in the ageing clock.",
                    largest,
                )
            )
            warnings.append(f"A {largest}-cycle gap is present in the history.")
        else:
            checks.append(
                QualityCheck("cycle_gaps", True, 0.0, f"Largest gap {largest} cycles.", largest)
            )

    # --- missing features --------------------------------------------------
    missing_fraction = 0.0
    if required_features:
        source = feature_frame if feature_frame is not None else history
        for name in required_features:
            if name not in source.columns or source[name].isna().all():
                missing_features.append(name)
        missing_fraction = len(missing_features) / max(len(required_features), 1)

        if missing_fraction > q.max_missing_fraction_acceptable:
            insufficient = True
            penalty = 1.0
            detail = (
                f"{len(missing_features)}/{len(required_features)} required features "
                f"({missing_fraction:.0%}) are absent — above the "
                f"{q.max_missing_fraction_acceptable:.0%} limit."
            )
            warnings.append(detail)
        elif missing_fraction > q.max_missing_fraction_good:
            penalty = 0.25
            detail = f"{missing_fraction:.0%} of required features are absent."
            warnings.append(
                f"{detail} Their values are filled from the training-time fallback, "
                "which weakens the prediction."
            )
        else:
            penalty = 0.0
            detail = f"{missing_fraction:.0%} of required features are absent."
        checks.append(
            QualityCheck(
                "missing_features",
                missing_fraction <= q.max_missing_fraction_good,
                penalty,
                detail,
                round(missing_fraction, 4),
            )
        )

    # --- physical plausibility ---------------------------------------------
    implausible: list[str] = []
    if "capacity_ah" in history.columns:
        capacity = pd.to_numeric(history["capacity_ah"], errors="coerce")
        nominal = cfg.data.nominal_capacity_ah
        if (capacity <= 0).any() or (capacity > 2.0 * nominal).any():
            implausible.append("capacity_ah")
    for column, (low, high) in (
        ("voltage_mean_v", cfg.validation.voltage_bounds_v),
        ("temperature_max_c", cfg.validation.temperature_bounds_c),
    ):
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce")
            if ((values < low) | (values > high)).any():
                implausible.append(column)
    if implausible:
        checks.append(
            QualityCheck(
                "value_plausibility",
                False,
                0.20,
                f"Physically implausible values in: {implausible}",
                implausible,
            )
        )
        warnings.append(f"Out-of-range sensor readings in {implausible}.")
    else:
        checks.append(QualityCheck("value_plausibility", True, 0.0, "All values in range."))

    # --- distribution shift -------------------------------------------------
    if reference_ranges and feature_frame is not None:
        for name, (low, high) in reference_ranges.items():
            if name not in feature_frame.columns:
                continue
            values = pd.to_numeric(feature_frame[name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size and (values.min() < low or values.max() > high):
                ood_flags.append(name)
        if ood_flags:
            checks.append(
                QualityCheck(
                    "distribution",
                    False,
                    min(0.05 * len(ood_flags), 0.25),
                    f"{len(ood_flags)} feature(s) fall outside the training reference "
                    "range; the model is extrapolating.",
                    ood_flags[:20],
                )
            )
            warnings.append(
                f"{len(ood_flags)} feature(s) are outside the training range — the "
                "prediction is an extrapolation."
            )
        else:
            checks.append(
                QualityCheck("distribution", True, 0.0, "Within the training reference range.")
            )

    score = float(np.clip(1.0 - sum(c.penalty for c in checks), 0.0, 1.0))
    quality_class = _classify(score, insufficient, cfg)
    if quality_class == "INSUFFICIENT":
        insufficient = True

    logger.info(
        "Data quality: %s (score %.2f) over %d cycles, %d warning(s)",
        quality_class,
        score,
        n_cycles,
        len(warnings),
    )
    return DataQualityAssessment(
        quality_score=round(score, 4),
        quality_class=quality_class,  # type: ignore[arg-type]
        n_cycles=n_cycles,
        missing_feature_fraction=round(missing_fraction, 4),
        warnings=warnings,
        missing_features=missing_features[:50],
        out_of_distribution_flags=ood_flags[:50],
        checks={c.name: c.to_dict() for c in checks},
        provenance=Provenance.DERIVED,
    )
