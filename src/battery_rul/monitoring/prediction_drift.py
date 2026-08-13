"""Prediction drift: has the model's *output* distribution moved?

Monitored quantities: predicted RUL, predicted SOH forecast, calibrated failure
risk, prediction-interval width, and the health / risk / maintenance-priority
class frequencies.

The interpretation is stated on every report and it is the important part:

    **Prediction drift does not prove model degradation.** It says the model is
    saying different things — which happens when the population changes (a fleet
    that has simply aged), when the input pipeline changes, or when the model
    changes. Distinguishing those needs labels, and labels are what
    :mod:`battery_rul.monitoring.performance` waits for.

A fleet whose cells are all a hundred cycles older than last month *should* show
prediction drift. Treating that as an incident trains an operations team to
ignore the alert that matters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from battery_rul.config import ExperimentConfig, PredictionDriftConfig
from battery_rul.fleet.domain import FleetBatteryRecord, MonitoringStatus
from battery_rul.monitoring.domain import PredictionDriftReport, PredictionDriftResult
from battery_rul.monitoring.drift import population_stability_index
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "PredictionSummary",
    "class_frequency_distance",
    "detect_prediction_drift",
    "summarise_predictions",
]


class PredictionSummary(dict):
    """A plain dict of output-distribution statistics.

    A ``dict`` subclass rather than a model: this is written into a reference
    artifact and compared field by field, and every consumer wants mapping
    semantics.
    """


def summarise_predictions(
    records: Sequence[FleetBatteryRecord], cfg: ExperimentConfig
) -> PredictionSummary:
    """Summarise a batch's outputs into a comparable shape."""
    evaluated = [r for r in records if r.is_evaluated]
    summary = PredictionSummary(
        n=len(evaluated),
        generated_from="fleet_snapshot",
    )
    quantiles = cfg.monitoring.prediction_drift.quantiles

    for name, attribute in (
        ("predicted_rul", "predicted_rul"),
        ("predicted_soh_forecast", "predicted_soh_forecast"),
        ("failure_risk", "failure_risk"),
        ("interval_width", "interval_width"),
    ):
        values = np.asarray(
            [
                float(getattr(r, attribute))
                for r in evaluated
                if getattr(r, attribute) is not None and np.isfinite(getattr(r, attribute))
            ],
            dtype=float,
        )
        if values.size == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "quantiles": {
                f"q{int(round(100 * q))}": float(np.quantile(values, q)) for q in quantiles
            },
            "min": float(values.min()),
            "max": float(values.max()),
            "histogram": _histogram(values, cfg.monitoring.prediction_drift.n_bins),
        }

    for name, attribute in (
        ("health_class", "health_class"),
        ("risk_class", "risk_class"),
        ("priority", "priority"),
    ):
        counts: dict[str, int] = {}
        for record in evaluated:
            value = getattr(record, attribute)
            label = value.value if hasattr(value, "value") else str(value)
            counts[label] = counts.get(label, 0) + 1
        total = max(sum(counts.values()), 1)
        summary[f"{name}_frequencies"] = {k: v / total for k, v in sorted(counts.items())}
    return summary


def _histogram(values: np.ndarray, n_bins: int) -> dict[str, list[float]]:
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        return {"edges": [low, high], "frequencies": [1.0]}
    counts, edges = np.histogram(values, bins=n_bins, range=(low, high))
    total = max(int(counts.sum()), 1)
    return {
        "edges": [float(e) for e in edges],
        "frequencies": [float(c / total) for c in counts],
    }


def class_frequency_distance(reference: dict[str, float], current: dict[str, float]) -> float:
    """Total-variation distance between two class-frequency vectors, in [0, 1].

    Chosen over a chi-square test because the reading is direct: 0.30 means
    30 % of the fleet's mass moved between classes, which is a sentence an
    operations team can act on.
    """
    labels = sorted(set(reference) | set(current))
    return 0.5 * float(
        sum(
            abs(float(current.get(label, 0.0)) - float(reference.get(label, 0.0)))
            for label in labels
        )
    )


def _severity(value: float | None, thresholds: tuple[float, float]) -> MonitoringStatus:
    if value is None or not np.isfinite(value):
        return MonitoringStatus.UNKNOWN
    warning, critical = thresholds
    if value >= critical:
        return MonitoringStatus.CRITICAL
    if value >= warning:
        return MonitoringStatus.WARNING
    return MonitoringStatus.OK


def detect_prediction_drift(
    current: PredictionSummary | dict,
    reference: PredictionSummary | dict,
    cfg: ExperimentConfig,
    *,
    model_version: str | None = None,
    reference_id: str | None = None,
) -> PredictionDriftReport:
    """Compare two prediction summaries."""
    policy: PredictionDriftConfig = cfg.monitoring.prediction_drift
    results: list[PredictionDriftResult] = []
    warnings: list[str] = []
    sample_size = int(current.get("n", 0) or 0)

    if sample_size < policy.min_sample_size:
        warnings.append(
            f"{sample_size} scored cell(s) is below the configured minimum of "
            f"{policy.min_sample_size}; prediction drift is not assessed."
        )
        return PredictionDriftReport(
            reference_id=reference_id,
            model_version=model_version,
            status=MonitoringStatus.UNKNOWN,
            sample_size=sample_size,
            warnings=warnings,
        )

    for name in ("predicted_rul", "predicted_soh_forecast", "failure_risk", "interval_width"):
        current_stats = current.get(name) or {}
        reference_stats = reference.get(name) or {}
        if not current_stats.get("count") or not reference_stats.get("count"):
            continue

        # --- standardised mean shift --------------------------------------
        scale = float(reference_stats.get("std") or 0.0)
        if scale <= 0:
            span = float(reference_stats.get("max", 0.0)) - float(reference_stats.get("min", 0.0))
            scale = span if span > 0 else 1.0
        shift = abs(float(current_stats["mean"]) - float(reference_stats["mean"])) / scale
        severity = _severity(shift, policy.standardised_mean_shift_thresholds)
        results.append(
            PredictionDriftResult(
                output_name=name,
                metric="standardised_mean_shift",
                reference_value=round(float(reference_stats["mean"]), 6),
                current_value=round(float(current_stats["mean"]), 6),
                drift_value=round(shift, 6),
                threshold=policy.standardised_mean_shift_thresholds[0],
                drift_detected=severity in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                severity=severity,
                reference_sample_size=int(reference_stats["count"]),
                sample_size=int(current_stats["count"]),
                detail={"scale_used": round(scale, 6)},
            )
        )

        # --- quantile shift ------------------------------------------------
        for level, reference_value in (reference_stats.get("quantiles") or {}).items():
            current_value = (current_stats.get("quantiles") or {}).get(level)
            if current_value is None:
                continue
            delta = abs(float(current_value) - float(reference_value)) / scale
            quantile_severity = _severity(delta, policy.standardised_mean_shift_thresholds)
            results.append(
                PredictionDriftResult(
                    output_name=f"{name}[{level}]",
                    metric="standardised_quantile_shift",
                    reference_value=round(float(reference_value), 6),
                    current_value=round(float(current_value), 6),
                    drift_value=round(delta, 6),
                    threshold=policy.standardised_mean_shift_thresholds[0],
                    drift_detected=quantile_severity
                    in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                    severity=quantile_severity,
                    reference_sample_size=int(reference_stats["count"]),
                    sample_size=int(current_stats["count"]),
                )
            )

        # --- distribution shift (PSI over the reference's own bins) ---------
        reference_histogram = reference_stats.get("histogram")
        current_histogram = current_stats.get("histogram")
        if reference_histogram and current_histogram:
            rebinned = _rebin(current_stats, reference_histogram)
            if rebinned is not None:
                psi = population_stability_index(reference_histogram["frequencies"], rebinned)
                psi_severity = _severity(psi, policy.psi_thresholds)
                results.append(
                    PredictionDriftResult(
                        output_name=name,
                        metric="psi",
                        drift_value=round(psi, 6),
                        threshold=policy.psi_thresholds[0],
                        drift_detected=psi_severity
                        in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                        severity=psi_severity,
                        reference_sample_size=int(reference_stats["count"]),
                        sample_size=int(current_stats["count"]),
                    )
                )

    # --- interval-width inflation -------------------------------------------
    reference_width = (reference.get("interval_width") or {}).get("median")
    current_width = (current.get("interval_width") or {}).get("median")
    if reference_width and current_width and reference_width > 0:
        ratio = float(current_width) / float(reference_width)
        severity = _severity(ratio, policy.interval_width_ratio_thresholds)
        results.append(
            PredictionDriftResult(
                output_name="interval_width",
                metric="width_ratio",
                reference_value=round(float(reference_width), 4),
                current_value=round(float(current_width), 4),
                drift_value=round(ratio, 4),
                threshold=policy.interval_width_ratio_thresholds[0],
                drift_detected=severity in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                severity=severity,
                detail={
                    "note": "A wider interval means the model is less certain about "
                    "this population, not that it is wrong about it."
                },
            )
        )

    # --- class-frequency shifts ---------------------------------------------
    for name in ("health_class_frequencies", "risk_class_frequencies", "priority_frequencies"):
        reference_frequencies = reference.get(name) or {}
        current_frequencies = current.get(name) or {}
        if not reference_frequencies or not current_frequencies:
            continue
        distance = class_frequency_distance(reference_frequencies, current_frequencies)
        severity = _severity(distance, policy.class_frequency_thresholds)
        results.append(
            PredictionDriftResult(
                output_name=name,
                metric="total_variation_distance",
                drift_value=round(distance, 6),
                threshold=policy.class_frequency_thresholds[0],
                drift_detected=severity in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                severity=severity,
                sample_size=sample_size,
                detail={"reference": reference_frequencies, "current": current_frequencies},
            )
        )

    drifted = [r for r in results if r.drift_detected]
    status = (
        MonitoringStatus.worst([r.severity for r in results])
        if results
        else MonitoringStatus.UNKNOWN
    )
    if not results:
        warnings.append(
            "No comparable output quantity was found in both the reference and the "
            "current batch; prediction drift could not be assessed."
        )

    return PredictionDriftReport(
        reference_id=reference_id,
        model_version=model_version,
        status=status,
        results=results,
        n_drifted=len(drifted),
        sample_size=sample_size,
        warnings=warnings,
    )


def _rebin(current_stats: dict, reference_histogram: dict) -> list[float] | None:
    """Re-express the current histogram on the reference's bin edges.

    Both histograms are stored as frequencies over their own edges, so they
    cannot be compared directly. The current distribution is approximated from
    its own edges and mapped onto the reference grid; when the supports do not
    overlap at all, ``None`` is returned rather than a fabricated comparison.
    """
    current_histogram = current_stats.get("histogram") or {}
    edges = current_histogram.get("edges") or []
    frequencies = current_histogram.get("frequencies") or []
    reference_edges = reference_histogram.get("edges") or []
    if len(edges) < 2 or len(reference_edges) < 2 or not frequencies:
        return None

    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(frequencies))]
    out = [0.0] * (len(reference_edges) - 1)
    for centre, frequency in zip(centres, frequencies, strict=True):
        index = int(np.searchsorted(reference_edges, centre, side="right")) - 1
        index = max(0, min(index, len(out) - 1))
        out[index] += float(frequency)
    total = sum(out)
    if total <= 0:
        return None
    return [value / total for value in out]
