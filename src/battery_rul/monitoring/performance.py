"""Model-performance monitoring with delayed labels.

The defining property of prognostics in production: the label arrives long after
the prediction. A prediction of "38 cycles remaining" made at cycle 90 can only
be scored once the cell has actually reached end of life, which may be months
later. Everything here is built around that delay.

The join
--------
A prediction is recorded when it is made (``prediction_id``, ``battery_id``,
``cycle_index``, ``model_version``). An outcome is recorded when it becomes
observable. They are joined on ``(battery_id, cycle_index)`` and **metrics are
attributed to the model version that made the prediction**, not to whatever is
in production when the label lands. Skipping that is how a new model inherits
its predecessor's errors.

What this is not
----------------
These metrics are not comparable with the Milestone 1/2 held-out test numbers.
Those describe a fixed partition of a laboratory dataset under a chosen split;
these describe whatever cells happened to be scored in production, at whatever
life stages they happened to be in. Both are reported, and they are reported
separately, with that sentence attached.

No metric here triggers a retrain. A threshold crossing produces an alert for a
human, because the cause might equally be a broken sensor, a fleet that changed
duty cycle, or twenty labels arriving from one unusual cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from battery_rul.calibration.probability import risk_metrics
from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.metrics import compute_metrics, soh_metrics
from battery_rul.monitoring.domain import PerformanceReport, PerformanceStatus
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "OutcomeLabel",
    "PredictionRecord",
    "evaluate_delayed_labels",
    "join_predictions_and_labels",
    "prediction_records_from_snapshot",
]

#: Life-stage bands for error breakdown, in observed remaining cycles. Prognostic
#: error is strongly heteroscedastic — wide early, tight near end of life — so a
#: single MAE hides the regime that matters most.
_LIFE_STAGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-50", 20.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100+", 100.0, float("inf")),
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


class PredictionRecord(_Model):
    """One prediction, recorded when it was made."""

    prediction_id: str
    battery_id: str
    cycle_index: int = Field(ge=0)
    model_version: str | None = None
    model_name: str | None = None
    fleet_id: str | None = None
    batch_id: str | None = None
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    predicted_rul: float | None = None
    rul_lower_bound: float | None = None
    rul_upper_bound: float | None = None
    interval_coverage_target: float | None = None
    predicted_soh_forecast: float | None = None
    soh_forecast_horizon_cycles: int | None = None
    failure_risk: float | None = None
    risk_horizon_cycles: int | None = None
    measured_soh: float | None = None


class OutcomeLabel(_Model):
    """One observed outcome, recorded when it became observable."""

    battery_id: str
    cycle_index: int = Field(ge=0)
    observed_at_utc: str | None = None
    #: Cycle at which the outcome became known; the difference from
    #: ``cycle_index`` is the evaluation delay.
    observed_at_cycle: int | None = Field(default=None, ge=0)
    observed_rul: float | None = Field(default=None, ge=0.0)
    observed_soh: float | None = None
    observed_eol_cycle: int | None = Field(default=None, ge=0)
    eol_within_horizon: bool | None = Field(
        default=None,
        description="Whether end of life actually occurred within the risk horizon "
        "used at prediction time. The only honest label for the risk head.",
    )
    label_source: str = "unspecified"


def prediction_records_from_snapshot(snapshot: Any) -> list[PredictionRecord]:
    """Extract prediction records from a fleet snapshot, for later scoring.

    Taking ``Any`` avoids a circular import between fleet and monitoring; the
    only requirement is the FleetSnapshot shape.
    """
    records: list[PredictionRecord] = []
    for record in snapshot.batteries:
        if not record.is_evaluated:
            continue
        records.append(
            PredictionRecord(
                prediction_id=f"{snapshot.snapshot_id}:{record.battery_id}",
                battery_id=record.battery_id,
                cycle_index=int(record.latest_cycle or 0),
                model_version=record.model_version,
                model_name=record.model_name,
                fleet_id=snapshot.fleet_id,
                batch_id=snapshot.batch_id,
                generated_at_utc=record.snapshot_generated_at_utc or snapshot.generated_at_utc,
                predicted_rul=record.predicted_rul,
                rul_lower_bound=record.rul_lower_bound,
                rul_upper_bound=record.rul_upper_bound,
                interval_coverage_target=record.interval_coverage_target,
                predicted_soh_forecast=record.predicted_soh_forecast,
                soh_forecast_horizon_cycles=record.soh_forecast_horizon_cycles,
                failure_risk=record.failure_risk,
                risk_horizon_cycles=record.risk_horizon_cycles,
                measured_soh=record.measured_soh,
            )
        )
    return records


def join_predictions_and_labels(
    predictions: Sequence[PredictionRecord], labels: Sequence[OutcomeLabel]
) -> pd.DataFrame:
    """Inner-join predictions to outcomes on ``(battery_id, cycle_index)``.

    An inner join by design: a prediction with no label is not a zero-error
    prediction, and a label with no prediction is not a miss. Both are counted
    in ``label_coverage`` instead.
    """
    if not predictions:
        return pd.DataFrame()
    left = pd.DataFrame([p.model_dump() for p in predictions])
    if not labels:
        return left.iloc[0:0].copy()
    right = pd.DataFrame([label.model_dump() for label in labels])
    joined = left.merge(
        right,
        on=["battery_id", "cycle_index"],
        how="inner",
        suffixes=("", "_label"),
        validate="many_to_one",
    )
    return joined.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)


def evaluate_delayed_labels(
    predictions: Sequence[PredictionRecord],
    labels: Sequence[OutcomeLabel],
    cfg: ExperimentConfig,
    *,
    model_version: str | None = None,
) -> PerformanceReport:
    """Score whatever labels have arrived, and say plainly when none have."""
    policy = cfg.monitoring.performance
    n_predictions = len(predictions)

    if not labels or not predictions:
        return PerformanceReport(
            model_version=model_version,
            status=PerformanceStatus.NO_LABELS,
            n_predictions=n_predictions,
            n_labels_joined=0,
            label_coverage=0.0,
            comparison_note=policy.label_delay_note,
            warnings=[
                "No outcome labels have been joined yet. Prognostic labels arrive "
                "only once a cell reaches end of life, so an empty report early in a "
                "deployment is expected, not a failure."
            ],
        )

    joined = join_predictions_and_labels(predictions, labels)
    n_joined = int(len(joined))
    coverage = n_joined / max(n_predictions, 1)

    if n_joined < policy.min_labels:
        return PerformanceReport(
            model_version=model_version,
            status=PerformanceStatus.INSUFFICIENT_LABELS,
            n_predictions=n_predictions,
            n_labels_joined=n_joined,
            label_coverage=round(coverage, 4),
            comparison_note=policy.label_delay_note,
            warnings=[
                f"{n_joined} joined label(s) is below the configured minimum of "
                f"{policy.min_labels}; no metric is published. A MAE over a handful "
                "of rows is dominated by which cells happened to finish first."
            ],
        )

    if model_version is not None and "model_version" in joined.columns:
        subset = joined.loc[joined["model_version"] == model_version]
        if len(subset) >= policy.min_labels:
            joined = subset
        elif len(subset) > 0:
            logger.warning(
                "Only %d label(s) belong to model version %s; scoring every version "
                "together and reporting the mix.",
                len(subset),
                model_version,
            )

    warnings: list[str] = []
    breaches: list[str] = []

    # -- RUL ---------------------------------------------------------------
    rul_metrics: dict[str, Any] = {}
    stage_rows: list[dict[str, Any]] = []
    interval_coverage: dict[str, Any] = {}
    mask = joined["observed_rul"].notna() & joined["predicted_rul"].notna()
    if int(mask.sum()) >= policy.min_labels:
        truth = joined.loc[mask, "observed_rul"].to_numpy(dtype=float)
        predicted = joined.loc[mask, "predicted_rul"].to_numpy(dtype=float)
        rul_metrics = compute_metrics(truth, predicted)
        rul_metrics["bias"] = round(float(np.mean(predicted - truth)), 4)
        rul_metrics["n"] = int(mask.sum())
        stage_rows = _error_by_life_stage(truth, predicted)
        interval_coverage = _interval_coverage(joined.loc[mask], cfg)

        mae = float(rul_metrics.get("mae", float("nan")))
        warning_level, degraded_level = policy.rul_mae_thresholds
        if np.isfinite(mae) and mae >= degraded_level:
            breaches.append(f"RUL MAE {mae:.2f} cycles >= degraded threshold {degraded_level}")
        elif np.isfinite(mae) and mae >= warning_level:
            breaches.append(f"RUL MAE {mae:.2f} cycles >= warning threshold {warning_level}")

        empirical = interval_coverage.get("empirical_coverage")
        nominal = interval_coverage.get("nominal_coverage")
        if (
            empirical is not None
            and nominal is not None
            and empirical < nominal - policy.coverage_tolerance
        ):
            breaches.append(
                f"Interval coverage {empirical:.2%} is more than "
                f"{policy.coverage_tolerance:.0%} below the nominal {nominal:.0%}"
            )
    else:
        warnings.append("Too few joined RUL labels to publish RUL metrics.")

    # -- SOH ---------------------------------------------------------------
    soh_report: dict[str, Any] = {}
    soh_mask = joined["observed_soh"].notna() & joined["predicted_soh_forecast"].notna()
    if int(soh_mask.sum()) >= policy.min_labels:
        truth = joined.loc[soh_mask, "observed_soh"].to_numpy(dtype=float)
        predicted = joined.loc[soh_mask, "predicted_soh_forecast"].to_numpy(dtype=float)
        persistence = (
            joined.loc[soh_mask, "measured_soh"].to_numpy(dtype=float)
            if "measured_soh" in joined.columns
            else None
        )
        soh_report = soh_metrics(truth, predicted, persistence=persistence)
        soh_report["bias"] = round(float(np.mean(predicted - truth)), 6)
        soh_report["n"] = int(soh_mask.sum())
        mae = float(soh_report.get("mae", float("nan")))
        warning_level, degraded_level = policy.soh_mae_thresholds
        if np.isfinite(mae) and mae >= degraded_level:
            breaches.append(f"SOH MAE {mae:.4f} >= degraded threshold {degraded_level}")
        elif np.isfinite(mae) and mae >= warning_level:
            breaches.append(f"SOH MAE {mae:.4f} >= warning threshold {warning_level}")
    else:
        warnings.append("Too few joined SOH labels to publish SOH metrics.")

    # -- risk --------------------------------------------------------------
    risk_report: dict[str, Any] = {}
    risk_mask = joined["eol_within_horizon"].notna() & joined["failure_risk"].notna()
    if int(risk_mask.sum()) >= policy.min_labels:
        truth = joined.loc[risk_mask, "eol_within_horizon"].astype(float).to_numpy()
        probability = joined.loc[risk_mask, "failure_risk"].to_numpy(dtype=float)
        if len(np.unique(truth)) < 2:
            warnings.append(
                "Every joined risk label has the same class, so PR-AUC, ROC-AUC and "
                "calibration error are undefined. Reported as unavailable rather than "
                "as a perfect or a zero score."
            )
            risk_report = {
                "n": int(risk_mask.sum()),
                "positive_rate": round(float(truth.mean()), 4),
                "note": "single-class labels; ranking and calibration metrics undefined",
            }
        else:
            risk_report = risk_metrics(
                truth,
                probability,
                threshold=cfg.risk.threshold if cfg.risk.threshold is not None else 0.5,
                n_bins=cfg.calibration.n_bins,
                cycle_index=joined.loc[risk_mask, "cycle_index"].to_numpy(),
            )
            risk_report["n"] = int(risk_mask.sum())
            brier = float(risk_report.get("brier_score", float("nan")))
            warning_level, degraded_level = policy.brier_thresholds
            if np.isfinite(brier) and brier >= degraded_level:
                breaches.append(f"Brier score {brier:.4f} >= degraded threshold {degraded_level}")
            elif np.isfinite(brier) and brier >= warning_level:
                breaches.append(f"Brier score {brier:.4f} >= warning threshold {warning_level}")
            pr_auc = float(risk_report.get("pr_auc", float("nan")))
            if np.isfinite(pr_auc) and pr_auc < policy.pr_auc_floor:
                breaches.append(f"PR-AUC {pr_auc:.3f} is below the floor {policy.pr_auc_floor}")
    else:
        warnings.append("Too few joined risk labels to publish risk metrics.")

    status = _status_from(breaches)
    delay = _delay_summary(joined)

    return PerformanceReport(
        model_version=model_version,
        status=status,
        n_predictions=n_predictions,
        n_labels_joined=n_joined,
        label_coverage=round(coverage, 4),
        evaluation_delay_cycles=delay,
        rul_metrics=rul_metrics,
        rul_error_by_life_stage=stage_rows,
        interval_coverage=interval_coverage,
        soh_metrics=soh_report,
        risk_metrics=risk_report,
        thresholds={
            "rul_mae": list(policy.rul_mae_thresholds),
            "soh_mae": list(policy.soh_mae_thresholds),
            "brier": list(policy.brier_thresholds),
            "pr_auc_floor": policy.pr_auc_floor,
            "coverage_tolerance": policy.coverage_tolerance,
            "min_labels": policy.min_labels,
        },
        breaches=breaches,
        comparison_note=policy.label_delay_note,
        warnings=warnings,
    )


def _status_from(breaches: Sequence[str]) -> PerformanceStatus:
    if not breaches:
        return PerformanceStatus.HEALTHY
    if any("degraded threshold" in b or "below the floor" in b for b in breaches):
        return PerformanceStatus.DEGRADED
    return PerformanceStatus.WARNING


def _error_by_life_stage(truth: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, low, high in _LIFE_STAGE_BANDS:
        mask = (truth >= low) & (truth < high)
        n = int(mask.sum())
        if n == 0:
            rows.append({"life_stage": label, "n": 0})
            continue
        errors = predicted[mask] - truth[mask]
        rows.append(
            {
                "life_stage": label,
                "n": n,
                "mae": round(float(np.mean(np.abs(errors))), 4),
                "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
                "bias": round(float(np.mean(errors)), 4),
            }
        )
    return rows


def _interval_coverage(frame: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, Any]:
    """Empirical coverage of the prediction intervals that were actually issued."""
    mask = frame["rul_lower_bound"].notna() & frame["rul_upper_bound"].notna()
    if not bool(mask.any()):
        return {
            "n": 0,
            "note": "No prediction intervals were recorded with these predictions.",
        }
    subset = frame.loc[mask]
    truth = subset["observed_rul"].to_numpy(dtype=float)
    inside = (truth >= subset["rul_lower_bound"].to_numpy(dtype=float)) & (
        truth <= subset["rul_upper_bound"].to_numpy(dtype=float)
    )
    nominal = subset["interval_coverage_target"].dropna()
    return {
        "n": int(len(subset)),
        "empirical_coverage": round(float(inside.mean()), 4),
        "nominal_coverage": float(nominal.iloc[0]) if len(nominal) else cfg.uncertainty.coverage,
        "mean_interval_width": round(
            float(
                (
                    subset["rul_upper_bound"].to_numpy(dtype=float)
                    - subset["rul_lower_bound"].to_numpy(dtype=float)
                ).mean()
            ),
            4,
        ),
        "note": (
            "Coverage measured on production predictions, which are not exchangeable "
            "with the conformal calibration cells in the way the nominal level assumes."
        ),
    }


def _delay_summary(joined: pd.DataFrame) -> dict[str, float]:
    if "observed_at_cycle" not in joined.columns:
        return {}
    delay = pd.to_numeric(joined["observed_at_cycle"], errors="coerce") - pd.to_numeric(
        joined["cycle_index"], errors="coerce"
    )
    delay = delay.dropna()
    if delay.empty:
        return {}
    return {
        "median": round(float(delay.median()), 2),
        "mean": round(float(delay.mean()), 2),
        "min": round(float(delay.min()), 2),
        "max": round(float(delay.max()), 2),
    }
