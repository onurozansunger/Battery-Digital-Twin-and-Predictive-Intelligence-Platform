r"""Split conformal prediction intervals for RUL (and SOH).

Why conformal rather than a model's own variance
------------------------------------------------
Quantile regression and MC dropout both produce intervals whose width is a
statement the *model* makes about itself; whether that statement is true is an
open question you then have to check anyway. Split conformal inverts the order:
it takes an arbitrary point predictor, measures how wrong it actually was on a
held-out calibration set, and builds the interval from those measured residuals.
The coverage guarantee comes from the calibration data, not from the model being
well-specified.

The guarantee and its assumption
--------------------------------
For a calibration set of size *n* and target coverage :math:`1-\alpha`, take

.. math::
    \hat q = \mathrm{Quantile}_{\lceil (n+1)(1-\alpha)\rceil / n}
             \left(\, |y_j - \hat y_j| \,\right)

over calibration rows *j*, and report :math:`[\hat y - \hat q,\ \hat y + \hat q]`.
Marginal coverage is then at least :math:`1-\alpha` **provided calibration and
test rows are exchangeable**.

That proviso does real work here and the honest statement is that it is only
approximately satisfied:

* rows within a cell are strongly autocorrelated, so *n* overstates the effective
  sample size and the realised coverage is noisier than the nominal level;
* calibration cells and the served cell are different physical cells, so
  exchangeability is a cross-cell assumption, not an i.i.d.-rows one.

Both are stated in ``docs/UNCERTAINTY_METHOD.md`` and in the interval's own
metadata, and empirical coverage is measured per battery and per life stage —
because a marginal 90 % that is 99 % early in life and 55 % near end of life is
worse than useless for a maintenance decision.

The output is a **prediction interval**, not a confidence interval: it covers a
future observation, not a parameter. The distinction is kept in the naming
throughout.

Degradation-stage conditioning
------------------------------
RUL residuals are strongly heteroscedastic — a fresh cell's remaining life is
nearly unpredictable, a nearly-dead cell's is not. A single global quantile is
therefore far too wide near end of life, which is precisely where a maintenance
decision is being made. With ``normalise_by_life_stage`` the calibration
residuals are bucketed and a quantile is fitted per bucket, falling back to the
global quantile when a bucket is too thin to support one. This is standard
Mondrian (group-conditional) conformal prediction; the coverage guarantee holds
within each group under the same exchangeability assumption.

**The bucketing variable is measured state of health, not life fraction.** Life
fraction is ``cycle / eol_cycle`` and therefore needs the end-of-life cycle — a
label. Conditioning on it would work perfectly in evaluation and then silently
degrade to one global quantile at serving time, because a served cell has no
label. SOH is measured, available at every cycle, and monotone in the same
direction, so the buckets mean the same thing on both sides of the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import UncertaintyConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ConformalIntervalEstimator",
    "PredictionInterval",
    "conformal_quantile",
    "coverage_report",
]


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """The finite-sample-corrected empirical quantile of absolute residuals.

    The ``(n + 1)`` correction is not cosmetic: without it the interval
    under-covers at small *n*, which at this cohort size is every interval.
    """
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan")
    level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    return float(np.quantile(values, level, method="higher"))


@dataclass(slots=True)
class PredictionInterval:
    """One interval, with everything needed to interpret it."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_coverage_target: float
    uncertainty_method: str
    calibration_sample_size: int
    life_stage: str | None = None
    is_prediction_interval: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.lower_bound > self.point_estimate or self.upper_bound < self.point_estimate:
            raise ValueError(
                "Interval must bracket the point estimate: "
                f"{self.lower_bound} <= {self.point_estimate} <= {self.upper_bound}"
            )

    @property
    def width(self) -> float:
        return float(self.upper_bound - self.lower_bound)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": round(float(self.point_estimate), 4),
            "lower_bound": round(float(self.lower_bound), 4),
            "upper_bound": round(float(self.upper_bound), 4),
            "width": round(self.width, 4),
            "interval_coverage_target": self.interval_coverage_target,
            "uncertainty_method": self.uncertainty_method,
            "calibration_sample_size": self.calibration_sample_size,
            "life_stage": self.life_stage,
            "interval_type": "prediction_interval",
            "note": self.note,
        }


@dataclass
class ConformalIntervalEstimator:
    """Split-conformal interval estimator, fitted on a calibration partition.

    Fit on **calibration data only** — rows that trained no model and tuned no
    threshold. Fitting on training residuals would produce intervals as wide as
    the model's optimism, which is to say far too narrow.
    """

    cfg: UncertaintyConfig
    target_name: str = "rul_cycles"
    global_quantile: float = float("nan")
    stage_quantiles: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    n_calibration: int = 0
    fitted: bool = False
    lower_clip: float | None = 0.0
    upper_clip: float | None = None

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        life_fraction: np.ndarray | None = None,
    ) -> ConformalIntervalEstimator:
        """Learn the conformal quantile(s) from calibration residuals.

        ``life_fraction`` is the **stage variable** — measured state of health.
        The parameter name is retained for backwards compatibility of callers.
        """
        truth = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        if truth.shape != pred.shape:
            raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")

        good = np.isfinite(truth) & np.isfinite(pred)
        residuals = np.abs(truth[good] - pred[good])
        self.n_calibration = int(residuals.size)
        if self.n_calibration < self.cfg.min_calibration_rows:
            raise ValueError(
                f"Conformal calibration needs at least {self.cfg.min_calibration_rows} "
                f"scoreable rows, got {self.n_calibration}. Widen the calibration "
                "partition rather than lowering the threshold — an interval fitted on "
                "a handful of residuals carries no useful guarantee."
            )

        self.global_quantile = conformal_quantile(residuals, self.cfg.coverage)
        self.stage_quantiles = {}
        self.stage_counts = {}

        if self.cfg.normalise_by_life_stage and life_fraction is not None:
            stages = self.life_stages(np.asarray(life_fraction, dtype=float)[good])
            for stage in sorted(set(stages)):
                subset = residuals[stages == stage]
                self.stage_counts[stage] = int(subset.size)
                if subset.size >= self.cfg.min_calibration_rows:
                    self.stage_quantiles[stage] = conformal_quantile(subset, self.cfg.coverage)
                else:
                    logger.info(
                        "Life stage %s has only %d calibration rows (< %d); it falls back "
                        "to the global quantile.",
                        stage,
                        subset.size,
                        self.cfg.min_calibration_rows,
                    )

        self.fitted = True
        logger.info(
            "Conformal estimator fitted on %d calibration rows: global q=%.3f at %.0f%% "
            "target coverage; per-stage q=%s",
            self.n_calibration,
            self.global_quantile,
            100 * self.cfg.coverage,
            {k: round(v, 2) for k, v in self.stage_quantiles.items()},
        )
        return self

    # -- degradation stages ---------------------------------------------------
    def life_stages(self, soh: np.ndarray) -> np.ndarray:
        """Bucket **state of health** into named stages using the configured edges.

        Edges are descending (default 0.90, 0.80): ``early`` is a healthy cell,
        ``late`` a worn one. Non-finite SOH yields ``unknown``, which falls back
        to the global quantile rather than being silently assigned a bucket.
        """
        edges = list(self.cfg.stage_edges)
        names = (
            ["early", "mid", "late"]
            if len(edges) == 2
            else [f"stage_{i}" for i in range(len(edges) + 1)]
        )
        values = np.asarray(soh, dtype=float)
        out = np.full(values.shape, names[-1], dtype=object)
        assigned = np.zeros(values.shape, dtype=bool)
        for i, edge in enumerate(edges):
            mask = (~assigned) & np.isfinite(values) & (values >= edge)
            out[mask] = names[i]
            assigned |= mask
        out[~np.isfinite(values)] = "unknown"
        return out.astype(str)

    # -- application --------------------------------------------------------
    def quantile_for(self, life_fraction: float | None) -> tuple[float, str | None]:
        """The quantile that applies to a row, plus the stage name it came from."""
        self._check_fitted()
        if not self.stage_quantiles or life_fraction is None:
            return self.global_quantile, None
        stage = str(self.life_stages(np.array([life_fraction]))[0])
        return self.stage_quantiles.get(stage, self.global_quantile), stage

    def interval(
        self, point_estimate: float, *, life_fraction: float | None = None
    ) -> PredictionInterval:
        """Build one interval around a point estimate."""
        self._check_fitted()
        quantile, stage = self.quantile_for(life_fraction)
        lower = point_estimate - quantile
        upper = point_estimate + quantile
        if self.lower_clip is not None:
            lower = max(lower, self.lower_clip)
        if self.upper_clip is not None:
            upper = min(upper, self.upper_clip)
        # Clipping can pull a bound past the point estimate when the estimate
        # itself sits outside the physical range; clip the estimate too rather
        # than emit an interval that does not contain it.
        point = float(min(max(point_estimate, lower), upper))
        return PredictionInterval(
            point_estimate=point,
            lower_bound=float(lower),
            upper_bound=float(upper),
            interval_coverage_target=self.cfg.coverage,
            uncertainty_method="split_conformal" + ("_by_life_stage" if stage is not None else ""),
            calibration_sample_size=self.stage_counts.get(stage or "", self.n_calibration),
            life_stage=stage,
            note=(
                "Prediction interval, not a confidence interval. Marginal coverage "
                "holds under exchangeability between calibration and served cells, "
                "which is only approximately true across physical cells."
            ),
        )

    def intervals(
        self, point_estimates: np.ndarray, *, life_fraction: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised lower/upper bounds for a batch of point estimates."""
        self._check_fitted()
        points = np.asarray(point_estimates, dtype=float)
        if life_fraction is None or not self.stage_quantiles:
            quantiles = np.full(points.shape, self.global_quantile)
        else:
            stages = self.life_stages(np.asarray(life_fraction, dtype=float))
            quantiles = np.array(
                [self.stage_quantiles.get(s, self.global_quantile) for s in stages]
            )
        lower = points - quantiles
        upper = points + quantiles
        if self.lower_clip is not None:
            lower = np.maximum(lower, self.lower_clip)
        if self.upper_clip is not None:
            upper = np.minimum(upper, self.upper_clip)
        return lower, upper

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "split_conformal",
            "target": self.target_name,
            "coverage_target": self.cfg.coverage,
            "n_calibration_rows": self.n_calibration,
            "global_quantile": (
                round(self.global_quantile, 5) if np.isfinite(self.global_quantile) else None
            ),
            "stage_quantiles": {k: round(v, 5) for k, v in self.stage_quantiles.items()},
            "stage_counts": self.stage_counts,
            "stage_variable": "measured state of health",
            "stage_edges": list(self.cfg.stage_edges),
            "clip": {"lower": self.lower_clip, "upper": self.upper_clip},
        }

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("ConformalIntervalEstimator is not fitted. Call fit() first.")


def coverage_report(
    predictions: pd.DataFrame,
    *,
    truth_col: str = "y_true",
    lower_col: str = "lower_bound",
    upper_col: str = "upper_bound",
    group_cols: tuple[str, ...] = ("battery_id", "life_stage"),
) -> dict[str, Any]:
    """Empirical coverage and interval width, overall and per group.

    A marginal number alone hides the failure mode that matters: intervals that
    over-cover where nothing is at stake and under-cover near end of life.
    """
    frame = predictions.dropna(subset=[truth_col, lower_col, upper_col])
    if frame.empty:
        return {"n": 0, "empirical_coverage": None, "mean_interval_width": None}

    covered = (frame[truth_col] >= frame[lower_col]) & (frame[truth_col] <= frame[upper_col])
    out: dict[str, Any] = {
        "n": int(len(frame)),
        "empirical_coverage": round(float(covered.mean()), 5),
        "mean_interval_width": round(float((frame[upper_col] - frame[lower_col]).mean()), 4),
        "median_interval_width": round(float((frame[upper_col] - frame[lower_col]).median()), 4),
    }
    for column in group_cols:
        if column not in frame.columns:
            continue
        grouped = frame.assign(_covered=covered).groupby(column, sort=True)
        out[f"by_{column}"] = {
            str(key): {
                "n": int(len(group)),
                "empirical_coverage": round(float(group["_covered"].mean()), 5),
                "mean_interval_width": round(
                    float((group[upper_col] - group[lower_col]).mean()), 4
                ),
            }
            for key, group in grouped
        }
    return out
