"""Probability calibration, reliability measurement and threshold tuning.

A classifier's raw score is a ranking device. The recommendation engine treats it
as a probability — "71 % chance of reaching end of life within 30 cycles" drives
a maintenance decision — and a ranking device is not entitled to that reading.
Tree ensembles in particular are systematically over-confident near 0 and 1, and
a threshold chosen on uncalibrated scores does not mean what its number says.

Two methods:

``platt``
    Logistic regression on the raw score. Two parameters, so it survives a small
    calibration set, but it can only apply a sigmoidal correction — if the
    miscalibration is a different shape, it will not fix it.

``isotonic`` (default)
    Monotone step-function fit. Strictly more expressive and the better choice
    when the calibration set can support it; it over-fits on very small sets,
    which is why ``calibration.min_calibration_rows`` is enforced rather than
    advisory.

**Calibration is fitted on validation/calibration rows only.** Fitting on the
final test labels would make every calibration metric a self-report. The
pipelines pass an explicit calibration partition and the fitted object records
its size, so a reviewer can see what the numbers rest on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import CalibrationConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ProbabilityCalibrator",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
    "risk_metrics",
    "tune_threshold",
]


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin."""
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(probability, dtype=float)
    good = np.isfinite(truth) & np.isfinite(prob)
    if not good.any():
        return float("nan")
    return float(np.mean((prob[good] - truth[good]) ** 2))


def reliability_curve(
    y_true: np.ndarray, probability: np.ndarray, *, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted-vs-observed frequency per probability bin (the reliability diagram)."""
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(probability, dtype=float)
    good = np.isfinite(truth) & np.isfinite(prob)
    truth, prob = truth[good], prob[good]
    if truth.size == 0:
        return pd.DataFrame(columns=["bin_lower", "bin_upper", "n", "mean_predicted", "observed"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = index == b
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(prob[mask].mean()) if mask.any() else float("nan"),
                "observed": float(truth[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, *, n_bins: int = 10
) -> float:
    """Sample-weighted mean gap between predicted probability and observed rate."""
    curve = reliability_curve(y_true, probability, n_bins=n_bins)
    populated = curve[curve["n"] > 0]
    if populated.empty:
        return float("nan")
    weights = populated["n"].to_numpy(dtype=float)
    gaps = np.abs(populated["mean_predicted"] - populated["observed"]).to_numpy(dtype=float)
    return float(np.sum(weights * gaps) / np.sum(weights))


def risk_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    n_bins: int = 10,
    cycle_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """The full classification report at one threshold, probabilities included.

    ``cycle_index`` enables the **trivial-baseline** columns, and supplying it is
    strongly recommended. The risk label is "RUL ≤ H", so within a single cell the
    positives are exactly the last H cycles — which means *cycle index alone*
    ranks them perfectly and scores AUC 1.0. On a single-cell evaluation
    partition every AUC here is therefore degenerate, and a model scoring 0.93
    is doing **worse than counting cycles**. Reporting the baseline next to the
    model's number is the only way that reading is available to a reader.
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(probability, dtype=float)
    good = np.isfinite(truth) & np.isfinite(prob)
    truth, prob = truth[good].astype(int), prob[good]
    n_pos = int(truth.sum())

    out: dict[str, Any] = {
        "n": int(truth.size),
        "n_positive": n_pos,
        "positive_rate": float(n_pos / truth.size) if truth.size else float("nan"),
        "threshold": float(threshold),
        "brier": brier_score(truth, prob),
        "ece": expected_calibration_error(truth, prob, n_bins=n_bins),
    }
    # AUCs need both classes present; reporting 0.5 or 1.0 for a degenerate set
    # would be a fabricated number, so it is NaN and the caller can see why.
    if 0 < n_pos < truth.size:
        out["roc_auc"] = float(roc_auc_score(truth, prob))
        out["pr_auc"] = float(average_precision_score(truth, prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        logger.warning(
            "Risk metrics computed on a single-class set (%d positives of %d); "
            "ROC-AUC and PR-AUC are undefined and reported as NaN.",
            n_pos,
            truth.size,
        )

    # --- the trivial baseline, on exactly these rows ------------------------
    if cycle_index is not None and 0 < n_pos < truth.size:
        cycles = np.asarray(cycle_index, dtype=float)[good]
        out["roc_auc_cycle_index_baseline"] = float(roc_auc_score(truth, cycles))
        out["pr_auc_cycle_index_baseline"] = float(average_precision_score(truth, cycles))
        out["beats_cycle_index_baseline"] = bool(out["pr_auc"] > out["pr_auc_cycle_index_baseline"])
        if not out["beats_cycle_index_baseline"]:
            logger.warning(
                "Risk model PR-AUC %.3f does not beat the cycle-index baseline (%.3f) "
                "on these rows. Because the label is 'RUL <= H', cycle index ranks the "
                "positives perfectly within any single cell — so this comparison is the "
                "one that carries information, not the absolute AUC.",
                out["pr_auc"],
                out["pr_auc_cycle_index_baseline"],
            )

    predicted = (prob >= threshold).astype(int)
    out["precision"] = float(precision_score(truth, predicted, zero_division=0))
    out["recall"] = float(recall_score(truth, predicted, zero_division=0))
    out["f1"] = float(f1_score(truth, predicted, zero_division=0))
    if truth.size:
        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        out["true_negative"] = int(matrix[0, 0])
        out["false_positive"] = int(matrix[0, 1])
        out["false_negative"] = int(matrix[1, 0])
        out["true_positive"] = int(matrix[1, 1])
    return out


def tune_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    objective: str = "f1",
    min_recall: float = 0.8,
) -> tuple[float, dict[str, Any]]:
    """Choose a decision threshold on **validation/calibration** rows.

    Never call this with test labels. The returned threshold is persisted in the
    model bundle so serving uses the same number the evaluation reported.

    Objectives
    ----------
    ``f1``                  maximise F1 — a balanced default.
    ``youden``              maximise ``recall + specificity - 1``.
    ``precision_at_recall`` the most precise threshold that still achieves
                            ``min_recall``. This is the one to use when a missed
                            end-of-life crossing costs more than a needless
                            inspection, which for a battery it usually does.
    """
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(probability, dtype=float)
    good = np.isfinite(truth) & np.isfinite(prob)
    truth, prob = truth[good].astype(int), prob[good]

    if truth.size == 0 or truth.sum() == 0 or truth.sum() == truth.size:
        logger.warning(
            "Cannot tune a threshold on a single-class calibration set; defaulting to 0.5"
        )
        return 0.5, {"reason": "single_class_calibration_set"}

    candidates = np.unique(np.clip(prob, 0.0, 1.0))
    candidates = np.concatenate([[0.0], candidates, [1.0]])
    best_threshold, best_score = 0.5, -np.inf
    best_stats: dict[str, Any] = {}

    for threshold in candidates:
        predicted = (prob >= threshold).astype(int)
        tp = float(np.sum((predicted == 1) & (truth == 1)))
        fp = float(np.sum((predicted == 1) & (truth == 0)))
        fn = float(np.sum((predicted == 0) & (truth == 1)))
        tn = float(np.sum((predicted == 0) & (truth == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if objective == "youden":
            score = recall + specificity - 1.0
        elif objective == "precision_at_recall":
            score = precision if recall >= min_recall else -1.0 + recall
        else:
            score = f1

        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
            best_stats = {
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "objective_value": float(score),
            }

    logger.info(
        "Threshold tuned on calibration rows: %.4f (objective=%s, precision=%.3f, "
        "recall=%.3f, f1=%.3f)",
        best_threshold,
        objective,
        best_stats.get("precision", float("nan")),
        best_stats.get("recall", float("nan")),
        best_stats.get("f1", float("nan")),
    )
    return best_threshold, {"objective": objective, **best_stats}


@dataclass
class ProbabilityCalibrator:
    """Fitted probability calibrator plus its before/after evidence."""

    cfg: CalibrationConfig
    method: str = "isotonic"
    model: Any | None = None
    fitted: bool = False
    n_calibration: int = 0
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    threshold_stats: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        y_true: np.ndarray,
        raw_probability: np.ndarray,
        *,
        tune: bool = True,
        objective: str = "f1",
        min_recall: float = 0.8,
    ) -> ProbabilityCalibrator:
        """Fit on calibration rows and record the improvement (or lack of it)."""
        truth = np.asarray(y_true, dtype=float)
        raw = np.asarray(raw_probability, dtype=float)
        good = np.isfinite(truth) & np.isfinite(raw)
        truth, raw = truth[good], raw[good]
        self.n_calibration = int(truth.size)
        self.method = self.cfg.method

        if self.n_calibration < self.cfg.min_calibration_rows:
            raise ValueError(
                f"Probability calibration needs at least {self.cfg.min_calibration_rows} "
                f"rows, got {self.n_calibration}."
            )

        self.metrics_before = {
            "brier": brier_score(truth, raw),
            "ece": expected_calibration_error(truth, raw, n_bins=self.cfg.n_bins),
        }

        if self.cfg.method == "none" or truth.sum() in (0, truth.size):
            if self.cfg.method != "none":
                logger.warning(
                    "Calibration set contains a single class; leaving probabilities "
                    "uncalibrated rather than fitting a degenerate mapping."
                )
                self.method = "none_single_class"
            self.model = None
        elif self.cfg.method == "platt":
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(C=1e6, solver="lbfgs")
            self.model.fit(raw.reshape(-1, 1), truth.astype(int))
        else:
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(raw, truth)

        self.fitted = True
        calibrated = self.transform(raw)
        self.metrics_after = {
            "brier": brier_score(truth, calibrated),
            "ece": expected_calibration_error(truth, calibrated, n_bins=self.cfg.n_bins),
        }

        if tune:
            self.threshold, self.threshold_stats = tune_threshold(
                truth, calibrated, objective=objective, min_recall=min_recall
            )

        logger.info(
            "Calibrator (%s) fitted on %d rows: Brier %.4f -> %.4f, ECE %.4f -> %.4f",
            self.method,
            self.n_calibration,
            self.metrics_before["brier"],
            self.metrics_after["brier"],
            self.metrics_before["ece"],
            self.metrics_after["ece"],
        )
        return self

    def transform(self, raw_probability: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities, clipped to [0, 1]."""
        if not self.fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted. Call fit() first.")
        raw = np.asarray(raw_probability, dtype=float)
        if self.model is None:
            return np.clip(raw, 0.0, 1.0)
        if hasattr(self.model, "predict_proba"):
            out = self.model.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            out = self.model.predict(raw)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_calibration_rows": self.n_calibration,
            "fitted_on": "validation/calibration partition only",
            "threshold": round(float(self.threshold), 5),
            "threshold_selection": self.threshold_stats,
            "metrics_before": {k: _round(v) for k, v in self.metrics_before.items()},
            "metrics_after": {k: _round(v) for k, v in self.metrics_after.items()},
            "n_bins": self.cfg.n_bins,
        }


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    return value
