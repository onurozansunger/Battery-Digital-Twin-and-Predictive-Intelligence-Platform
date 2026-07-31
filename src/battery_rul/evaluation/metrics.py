r"""Regression and prognostics metrics.

Beyond the standard four (MAE / RMSE / MAPE / R²) this module implements two
metrics that the prognostics literature uses and generic ML tooling does not:

**alpha-lambda accuracy**
    The fraction of predictions falling inside a *relative* cone around truth,
    :math:`|\hat y - y| \le \alpha\, y`. Unlike MAE it tightens as the cell
    approaches end of life, which is where a maintenance decision actually gets
    made — being 20 cycles wrong at RUL 100 is tolerable, at RUL 10 it is not.

**Prognostic Horizon (PH)**
    How many cycles before true end of life the prediction *first* enters and
    then stays inside the alpha cone. This is the number a reliability engineer
    cares about: "how much warning does this model give me?"

Also provided: a signed bias measure, because for maintenance the *direction* of
the error matters asymmetrically — over-predicting remaining life strands a cell
in service past its safe window, while under-predicting only wastes capacity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "METRIC_DIRECTION",
    "bootstrap_metric_ci",
    "compute_metrics",
    "per_battery_metrics",
    "prognostic_horizon",
    "residual_summary",
]

#: Whether lower or higher is better, used to pick the champion model.
METRIC_DIRECTION: dict[str, str] = {
    "mae": "min",
    "rmse": "min",
    "mse": "min",
    "mape": "min",
    "smape": "min",
    "median_ae": "min",
    "max_error": "min",
    "bias": "min",  # |bias|, see compute_metrics
    "r2": "max",
    "alpha_lambda": "max",
    "within_10_cycles": "max",
    "within_25_cycles": "max",
}


def _clean(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows either side cannot score (sequence models emit NaN warm-ups)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    mape_epsilon: float = 1.0,
    alpha: float = 0.20,
) -> dict[str, float]:
    """Full metric block for one set of predictions.

    ``mape_epsilon`` floors the MAPE denominator. RUL legitimately reaches zero
    at end of life, so raw MAPE would divide by zero on the single most important
    row of every cell; flooring at one cycle keeps the metric finite and is
    stated wherever MAPE is reported.
    """
    yt, yp = _clean(y_true, y_pred)
    n = yt.size
    if n == 0:
        return {k: float("nan") for k in METRIC_DIRECTION} | {"n": 0}

    residual = yp - yt
    abs_residual = np.abs(residual)

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    denominator = np.maximum(np.abs(yt), mape_epsilon)
    mape = float(np.mean(abs_residual / denominator) * 100.0)
    smape_denominator = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, mape_epsilon)
    smape = float(np.mean(abs_residual / smape_denominator) * 100.0)

    return {
        "n": int(n),
        "mae": float(np.mean(abs_residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mse": float(np.mean(residual**2)),
        "median_ae": float(np.median(abs_residual)),
        "max_error": float(abs_residual.max()),
        "mape": mape,
        "smape": smape,
        "r2": float(r2),
        # Signed: positive means the model is optimistic about remaining life,
        # which is the dangerous direction for maintenance planning.
        "bias": float(np.mean(residual)),
        "std_residual": float(np.std(residual)),
        "alpha_lambda": float(np.mean(abs_residual <= alpha * np.maximum(yt, 1.0))),
        "within_10_cycles": float(np.mean(abs_residual <= 10)),
        "within_25_cycles": float(np.mean(abs_residual <= 25)),
    }


def per_battery_metrics(
    frame: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    battery_col: str = "battery_id",
    mape_epsilon: float = 1.0,
    alpha: float = 0.20,
) -> pd.DataFrame:
    """Metrics computed independently for each held-out cell.

    Aggregate metrics hide the failure mode that matters most here: a model can
    look good overall while being badly wrong on one cell. With only two or three
    test cells, the per-cell table *is* the result.
    """
    rows = []
    for battery_id, group in frame.groupby(battery_col, sort=True):
        metrics = compute_metrics(
            group[y_true_col], group[y_pred_col], mape_epsilon=mape_epsilon, alpha=alpha
        )
        metrics[battery_col] = battery_id
        metrics["prognostic_horizon"] = prognostic_horizon(
            group[y_true_col].to_numpy(), group[y_pred_col].to_numpy(), alpha=alpha
        )
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[[battery_col] + [c for c in out.columns if c != battery_col]]


def prognostic_horizon(
    y_true: Sequence[float], y_pred: Sequence[float], *, alpha: float = 0.20
) -> float:
    """Cycles of advance warning before end of life.

    Walking backwards from EOL (RUL = 0), find the earliest point from which
    every subsequent prediction stays inside the alpha cone. The returned value
    is that point's true RUL: larger means the model became trustworthy sooner.

    Returns ``NaN`` when the model never stabilises inside the cone.
    """
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return float("nan")

    order = np.argsort(-yt)  # descending RUL == chronological order
    yt, yp = yt[order], yp[order]
    inside = np.abs(yp - yt) <= alpha * np.maximum(yt, 1.0)

    horizon = float("nan")
    for i in range(yt.size):
        if inside[i:].all():
            horizon = float(yt[i])
            break
    return horizon


def residual_summary(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, float]:
    """Distributional view of the errors, for the residual-analysis section."""
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return {}
    residual = yp - yt
    out = {f"q{int(q * 100):02d}": float(np.quantile(residual, q)) for q in quantiles}
    out["mean"] = float(residual.mean())
    out["std"] = float(residual.std())
    out["skew"] = float(pd.Series(residual).skew())
    out["kurtosis"] = float(pd.Series(residual).kurtosis())
    # Fraction of the variance explained by RUL level: a strong trend here means
    # the model is systematically biased at one end of the life curve.
    if yt.std() > 0:
        out["residual_rul_corr"] = float(np.corrcoef(residual, yt)[0, 1])
    return out


def bootstrap_metric_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    metric: str = "rmse",
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    mape_epsilon: float = 1.0,
    alpha: float = 0.20,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for one metric.

    With two or three held-out cells, a point estimate on its own invites
    over-reading. Note the resampling is over *rows*, which understates the true
    uncertainty because rows within a cell are correlated — the interval is a
    lower bound on the real spread, and the report says so.
    """
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0 or n_samples <= 0:
        return {}

    rng = np.random.default_rng(seed)
    values = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = rng.integers(0, yt.size, yt.size)
        values[i] = compute_metrics(yt[idx], yp[idx], mape_epsilon=mape_epsilon, alpha=alpha).get(
            metric, np.nan
        )

    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    tail = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "point": compute_metrics(yt, yp, mape_epsilon=mape_epsilon, alpha=alpha).get(metric),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "confidence": confidence,
        "n_bootstrap": int(values.size),
    }
