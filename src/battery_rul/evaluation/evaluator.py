"""Model evaluation: scoring, comparison tables and learning curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.metrics import (
    METRIC_DIRECTION,
    bootstrap_metric_ci,
    compute_metrics,
    per_battery_metrics,
    prognostic_horizon,
    residual_summary,
)
from battery_rul.features.target import inverse_transform_target
from battery_rul.models.base import BaseModel, TrainingData
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["EvaluationResult", "compare_models", "evaluate_model", "learning_curve"]


@dataclass
class EvaluationResult:
    """Everything known about one model on one partition."""

    model_name: str
    partition: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    per_battery: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: dict[str, float] = field(default_factory=dict)
    confidence_interval: dict[str, float] = field(default_factory=dict)
    n_unscored: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "partition": self.partition,
            "metrics": {k: _round(v) for k, v in self.metrics.items()},
            "residual_summary": {k: _round(v) for k, v in self.residuals.items()},
            "confidence_interval": self.confidence_interval,
            "n_unscored_rows": self.n_unscored,
            "per_battery": (
                self.per_battery.round(4).to_dict(orient="records")
                if not self.per_battery.empty
                else []
            ),
        }


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    return value


def evaluate_model(
    model: BaseModel,
    data: TrainingData,
    cfg: ExperimentConfig,
    *,
    partition: str = "test",
) -> EvaluationResult:
    """Score ``model`` on ``data`` and package every downstream artifact."""
    raw_pred = model.predict(data)
    y_pred = inverse_transform_target(raw_pred, cfg)
    y_true = inverse_transform_target(data.y, cfg)

    unscored = int(np.sum(~np.isfinite(y_pred)))
    if unscored:
        logger.info(
            "%s left %d/%d %s rows unscored (sequence warm-up)",
            model.name,
            unscored,
            len(data),
            partition,
        )

    predictions = pd.DataFrame(
        {
            "battery_id": data.battery_ids,
            "cycle_index": data.cycle_index,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["residual"].abs()
    predictions["model"] = model.name
    predictions["partition"] = partition
    for column in ("soh", "capacity_ah", "eol_cycle"):
        if column in data.frame.columns:
            predictions[column] = data.frame[column].to_numpy()

    eval_cfg = cfg.evaluation
    metrics = compute_metrics(
        y_true, y_pred, mape_epsilon=eval_cfg.mape_epsilon, alpha=eval_cfg.alpha
    )

    scored = predictions.dropna(subset=["y_pred"])
    per_battery = (
        per_battery_metrics(scored, mape_epsilon=eval_cfg.mape_epsilon, alpha=eval_cfg.alpha)
        if eval_cfg.per_battery_breakdown and not scored.empty
        else pd.DataFrame()
    )

    # Prognostic horizon is only meaningful *per cell* — it walks backwards from
    # one cell's end of life. Computing it on pooled rows from several cells
    # interleaves two life curves and produces a number that means nothing, so the
    # headline value is the mean of the per-cell horizons.
    if not per_battery.empty and "prognostic_horizon" in per_battery:
        horizons = per_battery["prognostic_horizon"].to_numpy(dtype=float)
        metrics["prognostic_horizon"] = (
            float(np.nanmean(horizons)) if np.isfinite(horizons).any() else float("nan")
        )
        metrics["prognostic_horizon_cells"] = int(np.isfinite(horizons).sum())
    else:
        metrics["prognostic_horizon"] = prognostic_horizon(y_true, y_pred, alpha=eval_cfg.alpha)

    ci = (
        bootstrap_metric_ci(
            y_true,
            y_pred,
            metric=cfg.models.select_by,
            n_samples=eval_cfg.bootstrap_samples,
            confidence=eval_cfg.confidence_level,
            seed=cfg.seed,
            mape_epsilon=eval_cfg.mape_epsilon,
            alpha=eval_cfg.alpha,
        )
        if eval_cfg.bootstrap_samples
        else {}
    )

    logger.info(
        "%-18s %-6s  MAE=%6.2f  RMSE=%6.2f  R2=%6.3f  MAPE=%6.1f%%  a-l=%.2f",
        model.name,
        partition,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["mape"],
        metrics["alpha_lambda"],
    )

    return EvaluationResult(
        model_name=model.name,
        partition=partition,
        metrics=metrics,
        predictions=predictions,
        per_battery=per_battery,
        residuals=residual_summary(y_true, y_pred),
        confidence_interval=ci,
        n_unscored=unscored,
    )


def compare_models(
    results: dict[str, EvaluationResult],
    *,
    select_by: str = "rmse",
    select_mode: str = "min",
) -> pd.DataFrame:
    """Ranked comparison table, sorted by the selection metric."""
    if not results:
        return pd.DataFrame()

    columns = [
        "model",
        "n",
        "mae",
        "rmse",
        "mape",
        "smape",
        "r2",
        "median_ae",
        "max_error",
        "bias",
        "alpha_lambda",
        "within_10_cycles",
        "within_25_cycles",
        "prognostic_horizon",
    ]
    rows = []
    for name, result in results.items():
        row = {"model": name}
        row.update({k: result.metrics.get(k, np.nan) for k in columns[1:]})
        row["n_unscored"] = result.n_unscored
        rows.append(row)

    table = pd.DataFrame(rows)
    ascending = METRIC_DIRECTION.get(select_by, select_mode) == "min"
    table = table.sort_values(select_by, ascending=ascending, na_position="last")
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.reset_index(drop=True).round(4)


def compare_models_common_rows(
    results: dict[str, EvaluationResult],
    *,
    select_by: str = "rmse",
    mape_epsilon: float = 1.0,
    alpha: float = 0.20,
) -> pd.DataFrame:
    """Like :func:`compare_models`, but on rows **every** model can score.

    Sequence models cannot score the first *w−1* cycles of a cell, so a naive
    table compares a Transformer on 156 rows against a Ridge on 194 — and the
    rows the sequence models skip are precisely the early-life ones, which are
    the hardest. That difference alone can reorder the ranking, which would make
    the comparison an artefact of the window length rather than a result.

    This table restricts every model to the intersection, so the ranking reflects
    the models rather than their input requirements. Both tables are reported.
    """
    if not results:
        return pd.DataFrame()

    keys: set[tuple[str, int]] | None = None
    for result in results.values():
        scored = result.predictions.dropna(subset=["y_pred"])
        rows = set(
            zip(scored["battery_id"].astype(str), scored["cycle_index"].astype(int), strict=True)
        )
        keys = rows if keys is None else (keys & rows)

    if not keys:
        logger.warning("Models share no commonly scoreable rows; skipping common-row comparison")
        return pd.DataFrame()

    table_rows = []
    for name, result in results.items():
        frame = result.predictions.copy()
        mask = [
            (str(b), int(c)) in keys
            for b, c in zip(frame["battery_id"], frame["cycle_index"], strict=True)
        ]
        subset = frame.loc[mask]
        metrics = compute_metrics(
            subset["y_true"], subset["y_pred"], mape_epsilon=mape_epsilon, alpha=alpha
        )
        table_rows.append(
            {
                "model": name,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "bias": metrics["bias"],
                "alpha_lambda": metrics["alpha_lambda"],
                "within_10_cycles": metrics["within_10_cycles"],
            }
        )

    table = pd.DataFrame(table_rows)
    ascending = METRIC_DIRECTION.get(select_by, "min") == "min"
    table = table.sort_values(select_by, ascending=ascending, na_position="last")
    table.insert(0, "rank", range(1, len(table) + 1))
    logger.info("Common-row comparison over %d rows scoreable by every model", len(keys))
    return table.reset_index(drop=True).round(4)


def learning_curve(
    model_factory,
    train: TrainingData,
    test: TrainingData,
    cfg: ExperimentConfig,
    *,
    fractions: list[float] | None = None,
) -> pd.DataFrame:
    """Train/test error as a function of training-set size.

    Subsampling is done by **cell**, not by row, and always keeps the earliest
    cycles of each retained cell. Sampling rows at random would silently rebuild
    the leaky split this repository exists to avoid, and would flatter the curve.
    """
    fractions = fractions or cfg.evaluation.learning_curve_fractions
    rows = []
    battery_ids = np.asarray(train.battery_ids)
    unique = sorted(set(battery_ids.tolist()))

    for fraction in fractions:
        n_cells = max(int(round(len(unique) * fraction)), 1)
        keep_cells = unique[:n_cells]
        mask = np.isin(battery_ids, keep_cells)

        # Within the retained cells, take the earliest cycles for sub-unit
        # fractions so the subset stays a prefix in time.
        if fraction < 1.0:
            per_cell_fraction = fraction * len(unique) / max(n_cells, 1)
            per_cell_fraction = min(per_cell_fraction, 1.0)
            keep = np.zeros(len(train), dtype=bool)
            positions = np.arange(len(train))
            for cell in keep_cells:
                idx = positions[battery_ids == cell]
                cut = max(int(len(idx) * per_cell_fraction), 5)
                keep[idx[:cut]] = True
            mask = mask & keep

        if mask.sum() < 20:
            continue

        subset = TrainingData(
            X=train.X[mask],
            y=train.y[mask],
            frame=train.frame.loc[mask].reset_index(drop=True),
            feature_names=train.feature_names,
        )
        try:
            model = model_factory().fit(subset)
        except Exception as exc:  # noqa: BLE001 - a failed point should not kill the curve
            logger.warning("Learning-curve point at fraction %.2f failed: %s", fraction, exc)
            continue

        train_metrics = compute_metrics(
            inverse_transform_target(subset.y, cfg),
            inverse_transform_target(model.predict(subset), cfg),
            mape_epsilon=cfg.evaluation.mape_epsilon,
        )
        test_metrics = compute_metrics(
            inverse_transform_target(test.y, cfg),
            inverse_transform_target(model.predict(test), cfg),
            mape_epsilon=cfg.evaluation.mape_epsilon,
        )
        rows.append(
            {
                "fraction": fraction,
                "n_train_rows": int(mask.sum()),
                "n_train_batteries": len(keep_cells),
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
            }
        )
    return pd.DataFrame(rows)
