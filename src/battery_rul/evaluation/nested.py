"""Nested, battery-aware model comparison and selection.

What was wrong before
---------------------
Milestone 1 chose the champion on a validation partition containing a **single**
cell, then cross-validated only the already-chosen model. Two problems compound
there. First, the choice rests on one cell, so it is close to arbitrary. Second —
and more seriously — the cross-validated number was then quoted as the headline,
even though the model it describes was picked using data that the
cross-validation also scores. That makes the headline conditional on a selection
step the interval does not account for, and it flatters the winner.

What this module does instead
-----------------------------
A textbook nested design, adapted to a five-cell cohort:

* **Outer loop** — leave one battery out. That cell is scored once, at the end,
  by a model that never saw it in any capacity.
* **Inner loop** — leave one battery out *within the outer training cells*.
  Model family (and any supplied hyperparameter grid) is chosen here, on
  training cells only.
* The chosen family is refitted on all outer-training cells and predicts the
  held-out cell. Pooling those out-of-fold predictions gives an estimate of
  *the whole procedure*, selection included — the number that may be quoted
  without an asterisk.

Every preprocessing artifact (fallback imputation, variance filter, correlation
pruning, supervised selection, scaler) is re-fitted inside each fold, inner and
outer, by constructing a fresh :class:`FeaturePipeline` per fold.

Alongside the nested estimate, every candidate is also refitted on each outer
fold's training cells and scored on that fold's held-out cell. That table shows
per-candidate pooled metrics, per-fold dispersion and how often each family won
the inner selection — a candidate that wins the headline once but is never
chosen by the inner loop is a coincidence, and the table says so.

Row comparability
-----------------
Sequence models cannot score a cell's first ``window - 1`` scoreable cycles.
Ranking a Transformer on the rows it likes against a Ridge on all rows compares
input requirements, not models. Metrics are therefore reported twice: on each
model's own scoreable rows (with the unscored count attached) and on the
intersection every candidate can score, which is what the ranking uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.metrics import METRIC_DIRECTION, compute_metrics
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.target import inverse_transform_target, transform_target
from battery_rul.models.base import TrainingData, build_model
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["NestedResult", "nested_model_comparison"]


@dataclass
class NestedResult:
    """Everything the nested evaluation produced."""

    #: Out-of-fold predictions of the *selected-per-fold* procedure.
    nested_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Pooled metrics of that procedure — the quotable headline.
    nested_metrics: dict[str, float] = field(default_factory=dict)
    #: Which family the inner loop chose in each outer fold.
    selection_by_fold: dict[str, str] = field(default_factory=dict)
    #: How often each family was selected across outer folds.
    selection_frequency: dict[str, int] = field(default_factory=dict)
    #: Per-candidate out-of-fold predictions, stacked.
    candidate_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-candidate pooled metrics on each candidate's own scoreable rows.
    candidate_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-candidate pooled metrics on rows every candidate can score.
    candidate_metrics_common: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-candidate, per-outer-fold metrics — the dispersion story.
    per_fold: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: dict[str, str] = field(default_factory=dict)
    n_common_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": "nested leave-one-battery-out (inner selection, outer scoring)",
            "nested_metrics": {k: _round(v) for k, v in self.nested_metrics.items()},
            "selection_by_fold": self.selection_by_fold,
            "selection_frequency": self.selection_frequency,
            "n_common_scoreable_rows": self.n_common_rows,
            "candidate_metrics": self.candidate_metrics.to_dict(orient="records"),
            "candidate_metrics_common_rows": self.candidate_metrics_common.to_dict(
                orient="records"
            ),
            "per_fold": self.per_fold.to_dict(orient="records"),
            "failures": self.failures,
        }


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    return value


def _fold_data(
    frame: pd.DataFrame,
    y_all: np.ndarray,
    features: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[TrainingData, TrainingData]:
    """Fit a fresh preprocessing pipeline on the fold's training rows only."""
    pipeline = FeaturePipeline(cfg=cfg.features)
    pipeline.fit(frame.loc[train_mask, features], y_all[train_mask])

    def _make(mask: np.ndarray) -> TrainingData:
        subset = frame.loc[mask].reset_index(drop=True)
        return TrainingData(
            X=pipeline.transform(subset[features]),
            y=y_all[mask],
            frame=subset,
            feature_names=pipeline.feature_names,
        )

    return _make(train_mask), _make(test_mask)


def _score(
    candidate: str,
    cfg: ExperimentConfig,
    train: TrainingData,
    test: TrainingData,
    params: dict[str, Any] | None,
) -> pd.DataFrame | None:
    """Fit one candidate on ``train`` and return row-aligned predictions on ``test``."""
    try:
        model = build_model(candidate, cfg, **(params or {}))
        model.fit(train)
        raw = model.predict(test)
    except Exception as exc:  # noqa: BLE001 - a failed candidate must not sink the design
        logger.warning("Candidate %s failed on a fold: %s: %s", candidate, type(exc).__name__, exc)
        return None
    return pd.DataFrame(
        {
            "model": candidate,
            "battery_id": test.battery_ids,
            "cycle_index": test.cycle_index,
            "y_true": inverse_transform_target(test.y, cfg),
            "y_pred": inverse_transform_target(raw, cfg),
        }
    )


def nested_model_comparison(
    prepared: Any,
    cfg: ExperimentConfig,
    *,
    candidates: list[str],
    param_grid: dict[str, dict[str, Any]] | None = None,
    select_by: str = "mae",
) -> NestedResult:
    """Run the nested leave-one-battery-out design over ``candidates``.

    Parameters
    ----------
    prepared:
        A :class:`~battery_rul.pipelines.prepare_data.PreparedData`.
    candidates:
        Model registry keys to compare. Include the baselines.
    param_grid:
        Optional fixed hyperparameters per candidate. Tuning *inside* the inner
        loop is supported by passing several registry aliases; a full inner
        search is deliberately out of scope at this cohort size, and the
        remaining selection bias is reported rather than hidden.
    select_by:
        Metric the inner loop minimises (or maximises, per ``METRIC_DIRECTION``).
    """
    frame = prepared.frame
    features = list(prepared.feature_names)
    y_all = transform_target(frame[cfg.target.name].to_numpy(dtype=float), cfg)
    battery_col = frame["battery_id"].to_numpy()
    cells = sorted(pd.unique(battery_col).tolist())
    result = NestedResult()

    if len(cells) < 3:
        logger.warning("Nested CV needs >= 3 cells, found %d; skipping", len(cells))
        return result

    ascending = METRIC_DIRECTION.get(select_by, "min") == "min"
    candidate_frames: list[pd.DataFrame] = []
    nested_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for outer_cell in cells:
        outer_test = battery_col == outer_cell
        outer_train = ~outer_test
        inner_cells = [c for c in cells if c != outer_cell]

        # ---- inner loop: choose a family on training cells only --------------
        inner_scores: dict[str, list[float]] = {c: [] for c in candidates}
        for inner_cell in inner_cells:
            inner_test = battery_col == inner_cell
            inner_train = outer_train & ~inner_test
            if inner_train.sum() < 20 or inner_test.sum() < 5:
                continue
            tr, te = _fold_data(frame, y_all, features, inner_train, inner_test, cfg)
            for candidate in candidates:
                predictions = _score(candidate, cfg, tr, te, (param_grid or {}).get(candidate))
                if predictions is None:
                    continue
                scored = predictions.dropna(subset=["y_pred"])
                if scored.empty:
                    continue
                metrics = compute_metrics(
                    scored["y_true"].to_numpy(),
                    scored["y_pred"].to_numpy(),
                    mape_epsilon=cfg.evaluation.mape_epsilon,
                    alpha=cfg.evaluation.alpha,
                )
                inner_scores[candidate].append(float(metrics.get(select_by, np.nan)))

        ranked = {
            name: float(np.nanmean(values))
            for name, values in inner_scores.items()
            if values and np.isfinite(values).any()
        }
        if not ranked:
            result.failures[str(outer_cell)] = "no candidate produced an inner score"
            continue
        selected = (min if ascending else max)(ranked, key=lambda k: ranked[k])
        result.selection_by_fold[str(outer_cell)] = selected
        logger.info(
            "Outer fold %s: inner loop selected %s (%s=%.3f over %d inner folds)",
            outer_cell,
            selected,
            select_by,
            ranked[selected],
            len(inner_scores[selected]),
        )

        # ---- outer scoring ---------------------------------------------------
        tr, te = _fold_data(frame, y_all, features, outer_train, outer_test, cfg)
        for candidate in candidates:
            predictions = _score(candidate, cfg, tr, te, (param_grid or {}).get(candidate))
            if predictions is None:
                continue
            predictions["outer_fold"] = str(outer_cell)
            candidate_frames.append(predictions)
            if candidate == selected:
                nested = predictions.copy()
                nested["selected_model"] = selected
                nested_frames.append(nested)

            scored = predictions.dropna(subset=["y_pred"])
            if scored.empty:
                continue
            metrics = compute_metrics(
                scored["y_true"].to_numpy(),
                scored["y_pred"].to_numpy(),
                mape_epsilon=cfg.evaluation.mape_epsilon,
                alpha=cfg.evaluation.alpha,
            )
            fold_rows.append(
                {
                    "outer_fold": str(outer_cell),
                    "model": candidate,
                    "selected_by_inner_loop": candidate == selected,
                    "n_scored": int(len(scored)),
                    "n_unscored": int(len(predictions) - len(scored)),
                    **{k: metrics.get(k) for k in ("mae", "rmse", "r2", "bias", "alpha_lambda")},
                }
            )

    if not candidate_frames:
        logger.warning("Nested comparison produced no predictions")
        return result

    result.candidate_predictions = pd.concat(candidate_frames, ignore_index=True)
    result.per_fold = pd.DataFrame(fold_rows).round(4)

    if nested_frames:
        result.nested_predictions = pd.concat(nested_frames, ignore_index=True)
        scored = result.nested_predictions.dropna(subset=["y_pred"])
        result.nested_metrics = compute_metrics(
            scored["y_true"].to_numpy(),
            scored["y_pred"].to_numpy(),
            mape_epsilon=cfg.evaluation.mape_epsilon,
            alpha=cfg.evaluation.alpha,
        )
        result.nested_metrics["n_unscored"] = int(len(result.nested_predictions) - len(scored))
    counts = pd.Series(list(result.selection_by_fold.values())).value_counts()
    result.selection_frequency = {str(k): int(v) for k, v in counts.items()}

    result.candidate_metrics = _pool(result.candidate_predictions, cfg, select_by, ascending)
    common = _common_rows(result.candidate_predictions)
    result.n_common_rows = len(common)
    if common:
        keys = result.candidate_predictions.apply(
            lambda r: (str(r["battery_id"]), int(r["cycle_index"])) in common, axis=1
        )
        result.candidate_metrics_common = _pool(
            result.candidate_predictions.loc[keys], cfg, select_by, ascending
        )
    return result


def _pool(
    predictions: pd.DataFrame, cfg: ExperimentConfig, select_by: str, ascending: bool
) -> pd.DataFrame:
    """Pooled out-of-fold metrics per candidate, plus per-fold dispersion."""
    rows: list[dict[str, Any]] = []
    for name, group in predictions.groupby("model", sort=True):
        scored = group.dropna(subset=["y_pred"])
        if scored.empty:
            continue
        metrics = compute_metrics(
            scored["y_true"].to_numpy(),
            scored["y_pred"].to_numpy(),
            mape_epsilon=cfg.evaluation.mape_epsilon,
            alpha=cfg.evaluation.alpha,
        )
        per_fold_mae = [
            compute_metrics(
                f["y_true"].to_numpy(),
                f["y_pred"].to_numpy(),
                mape_epsilon=cfg.evaluation.mape_epsilon,
            )["mae"]
            for _, f in scored.groupby("outer_fold")
        ]
        rows.append(
            {
                "model": str(name),
                "n_scored": int(len(scored)),
                "n_unscored": int(len(group) - len(scored)),
                "coverage": round(len(scored) / max(len(group), 1), 4),
                **{
                    k: metrics.get(k)
                    for k in (
                        "mae",
                        "rmse",
                        "mape",
                        "r2",
                        "bias",
                        "alpha_lambda",
                        "within_10_cycles",
                    )
                },
                "mae_across_folds_std": (
                    float(np.nanstd(per_fold_mae, ddof=1))
                    if len(per_fold_mae) > 1
                    else float("nan")
                ),
                "mae_worst_fold": float(np.nanmax(per_fold_mae)) if per_fold_mae else float("nan"),
                "n_folds": len(per_fold_mae),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(select_by, ascending=ascending, na_position="last")
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.reset_index(drop=True).round(4)


def _common_rows(predictions: pd.DataFrame) -> set[tuple[str, int]]:
    """Rows every candidate managed to score."""
    keys: set[tuple[str, int]] | None = None
    for _, group in predictions.groupby("model", sort=False):
        scored = group.dropna(subset=["y_pred"])
        rows = set(
            zip(scored["battery_id"].astype(str), scored["cycle_index"].astype(int), strict=True)
        )
        keys = rows if keys is None else (keys & rows)
    return keys or set()
