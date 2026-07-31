"""Hyperparameter optimisation with Optuna.

The objective is **battery-grouped cross-validated RMSE on the training
partition**. Three details make it honest:

1. Folds group by cell, so a trial cannot win by memorising a cell.
2. The feature pipeline is re-fit inside every fold, so scaler and feature
   selection statistics never cross the fold boundary.
3. The test partition is not touched at any point.

Search spaces live in :mod:`battery_rul.models.search_spaces`, versioned with the
code, so a study is reproducible from the git revision alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.metrics import METRIC_DIRECTION, compute_metrics
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.splitting import iter_group_folds
from battery_rul.features.target import inverse_transform_target, transform_target
from battery_rul.models.base import TrainingData, build_model
from battery_rul.models.search_spaces import SEARCH_SPACES, describe_spaces, suggest_params
from battery_rul.pipelines.prepare_data import PreparedData, load_prepared
from battery_rul.utils.io import save_json
from battery_rul.utils.logging import get_logger, log_section
from battery_rul.utils.seed import seed_everything

logger = get_logger(__name__)

__all__ = ["TuningResult", "run"]


@dataclass
class TuningResult:
    """Outcome of one model's study."""

    model: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float = float("nan")
    n_trials: int = 0
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "best_params": self.best_params,
            "best_value": None if not np.isfinite(self.best_value) else round(self.best_value, 6),
            "n_trials": self.n_trials,
            "trials": self.trials,
        }


def _make_sampler(cfg: ExperimentConfig) -> optuna.samplers.BaseSampler:
    if cfg.tuning.sampler == "random":
        return optuna.samplers.RandomSampler(seed=cfg.tuning.seed)
    if cfg.tuning.sampler == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=cfg.tuning.seed)
    return optuna.samplers.TPESampler(seed=cfg.tuning.seed, multivariate=True)


def _make_pruner(cfg: ExperimentConfig) -> optuna.pruners.BasePruner:
    if cfg.tuning.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if cfg.tuning.pruner == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)


def _cv_score(
    model_name: str,
    params: dict[str, Any],
    prepared: PreparedData,
    cfg: ExperimentConfig,
    trial: optuna.Trial | None = None,
) -> float:
    """Mean grouped-CV score for one hyperparameter vector."""
    frame = prepared.frame.loc[prepared.split.train].reset_index(drop=True)
    features = prepared.feature_names
    y_all = transform_target(frame[cfg.target.name].to_numpy(dtype=float), cfg)

    metric = cfg.tuning.metric
    scores: list[float] = []

    for step, (train_mask, test_mask) in enumerate(
        iter_group_folds(frame, cfg.tuning.cv_folds, seed=cfg.tuning.seed)
    ):
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue

        # Re-fit the transform inside the fold: sharing one pipeline across folds
        # would leak the held-out cells' statistics into every trial's score.
        pipeline = FeaturePipeline(cfg=cfg.features)
        pipeline.fit(frame.loc[train_mask, features], y_all[train_mask])

        fold_train = TrainingData(
            X=pipeline.transform(frame.loc[train_mask, features]),
            y=y_all[train_mask],
            frame=frame.loc[train_mask].reset_index(drop=True),
            feature_names=pipeline.feature_names,
        )
        fold_test = TrainingData(
            X=pipeline.transform(frame.loc[test_mask, features]),
            y=y_all[test_mask],
            frame=frame.loc[test_mask].reset_index(drop=True),
            feature_names=pipeline.feature_names,
        )

        try:
            model = build_model(model_name, cfg, **params).fit(fold_train)
            predictions = inverse_transform_target(model.predict(fold_test), cfg)
        except Exception as exc:  # noqa: BLE001 - a bad hyperparameter draw is not a bug
            logger.debug("Trial fold failed for %s: %s", model_name, exc)
            raise optuna.TrialPruned() from exc

        fold_metrics = compute_metrics(
            inverse_transform_target(fold_test.y, cfg),
            predictions,
            mape_epsilon=cfg.evaluation.mape_epsilon,
            alpha=cfg.evaluation.alpha,
        )
        value = fold_metrics.get(metric, np.nan)
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        scores.append(float(value))

        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if not scores:
        raise optuna.TrialPruned()
    return float(np.mean(scores))


def run(cfg: ExperimentConfig, *, prepared: PreparedData | None = None) -> dict[str, TuningResult]:
    """Run one Optuna study per configured model. Returns the best params."""
    log_section(logger, "stage 1b — hyperparameter optimisation")
    if not cfg.tuning.enabled:
        logger.info("Tuning disabled (tuning.enabled=false); using config defaults")
        return {}

    seed_everything(cfg.tuning.seed)
    prepared = prepared or load_prepared(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    direction = (
        "minimize"
        if METRIC_DIRECTION.get(cfg.tuning.metric, cfg.tuning.direction[:3]) == "min"
        else "maximize"
    )
    results: dict[str, TuningResult] = {}

    for model_name in cfg.tuning.models:
        if model_name not in SEARCH_SPACES:
            logger.warning("No search space for %s; skipping", model_name)
            continue

        logger.info(
            "Optimising %s: %d trials, %s %s, %d grouped folds",
            model_name,
            cfg.tuning.n_trials,
            direction,
            cfg.tuning.metric,
            cfg.tuning.cv_folds,
        )
        study = optuna.create_study(
            study_name=f"{cfg.tuning.study_name}_{model_name}",
            direction=direction,
            sampler=_make_sampler(cfg),
            pruner=_make_pruner(cfg),
            storage=cfg.tuning.storage,
            load_if_exists=cfg.tuning.storage is not None,
        )

        def objective(trial: optuna.Trial, _name: str = model_name) -> float:
            return _cv_score(_name, suggest_params(_name, trial), prepared, cfg, trial)

        study.optimize(
            objective,
            n_trials=cfg.tuning.n_trials,
            timeout=cfg.tuning.timeout_s,
            show_progress_bar=False,
            catch=(ValueError, RuntimeError),
        )

        completed = [t for t in study.trials if t.value is not None]
        if not completed:
            logger.warning("%s: every trial failed or was pruned", model_name)
            results[model_name] = TuningResult(model=model_name, n_trials=len(study.trials))
            continue

        results[model_name] = TuningResult(
            model=model_name,
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            trials=[
                {
                    "number": t.number,
                    "value": None if t.value is None else round(float(t.value), 6),
                    "state": str(t.state),
                    "params": t.params,
                }
                for t in study.trials
            ],
        )
        logger.info(
            "%s best %s = %.4f with %s",
            model_name,
            cfg.tuning.metric,
            study.best_value,
            study.best_params,
        )

    payload = {
        "metric": cfg.tuning.metric,
        "direction": direction,
        "cv_folds": cfg.tuning.cv_folds,
        "sampler": cfg.tuning.sampler,
        "pruner": cfg.tuning.pruner,
        "n_trials": cfg.tuning.n_trials,
        "search_spaces": describe_spaces(),
        "studies": {name: result.to_dict() for name, result in results.items()},
    }
    save_json(payload, cfg.paths.reports_dir / "tuning.json")
    logger.info("Tuning results -> %s", cfg.paths.reports_dir / "tuning.json")
    return results
