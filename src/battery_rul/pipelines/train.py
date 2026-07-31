"""Stage 2 — fit the model zoo and crown a champion.

Model selection uses the **validation** partition; the test partition is scored
once, at the end, and never influences a choice. That ordering is the whole point
of holding three partitions instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.evaluator import (
    EvaluationResult,
    compare_models,
    compare_models_common_rows,
    cross_validate_by_battery,
    evaluate_model,
)
from battery_rul.evaluation.metrics import METRIC_DIRECTION
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.target import transform_target
from battery_rul.models.base import BaseModel, TrainingData, build_model
from battery_rul.pipelines.prepare_data import PreparedData, load_prepared
from battery_rul.utils.io import environment_fingerprint, save_json, save_pickle, write_table
from battery_rul.utils.logging import get_logger, log_section
from battery_rul.utils.seed import seed_everything
from battery_rul.utils.timing import StageTimer

logger = get_logger(__name__)

__all__ = ["TrainingArtifacts", "build_partitions", "run"]

MODEL_FILENAME = "trained_model.pkl"
PIPELINE_FILENAME = "feature_pipeline.pkl"
METRICS_FILENAME = "metrics.json"


@dataclass
class TrainingArtifacts:
    """Everything stage 2 produced."""

    models: dict[str, BaseModel] = field(default_factory=dict)
    val_results: dict[str, EvaluationResult] = field(default_factory=dict)
    test_results: dict[str, EvaluationResult] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_common: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_metrics: dict[str, Any] = field(default_factory=dict)
    cv_per_fold: pd.DataFrame = field(default_factory=pd.DataFrame)
    champion: str = ""
    feature_pipeline: FeaturePipeline | None = None
    partitions: dict[str, TrainingData] = field(default_factory=dict)
    prepared: PreparedData | None = None
    failures: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def champion_model(self) -> BaseModel:
        return self.models[self.champion]

    @property
    def champion_result(self) -> EvaluationResult:
        return self.test_results[self.champion]


def build_partitions(
    prepared: PreparedData, cfg: ExperimentConfig
) -> tuple[dict[str, TrainingData], FeaturePipeline]:
    """Fit the feature pipeline on train rows and project all three partitions.

    The fit sees training rows only. Val and test are *transformed* with the
    statistics learned from train — the standard discipline, and the reason this
    lives in one place rather than being repeated per model.
    """
    frame = prepared.frame
    target = cfg.target.name
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} missing from the prepared dataset")

    masks = {
        "train": prepared.split.train,
        "val": prepared.split.val,
        "test": prepared.split.test,
    }
    features = prepared.feature_names

    y_all = transform_target(frame[target].to_numpy(dtype=float), cfg)

    pipeline = FeaturePipeline(cfg=cfg.features)
    pipeline.fit(frame.loc[masks["train"], features], y_all[masks["train"]])

    partitions: dict[str, TrainingData] = {}
    for name, mask in masks.items():
        if not mask.any():
            logger.info("Partition %s is empty; skipping", name)
            continue
        subset = frame.loc[mask].reset_index(drop=True)
        partitions[name] = TrainingData(
            X=pipeline.transform(subset[features]),
            y=y_all[mask],
            frame=subset,
            feature_names=pipeline.feature_names,
        )
        logger.info(
            "Partition %-5s: %4d rows, cells %s",
            name,
            len(partitions[name]),
            sorted(subset["battery_id"].unique().tolist()),
        )
    return partitions, pipeline


def run(
    cfg: ExperimentConfig,
    *,
    prepared: PreparedData | None = None,
    tuned_params: dict[str, dict[str, Any]] | None = None,
) -> TrainingArtifacts:
    """Fit every enabled model, select a champion, persist the artifacts."""
    log_section(logger, "stage 2 — train")
    seed_everything(cfg.seed)
    cfg.paths.ensure()
    timer = StageTimer()

    prepared = prepared or load_prepared(cfg)
    with timer("build_partitions"):
        partitions, pipeline = build_partitions(prepared, cfg)

    train = partitions["train"]
    val = partitions.get("val")
    test = partitions.get("test")
    if test is None:
        raise ValueError("No test partition; check split configuration")

    artifacts = TrainingArtifacts(
        feature_pipeline=pipeline, partitions=partitions, prepared=prepared
    )

    for name in cfg.models.enabled:
        overrides = (tuned_params or {}).get(name, {})
        try:
            with timer(f"fit:{name}"):
                model = build_model(name, cfg, **overrides)
                model.fit(train, val)
        except Exception as exc:  # noqa: BLE001 - one failing model must not sink the run
            logger.exception("Model %s failed to train: %s", name, exc)
            artifacts.failures[name] = f"{type(exc).__name__}: {exc}"
            continue

        artifacts.models[name] = model
        if val is not None and not val.is_empty:
            artifacts.val_results[name] = evaluate_model(model, val, cfg, partition="val")
        artifacts.test_results[name] = evaluate_model(model, test, cfg, partition="test")

    if not artifacts.models:
        raise RuntimeError(f"Every model failed to train: {artifacts.failures}")

    artifacts.champion = _select_champion(artifacts, cfg)
    artifacts.comparison = compare_models(
        artifacts.test_results, select_by=cfg.models.select_by, select_mode=cfg.models.select_mode
    )
    artifacts.comparison_common = compare_models_common_rows(
        artifacts.test_results,
        select_by=cfg.models.select_by,
        mape_epsilon=cfg.evaluation.mape_epsilon,
        alpha=cfg.evaluation.alpha,
    )
    # With a cohort this small the single holdout puts one cell in test, so the
    # headline number is one sample. Leave-one-battery-out pools out-of-fold
    # predictions over every cell and is the number worth quoting.
    with timer("cross_validate"):
        artifacts.cv_metrics, artifacts.cv_per_fold = cross_validate_by_battery(
            artifacts.champion,
            prepared,
            cfg,
            params=artifacts.champion_model.params,
        )

    artifacts.timings = timer.as_dict()

    _persist(artifacts, cfg)
    log_section(logger, f"champion: {artifacts.champion}")
    logger.info("\n%s", artifacts.comparison.to_string(index=False))
    return artifacts


def _select_champion(artifacts: TrainingArtifacts, cfg: ExperimentConfig) -> str:
    """Pick the best model **on validation**, falling back to test only if there
    is no validation partition (in which case the run is explicitly flagged)."""
    metric = cfg.models.select_by
    direction = METRIC_DIRECTION.get(metric, cfg.models.select_mode)

    pool = artifacts.val_results or artifacts.test_results
    if not artifacts.val_results:
        logger.warning(
            "No validation partition — the champion is being chosen on the test "
            "partition, so the reported test metric is optimistic. Configure "
            "split.val_size > 0 for an honest number."
        )

    scores = {
        name: result.metrics.get(metric, np.nan)
        for name, result in pool.items()
        if np.isfinite(result.metrics.get(metric, np.nan))
    }
    if not scores:
        return next(iter(artifacts.models))

    champion = (min if direction == "min" else max)(scores, key=scores.get)
    logger.info(
        "Champion %s selected on %s %s = %.4f",
        champion,
        "validation" if artifacts.val_results else "test",
        metric,
        scores[champion],
    )
    return champion


def _persist(artifacts: TrainingArtifacts, cfg: ExperimentConfig) -> None:
    """Write model, pipeline, metrics and predictions."""
    models_dir = cfg.paths.models_dir
    reports_dir = cfg.paths.reports_dir

    artifacts.champion_model.save(models_dir / MODEL_FILENAME)
    if artifacts.feature_pipeline is not None:
        artifacts.feature_pipeline.save(models_dir / PIPELINE_FILENAME)

    # Every model is kept too — a comparison table nobody can re-check is a claim,
    # not a result.
    for name, model in artifacts.models.items():
        save_pickle(model, models_dir / "zoo" / f"{name}.pkl")

    metrics = {
        "experiment_name": cfg.experiment_name,
        "environment": environment_fingerprint(),
        "champion": artifacts.champion,
        "selection_metric": cfg.models.select_by,
        "selected_on": "validation" if artifacts.val_results else "test",
        "config": cfg.to_dict(),
        "split": artifacts.prepared.split.to_dict() if artifacts.prepared else {},
        "feature_pipeline": (
            artifacts.feature_pipeline.fit_stats if artifacts.feature_pipeline else {}
        ),
        "models": {name: model.describe() for name, model in artifacts.models.items()},
        "validation": {n: r.to_dict() for n, r in artifacts.val_results.items()},
        "test": {n: r.to_dict() for n, r in artifacts.test_results.items()},
        "comparison": artifacts.comparison.to_dict(orient="records"),
        "comparison_common_rows": artifacts.comparison_common.to_dict(orient="records"),
        "cross_validation": {
            "scheme": "leave-one-battery-out",
            "model": artifacts.champion,
            "pooled": artifacts.cv_metrics,
            "per_fold": (
                artifacts.cv_per_fold.to_dict(orient="records")
                if not artifacts.cv_per_fold.empty
                else []
            ),
        },
        "failures": artifacts.failures,
        "timings_s": artifacts.timings,
    }
    save_json(metrics, reports_dir / METRICS_FILENAME)

    predictions = pd.concat(
        [r.predictions for r in artifacts.test_results.values()], ignore_index=True
    )
    write_table(predictions, reports_dir / "predictions_test.parquet")
    write_table(predictions, reports_dir / "predictions_test.csv")
    if artifacts.comparison is not None and not artifacts.comparison.empty:
        write_table(artifacts.comparison, reports_dir / "model_comparison.csv")
    if artifacts.cv_per_fold is not None and not artifacts.cv_per_fold.empty:
        write_table(artifacts.cv_per_fold, reports_dir / "cross_validation_by_battery.csv")
    if artifacts.comparison_common is not None and not artifacts.comparison_common.empty:
        write_table(artifacts.comparison_common, reports_dir / "model_comparison_common_rows.csv")

    logger.info("Champion -> %s", models_dir / MODEL_FILENAME)
    logger.info("Metrics  -> %s", reports_dir / METRICS_FILENAME)


def load_champion(cfg: ExperimentConfig) -> tuple[BaseModel, FeaturePipeline]:
    """Load the persisted champion model and its feature pipeline."""
    model = BaseModel.load(cfg.paths.models_dir / MODEL_FILENAME)
    pipeline = FeaturePipeline.load(cfg.paths.models_dir / PIPELINE_FILENAME)
    return model, pipeline
