"""Stage 3 — figures, explanations and the evaluation report.

Reads what stage 2 wrote (or accepts it in memory) and produces the human-facing
deliverables: figures, SHAP/permutation explanations, and
``reports/evaluation_report.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.evaluator import learning_curve
from battery_rul.evaluation.reporting import render_evaluation_report
from battery_rul.explainability.explain import ExplanationResult, explain_model
from battery_rul.models.base import build_model
from battery_rul.pipelines.train import TrainingArtifacts
from battery_rul.utils.io import load_json, save_json, write_table
from battery_rul.utils.logging import get_logger, log_section
from battery_rul.utils.timing import StageTimer
from battery_rul.visualization.eda import generate_eda_figures
from battery_rul.visualization.results import (
    plot_learning_curve,
    plot_model_comparison,
    plot_predictions,
    plot_residual_analysis,
    plot_training_history,
)

logger = get_logger(__name__)

__all__ = ["EvaluationArtifacts", "run"]


@dataclass
class EvaluationArtifacts:
    figures: list[Path] = field(default_factory=list)
    explanation: ExplanationResult | None = None
    learning_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    report_path: Path | None = None


def run(
    cfg: ExperimentConfig,
    artifacts: TrainingArtifacts,
    *,
    tuning_info: dict[str, Any] | None = None,
    with_eda: bool = True,
    with_learning_curve: bool = True,
) -> EvaluationArtifacts:
    """Produce every figure and the markdown evaluation report."""
    log_section(logger, "stage 3 — evaluate & explain")
    cfg.paths.ensure()
    timer = StageTimer()
    out = EvaluationArtifacts()

    prepared = artifacts.prepared
    if prepared is None:
        raise ValueError("TrainingArtifacts.prepared is required for evaluation")

    # -- EDA ---------------------------------------------------------------
    if with_eda:
        with timer("eda_figures"):
            out.figures.extend(generate_eda_figures(prepared.cycles, cfg))

    # -- result figures ------------------------------------------------------
    with timer("result_figures"):
        for name, result in artifacts.test_results.items():
            out.figures.extend(plot_predictions(result, cfg))
            residual = plot_residual_analysis(result, cfg)
            if residual:
                out.figures.append(residual)
            model = artifacts.models.get(name)
            if model is not None and model.train_history:
                history = plot_training_history(model.train_history, name, cfg)
                if history:
                    out.figures.append(history)
        comparison = plot_model_comparison(artifacts.comparison, cfg)
        if comparison:
            out.figures.append(comparison)

    # -- learning curve -------------------------------------------------------
    if with_learning_curve and artifacts.champion:
        with timer("learning_curve"):
            champion_params = artifacts.champion_model.params
            out.learning_curve = learning_curve(
                lambda: build_model(artifacts.champion, cfg, **champion_params),
                artifacts.partitions["train"],
                artifacts.partitions["test"],
                cfg,
            )
            if not out.learning_curve.empty:
                write_table(out.learning_curve, cfg.paths.reports_dir / "learning_curve.csv")
                path = plot_learning_curve(out.learning_curve, artifacts.champion, cfg)
                if path:
                    out.figures.append(path)

    # -- explainability --------------------------------------------------------
    if cfg.explainability.enabled and artifacts.champion:
        with timer("explainability"):
            out.explanation = explain_model(
                artifacts.champion_model,
                artifacts.partitions["train"],
                artifacts.partitions["test"],
                cfg,
                predictions=artifacts.champion_result.predictions,
            )
            out.figures.extend(out.explanation.figures)
            save_json(out.explanation.to_dict(), cfg.paths.reports_dir / "explainability.json")
            if not out.explanation.permutation_importance.empty:
                write_table(
                    out.explanation.permutation_importance,
                    cfg.paths.reports_dir / "permutation_importance.csv",
                )

    # -- report -----------------------------------------------------------------
    with timer("render_report"):
        manifest = prepared.manifest or {}
        feature_info = dict(manifest.get("features", {}))
        if artifacts.feature_pipeline is not None:
            feature_info["n_selected"] = len(artifacts.feature_pipeline.feature_names)

        out.report_path = render_evaluation_report(
            cfg=cfg,
            comparison=artifacts.comparison,
            comparison_common=artifacts.comparison_common,
            results=artifacts.test_results,
            champion=artifacts.champion,
            dataset_summary=pd.DataFrame(manifest.get("battery_summary", [])),
            split_info=prepared.split.to_dict(),
            target_info=manifest.get("target", {}),
            feature_info=feature_info,
            validation_info=manifest.get("validation", {}),
            tuning_info=tuning_info,
            learning_curves=out.learning_curve,
            timings={**artifacts.timings, **timer.as_dict()},
            figures=[str(p.relative_to(cfg.paths.root)) for p in out.figures],
        )

    logger.info("Stage 3 complete: %d figures, report at %s", len(out.figures), out.report_path)
    return out


def run_from_disk(cfg: ExperimentConfig) -> EvaluationArtifacts:
    """Re-render figures and the report from persisted artifacts.

    Used by ``scripts/evaluate.py`` so the report can be regenerated without
    re-training — the models in ``models/zoo`` are reloaded and re-scored.
    """
    from battery_rul.evaluation.evaluator import (
        compare_models,
        compare_models_common_rows,
        evaluate_model,
    )
    from battery_rul.features.pipeline import FeaturePipeline
    from battery_rul.models.base import BaseModel
    from battery_rul.pipelines.prepare_data import load_prepared
    from battery_rul.pipelines.train import build_partitions

    prepared = load_prepared(cfg)
    partitions, _ = build_partitions(prepared, cfg)
    pipeline = FeaturePipeline.load(cfg.paths.models_dir / "feature_pipeline.pkl")

    zoo_dir = cfg.paths.models_dir / "zoo"
    if not zoo_dir.is_dir():
        raise FileNotFoundError(f"{zoo_dir} not found. Run the training pipeline first.")

    artifacts = TrainingArtifacts(
        feature_pipeline=pipeline, partitions=partitions, prepared=prepared
    )
    for path in sorted(zoo_dir.glob("*.pkl")):
        model = BaseModel.load(path)
        artifacts.models[model.name] = model
        if "val" in partitions:
            artifacts.val_results[model.name] = evaluate_model(
                model, partitions["val"], cfg, partition="val"
            )
        artifacts.test_results[model.name] = evaluate_model(
            model, partitions["test"], cfg, partition="test"
        )

    metrics_path = cfg.paths.reports_dir / "metrics.json"
    if metrics_path.is_file():
        artifacts.champion = str(load_json(metrics_path).get("champion", ""))
    if artifacts.champion not in artifacts.models:
        artifacts.champion = next(iter(artifacts.models))

    artifacts.comparison = compare_models(
        artifacts.test_results, select_by=cfg.models.select_by, select_mode=cfg.models.select_mode
    )
    artifacts.comparison_common = compare_models_common_rows(
        artifacts.test_results,
        select_by=cfg.models.select_by,
        mape_epsilon=cfg.evaluation.mape_epsilon,
        alpha=cfg.evaluation.alpha,
    )
    return run(cfg, artifacts)
