"""End-to-end orchestration: prepare -> (tune) -> train -> evaluate -> predict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from battery_rul.config import ExperimentConfig
from battery_rul.pipelines import evaluate as evaluate_stage
from battery_rul.pipelines import predict as predict_stage
from battery_rul.pipelines import prepare_data as prepare_stage
from battery_rul.pipelines import train as train_stage
from battery_rul.pipelines import tune as tune_stage
from battery_rul.utils.io import save_json
from battery_rul.utils.logging import get_logger, log_section, setup_logging
from battery_rul.utils.seed import seed_everything
from battery_rul.utils.timing import StageTimer

logger = get_logger(__name__)

__all__ = ["PipelineResult", "run"]


@dataclass
class PipelineResult:
    prepared: prepare_stage.PreparedData
    training: train_stage.TrainingArtifacts
    evaluation: evaluate_stage.EvaluationArtifacts
    tuning: dict[str, tune_stage.TuningResult]
    prediction: predict_stage.PredictionResult | None
    timings: dict[str, float]

    def summary(self) -> dict[str, Any]:
        champion = self.training.champion
        metrics = self.training.test_results[champion].metrics
        return {
            "champion": champion,
            "test_mae": round(metrics["mae"], 3),
            "test_rmse": round(metrics["rmse"], 3),
            "test_r2": round(metrics["r2"], 4),
            "test_mape": round(metrics["mape"], 3),
            "n_models_trained": len(self.training.models),
            "n_figures": len(self.evaluation.figures),
            "report": str(self.evaluation.report_path),
            "total_seconds": round(sum(self.timings.values()), 1),
        }


def run(
    cfg: ExperimentConfig,
    *,
    skip_tuning: bool = False,
    skip_eda: bool = False,
    skip_predict: bool = False,
    verify_leakage: bool = True,
) -> PipelineResult:
    """Run every stage in order and write a run summary."""
    cfg.paths.ensure()
    setup_logging(log_file=cfg.paths.reports_dir / "pipeline.log")
    seed_everything(cfg.seed)
    timer = StageTimer()

    log_section(logger, f"battery rul pipeline — {cfg.experiment_name}")
    logger.info("seed=%d  source=%s  split=%s", cfg.seed, cfg.data.source, cfg.split.strategy)

    with timer("stage1_prepare"):
        prepared = prepare_stage.run(cfg, verify_leakage=verify_leakage)

    tuning: dict[str, tune_stage.TuningResult] = {}
    tuned_params: dict[str, dict[str, Any]] = {}
    if cfg.tuning.enabled and not skip_tuning:
        with timer("stage1b_tune"):
            tuning = tune_stage.run(cfg, prepared=prepared)
            tuned_params = {
                name: result.best_params for name, result in tuning.items() if result.best_params
            }

    with timer("stage2_train"):
        training = train_stage.run(cfg, prepared=prepared, tuned_params=tuned_params)

    with timer("stage3_evaluate"):
        evaluation = evaluate_stage.run(
            cfg,
            training,
            tuning_info={name: r.to_dict() for name, r in tuning.items()} if tuning else None,
            with_eda=not skip_eda,
        )

    prediction = None
    if not skip_predict:
        with timer("stage4_predict"):
            try:
                prediction = predict_stage.run(cfg)
            except Exception as exc:  # noqa: BLE001 - demo stage, never fatal
                logger.warning("Prediction demo failed: %s", exc)

    result = PipelineResult(
        prepared=prepared,
        training=training,
        evaluation=evaluation,
        tuning=tuning,
        prediction=prediction,
        timings=timer.as_dict(),
    )

    save_json(result.summary(), cfg.paths.reports_dir / "run_summary.json")
    log_section(logger, "pipeline complete")
    for key, value in result.summary().items():
        logger.info("%-20s %s", key, value)
    return result
