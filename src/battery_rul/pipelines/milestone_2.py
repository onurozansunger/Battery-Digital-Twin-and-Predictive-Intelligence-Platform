"""Milestone 2 pipelines: multi-task data, SOH, risk, calibration, bundles.

Each stage is a function with a CLI entry point. They are separate commands
because they have genuinely different costs — retraining the multi-task encoder
is minutes, refitting a calibrator is seconds — and an orchestration command that
always did everything would train people to skip it.

Partition discipline
--------------------
The Milestone 1 battery-holdout split is reused unchanged: the same cells are in
train, validation and test as in Milestone 1, so the two milestones' numbers
describe the same partition.

Calibration — both the risk probability calibrator and the conformal interval
estimator — is fitted on **out-of-fold predictions over the non-test cells**,
produced by leave-one-battery-out within train+validation. Two reasons for that
rather than a fixed held-out slice:

* the cohort is five cells, so a dedicated calibration cell would cost 20 % of
  the training data and still give one cell's worth of residuals;
* out-of-fold predictions are honest — each row is scored by a model that never
  saw its cell — so the residuals are the ones the deployed model would make.

Test cells are excluded from every calibration fit, every threshold search and
every selection decision. They are scored once, at the end.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.calibration.probability import (
    ProbabilityCalibrator,
    reliability_curve,
    risk_metrics,
)
from battery_rul.config import ExperimentConfig, load_config
from battery_rul.evaluation.metrics import compute_metrics
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.target import inverse_transform_target, transform_target
from battery_rul.features.warmup import WarmupPolicy
from battery_rul.models.base import TrainingData, build_model
from battery_rul.models.bundle import save_bundle
from battery_rul.pipelines.prepare_data import PreparedData, load_prepared
from battery_rul.targets.risk import attach_failure_risk_target
from battery_rul.targets.soh import attach_soh_target
from battery_rul.uncertainty.conformal import ConformalIntervalEstimator, coverage_report
from battery_rul.utils.io import environment_fingerprint, save_json, write_table
from battery_rul.utils.logging import get_logger, log_section, setup_logging
from battery_rul.utils.seed import seed_everything
from battery_rul.utils.timing import StageTimer

logger = get_logger(__name__)

__all__ = [
    "MultiTaskData",
    "build_bundles",
    "calibrate_risk",
    "calibrate_uncertainty",
    "evaluate_milestone_2",
    "main",
    "prepare_multitask_data",
    "run_milestone_2",
    "train_multitask",
    "train_risk",
    "train_soh",
]

MULTITASK_DATASET = "multitask_dataset.parquet"


# ---------------------------------------------------------------------------
# Stage 1 — targets
# ---------------------------------------------------------------------------
@dataclass
class MultiTaskData:
    """The Milestone 1 feature frame with SOH and risk targets attached."""

    frame: pd.DataFrame
    feature_names: list[str]
    split: Any
    reports: dict[str, Any] = field(default_factory=dict)

    def mask(self, name: str) -> np.ndarray:
        return self.frame["split"].to_numpy() == name

    @property
    def non_test(self) -> np.ndarray:
        return ~self.mask("test")

    def cells(self, mask: np.ndarray) -> list[str]:
        return sorted(self.frame.loc[mask, "battery_id"].unique().tolist())


def prepare_multitask_data(
    cfg: ExperimentConfig, *, prepared: PreparedData | None = None
) -> MultiTaskData:
    """Attach the SOH and risk targets to the Milestone 1 modelling dataset.

    The SOH reference is computed on the **full** cycle record (``cycles.parquet``),
    not the warm-up-trimmed feature frame: the reference is by definition the
    cell's opening capacity, and trimming the opening cycles before computing it
    would silently change what SOH means.
    """
    log_section(logger, "milestone 2 — prepare multi-task data")
    prepared = prepared or load_prepared(cfg)

    labelled, soh_report = attach_soh_target(prepared.cycles, cfg)
    labelled, risk_report = attach_failure_risk_target(labelled, cfg)

    target_columns = [
        cfg.soh.target_name,
        "soh_reference_capacity_ah",
        "soh_health_class",
        cfg.risk.target_name,
        *[c for c in labelled.columns if c.startswith(f"{cfg.risk.target_name}_")],
    ]
    target_columns = list(dict.fromkeys(target_columns))
    merged = prepared.frame.merge(
        labelled[["battery_id", "cycle_index", *target_columns]],
        on=["battery_id", "cycle_index"],
        how="left",
        validate="one_to_one",
    )

    missing = merged[cfg.soh.target_name].isna().sum()
    if missing:
        logger.warning("%d feature rows have no SOH target after the merge", int(missing))

    outpath = cfg.paths.processed_dir / MULTITASK_DATASET
    write_table(merged, outpath)
    logger.info("Multi-task dataset -> %s (%d rows)", outpath, len(merged))

    return MultiTaskData(
        frame=merged,
        feature_names=list(prepared.feature_names),
        split=prepared.split,
        reports={"soh": soh_report.to_dict(), "risk": risk_report.to_dict()},
    )


# ---------------------------------------------------------------------------
# Shared fold machinery
# ---------------------------------------------------------------------------
def _fit_pipeline(
    data: MultiTaskData, cfg: ExperimentConfig, mask: np.ndarray, y: np.ndarray
) -> FeaturePipeline:
    pipeline = FeaturePipeline(cfg=cfg.features)
    pipeline.fit(data.frame.loc[mask, data.feature_names], y[mask])
    return pipeline


def _training_data(
    data: MultiTaskData, pipeline: FeaturePipeline, mask: np.ndarray, y: np.ndarray
) -> TrainingData:
    subset = data.frame.loc[mask].reset_index(drop=True)
    return TrainingData(
        X=pipeline.transform(subset[data.feature_names]),
        y=y[mask],
        frame=subset,
        feature_names=pipeline.feature_names,
    )


def _out_of_fold(
    data: MultiTaskData,
    cfg: ExperimentConfig,
    *,
    y: np.ndarray,
    model_name: str,
    predict: str = "regression",
) -> pd.DataFrame:
    """Leave-one-battery-out out-of-fold predictions over the non-test cells.

    Each row is predicted by a model that never saw its cell, and every
    preprocessing artifact is re-fitted inside the fold. These are the rows the
    calibrators are fitted on.
    """
    frame = data.frame
    battery_col = frame["battery_id"].to_numpy()
    pool = data.non_test
    cells = sorted(pd.unique(battery_col[pool]).tolist())
    rows: list[pd.DataFrame] = []

    for cell in cells:
        test_mask = pool & (battery_col == cell)
        train_mask = pool & (battery_col != cell)
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue
        valid = np.isfinite(y)
        pipeline = _fit_pipeline(data, cfg, train_mask & valid, y)
        train = _training_data(data, pipeline, train_mask & valid, y)
        test = _training_data(data, pipeline, test_mask, y)

        model = build_model(model_name, cfg)
        if predict == "classification":
            estimator = _fit_classifier(train, cfg)
            probabilities = estimator.predict_proba(test.X)[:, 1]
            prediction = probabilities
        else:
            model.fit(train)
            prediction = np.asarray(model.predict(test), dtype=float)

        rows.append(
            pd.DataFrame(
                {
                    "battery_id": test.battery_ids,
                    "cycle_index": test.cycle_index,
                    "y_true": test.y,
                    "y_pred": prediction,
                    # The conformal stage variable: measured state of health,
                    # which unlike life fraction is available at serving time.
                    "stage_value": _stage_value(test.frame, cfg),
                    "fold": str(cell),
                }
            )
        )

    if not rows:
        raise RuntimeError(
            f"Out-of-fold prediction produced nothing for {model_name}; the non-test "
            "cohort is too small to leave a cell out."
        )
    return pd.concat(rows, ignore_index=True)


def _fit_classifier(train: TrainingData, cfg: ExperimentConfig) -> Any:
    """Fit a scikit-learn-style classifier for the risk task.

    The registry's regressors are the wrong estimator class here, so the risk
    task builds its classifier directly rather than pretending a regressor is
    one. Class weighting handles the imbalance; no row is duplicated.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    name = cfg.risk.model
    weight = "balanced" if cfg.risk.class_weight_balanced else None
    labels = np.asarray(train.y, dtype=float)
    good = np.isfinite(labels)

    if name == "logistic_regression":
        estimator: Any = LogisticRegression(
            max_iter=5000, class_weight=weight, random_state=cfg.seed
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight=weight,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier

        positives = float(np.sum(labels[good] == 1.0))
        negatives = float(np.sum(labels[good] == 0.0))
        estimator = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=cfg.seed,
            scale_pos_weight=(negatives / positives) if positives else 1.0,
        )
    else:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=400,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=10,
            class_weight=weight,
            random_state=cfg.seed,
            verbose=-1,
        )

    estimator.fit(train.X[good], labels[good].astype(int))
    return estimator


# ---------------------------------------------------------------------------
# Stage 2 — SOH model
# ---------------------------------------------------------------------------
def train_soh(cfg: ExperimentConfig, data: MultiTaskData | None = None) -> dict[str, Any]:
    """Fit and evaluate the independent SOH regressor, then bundle it."""
    log_section(logger, "milestone 2 — train SOH")
    seed_everything(cfg.seed)
    data = data or prepare_multitask_data(cfg)

    y = data.frame[cfg.soh.target_name].to_numpy(dtype=float)
    valid = np.isfinite(y)
    train_mask = data.mask("train") & valid
    val_mask = data.mask("val") & valid
    test_mask = data.mask("test") & valid

    candidates = ["elastic_net", "random_forest", "lightgbm"]
    pipeline = _fit_pipeline(data, cfg, train_mask, y)
    train = _training_data(data, pipeline, train_mask, y)
    validation = _training_data(data, pipeline, val_mask, y) if val_mask.any() else None

    scores: dict[str, float] = {}
    models: dict[str, Any] = {}
    for name in candidates:
        model = build_model(name, cfg).fit(train)
        models[name] = model
        if validation is not None:
            predicted = np.asarray(model.predict(validation), dtype=float)
            scores[name] = compute_metrics(validation.y, predicted)["mae"]
    selected = min(scores, key=lambda k: scores[k]) if scores else candidates[0]
    logger.info("SOH model selected on validation: %s (MAE %s)", selected, scores)

    model = models[selected]
    out_of_fold = _out_of_fold(data, cfg, y=y, model_name=selected)
    oof_metrics = compute_metrics(
        out_of_fold["y_true"].to_numpy(), out_of_fold["y_pred"].to_numpy()
    )
    oof_metrics["max_absolute_error"] = float(
        np.nanmax(np.abs(out_of_fold["y_true"] - out_of_fold["y_pred"]))
    )

    test_metrics: dict[str, float] = {}
    per_battery: list[dict[str, Any]] = []
    if test_mask.any():
        test = _training_data(data, pipeline, test_mask, y)
        predicted = np.asarray(model.predict(test), dtype=float)
        test_metrics = compute_metrics(test.y, predicted)
        frame = pd.DataFrame(
            {"battery_id": test.battery_ids, "y_true": test.y, "y_pred": predicted}
        )
        for cell, group in frame.groupby("battery_id"):
            metrics = compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            per_battery.append({"battery_id": str(cell), "partition": "test", **metrics})

    for cell, group in out_of_fold.groupby("battery_id"):
        metrics = compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        per_battery.append({"battery_id": str(cell), "partition": "out_of_fold", **metrics})

    save_bundle(
        cfg.artifacts.soh_dir,
        model=model,
        preprocessing=pipeline,
        cfg=cfg,
        task="soh_regression",
        model_name=selected,
        dataset_fingerprint=cfg.data_fingerprint(),
        metrics={"out_of_fold": oof_metrics, "test": test_metrics},
        thresholds={
            "healthy_min": cfg.soh.healthy_min,
            "slightly_degraded_min": cfg.soh.slightly_degraded_min,
            "warning_min": cfg.soh.warning_min,
            "feature_reference_ranges": _reference_ranges(data, train_mask),
        },
        first_scoreable_cycle=WarmupPolicy(
            cfg.features.drop_warmup_cycles, None
        ).first_scoreable_cycle,
        warmup_policy=WarmupPolicy(cfg.features.drop_warmup_cycles, None).to_dict(),
        notes="SOH is a fraction in [0, 1]. Reference strategy is in target_definition.",
    )

    result = {
        "selected_model": selected,
        "validation_selection_mae": scores,
        "out_of_fold_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "per_battery": per_battery,
        "n_train_rows": int(train_mask.sum()),
        "representation": "fraction in [0, 1]",
    }
    write_table(out_of_fold, cfg.paths.reports_dir / "milestone_2" / "soh_out_of_fold.csv")
    return result


def _reference_ranges(data: MultiTaskData, train_mask: np.ndarray) -> dict[str, list[float]]:
    """Per-feature training min/max, for the serving out-of-distribution flag."""
    subset = data.frame.loc[train_mask, data.feature_names]
    lows = subset.min(numeric_only=True)
    highs = subset.max(numeric_only=True)
    return {
        str(name): [float(lows[name]), float(highs[name])]
        for name in lows.index
        if np.isfinite(lows[name]) and np.isfinite(highs[name])
    }


# ---------------------------------------------------------------------------
# Stage 3 — risk model + calibration
# ---------------------------------------------------------------------------
class RiskClassifierAdapter:
    """Adapts a scikit-learn classifier to the bundle's ``predict`` interface.

    Module level, not a closure: a class defined inside a function cannot be
    pickled, and the bundle has to be loadable in a fresh process.
    """

    def __init__(self, estimator: Any, *, is_tree: bool = False) -> None:
        self.estimator = estimator
        self.is_tree = is_tree

    def predict(self, data: TrainingData) -> np.ndarray:
        """Positive-class probability, row-aligned with ``data``."""
        return np.asarray(self.estimator.predict_proba(data.X), dtype=float)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X), dtype=float)


def train_risk(cfg: ExperimentConfig, data: MultiTaskData | None = None) -> dict[str, Any]:
    """Fit the failure-risk classifier, calibrate it, tune its threshold, bundle."""
    log_section(logger, "milestone 2 — train failure risk")
    seed_everything(cfg.seed)
    data = data or prepare_multitask_data(cfg)

    y = pd.to_numeric(data.frame[cfg.risk.target_name], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    train_mask = data.mask("train") & valid
    test_mask = data.mask("test") & valid

    pipeline = _fit_pipeline(data, cfg, train_mask, y)
    train = _training_data(data, pipeline, train_mask, y)
    estimator = _fit_classifier(train, cfg)

    # --- calibration on out-of-fold, non-test rows only ---------------------
    out_of_fold = _out_of_fold(data, cfg, y=y, model_name=cfg.risk.model, predict="classification")
    calibrator = ProbabilityCalibrator(cfg=cfg.calibration)
    calibrator.fit(
        out_of_fold["y_true"].to_numpy(),
        out_of_fold["y_pred"].to_numpy(),
        tune=cfg.risk.threshold is None,
        objective=cfg.risk.threshold_objective,
        min_recall=cfg.risk.min_recall,
    )
    threshold = float(
        cfg.risk.threshold if cfg.risk.threshold is not None else calibrator.threshold
    )

    calibrated_oof = calibrator.transform(out_of_fold["y_pred"].to_numpy())
    oof_cycles = out_of_fold["cycle_index"].to_numpy()
    oof_metrics_raw = risk_metrics(
        out_of_fold["y_true"].to_numpy(),
        out_of_fold["y_pred"].to_numpy(),
        threshold=threshold,
        n_bins=cfg.calibration.n_bins,
        cycle_index=oof_cycles,
    )
    oof_metrics_calibrated = risk_metrics(
        out_of_fold["y_true"].to_numpy(),
        calibrated_oof,
        threshold=threshold,
        n_bins=cfg.calibration.n_bins,
        cycle_index=oof_cycles,
    )
    # The calibrator was fitted on these very rows, so its post-calibration Brier
    # and ECE here are in-sample and optimistic by construction — ECE will often
    # be exactly zero. The honest calibration evidence is the test row below.
    oof_metrics_calibrated["calibration_is_in_sample"] = True
    oof_metrics_calibrated["note"] = (
        "Calibrated on these rows; Brier and ECE here are in-sample. Read the "
        "test-partition figures for out-of-sample calibration quality."
    )

    test_metrics: dict[str, float] = {}
    if test_mask.any():
        test = _training_data(data, pipeline, test_mask, y)
        raw = estimator.predict_proba(test.X)[:, 1]
        test_metrics = risk_metrics(
            test.y,
            calibrator.transform(raw),
            threshold=threshold,
            n_bins=cfg.calibration.n_bins,
            cycle_index=test.cycle_index,
        )
        test_metrics["n_test_cells"] = int(len(np.unique(test.battery_ids)))

    curve = reliability_curve(
        out_of_fold["y_true"].to_numpy(), calibrated_oof, n_bins=cfg.calibration.n_bins
    )

    save_bundle(
        cfg.artifacts.risk_dir,
        model=RiskClassifierAdapter(
            estimator, is_tree=cfg.risk.model in {"random_forest", "lightgbm", "xgboost"}
        ),
        preprocessing=pipeline,
        cfg=cfg,
        task="failure_risk_classification",
        model_name=cfg.risk.model,
        dataset_fingerprint=cfg.data_fingerprint(),
        metrics={
            "out_of_fold_raw": oof_metrics_raw,
            "out_of_fold_calibrated": oof_metrics_calibrated,
            "test_calibrated": test_metrics,
        },
        thresholds={
            "decision_threshold": threshold,
            "objective": cfg.risk.threshold_objective,
            "tuned_on": "out-of-fold predictions over non-test cells",
            "risk_bands": {
                "low_max": cfg.risk.low_max,
                "medium_max": cfg.risk.medium_max,
                "high_max": cfg.risk.high_max,
            },
            "feature_reference_ranges": _reference_ranges(data, train_mask),
        },
        calibrator=calibrator,
        first_scoreable_cycle=WarmupPolicy(
            cfg.features.drop_warmup_cycles, None
        ).first_scoreable_cycle,
        warmup_policy=WarmupPolicy(cfg.features.drop_warmup_cycles, None).to_dict(),
        notes=(
            "Derived label: projected end-of-life crossing within the horizon. "
            "Not an observed safety failure."
        ),
    )

    reports = cfg.paths.reports_dir / "milestone_2"
    write_table(
        out_of_fold.assign(y_pred_calibrated=calibrated_oof), reports / "risk_out_of_fold.csv"
    )
    write_table(curve, reports / "risk_reliability_curve.csv")

    return {
        "model": cfg.risk.model,
        "horizon_cycles": cfg.risk.horizon_cycles,
        "calibration": calibrator.to_dict(),
        "threshold": threshold,
        "out_of_fold_raw": oof_metrics_raw,
        "out_of_fold_calibrated": oof_metrics_calibrated,
        "test_calibrated": test_metrics,
        "reliability_curve": curve.to_dict(orient="records"),
    }


def calibrate_risk(cfg: ExperimentConfig) -> dict[str, Any]:
    """Refit only the probability calibrator against the existing risk bundle."""
    return train_risk(cfg)


# ---------------------------------------------------------------------------
# Stage 4 — RUL bundle + conformal uncertainty
# ---------------------------------------------------------------------------
def calibrate_uncertainty(
    cfg: ExperimentConfig, data: MultiTaskData | None = None
) -> dict[str, Any]:
    """Fit the split-conformal estimator for RUL and bundle the RUL model."""
    log_section(logger, "milestone 2 — conformal uncertainty for RUL")
    seed_everything(cfg.seed)
    data = data or prepare_multitask_data(cfg)

    y = transform_target(data.frame[cfg.target.name].to_numpy(dtype=float), cfg)
    valid = np.isfinite(y)
    train_mask = data.mask("train") & valid
    test_mask = data.mask("test") & valid

    model_name = _preferred_rul_model(cfg)
    pipeline = _fit_pipeline(data, cfg, train_mask, y)
    train = _training_data(data, pipeline, train_mask, y)
    model = build_model(model_name, cfg).fit(train)

    out_of_fold = _out_of_fold(data, cfg, y=y, model_name=model_name)
    truth = inverse_transform_target(out_of_fold["y_true"].to_numpy(), cfg)
    predicted = inverse_transform_target(out_of_fold["y_pred"].to_numpy(), cfg)

    stage_values = out_of_fold["stage_value"].to_numpy()
    estimator = ConformalIntervalEstimator(cfg=cfg.uncertainty, target_name=cfg.target.name)
    estimator.fit(truth, predicted, life_fraction=stage_values)

    lower, upper = estimator.intervals(predicted, life_fraction=stage_values)
    stages = estimator.life_stages(stage_values)
    oof_frame = out_of_fold.assign(
        y_true=truth, y_pred=predicted, lower_bound=lower, upper_bound=upper, life_stage=stages
    )
    oof_coverage = coverage_report(oof_frame)

    test_coverage: dict[str, Any] = {}
    if test_mask.any():
        test = _training_data(data, pipeline, test_mask, y)
        test_pred = inverse_transform_target(np.asarray(model.predict(test), dtype=float), cfg)
        test_truth = inverse_transform_target(test.y, cfg)
        life = _stage_value(test.frame, cfg)
        t_lower, t_upper = estimator.intervals(test_pred, life_fraction=life)
        test_frame = pd.DataFrame(
            {
                "battery_id": test.battery_ids,
                "cycle_index": test.cycle_index,
                "y_true": test_truth,
                "y_pred": test_pred,
                "lower_bound": t_lower,
                "upper_bound": t_upper,
                "life_stage": estimator.life_stages(life),
            }
        ).dropna(subset=["y_pred"])
        test_coverage = coverage_report(test_frame)
        write_table(test_frame, cfg.paths.reports_dir / "milestone_2" / "rul_test_intervals.csv")

    save_bundle(
        cfg.artifacts.rul_dir,
        model=model,
        preprocessing=pipeline,
        cfg=cfg,
        task="rul_regression",
        model_name=model_name,
        dataset_fingerprint=cfg.data_fingerprint(),
        metrics={
            "out_of_fold": compute_metrics(truth, predicted),
            "out_of_fold_coverage": oof_coverage,
            "test_coverage": test_coverage,
        },
        thresholds={"feature_reference_ranges": _reference_ranges(data, train_mask)},
        uncertainty=estimator,
        first_scoreable_cycle=WarmupPolicy(
            cfg.features.drop_warmup_cycles, None
        ).first_scoreable_cycle,
        warmup_policy=WarmupPolicy(cfg.features.drop_warmup_cycles, None).to_dict(),
        notes="Intervals are prediction intervals from split conformal calibration.",
    )

    write_table(oof_frame, cfg.paths.reports_dir / "milestone_2" / "rul_out_of_fold_intervals.csv")
    return {
        "model": model_name,
        "estimator": estimator.to_dict(),
        "out_of_fold_coverage": oof_coverage,
        "test_coverage": test_coverage,
    }


def _stage_value(frame: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    """The conformal conditioning variable: measured state of health.

    Falls back through the columns that carry it. Returns NaN where none is
    present, which the estimator maps to the global quantile rather than to a
    guessed bucket.
    """
    for column in (cfg.soh.target_name, "soh"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), np.nan)


def _preferred_rul_model(cfg: ExperimentConfig) -> str:
    """Pick the RUL family to deploy from the nested comparison, if one exists.

    Falls back to the configured first enabled tabular model. The choice is
    recorded in the bundle metadata either way, so a reader can see whether it
    came from evidence or from a default.
    """
    metrics_path = cfg.paths.reports_dir / "metrics.json"
    if metrics_path.is_file():
        from battery_rul.utils.io import load_json

        payload = load_json(metrics_path)
        frequency = payload.get("nested_evaluation", {}).get("selection_frequency", {})
        tabular = {
            name: count
            for name, count in frequency.items()
            if name not in {"lstm", "gru", "transformer"}
        }
        if tabular:
            winner = max(tabular, key=lambda k: tabular[k])
            logger.info(
                "Deploying RUL family %s: selected by the inner loop in %d of the "
                "nested outer folds.",
                winner,
                tabular[winner],
            )
            return winner
    fallback = next(
        (m for m in cfg.models.enabled if m not in {"lstm", "gru", "transformer"}), "ridge"
    )
    logger.warning(
        "No nested selection frequencies available; deploying the configured "
        "fallback family %r. Run the training pipeline to make this evidence-based.",
        fallback,
    )
    return fallback


# ---------------------------------------------------------------------------
# Stage 5 — multi-task model
# ---------------------------------------------------------------------------
def train_multitask(cfg: ExperimentConfig, data: MultiTaskData | None = None) -> dict[str, Any]:
    """Fit the shared-encoder multi-task model and evaluate all three heads."""
    log_section(logger, "milestone 2 — train multi-task model")
    if not cfg.multitask.enabled:
        logger.info("multitask.enabled is false; skipping")
        return {"skipped": True}

    from battery_rul.models.multitask import MultiTaskSequenceModel

    seed_everything(cfg.multitask.seed)
    data = data or prepare_multitask_data(cfg)

    y_rul = transform_target(data.frame[cfg.target.name].to_numpy(dtype=float), cfg)
    train_mask = data.mask("train")
    val_mask = data.mask("val")
    test_mask = data.mask("test")

    pipeline = _fit_pipeline(data, cfg, train_mask, y_rul)
    train_frame = data.frame.loc[train_mask].reset_index(drop=True)
    train_values = pipeline.transform(train_frame[data.feature_names])
    val_frame = data.frame.loc[val_mask].reset_index(drop=True) if val_mask.any() else None
    val_values = (
        pipeline.transform(val_frame[data.feature_names]) if val_frame is not None else None
    )

    model = MultiTaskSequenceModel(cfg=cfg)
    model.fit(
        train_frame,
        train_values,
        val_frame=val_frame,
        val_values=val_values,
        feature_names=list(pipeline.feature_names),
    )

    results: dict[str, Any] = {"fit": model.fit_metadata, "partitions": {}}
    for name, mask in (("val", val_mask), ("test", test_mask)):
        if not mask.any():
            continue
        frame = data.frame.loc[mask].reset_index(drop=True)
        values = pipeline.transform(frame[data.feature_names])
        prediction = model.predict(frame, values)
        scored = prediction.scoreable
        n_scored = int(scored.sum())
        entry: dict[str, Any] = {
            "n_rows": int(len(frame)),
            "n_scored": n_scored,
            "n_unscored": int(len(frame) - n_scored),
            "coverage": round(n_scored / max(len(frame), 1), 4),
        }
        if n_scored:
            entry["rul"] = compute_metrics(
                frame[cfg.target.name].to_numpy(dtype=float)[scored], prediction.rul[scored]
            )
            entry["soh"] = compute_metrics(
                frame[cfg.soh.target_name].to_numpy(dtype=float)[scored], prediction.soh[scored]
            )
            risk_truth = pd.to_numeric(frame[cfg.risk.target_name], errors="coerce").to_numpy(
                dtype=float
            )[scored]
            entry["risk"] = risk_metrics(
                risk_truth,
                prediction.risk_probability[scored],
                threshold=0.5,
                n_bins=cfg.calibration.n_bins,
                cycle_index=frame["cycle_index"].to_numpy()[scored],
            )
            entry["n_cells"] = int(frame["battery_id"].nunique())
        results["partitions"][name] = entry

    cfg.artifacts.ensure()
    model.save(cfg.artifacts.multitask_dir / "model.pkl")
    pipeline.save(cfg.artifacts.multitask_dir / "preprocessing.pkl")
    save_json(
        {
            "model_name": "multitask_" + cfg.multitask.encoder,
            "task": "multitask_rul_soh_risk",
            "schema_version": "2.0",
            "environment": environment_fingerprint(),
            "config": cfg.multitask.model_dump(mode="json"),
            "feature_names": list(pipeline.feature_names),
            "results": results,
            "warmup_policy": WarmupPolicy(
                cfg.features.drop_warmup_cycles, cfg.multitask.window
            ).to_dict(),
        },
        cfg.artifacts.multitask_dir / "metadata.json",
    )
    return results


# ---------------------------------------------------------------------------
# Stage 6 — evaluation and orchestration
# ---------------------------------------------------------------------------
def evaluate_milestone_2(cfg: ExperimentConfig, results: dict[str, Any]) -> dict[str, Any]:
    """Write the Milestone 2 metric artifacts and the evaluation report."""
    from battery_rul.evaluation.reporting_m2 import write_milestone_2_report

    reports = cfg.paths.reports_dir / "milestone_2"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": cfg.experiment_name,
        "environment": environment_fingerprint(),
        "config": cfg.to_dict(),
        **results,
    }
    save_json(payload, reports / "metrics.json")
    write_milestone_2_report(cfg, payload)
    return payload


def build_bundles(cfg: ExperimentConfig, data: MultiTaskData | None = None) -> dict[str, Any]:
    """Build every deployable bundle (RUL + uncertainty, SOH, risk + calibration)."""
    data = data or prepare_multitask_data(cfg)
    return {
        "rul": calibrate_uncertainty(cfg, data),
        "soh": train_soh(cfg, data),
        "risk": train_risk(cfg, data),
    }


def run_milestone_2(cfg: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    """Run every Milestone 2 stage, skipping expensive ones with valid artifacts."""
    log_section(logger, "MILESTONE 2 — battery digital twin")
    timer = StageTimer()
    cfg.paths.ensure()
    cfg.artifacts.ensure()

    with timer("prepare"):
        data = prepare_multitask_data(cfg)

    results: dict[str, Any] = {"targets": data.reports}
    with timer("rul_uncertainty"):
        results["rul"] = calibrate_uncertainty(cfg, data)
    with timer("soh"):
        results["soh"] = train_soh(cfg, data)
    with timer("risk"):
        results["risk"] = train_risk(cfg, data)

    multitask_path = cfg.artifacts.multitask_dir / "model.pkl"
    if multitask_path.is_file() and not force:
        logger.info(
            "Multi-task artifact already exists at %s; skipping the expensive retrain. "
            "Pass --force to rebuild it.",
            multitask_path,
        )
        from battery_rul.utils.io import load_json

        metadata_path = cfg.artifacts.multitask_dir / "metadata.json"
        results["multitask"] = (
            load_json(metadata_path).get("results", {}) if metadata_path.is_file() else {}
        )
        results["multitask_reused_existing_artifact"] = True
    else:
        with timer("multitask"):
            results["multitask"] = train_multitask(cfg, data)

    results["timings_s"] = timer.as_dict()
    return evaluate_milestone_2(cfg, results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_STAGES = {
    "prepare-multitask-data": lambda cfg, args: prepare_multitask_data(cfg).reports,
    "train-soh": lambda cfg, args: train_soh(cfg),
    "train-risk": lambda cfg, args: train_risk(cfg),
    "train-multitask": lambda cfg, args: train_multitask(cfg),
    "calibrate-risk": lambda cfg, args: calibrate_risk(cfg),
    "calibrate-uncertainty": lambda cfg, args: calibrate_uncertainty(cfg),
    "build-model-bundle": lambda cfg, args: build_bundles(cfg),
    "run": lambda cfg, args: run_milestone_2(cfg, force=args.force),
}


def main(argv: list[str] | None = None) -> int:
    """Entry point shared by every ``python -m battery_rul.pipelines.*`` alias."""
    parser = argparse.ArgumentParser(
        prog="battery-rul milestone-2", description=__doc__.splitlines()[0]
    )
    parser.add_argument("stage", choices=sorted(_STAGES), nargs="?", default="run")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts exist.")
    args = parser.parse_args(argv)

    overrides = {}
    for pair in args.set:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()

    from pathlib import Path

    path = Path(args.config)
    cfg = load_config(path if path.is_file() else None, overrides=overrides)
    cfg.paths.ensure()
    setup_logging(log_file=cfg.paths.reports_dir / "pipeline.log", force=True)

    try:
        result = _STAGES[args.stage](cfg, args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and exit non-zero
        logger.exception("Stage %s failed: %s", args.stage, exc)
        return 1
    logger.info("Stage %s complete.", args.stage)
    if isinstance(result, dict) and "metrics" not in result:
        save_json(result, cfg.paths.reports_dir / "milestone_2" / f"{args.stage}.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Re-import under the module's real name before running. Executed via
    # ``python -m``, this file is ``__main__``, and anything it pickles — the
    # risk-classifier adapter, in particular — would be stamped with a module
    # path that does not exist in any other process. Delegating to the imported
    # copy makes the artifacts loadable everywhere.
    from battery_rul.pipelines.milestone_2 import main as _main

    sys.exit(_main())
