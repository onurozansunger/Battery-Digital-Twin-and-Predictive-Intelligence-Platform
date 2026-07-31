"""End-to-end pipeline behaviour: prepare, train, predict, artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.pipelines import predict as predict_stage
from battery_rul.pipelines import prepare_data, train

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def _module_cfg(tmp_path_factory) -> ExperimentConfig:
    from battery_rul.config import load_config

    return load_config(
        overrides={
            "experiment_name": "pipeline_test",
            "seed": 3,
            "paths.root": str(tmp_path_factory.mktemp("pipeline")),
            "data.source": "synthetic",
            "data.subdir": "synthetic",
            "data.cache_interim": False,
            "features.rolling_windows": [3, 5],
            "features.lags": [1, 3],
            "features.slope_windows": [5],
            "features.ewm_halflives": [5],
            "features.drop_warmup_cycles": 5,
            "features.max_features": 25,
            "models.enabled": ["ridge", "random_forest", "gru"],
            "models.training.epochs": 3,
            "models.training.device": "cpu",
            "models.sequence.window": 8,
            "models.sequence.hidden_size": 16,
            "evaluation.bootstrap_samples": 0,
            "explainability.enabled": False,
        }
    )


@pytest.fixture(scope="module")
def prepared(_module_cfg):
    return prepare_data.run(_module_cfg)


@pytest.fixture(scope="module")
def trained(_module_cfg, prepared):
    return train.run(_module_cfg, prepared=prepared)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def test_prepare_writes_expected_artifacts(_module_cfg, prepared):
    outdir = _module_cfg.paths.processed_dir
    assert (outdir / "dataset.parquet").is_file()
    assert (outdir / "cycles.parquet").is_file()
    assert (outdir / "manifest.json").is_file()

    manifest = json.loads((outdir / "manifest.json").read_text())
    assert {"dataset", "validation", "target", "features", "split", "environment"} <= set(manifest)


def test_prepared_dataset_is_well_formed(prepared, _module_cfg):
    frame = prepared.frame
    assert not frame.empty
    assert prepared.feature_names
    assert _module_cfg.target.name in frame.columns
    assert set(frame["split"].unique()) <= {"train", "val", "test", "unused"}
    assert np.isfinite(frame[prepared.feature_names].to_numpy()).all()


def test_load_prepared_round_trips(_module_cfg, prepared):
    reloaded = prepare_data.load_prepared(_module_cfg)
    assert len(reloaded.frame) == len(prepared.frame)
    assert reloaded.feature_names == prepared.feature_names
    np.testing.assert_array_equal(reloaded.split.test, prepared.split.test)


def test_prepare_is_reproducible(_module_cfg):
    """Same config, same data, same output — the reproducibility claim."""
    a = prepare_data.run(_module_cfg, verify_leakage=False)
    b = prepare_data.run(_module_cfg, verify_leakage=False)
    pd.testing.assert_frame_equal(a.frame, b.frame)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def test_train_produces_a_champion(trained):
    assert trained.champion in trained.models
    assert trained.test_results
    assert not trained.comparison.empty
    assert "rank" in trained.comparison.columns


def test_train_writes_expected_artifacts(_module_cfg, trained):
    models_dir = _module_cfg.paths.models_dir
    reports_dir = _module_cfg.paths.reports_dir
    assert (models_dir / "trained_model.pkl").is_file()
    assert (models_dir / "feature_pipeline.pkl").is_file()
    assert (reports_dir / "metrics.json").is_file()
    assert (reports_dir / "predictions_test.csv").is_file()

    metrics = json.loads((reports_dir / "metrics.json").read_text())
    assert metrics["champion"] == trained.champion
    assert "environment" in metrics
    assert metrics["test"]


def test_every_model_is_persisted(_module_cfg, trained):
    zoo = _module_cfg.paths.models_dir / "zoo"
    saved = {p.stem for p in zoo.glob("*.pkl")}
    assert saved == set(trained.models)


def test_champion_is_chosen_on_validation(_module_cfg, trained):
    metrics = json.loads((_module_cfg.paths.reports_dir / "metrics.json").read_text())
    assert metrics["selected_on"] == "validation"


def test_feature_pipeline_fitted_on_train_rows_only(trained):
    pipeline = trained.feature_pipeline
    assert pipeline.fit_stats["n_rows_fit"] == len(trained.partitions["train"])


def test_partitions_share_no_cells(trained):
    cells = {
        name: set(data.frame["battery_id"].unique()) for name, data in trained.partitions.items()
    }
    assert not (cells["train"] & cells["test"])
    if "val" in cells:
        assert not (cells["train"] & cells["val"])
        assert not (cells["val"] & cells["test"])


def test_predictions_are_row_aligned(trained):
    for result in trained.test_results.values():
        assert len(result.predictions) == len(trained.partitions["test"])
        assert (
            result.predictions["battery_id"].to_numpy() == trained.partitions["test"].battery_ids
        ).all()


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------
def test_predict_from_persisted_artifacts(_module_cfg, trained, prepared):
    """The serving path must reproduce the training path exactly."""
    predictor = predict_stage.RULPredictor.from_artifacts(_module_cfg)
    test_cells = prepared.split.test_batteries or ["S0001"]
    cycles = prepared.cycles[prepared.cycles["battery_id"].isin(test_cells)]

    result = predictor.predict(cycles)
    assert len(result.predictions) == len(cycles)
    assert result.predictions["predicted_rul_cycles"].notna().any()
    assert not result.per_battery.empty
    assert "predicted_eol_cycle" in result.predictions.columns


def test_predict_matches_training_time_scores(_module_cfg, trained, prepared):
    """No training/serving skew: the serving path must reproduce the numbers the
    evaluator reported, on the rows both can score."""
    predictor = predict_stage.RULPredictor.from_artifacts(_module_cfg)
    test_cells = prepared.split.test_batteries
    cycles = prepared.cycles[prepared.cycles["battery_id"].isin(test_cells)]
    served = predictor.predict(cycles).predictions

    trained_preds = trained.champion_result.predictions
    merged = trained_preds.merge(
        served, on=["battery_id", "cycle_index"], how="inner", suffixes=("_t", "_s")
    ).dropna(subset=["y_pred", "predicted_rul_cycles"])

    assert len(merged) > 10
    np.testing.assert_allclose(
        merged["y_pred"].to_numpy(), merged["predicted_rul_cycles"].to_numpy(), rtol=1e-4, atol=1e-3
    )


def test_predict_rejects_empty_input(_module_cfg, trained):
    predictor = predict_stage.RULPredictor.from_artifacts(_module_cfg)
    with pytest.raises(ValueError, match="empty"):
        predictor.predict(pd.DataFrame())


def test_predict_rejects_missing_id_columns(_module_cfg, trained, prepared):
    predictor = predict_stage.RULPredictor.from_artifacts(_module_cfg)
    with pytest.raises(KeyError, match="missing required columns"):
        predictor.predict(prepared.cycles.drop(columns=["battery_id"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_parser_accepts_every_command():
    from battery_rul.cli import build_parser

    parser = build_parser()
    for command in ("prepare", "tune", "train", "evaluate", "predict", "all"):
        args = parser.parse_args([command])
        assert args.command == command


def test_cli_set_override_is_parsed():
    from battery_rul.cli import build_parser

    args = build_parser().parse_args(["train", "--set", "seed=99"])
    assert args.set == ["seed=99"]
