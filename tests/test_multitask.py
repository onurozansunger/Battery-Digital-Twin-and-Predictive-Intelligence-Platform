"""Multi-task sequence model: windowing, losses, output shapes and ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from battery_rul.config import ExperimentConfig
from battery_rul.models.multitask import (
    MultiTaskNetwork,
    MultiTaskSequenceModel,
    make_multitask_windows,
)


@pytest.fixture
def mt_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg.multitask.window = 6
    cfg.multitask.d_model = 16
    cfg.multitask.nhead = 2
    cfg.multitask.num_layers = 1
    cfg.multitask.dim_feedforward = 32
    cfg.multitask.hidden_size = 16
    cfg.multitask.head_hidden = 8
    cfg.multitask.epochs = 3
    cfg.multitask.batch_size = 16
    cfg.multitask.device = "cpu"
    return cfg


@pytest.fixture
def multitask_frame(labelled_cycles: pd.DataFrame, mt_cfg: ExperimentConfig) -> pd.DataFrame:
    from battery_rul.targets.risk import attach_failure_risk_target
    from battery_rul.targets.soh import attach_soh_target

    frame, _ = attach_soh_target(labelled_cycles, mt_cfg)
    frame, _ = attach_failure_risk_target(frame, mt_cfg)
    return frame


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_windows_never_cross_a_battery_boundary(multitask_frame, mt_cfg):
    values = np.random.default_rng(0).normal(size=(len(multitask_frame), 4))
    dataset = make_multitask_windows(multitask_frame, values, window=6)
    battery = multitask_frame["battery_id"].to_numpy()
    for i, position in enumerate(dataset.row_positions):
        window_rows = range(position - 5, position + 1)
        assert len({battery[r] for r in window_rows}) == 1, f"window {i} spans two cells"


def test_window_label_is_the_final_row(multitask_frame, mt_cfg):
    values = np.zeros((len(multitask_frame), 3))
    rul = multitask_frame[mt_cfg.target.name].to_numpy(dtype=float)
    soh = multitask_frame[mt_cfg.soh.target_name].to_numpy(dtype=float)
    dataset = make_multitask_windows(multitask_frame, values, window=6, rul=rul, soh=soh)
    for i, position in enumerate(dataset.row_positions):
        assert dataset.y_rul[i] == pytest.approx(rul[position], rel=1e-5)
        assert dataset.y_soh[i] == pytest.approx(soh[position], rel=1e-5)


def test_window_cycles_are_strictly_increasing(multitask_frame, mt_cfg):
    values = np.zeros((len(multitask_frame), 3))
    dataset = make_multitask_windows(multitask_frame, values, window=6)
    for cell in np.unique(dataset.battery_ids):
        cycles = dataset.cycle_index[dataset.battery_ids == cell]
        assert np.all(np.diff(cycles) > 0)


def test_short_cell_produces_no_windows(mt_cfg):
    frame = pd.DataFrame({"battery_id": ["A"] * 3, "cycle_index": [1, 2, 3]})
    with pytest.raises(ValueError, match="No windows produced"):
        make_multitask_windows(frame, np.zeros((3, 2)), window=10)


def test_row_count_mismatch_is_rejected(multitask_frame):
    with pytest.raises(ValueError, match="Row-count mismatch"):
        make_multitask_windows(multitask_frame, np.zeros((5, 2)), window=6)


def test_nan_targets_are_preserved_not_imputed(multitask_frame, mt_cfg):
    """A missing label must cost the task one sample, not teach it a made-up one."""
    risk = np.full(len(multitask_frame), np.nan)
    dataset = make_multitask_windows(
        multitask_frame, np.zeros((len(multitask_frame), 2)), window=6, risk=risk
    )
    assert np.all(np.isnan(dataset.y_risk))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("encoder", ["transformer", "lstm", "gru"])
def test_network_emits_three_heads_with_correct_shapes(encoder, mt_cfg):
    mt_cfg.multitask.encoder = encoder
    network = MultiTaskNetwork(n_features=7, cfg=mt_cfg.multitask)
    out = network(torch.randn(4, mt_cfg.multitask.window, 7))
    assert set(out) == {"rul", "soh", "risk_logit", "representation"}
    for key in ("rul", "soh", "risk_logit"):
        assert out[key].shape == (4,)


def test_heads_share_one_representation(mt_cfg):
    network = MultiTaskNetwork(n_features=5, cfg=mt_cfg.multitask)
    out = network(torch.randn(3, mt_cfg.multitask.window, 5))
    assert out["representation"].shape[0] == 3
    assert out["representation"].shape[1] == mt_cfg.multitask.d_model


def test_loss_weights_are_validated():
    from battery_rul.config import MultiTaskConfig

    with pytest.raises(ValueError, match="must be positive"):
        MultiTaskConfig(rul_weight=0.0, soh_weight=0.0, risk_weight=0.0)


def test_head_divisibility_is_validated():
    from battery_rul.config import MultiTaskConfig

    with pytest.raises(ValueError, match="divisible"):
        MultiTaskConfig(d_model=10, nhead=3)


def test_combined_loss_is_the_weighted_sum_of_its_parts(mt_cfg):
    model = MultiTaskSequenceModel(cfg=mt_cfg)
    model.trained_config = mt_cfg.multitask
    outputs = {
        "rul": torch.tensor([1.0, 2.0]),
        "soh": torch.tensor([0.9, 0.8]),
        "risk_logit": torch.tensor([0.5, -0.5]),
    }
    batch = {
        "rul": torch.tensor([1.5, 2.5]),
        "soh": torch.tensor([0.85, 0.75]),
        "risk": torch.tensor([1.0, 0.0]),
    }
    total, components = model._compute_losses(outputs, batch, torch.tensor(1.0))
    expected = (
        mt_cfg.multitask.rul_weight * components["rul_loss"]
        + mt_cfg.multitask.soh_weight * components["soh_loss"]
        + mt_cfg.multitask.risk_weight * components["risk_loss"]
    )
    assert float(total.item()) == pytest.approx(expected, rel=1e-5)
    assert components["total_loss"] == pytest.approx(expected, rel=1e-5)


def test_each_component_loss_is_logged_separately(mt_cfg):
    model = MultiTaskSequenceModel(cfg=mt_cfg)
    model.trained_config = mt_cfg.multitask
    outputs = {
        "rul": torch.tensor([1.0]),
        "soh": torch.tensor([0.9]),
        "risk_logit": torch.tensor([0.1]),
    }
    batch = {
        "rul": torch.tensor([1.2]),
        "soh": torch.tensor([0.88]),
        "risk": torch.tensor([1.0]),
    }
    _, components = model._compute_losses(outputs, batch, torch.tensor(1.0))
    assert {"rul_loss", "soh_loss", "risk_loss", "total_loss"} <= set(components)


def test_a_task_with_no_valid_labels_is_masked_out(mt_cfg):
    model = MultiTaskSequenceModel(cfg=mt_cfg)
    model.trained_config = mt_cfg.multitask
    outputs = {
        "rul": torch.tensor([1.0]),
        "soh": torch.tensor([0.9]),
        "risk_logit": torch.tensor([0.1]),
    }
    batch = {
        "rul": torch.tensor([1.2]),
        "soh": torch.tensor([0.88]),
        "risk": torch.tensor([float("nan")]),
    }
    _, components = model._compute_losses(outputs, batch, torch.tensor(1.0))
    assert np.isnan(components["risk_loss"])
    assert np.isfinite(components["total_loss"])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def _fit(multitask_frame: pd.DataFrame, mt_cfg: ExperimentConfig):
    from battery_rul.features.engineering import build_features, feature_columns
    from battery_rul.features.pipeline import FeaturePipeline

    frame, _ = build_features(multitask_frame, mt_cfg.features)
    features = feature_columns(frame)
    pipeline = FeaturePipeline(cfg=mt_cfg.features).fit(
        frame[features], frame[mt_cfg.target.name].to_numpy()
    )
    values = pipeline.transform(frame[features])
    model = MultiTaskSequenceModel(cfg=mt_cfg).fit(
        frame, values, feature_names=list(pipeline.feature_names)
    )
    return model, frame, values


def test_fit_and_predict_produce_row_aligned_outputs(multitask_frame, mt_cfg):
    model, frame, values = _fit(multitask_frame, mt_cfg)
    prediction = model.predict(frame, values)
    assert prediction.rul.shape == (len(frame),)
    assert prediction.soh.shape == (len(frame),)
    assert prediction.risk_probability.shape == (len(frame),)


def test_warmup_rows_are_nan_not_fabricated(multitask_frame, mt_cfg):
    model, frame, values = _fit(multitask_frame, mt_cfg)
    prediction = model.predict(frame, values)
    assert not prediction.scoreable.all(), "expected some rows without a full window"
    assert np.all(np.isnan(prediction.rul[~prediction.scoreable]))


def test_outputs_respect_their_physical_ranges(multitask_frame, mt_cfg):
    model, frame, values = _fit(multitask_frame, mt_cfg)
    prediction = model.predict(frame, values)
    scored = prediction.scoreable
    assert np.all(prediction.rul[scored] >= 0.0)
    assert np.all(prediction.risk_probability[scored] >= 0.0)
    assert np.all(prediction.risk_probability[scored] <= 1.0)
    assert np.all(prediction.soh[scored] >= mt_cfg.soh.plausible_min)
    assert np.all(prediction.soh[scored] <= mt_cfg.soh.plausible_max)


def test_model_round_trips_through_disk(multitask_frame, mt_cfg, tmp_path):
    model, frame, values = _fit(multitask_frame, mt_cfg)
    before = model.predict(frame, values)
    path = model.save(tmp_path / "multitask.pkl")

    restored = MultiTaskSequenceModel.load(path, mt_cfg)
    after = restored.predict(frame, values)
    np.testing.assert_allclose(before.rul[before.scoreable], after.rul[after.scoreable], rtol=1e-4)
    assert restored.feature_names == model.feature_names


def test_loaded_model_uses_its_own_training_window(multitask_frame, mt_cfg, tmp_path):
    """A runtime configuration change must not silently rewindow a trained model."""
    model, frame, values = _fit(multitask_frame, mt_cfg)
    path = model.save(tmp_path / "multitask.pkl")

    mt_cfg.multitask.window = 12  # runtime drift
    restored = MultiTaskSequenceModel.load(path, mt_cfg)
    assert restored.mt.window == 6


def test_unfitted_model_refuses_to_predict(mt_cfg, multitask_frame):
    model = MultiTaskSequenceModel(cfg=mt_cfg)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(multitask_frame, np.zeros((len(multitask_frame), 3)))


def test_unfitted_model_refuses_to_save(mt_cfg, tmp_path):
    model = MultiTaskSequenceModel(cfg=mt_cfg)
    with pytest.raises(RuntimeError, match="unfitted"):
        model.save(tmp_path / "x.pkl")


def test_component_losses_are_recorded_in_history(multitask_frame, mt_cfg):
    model, _, _ = _fit(multitask_frame, mt_cfg)
    for key in ("total_loss", "rul_loss", "soh_loss", "risk_loss"):
        assert key in model.history and len(model.history[key]) > 0
