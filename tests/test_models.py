"""Model construction, fitting, persistence and the sequence-model contract."""

from __future__ import annotations

import numpy as np
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.features.engineering import build_features, feature_columns
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.models import available_models, build_model
from battery_rul.models.base import BaseModel, TrainingData
from battery_rul.models.search_spaces import SEARCH_SPACES, describe_spaces

TABULAR = ["linear_regression", "ridge", "random_forest", "xgboost", "lightgbm"]
SEQUENCE = ["lstm", "gru", "transformer"]


@pytest.fixture
def partitions(labelled_cycles, cfg: ExperimentConfig):
    """Train/test TrainingData built exactly as the pipeline builds them."""
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    y = frame[cfg.target.name].to_numpy(dtype=float)

    cells = frame["battery_id"].to_numpy()
    test_mask = np.isin(cells, ["T0004"])
    train_mask = ~test_mask

    pipeline = FeaturePipeline(cfg=cfg.features)
    pipeline.fit(frame.loc[train_mask, names], y[train_mask])

    def _make(mask):
        subset = frame.loc[mask].reset_index(drop=True)
        return TrainingData(
            X=pipeline.transform(subset[names]),
            y=y[mask],
            frame=subset,
            feature_names=pipeline.feature_names,
        )

    return _make(train_mask), _make(test_mask)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contains_every_advertised_model():
    expected = {
        "linear_regression",
        "ridge",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "lstm",
        "gru",
        "transformer",
    }
    assert expected <= set(available_models())


def test_models_report_their_own_name(cfg: ExperimentConfig):
    """Regression guard: model metadata must not be shadowed by dataclass fields."""
    for name in available_models():
        assert build_model(name, cfg).name == name


def test_unknown_model_raises(cfg: ExperimentConfig):
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("does_not_exist", cfg)


def test_overrides_reach_the_estimator(cfg: ExperimentConfig):
    model = build_model("ridge", cfg, alpha=123.0)
    assert model.params["alpha"] == 123.0


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", TABULAR)
def test_tabular_model_fits_and_predicts(name, partitions, cfg: ExperimentConfig):
    train, test = partitions
    model = build_model(name, cfg).fit(train, test)
    assert model.fitted

    pred = model.predict(test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", TABULAR)
def test_tabular_model_beats_the_mean_on_training_data(name, partitions, cfg: ExperimentConfig):
    """A sanity floor: any working regressor should fit its own training data."""
    train, test = partitions
    model = build_model(name, cfg).fit(train, test)
    pred = model.predict(train)
    baseline = np.full_like(train.y, train.y.mean())
    assert np.mean(np.abs(pred - train.y)) < np.mean(np.abs(baseline - train.y))


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "lightgbm", "ridge"])
def test_feature_importance_is_named_and_aligned(name, partitions, cfg: ExperimentConfig):
    train, test = partitions
    model = build_model(name, cfg).fit(train, test)
    importance = model.feature_importance()
    assert importance is not None
    assert len(importance) == len(train.feature_names)
    assert set(importance.index) == set(train.feature_names)


def test_model_persists_and_reloads(partitions, cfg: ExperimentConfig, tmp_path):
    train, test = partitions
    model = build_model("random_forest", cfg).fit(train, test)
    path = model.save(tmp_path / "m.pkl")

    reloaded = BaseModel.load(path)
    assert reloaded.name == "random_forest"
    np.testing.assert_allclose(reloaded.predict(test), model.predict(test))


def test_unfitted_model_refuses_to_save(cfg: ExperimentConfig, tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        build_model("ridge", cfg).save(tmp_path / "x.pkl")


def test_unfitted_model_refuses_to_predict(partitions, cfg: ExperimentConfig):
    _, test = partitions
    with pytest.raises(RuntimeError, match="not fitted"):
        build_model("ridge", cfg).predict(test)


# ---------------------------------------------------------------------------
# Sequence models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_model_fits_and_is_row_aligned(name, partitions, cfg: ExperimentConfig):
    train, test = partitions
    model = build_model(name, cfg).fit(train, test)
    assert model.fitted
    assert model.is_sequence

    pred = model.predict(test)
    # One prediction per input row, NaN where no full window exists.
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).any()


@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_warmup_rows_are_nan_not_dropped(name, partitions, cfg: ExperimentConfig):
    train, test = partitions
    window = cfg.models.sequence.window
    model = build_model(name, cfg).fit(train, test)
    pred = model.predict(test)

    ordered = test.frame.sort_values(["battery_id", "cycle_index"])
    first_positions = [
        ordered.index[ordered["battery_id"] == cell][: window - 1]
        for cell in ordered["battery_id"].unique()
    ]
    for positions in first_positions:
        assert np.isnan(pred[list(positions)]).all()


def test_sequence_model_records_training_history(partitions, cfg: ExperimentConfig):
    train, test = partitions
    model = build_model("gru", cfg).fit(train, test)
    assert model.train_history["train_loss"]
    assert model.fit_metadata["n_parameters"] > 0
    assert model.fit_metadata["window"] == cfg.models.sequence.window


def test_transformer_rounds_incompatible_d_model(cfg: ExperimentConfig, partitions):
    train, test = partitions
    model = build_model("transformer", cfg, d_model=30, nhead=4).fit(train, test)
    assert model.fitted


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def test_every_tunable_model_is_registered():
    assert set(SEARCH_SPACES) <= set(available_models())


def test_search_spaces_describe_their_parameters():
    described = describe_spaces()
    assert set(described) == set(SEARCH_SPACES)
    for name, params in described.items():
        assert params, f"{name} search space reports no parameters"


def test_search_space_sampling_produces_valid_params(cfg: ExperimentConfig):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in ("xgboost", "lightgbm", "random_forest"):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial, _n=name: float(len(SEARCH_SPACES[_n](trial))), n_trials=2)
        assert study.best_value > 0


# ---------------------------------------------------------------------------
def test_training_data_validates_lengths(partitions):
    train, _ = partitions
    with pytest.raises(ValueError, match="length mismatch"):
        TrainingData(
            X=train.X[:-3], y=train.y, frame=train.frame, feature_names=train.feature_names
        )
