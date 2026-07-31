"""Feature engineering, leakage guarantees, target construction and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.features.engineering import (
    LEAKY_COLUMNS,
    NON_FEATURE_COLUMNS,
    assert_no_leakage,
    build_features,
    feature_columns,
)
from battery_rul.features.pipeline import FeaturePipeline, FeatureUnseenColumnsError
from battery_rul.features.sequences import make_sequences
from battery_rul.features.target import attach_target, find_eol_cycle


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def test_rul_is_eol_minus_cycle(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    df = labelled_cycles
    expected = (df["eol_cycle"] - df["cycle_index"]).clip(lower=0)
    np.testing.assert_allclose(df[cfg.target.name], expected, atol=1e-5)


def test_rul_is_zero_at_end_of_life(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    for _, group in labelled_cycles.groupby("battery_id"):
        at_eol = group[group["cycle_index"] == group["eol_cycle"]]
        if not at_eol.empty:
            assert at_eol[cfg.target.name].iloc[0] == pytest.approx(0.0)


def test_rul_decreases_monotonically_within_a_cell(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    for _, group in labelled_cycles.groupby("battery_id"):
        values = group.sort_values("cycle_index")[cfg.target.name].to_numpy()
        assert np.all(np.diff(values) <= 0)


def test_no_negative_rul(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    assert (labelled_cycles[cfg.target.name] >= 0).all()


def test_eol_requires_persistence(cfg: ExperimentConfig):
    """A single dip below threshold (capacity recovery) must not trigger EOL."""
    threshold = cfg.eol_capacity_ah
    capacity = np.full(40, threshold + 0.2)
    capacity[10] = threshold - 0.05  # transient dip, recovers immediately
    capacity[25:] = threshold - 0.05  # genuine, persistent crossing

    group = pd.DataFrame(
        {
            "cycle_index": np.arange(1, 41),
            "capacity_smooth_ah": capacity,
            "reference_capacity_ah": cfg.data.nominal_capacity_ah,
        }
    )
    assert find_eol_cycle(group, cfg) == 26


def test_eol_returns_none_when_never_reached(cfg: ExperimentConfig):
    group = pd.DataFrame(
        {
            "cycle_index": np.arange(1, 31),
            "capacity_smooth_ah": np.full(30, cfg.eol_capacity_ah + 0.5),
            "reference_capacity_ah": cfg.data.nominal_capacity_ah,
        }
    )
    assert find_eol_cycle(group, cfg) is None


def test_censored_cells_are_excluded(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    from battery_rul.data.loader import _derive_health

    healthy = raw_cycles.copy()
    mask = healthy["battery_id"] == "T0002"
    healthy.loc[mask, "capacity_ah"] = 1.95  # never degrades

    df = _derive_health(healthy, cfg)
    out, report = attach_target(df, cfg)
    assert "T0002" in report.censored_batteries
    assert "T0002" not in set(out["battery_id"])


def test_target_cap_is_applied(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    from battery_rul.data.loader import _derive_health

    cfg.target.cap_at = 40
    out, _ = attach_target(_derive_health(raw_cycles, cfg), cfg)
    assert out[cfg.target.name].max() <= 40


def test_log_transform_round_trips(cfg: ExperimentConfig):
    from battery_rul.features.target import inverse_transform_target, transform_target

    cfg.target.log_transform = True
    y = np.array([0.0, 1.0, 25.0, 130.0])
    np.testing.assert_allclose(
        inverse_transform_target(transform_target(y, cfg), cfg), y, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_build_features_produces_many_columns(labelled_cycles, cfg: ExperimentConfig):
    frame, report = build_features(labelled_cycles, cfg.features)
    assert report.n_generated > 50
    assert len(feature_columns(frame)) > 20
    assert not frame.empty


def test_no_nan_or_inf_in_features(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    values = frame[feature_columns(frame)].to_numpy()
    assert np.isfinite(values).all()


def test_label_columns_are_never_features(labelled_cycles, cfg: ExperimentConfig):
    """The single most important guard in the project: no target leakage."""
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = set(feature_columns(frame))
    assert not (names & LEAKY_COLUMNS)
    assert not (names & NON_FEATURE_COLUMNS)
    assert cfg.target.name not in names
    assert "eol_cycle" not in names
    assert "life_fraction" not in names


def test_features_are_causal(labelled_cycles, cfg: ExperimentConfig):
    """Truncating the future must not change already-computed rows."""
    assert_no_leakage(labelled_cycles, cfg.features, battery_id="T0001")


def test_causality_holds_for_every_cell(labelled_cycles, cfg: ExperimentConfig):
    for battery in labelled_cycles["battery_id"].unique():
        assert_no_leakage(labelled_cycles, cfg.features, battery_id=str(battery))


def test_leakage_detector_catches_a_planted_violation(labelled_cycles, cfg: ExperimentConfig):
    """The detector must actually detect.

    A guard that has never been observed to fail proves nothing, so this plants a
    genuinely non-causal feature — one computed *at build time* from the rows in
    hand, which is how real leakage enters — and requires the checker to raise.
    Note the violation cannot be baked into the input frame beforehand: it has to
    depend on what the builder can see, or truncation would not change it.
    """

    def leaky_builder(df, feature_cfg):
        frame, report = build_features(df, feature_cfg)
        source = feature_columns(frame)[0]
        # Reverse cumulative minimum: at cycle k this reads every cycle AFTER k.
        frame["planted_future_min"] = (
            frame.iloc[::-1].groupby("battery_id")[source].cummin().iloc[::-1]
        )
        return frame, report

    with pytest.raises(AssertionError, match="[Tt]emporal leakage"):
        assert_no_leakage(labelled_cycles, cfg.features, battery_id="T0001", builder=leaky_builder)


def test_leakage_detector_passes_the_real_builder(labelled_cycles, cfg: ExperimentConfig):
    """The companion to the test above: the production builder must survive it."""
    assert_no_leakage(labelled_cycles, cfg.features, battery_id="T0002", builder=build_features)


def test_cells_do_not_influence_each_other(labelled_cycles, cfg: ExperimentConfig):
    """Removing one cell must not change another cell's features."""
    both, _ = build_features(labelled_cycles, cfg.features)
    subset = labelled_cycles[labelled_cycles["battery_id"].isin(["T0001", "T0002"])]
    partial, _ = build_features(subset, cfg.features)

    shared = [c for c in feature_columns(partial) if c in both.columns]
    a = both[both["battery_id"] == "T0001"][shared].reset_index(drop=True)
    b = partial[partial["battery_id"] == "T0001"][shared].reset_index(drop=True)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5, atol=1e-6)


def test_warmup_rows_are_dropped(labelled_cycles, cfg: ExperimentConfig):
    cfg.features.drop_warmup_cycles = 8
    frame, report = build_features(labelled_cycles, cfg.features)
    assert frame["cycle_index"].min() > 8
    assert report.warmup_rows_dropped > 0


def test_prune_false_yields_a_superset(labelled_cycles, cfg: ExperimentConfig):
    """The serving path relies on this: unpruned output must contain every
    column the pruned (training) output produced."""
    pruned, _ = build_features(labelled_cycles, cfg.features, prune=True)
    unpruned, _ = build_features(labelled_cycles, cfg.features, prune=False)
    assert set(feature_columns(pruned)) <= set(feature_columns(unpruned))


def test_empty_input_raises(cfg: ExperimentConfig):
    with pytest.raises(ValueError, match="empty"):
        build_features(pd.DataFrame(), cfg.features)


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------
def test_pipeline_fit_transform_shape(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    y = frame[cfg.target.name].to_numpy()

    pipeline = FeaturePipeline(cfg=cfg.features)
    X = pipeline.fit_transform(frame[names], y)
    assert X.shape == (len(frame), len(pipeline.feature_names))
    assert len(pipeline.feature_names) <= cfg.features.max_features


def test_pipeline_enforces_column_order(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame[names], frame[cfg.target.name].to_numpy()
    )

    shuffled = frame[names[::-1]]
    np.testing.assert_allclose(pipeline.transform(frame[names]), pipeline.transform(shuffled))


def test_pipeline_rejects_missing_columns(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame[names], frame[cfg.target.name].to_numpy()
    )
    with pytest.raises(FeatureUnseenColumnsError):
        pipeline.transform(frame[names].drop(columns=pipeline.feature_names[:2]))


def test_pipeline_raises_before_fit(cfg: ExperimentConfig):
    with pytest.raises(RuntimeError, match="not fitted"):
        FeaturePipeline(cfg=cfg.features).transform(pd.DataFrame({"a": [1.0]}))


def test_pipeline_round_trips_through_disk(labelled_cycles, cfg: ExperimentConfig, tmp_path):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame[names], frame[cfg.target.name].to_numpy()
    )
    path = pipeline.save(tmp_path / "pipe.pkl")
    reloaded = FeaturePipeline.load(path)

    assert reloaded.feature_names == pipeline.feature_names
    np.testing.assert_allclose(reloaded.transform(frame[names]), pipeline.transform(frame[names]))


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------
def test_windows_never_cross_cell_boundaries(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    values = frame[names].to_numpy(dtype=float)
    y = frame[cfg.target.name].to_numpy(dtype=float)

    batch = make_sequences(frame, values, y, window=10, feature_names=names)
    assert batch.X.shape[1] == 10
    assert batch.X.shape[2] == len(names)

    # Each window's count must be consistent with per-cell lengths.
    per_cell = frame.groupby("battery_id").size()
    assert len(batch) == int(sum(max(n - 10 + 1, 0) for n in per_cell))


def test_window_label_is_the_last_cycle(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    y = frame[cfg.target.name].to_numpy(dtype=float)
    batch = make_sequences(frame, frame[names].to_numpy(float), y, window=8, feature_names=names)

    lookup = {
        (b, c): value
        for b, c, value in zip(frame["battery_id"], frame["cycle_index"], y, strict=True)
    }
    for i in range(0, len(batch), 25):
        key = (batch.battery_ids[i], int(batch.cycle_index[i]))
        assert batch.y[i] == pytest.approx(lookup[key])


def test_window_too_long_raises(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    with pytest.raises(ValueError, match="No sequences"):
        make_sequences(
            frame, frame[names].to_numpy(float), frame[cfg.target.name].to_numpy(float), window=9999
        )


def test_sequence_row_count_mismatch_raises(labelled_cycles, cfg: ExperimentConfig):
    frame, _ = build_features(labelled_cycles, cfg.features)
    names = feature_columns(frame)
    with pytest.raises(ValueError, match="Row-count mismatch"):
        make_sequences(
            frame,
            frame[names].to_numpy(float)[:-5],
            frame[cfg.target.name].to_numpy(float),
            window=5,
        )
