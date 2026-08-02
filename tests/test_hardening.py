"""Milestone 1.1 regression tests.

Each test here corresponds to a specific defect found in the Milestone 1 audit.
They are grouped by the audit item so a reviewer can map a test to the claim it
substantiates:

* 1.1  split-safe, causal imputation
* 1.2  train-only feature pruning
* 1.3  exact end-of-life persistence
* 1.5  training/serving warm-up parity
* 1.6  artifact and configuration compatibility

A guard that has never been observed to fail proves nothing, so several of these
plant the defect and require the guard to catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.data.validation import validate_cycles
from battery_rul.features.engineering import build_features, feature_columns
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.target import attach_target, find_eol_cycle
from battery_rul.features.warmup import WarmupPolicy, first_scoreable_cycle, scoreable_mask


# ===========================================================================
# 1.1 — split-safe and causal imputation
# ===========================================================================
def test_imputation_never_uses_a_full_dataset_statistic(
    raw_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """Changing a *held-out* cell's readings must not move a training cell's.

    This is the planted-leakage test for ingestion. The old implementation fell
    back to ``df[col].median()`` over every loaded row, so editing one cell
    shifted another cell's imputed values — across what would later become an
    evaluation boundary.
    """
    frame = raw_cycles.copy()
    frame.loc[frame["battery_id"] == "T0001", "internal_resistance_ohm"] = np.nan
    frame.loc[frame["battery_id"] == "T0002", "internal_resistance_ohm"] = 0.05

    baseline, _ = validate_cycles(frame.copy(), data_cfg=cfg.data, validation_cfg=cfg.validation)

    tampered = frame.copy()
    tampered.loc[tampered["battery_id"] == "T0002", "internal_resistance_ohm"] = 999.0
    after, _ = validate_cycles(tampered, data_cfg=cfg.data, validation_cfg=cfg.validation)

    a = baseline.loc[baseline["battery_id"] == "T0001", "internal_resistance_ohm"]
    b = after.loc[after["battery_id"] == "T0001", "internal_resistance_ohm"]
    pd.testing.assert_series_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_imputation_never_reads_a_future_cycle(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    """Truncating the future must not change an already-imputed earlier value."""
    frame = raw_cycles.loc[raw_cycles["battery_id"] == "T0001"].copy()
    frame.loc[frame.index[5:8], "temperature_max_c"] = np.nan

    full, _ = validate_cycles(frame.copy(), data_cfg=cfg.data, validation_cfg=cfg.validation)
    prefix, _ = validate_cycles(
        frame.iloc[:40].copy(), data_cfg=cfg.data, validation_cfg=cfg.validation
    )
    shared = min(len(full), len(prefix))
    np.testing.assert_allclose(
        full["temperature_max_c"].to_numpy()[:shared],
        prefix["temperature_max_c"].to_numpy()[:shared],
        rtol=1e-6,
    )


def test_missingness_indicators_are_emitted(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    """An absent reading must survive into the model as information."""
    frame = raw_cycles.copy()
    frame.loc[frame.index[:20], "internal_resistance_ohm"] = np.nan
    out, report = validate_cycles(frame, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert "internal_resistance_ohm_is_missing" in out.columns
    assert "internal_resistance_ohm_is_missing" in report.missingness_indicators
    assert out["internal_resistance_ohm_is_missing"].sum() == 20


def test_indicators_can_be_disabled(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    cfg.validation.missingness_indicators = False
    frame = raw_cycles.copy()
    frame.loc[frame.index[:5], "internal_resistance_ohm"] = np.nan
    out, _ = validate_cycles(frame, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert "internal_resistance_ohm_is_missing" not in out.columns


def test_pipeline_fallback_is_learned_from_training_rows_only(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """The fleet fallback is a *training* statistic, persisted and replayed."""
    frame, _ = build_features(labelled_cycles, cfg.features)
    features = feature_columns(frame)
    train = frame["battery_id"].isin(["T0001", "T0002"])

    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame.loc[train, features], frame.loc[train, cfg.target.name].to_numpy()
    )
    column = pipeline.feature_names[0]
    expected = float(np.nanmedian(frame.loc[train, column].to_numpy(dtype=float)))
    assert pipeline.fallback_values[column] == pytest.approx(expected, rel=1e-6)

    # A frame consisting entirely of NaN must transform to the persisted values,
    # not to a statistic of itself.
    blank = frame.loc[~train, pipeline.feature_names].copy()
    blank[:] = np.nan
    transformed = pipeline.transform(blank)
    reference = pipeline.transform(pd.DataFrame([pipeline.fallback_values])[pipeline.feature_names])
    np.testing.assert_allclose(transformed[0], reference[0], rtol=1e-6)


# ===========================================================================
# 1.2 — train-only feature pruning
# ===========================================================================
def test_held_out_battery_cannot_change_the_training_schema(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """The planted-leakage test for preprocessing.

    Corrupt a held-out cell beyond recognition. Because variance filtering,
    correlation pruning and supervised selection are all fitted on training rows
    only, the surviving feature schema must be byte-identical.
    """
    frame, _ = build_features(labelled_cycles, cfg.features)
    features = feature_columns(frame)
    train = frame["battery_id"].isin(["T0001", "T0002"])
    y = frame[cfg.target.name].to_numpy()

    baseline = FeaturePipeline(cfg=cfg.features).fit(
        frame.loc[train, features], y[train.to_numpy()]
    )

    tampered = frame.copy()
    held_out = tampered["battery_id"] == "T0004"
    rng = np.random.default_rng(0)
    # Widen before writing: the engineered columns are float32, and pandas 3.0
    # raises on a lossy setitem instead of silently downcasting the way 2.x did.
    # The cast is the test's business, not the pipeline's — what is under test is
    # that the *training* schema is unmoved by whatever this cell contains.
    tampered[features] = tampered[features].astype("float64")
    tampered.loc[held_out, features] = rng.normal(
        1000.0, 500.0, size=(int(held_out.sum()), len(features))
    )
    after = FeaturePipeline(cfg=cfg.features).fit(
        tampered.loc[train, features], y[train.to_numpy()]
    )

    assert baseline.feature_names == after.feature_names
    assert baseline.dropped_constant == after.dropped_constant
    assert baseline.fingerprint() == after.fingerprint()


def test_feature_generation_does_not_prune(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    """Generation must be stateless: same columns regardless of which cells are in."""
    everything, _ = build_features(labelled_cycles, cfg.features)
    subset, _ = build_features(
        labelled_cycles[labelled_cycles["battery_id"].isin(["T0001"])], cfg.features
    )
    assert feature_columns(everything) == feature_columns(subset)


def test_correlated_features_are_pruned_by_the_pipeline(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """Pruning still happens — it simply happens behind the evaluation boundary."""
    frame, _ = build_features(labelled_cycles, cfg.features)
    features = feature_columns(frame)
    cfg.features.correlation_prune_threshold = 0.95
    cfg.features.max_features = None
    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame[features], frame[cfg.target.name].to_numpy()
    )
    assert pipeline.dropped_correlated, "expected some near-duplicate columns to be dropped"
    assert len(pipeline.feature_names) < len(features)


def test_pipeline_schema_round_trips(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig, tmp_path
):
    frame, _ = build_features(labelled_cycles, cfg.features)
    features = feature_columns(frame)
    pipeline = FeaturePipeline(cfg=cfg.features).fit(
        frame[features], frame[cfg.target.name].to_numpy()
    )
    path = pipeline.save(tmp_path / "pipeline.pkl")
    restored = FeaturePipeline.load(path)
    assert restored.feature_names == pipeline.feature_names
    assert restored.fingerprint() == pipeline.fingerprint()
    assert restored.schema()["fallback_values"] == pipeline.schema()["fallback_values"]


# ===========================================================================
# 1.3 — exact end-of-life persistence
# ===========================================================================
def _cell(capacities: list[float], cfg: ExperimentConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": "X",
            "cycle_index": np.arange(1, len(capacities) + 1),
            "capacity_smooth_ah": capacities,
            "capacity_ah": capacities,
        }
    )


def test_transient_dip_is_not_end_of_life(cfg: ExperimentConfig):
    """One cycle below threshold, then recovery: not a crossing."""
    threshold = cfg.eol_capacity_ah
    capacities = [1.9] * 10 + [threshold - 0.01] + [1.8] * 10
    assert find_eol_cycle(_cell(capacities, cfg), cfg) is None


def test_exact_p_cycle_crossing_is_detected(cfg: ExperimentConfig):
    """Exactly P consecutive below-threshold cycles counts, and at the first one."""
    cfg.target.eol_persistence = 3
    threshold = cfg.eol_capacity_ah
    capacities = [1.9] * 10 + [threshold - 0.01] * 3 + [1.9] * 5
    assert find_eol_cycle(_cell(capacities, cfg), cfg) == 11


def test_incomplete_end_of_record_crossing_is_censored(cfg: ExperimentConfig):
    """The exact defect: two below-threshold rows at the very end of a record.

    The old implementation accepted "holds for every remaining observation" and
    labelled this as end of life at cycle 11. It is right-censored — the record
    stops before persistence can be confirmed.
    """
    cfg.target.eol_persistence = 3
    threshold = cfg.eol_capacity_ah
    capacities = [1.9] * 10 + [threshold - 0.01] * 2
    assert find_eol_cycle(_cell(capacities, cfg), cfg) is None


def test_recovery_after_a_dip_defers_the_crossing(cfg: ExperimentConfig):
    cfg.target.eol_persistence = 3
    threshold = cfg.eol_capacity_ah
    capacities = [1.9] * 5 + [threshold - 0.01, threshold - 0.01, 1.9] + [threshold - 0.02] * 4
    assert find_eol_cycle(_cell(capacities, cfg), cfg) == 9


def test_persistence_is_configurable(cfg: ExperimentConfig):
    threshold = cfg.eol_capacity_ah
    capacities = [1.9] * 5 + [threshold - 0.01] * 2 + [1.9] * 5
    cfg.target.eol_persistence = 2
    assert find_eol_cycle(_cell(capacities, cfg), cfg) == 6
    cfg.target.eol_persistence = 3
    assert find_eol_cycle(_cell(capacities, cfg), cfg) is None


def test_record_shorter_than_persistence_is_censored(cfg: ExperimentConfig):
    cfg.target.eol_persistence = 5
    threshold = cfg.eol_capacity_ah
    assert find_eol_cycle(_cell([threshold - 0.1] * 3, cfg), cfg) is None


def test_censored_cells_are_reported_by_attach_target(
    raw_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """A cohort where no cell can confirm a crossing must fail loudly."""
    from battery_rul.data.loader import derive_health

    frame = derive_health(raw_cycles, cfg)
    cfg.data.eol_threshold = 0.01  # unreachable
    with pytest.raises(ValueError, match="end-of-life threshold"):
        attach_target(frame, cfg)


# ===========================================================================
# 1.5 — warm-up and serving parity
# ===========================================================================
def test_first_scoreable_cycle_matches_the_warmup_trim(cfg: ExperimentConfig):
    cfg.features.drop_warmup_cycles = 5
    assert first_scoreable_cycle(cfg, family="tabular") == 6


def test_sequence_first_scoreable_cycle_includes_the_window(cfg: ExperimentConfig):
    cfg.features.drop_warmup_cycles = 5
    cfg.models.sequence.window = 8
    assert first_scoreable_cycle(cfg, family="sequence") == 13


def test_warmup_policy_min_history_matches_first_scoreable(cfg: ExperimentConfig):
    policy = WarmupPolicy(drop_warmup_cycles=10, sequence_window=20)
    assert policy.first_scoreable_cycle == 30
    assert policy.min_history_cycles == 30


def test_training_and_serving_agree_on_the_first_scoreable_cycle(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    """The row set the training builder keeps is the row set the policy predicts."""
    frame, _ = build_features(labelled_cycles, cfg.features)
    assert int(frame["cycle_index"].min()) == first_scoreable_cycle(cfg, family="tabular")


def test_scoreable_mask_excludes_pre_warmup_rows(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    mask = scoreable_mask(labelled_cycles, cfg, family="tabular")
    kept = labelled_cycles.loc[mask, "cycle_index"]
    assert kept.min() >= first_scoreable_cycle(cfg, family="tabular")
    assert (~mask).sum() > 0


def test_unknown_family_is_rejected(cfg: ExperimentConfig):
    with pytest.raises(ValueError, match="Unknown model family"):
        first_scoreable_cycle(cfg, family="quantum")


# ===========================================================================
# 1.6 — cache keying and configuration fingerprints
# ===========================================================================
def test_data_fingerprint_changes_with_a_data_affecting_field():
    base = load_config()
    other = load_config(overrides={"data.eol_threshold": 0.75})
    assert base.data_fingerprint() != other.data_fingerprint()


def test_data_fingerprint_ignores_cosmetic_fields():
    base = load_config()
    other = load_config(overrides={"viz.dpi": 300, "evaluation.bootstrap_samples": 7})
    assert base.data_fingerprint() == other.data_fingerprint()


def test_eol_persistence_is_data_affecting():
    base = load_config()
    other = load_config(overrides={"target.eol_persistence": 5})
    assert base.data_fingerprint() != other.data_fingerprint()


def test_cache_path_embeds_the_fingerprint(cfg: ExperimentConfig, monkeypatch, tmp_path):
    """A changed data-affecting setting must not reuse the old interim table."""
    from battery_rul.data import loader

    seen: list[str] = []
    original = loader.read_table

    def _spy(path, **kwargs):
        seen.append(str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(loader, "read_table", _spy)
    cfg.data.cache_interim = True
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)

    loader.load_cycles(cfg)
    first = sorted(cfg.paths.interim_dir.glob("cycles_*.parquet"))
    cfg.data.capacity_smoothing_window = 9
    loader.load_cycles(cfg)
    second = sorted(cfg.paths.interim_dir.glob("cycles_*.parquet"))

    assert len(second) == len(first) + 1, "changing a data-affecting field reused the cache"
