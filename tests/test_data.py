"""Data loading, schema coercion and the validation gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.data import available_sources, load_cycles
from battery_rul.data.loader import _trim_leading_artifacts, _truncate_at_collapse
from battery_rul.data.schema import REQUIRED_COLUMNS, coerce_schema, schema_frame
from battery_rul.data.validation import validate_cycles


# ---------------------------------------------------------------------------
# Registry / schema
# ---------------------------------------------------------------------------
def test_sources_are_registered():
    assert {"nasa", "synthetic"} <= set(available_sources())


def test_schema_frame_documents_every_column():
    frame = schema_frame()
    assert not frame.empty
    assert set(frame.columns) == {"column", "dtype", "unit", "required", "description"}
    assert frame["description"].str.len().min() > 0


def test_coerce_schema_fills_optional_columns(raw_cycles: pd.DataFrame):
    trimmed = raw_cycles.drop(columns=["internal_resistance_ohm"])
    out = coerce_schema(trimmed)
    assert "internal_resistance_ohm" in out.columns
    assert out["internal_resistance_ohm"].isna().all()


def test_coerce_schema_rejects_missing_required(raw_cycles: pd.DataFrame):
    with pytest.raises(ValueError, match="required schema columns"):
        coerce_schema(raw_cycles.drop(columns=["capacity_ah"]))


def test_required_columns_present_in_synthetic(raw_cycles: pd.DataFrame):
    assert set(REQUIRED_COLUMNS) <= set(raw_cycles.columns)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_load_synthetic_end_to_end(cfg: ExperimentConfig):
    dataset = load_cycles(cfg)
    assert not dataset.frame.empty
    assert dataset.metadata.synthetic is True
    assert len(dataset.batteries) >= 3
    assert {"soh", "capacity_smooth_ah", "reference_capacity_ah"} <= set(dataset.frame.columns)


def test_rows_are_sorted_by_cell_then_cycle(cfg: ExperimentConfig):
    """Ordering is load-bearing: every rolling feature assumes it."""
    frame = load_cycles(cfg).frame
    for _, group in frame.groupby("battery_id"):
        assert group["cycle_index"].is_monotonic_increasing


def test_capacity_smoothing_is_causal(cfg: ExperimentConfig):
    """A trailing median must not change when future cycles are removed."""
    from battery_rul.data.loader import _derive_health

    frame = load_cycles(cfg).frame
    battery = frame["battery_id"].iloc[0]
    full = frame[frame["battery_id"] == battery].reset_index(drop=True)
    truncated = full.iloc[:50].reset_index(drop=True)

    a = _derive_health(full, cfg)["capacity_smooth_ah"].to_numpy()[:50]
    b = _derive_health(truncated, cfg)["capacity_smooth_ah"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_soh_matches_reference_capacity(cfg: ExperimentConfig):
    frame = load_cycles(cfg).frame
    expected = frame["capacity_smooth_ah"] / frame["reference_capacity_ah"]
    np.testing.assert_allclose(frame["soh"], expected, rtol=1e-4)


def test_missing_source_falls_back_to_synthetic(cfg: ExperimentConfig):
    cfg.data.source = "nasa"
    cfg.data.subdir = "nowhere"
    cfg.data.allow_synthetic_fallback = True
    dataset = load_cycles(cfg)
    assert dataset.metadata.synthetic is True


def test_missing_source_raises_when_fallback_disabled(cfg: ExperimentConfig):
    cfg.data.source = "nasa"
    cfg.data.subdir = "nowhere"
    cfg.data.allow_synthetic_fallback = False
    with pytest.raises(FileNotFoundError):
        load_cycles(cfg)


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------
def test_validation_drops_duplicates(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    doubled = pd.concat([raw_cycles, raw_cycles.head(20)], ignore_index=True)
    out, report = validate_cycles(doubled, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert len(out) < len(doubled)
    assert any(i.check == "duplicate_cycles" for i in report.issues)


def test_validation_nullifies_impossible_voltage(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    corrupted = raw_cycles.copy()
    corrupted.loc[5, "voltage_mean_v"] = 999.0
    out, report = validate_cycles(corrupted, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert out["voltage_mean_v"].max() < 10
    assert any(i.check == "physical_bounds" for i in report.issues)


def test_validation_drops_nonpositive_capacity(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    corrupted = raw_cycles.copy()
    corrupted.loc[10, "capacity_ah"] = -1.0
    out, report = validate_cycles(corrupted, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert (out["capacity_ah"] > 0).all()
    assert any(i.check == "capacity_positive" for i in report.issues)


def test_validation_imputation_leaves_no_nans(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    corrupted = raw_cycles.copy()
    corrupted.loc[3:9, "internal_resistance_ohm"] = np.nan
    out, report = validate_cycles(corrupted, data_cfg=cfg.data, validation_cfg=cfg.validation)
    assert out["internal_resistance_ohm"].notna().all()
    assert report.imputed_cells


def test_validation_fail_fast_raises_on_error(cfg: ExperimentConfig):
    from battery_rul.data.validation import DataValidationError

    cfg.validation.fail_fast = True
    with pytest.raises(DataValidationError):
        validate_cycles(
            pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            data_cfg=cfg.data,
            validation_cfg=cfg.validation,
        )


def test_validation_report_serialises(raw_cycles: pd.DataFrame, cfg: ExperimentConfig):
    _, report = validate_cycles(raw_cycles, data_cfg=cfg.data, validation_cfg=cfg.validation)
    payload = report.to_dict()
    assert {"passed", "n_rows_in", "n_rows_out", "issues"} <= set(payload)


# ---------------------------------------------------------------------------
# Leading-artifact trim
# ---------------------------------------------------------------------------
def test_trim_removes_leading_artifacts_only(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    corrupted = raw_cycles.copy()
    first = corrupted.index[corrupted["battery_id"] == "T0001"][:3]
    corrupted.loc[first, "capacity_ah"] = 0.2  # aborted rig runs

    out = _trim_leading_artifacts(corrupted, cfg)
    kept = out[out["battery_id"] == "T0001"]
    assert len(kept) == len(raw_cycles[raw_cycles["battery_id"] == "T0001"]) - 3
    assert kept["capacity_ah"].iloc[0] > 1.0
    # Other cells untouched.
    assert len(out[out["battery_id"] == "T0002"]) == len(
        raw_cycles[raw_cycles["battery_id"] == "T0002"]
    )


def test_trim_reindexes_cycles_from_one(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    corrupted = raw_cycles.copy()
    corrupted.loc[corrupted.index[corrupted["battery_id"] == "T0001"][:2], "capacity_ah"] = 0.1
    out = _trim_leading_artifacts(corrupted, cfg)
    for _, group in out.groupby("battery_id"):
        cycles = group["cycle_index"].to_numpy()
        assert cycles[0] == 1
        assert np.array_equal(cycles, np.arange(1, len(cycles) + 1))


def test_trim_is_a_noop_on_clean_data(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    out = _trim_leading_artifacts(raw_cycles, cfg)
    assert len(out) == len(raw_cycles)


# ---------------------------------------------------------------------------
# Sustained-collapse truncation
# ---------------------------------------------------------------------------
def test_collapse_truncates_the_tail(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    """The B0042-B0044 failure mode: a regime change read as end of life.

    Capacity drops 20x in one step and stays down. That is a chamber/protocol
    change, not degradation, and the record must end there.
    """
    corrupted = raw_cycles.copy()
    mask = (corrupted["battery_id"] == "T0001") & (corrupted["cycle_index"] >= 60)
    corrupted.loc[mask, "capacity_ah"] = 0.07

    out = _truncate_at_collapse(corrupted, cfg)
    kept = out[out["battery_id"] == "T0001"]
    assert kept["cycle_index"].max() == 59
    assert kept["capacity_ah"].min() > 1.0
    # Other cells untouched.
    assert len(out[out["battery_id"] == "T0002"]) == len(
        raw_cycles[raw_cycles["battery_id"] == "T0002"]
    )


def test_collapse_ignores_a_single_bad_reading(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    """Persistence matters: one corrupt cycle must not truncate a whole record."""
    corrupted = raw_cycles.copy()
    corrupted.loc[
        (corrupted["battery_id"] == "T0001") & (corrupted["cycle_index"] == 70), "capacity_ah"
    ] = 0.05

    out = _truncate_at_collapse(corrupted, cfg)
    assert len(out[out["battery_id"] == "T0001"]) == len(
        raw_cycles[raw_cycles["battery_id"] == "T0001"]
    )


def test_collapse_is_a_noop_on_clean_data(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    assert len(_truncate_at_collapse(raw_cycles, cfg)) == len(raw_cycles)


def test_collapse_prevents_a_false_eol_label(cfg: ExperimentConfig, raw_cycles: pd.DataFrame):
    """End to end: without truncation the collapse would be labelled as EOL.

    This is the regression guard for the bug itself — a single-step jump check
    cannot see a level shift, so the EOL detector read the collapse as a
    persistent threshold crossing and mislabelled the entire cell.
    """
    from battery_rul.data.loader import _derive_health
    from battery_rul.features.target import find_eol_cycle

    corrupted = raw_cycles.copy()
    mask = (corrupted["battery_id"] == "T0001") & (corrupted["cycle_index"] >= 40)
    corrupted.loc[mask, "capacity_ah"] = 0.07

    # Without truncation: EOL is detected right at the collapse.
    untruncated = _derive_health(corrupted, cfg)
    naive_eol = find_eol_cycle(untruncated[untruncated["battery_id"] == "T0001"], cfg)
    assert naive_eol is not None and naive_eol <= 45

    # With truncation: the corrupted region is gone, so no false crossing exists.
    truncated = _derive_health(_truncate_at_collapse(corrupted, cfg), cfg)
    group = truncated[truncated["battery_id"] == "T0001"]
    assert group["capacity_ah"].min() > 1.0
    assert find_eol_cycle(group, cfg) is None


# ---------------------------------------------------------------------------
# Real NASA data (skipped when absent)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_nasa_loader_parses_real_files(nasa_available: bool, repo_root):
    if not nasa_available:
        pytest.skip("NASA .mat files not downloaded")

    from battery_rul.config import load_config

    cfg = load_config(
        overrides={
            "paths.root": str(repo_root),
            "data.batteries": ["B0005"],
            "data.cache_interim": False,
        }
    )
    dataset = load_cycles(cfg)
    assert dataset.metadata.synthetic is False
    assert dataset.metadata.n_batteries == 1
    assert dataset.metadata.n_cycles > 100
    frame = dataset.frame
    # Physical sanity of the parsed traces.
    assert frame["capacity_ah"].between(0.5, 2.5).all()
    assert frame["voltage_min_v"].between(1.0, 4.5).all()
    assert frame["capacity_smooth_ah"].iloc[-1] < frame["capacity_smooth_ah"].iloc[0]

    # Impedance is forward-filled from the last EIS sweep at or before each cycle,
    # so B0005's opening cycles — which precede the rig's first sweep — have no
    # value at all. That gap is deliberate since Milestone 1.1: it used to be
    # filled by a median over the whole loaded dataset, which crossed the
    # evaluation boundary. The values that do exist must be physical, and the
    # missing ones must be a leading run, never a hole in the middle.
    resistance = frame["internal_resistance_ohm"]
    observed = resistance.dropna()
    assert not observed.empty
    assert observed.between(0.0, 1.0).all()

    missing = resistance.isna().to_numpy()
    if missing.any():
        first_observed = int(np.argmax(~missing))
        assert missing[:first_observed].all(), "missing impedance is not a leading run"
        assert not missing[first_observed:].any(), "impedance goes missing again mid-record"
