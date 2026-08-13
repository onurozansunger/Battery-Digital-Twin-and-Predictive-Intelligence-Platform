"""Shared fixtures.

The suite runs entirely on the synthetic generator so it is fast (seconds) and
requires no dataset download. Tests that specifically need the real NASA files
are marked and skip cleanly when the files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.data.synthetic import make_synthetic_cycles

#: Cloud-sync duplicates ("test_api 2.py"). They are byte copies of real test
#: modules, so collecting them runs every test twice and — because the copy is a
#: stale snapshot — fails against current code. Ignored here as well as in
#: .gitignore, because pytest does not read .gitignore.
collect_ignore_glob = ["* [0-9].py"]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path: Path) -> ExperimentConfig:
    """A small, fast configuration writing everything under ``tmp_path``."""
    return load_config(
        overrides={
            "experiment_name": "test",
            "seed": 7,
            "paths.root": str(tmp_path),
            "data.source": "synthetic",
            "data.subdir": "synthetic",
            "data.cache_interim": False,
            "features.rolling_windows": [3, 5],
            "features.lags": [1, 3],
            "features.slope_windows": [5],
            "features.ewm_halflives": [5],
            "features.drop_warmup_cycles": 5,
            "features.max_features": 30,
            "models.enabled": ["ridge", "random_forest"],
            "models.training.epochs": 3,
            "models.sequence.window": 8,
            "evaluation.bootstrap_samples": 0,
            "evaluation.nested_enabled": False,
            "explainability.enabled": False,
        }
    )


@pytest.fixture
def raw_cycles() -> pd.DataFrame:
    """Four synthetic cells in canonical-schema form."""
    frames = [
        make_synthetic_cycles("T0001", n_cycles=140, seed=1),
        make_synthetic_cycles("T0002", n_cycles=120, seed=2),
        make_synthetic_cycles("T0003", n_cycles=160, seed=3),
        make_synthetic_cycles("T0004", n_cycles=130, seed=4),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = "synthetic"
    return df


@pytest.fixture
def labelled_cycles(raw_cycles: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Health-derived and RUL-labelled cycles, ready for feature engineering."""
    from battery_rul.data.loader import _derive_health
    from battery_rul.features.target import attach_target

    df = _derive_health(raw_cycles, cfg)
    labelled, _ = attach_target(df, cfg)
    return labelled


@pytest.fixture(scope="session")
def nasa_available(repo_root: Path) -> bool:
    mat_dir = repo_root / "data" / "raw" / "nasa" / "mat"
    return mat_dir.is_dir() and any(mat_dir.glob("*.mat"))


# ---------------------------------------------------------------------------
# Milestone 3 — a trained platform, built once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def m3_config(tmp_path_factory) -> ExperimentConfig:
    """A small configuration rooted in a temporary directory.

    Everything Milestone 3 writes — registry, database, monitoring artifacts —
    is anchored to ``paths.root``, so a test session never touches the
    developer's real artifacts. That is the whole reason those paths resolve
    against the configuration rather than the repository root.
    """
    root = tmp_path_factory.mktemp("m3")
    return load_config(
        "configs/synthetic.yaml",
        overrides={
            "paths.root": str(root),
            "data.cache_interim": False,
            "evaluation.nested_enabled": False,
            "models.enabled": ["ridge", "random_forest"],
            "features.max_features": 25,
            "features.rolling_windows": [3, 5],
            "features.lags": [1, 3],
            "features.slope_windows": [5],
            "features.ewm_halflives": [5],
            "multitask.enabled": False,
            "uncertainty.min_calibration_rows": 10,
            "calibration.min_calibration_rows": 10,
            "explainability.enabled": False,
            "persistence.backend": "sqlite",
            "monitoring.drift.min_sample_size": 20,
            "monitoring.prediction_drift.min_sample_size": 3,
            "monitoring.performance.min_labels": 5,
        },
    )


@pytest.fixture(scope="session")
def m3_platform(m3_config: ExperimentConfig):
    """Real bundles trained on synthetic cells, plus the processed cycles.

    Session-scoped because building four small models per test module would
    dominate the suite's runtime. Fixture metrics are **not** model performance
    and no test asserts a quality number: these tests assert behaviour.
    """
    from battery_rul.pipelines import prepare_data
    from battery_rul.pipelines.milestone_2 import build_bundles, prepare_multitask_data

    prepared = prepare_data.run(m3_config, verify_leakage=False)
    data = prepare_multitask_data(m3_config, prepared=prepared)
    build_bundles(m3_config, data)
    return m3_config, prepared.cycles


@pytest.fixture(scope="session")
def fleet_service(m3_platform):
    from battery_rul.fleet.inference import FleetInferenceService

    cfg, _ = m3_platform
    return FleetInferenceService.create(cfg, strict=True)


@pytest.fixture(scope="session")
def fleet_histories(m3_platform):
    """Validated per-battery histories for the synthetic cohort."""
    from battery_rul.fleet.ingestion import FleetIngestor

    cfg, cycles = m3_platform
    label_prefixes = (
        "rul_",
        "eol_",
        "life_",
        "is_censored",
        "soh",
        "capacity_smooth",
        "reference_capacity",
        "capacity_fade",
        "equivalent_full",
        "failure_within",
        "split",
    )
    frame = cycles.drop(
        columns=[c for c in cycles.columns if c.startswith(label_prefixes)], errors="ignore"
    )
    return FleetIngestor(cfg=cfg).from_frame("TEST-FLEET", frame, source="fixture")


@pytest.fixture(scope="session")
def fleet_snapshot(fleet_service, fleet_histories):
    ingestion, histories = fleet_histories
    return fleet_service.create_fleet_snapshot(
        "TEST-FLEET", histories, ingestion=ingestion, batch_id="test-batch-0001"
    )
