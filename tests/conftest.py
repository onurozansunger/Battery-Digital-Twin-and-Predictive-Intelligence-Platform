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
