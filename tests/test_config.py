"""Configuration loading, validation and override semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from battery_rul.config import ExperimentConfig, load_config, project_root


def test_defaults_are_valid():
    cfg = ExperimentConfig()
    assert cfg.seed == 42
    assert cfg.data.source == "nasa"
    assert 0 < cfg.data.eol_threshold < 1


def test_paths_are_absolute_and_rooted(tmp_path: Path):
    cfg = load_config(overrides={"paths.root": str(tmp_path)})
    assert cfg.paths.processed_dir.is_absolute()
    assert cfg.paths.processed_dir == tmp_path / "data/processed"


def test_ensure_creates_directories(tmp_path: Path):
    cfg = load_config(overrides={"paths.root": str(tmp_path)})
    cfg.paths.ensure()
    for path in (cfg.paths.raw_dir, cfg.paths.models_dir, cfg.paths.figures_dir):
        assert path.is_dir()


def test_unknown_key_is_rejected():
    """Typos in YAML must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        ExperimentConfig(dat={"source": "nasa"})


def test_out_of_range_value_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(data={"eol_threshold": 1.5})


def test_dotted_overrides_are_nested_and_yaml_parsed():
    cfg = load_config(
        overrides={
            "models.training.epochs": "9",
            "models.enabled": "[ridge, xgboost]",
            "target.log_transform": "true",
        }
    )
    assert cfg.models.training.epochs == 9
    assert cfg.models.enabled == ["ridge", "xgboost"]
    assert cfg.target.log_transform is True


def test_extends_merges_parent(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump({"seed": 11, "data": {"min_cycles": 99}}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"extends": "parent.yaml", "data": {"min_cycles": 5}}))

    cfg = load_config(child)
    assert cfg.seed == 11  # inherited
    assert cfg.data.min_cycles == 5  # overridden


def test_seed_propagates_to_stages():
    cfg = load_config(overrides={"seed": 1234})
    assert cfg.split.seed == 1234
    assert cfg.tuning.seed == 1234
    assert cfg.models.training.seed == 1234


def test_eol_capacity_derivation():
    cfg = load_config(overrides={"data.nominal_capacity_ah": 2.0, "data.eol_threshold": 0.7})
    assert cfg.eol_capacity_ah == pytest.approx(1.4)


def test_sequence_head_divisibility_is_enforced():
    with pytest.raises(ValidationError):
        ExperimentConfig(models={"sequence": {"d_model": 30, "nhead": 4}})


def test_shipped_configs_all_load(repo_root: Path):
    """Every config in configs/ must parse — a broken one is a broken README."""
    configs = sorted((repo_root / "configs").glob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        cfg = load_config(path)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.experiment_name


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_round_trip_save_and_load(tmp_path: Path):
    cfg = load_config(overrides={"experiment_name": "roundtrip", "seed": 5})
    path = cfg.save(tmp_path / "out.yaml")
    reloaded = load_config(path)
    assert reloaded.experiment_name == "roundtrip"
    assert reloaded.seed == 5


def test_project_root_is_a_directory():
    assert project_root().is_dir()
