"""Stage 1 — raw files to a modelling-ready dataset.

Responsibilities: load, validate, label, engineer features, plan the split, and
write everything to ``data/processed`` with a manifest recording exactly what was
produced and from what. Downstream stages read the manifest, never the raw files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.data.loader import battery_summary_table, load_cycles
from battery_rul.features.engineering import assert_no_leakage, build_features, feature_columns
from battery_rul.features.splitting import make_split, walk_forward_folds
from battery_rul.features.target import attach_target
from battery_rul.utils.io import environment_fingerprint, read_table, save_json, write_table
from battery_rul.utils.logging import get_logger, log_section
from battery_rul.utils.seed import seed_everything
from battery_rul.utils.timing import StageTimer

logger = get_logger(__name__)

__all__ = ["PreparedData", "load_prepared", "run"]

DATASET_FILENAME = "dataset.parquet"
CYCLES_FILENAME = "cycles.parquet"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class PreparedData:
    """The output of stage 1, in memory."""

    frame: pd.DataFrame
    cycles: pd.DataFrame
    feature_names: list[str]
    split: Any
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def target_column(self) -> str:
        return str(self.manifest.get("target", {}).get("name", "rul_cycles"))

    def summary(self) -> str:
        sizes = self.split.sizes
        return (
            f"{len(self.frame)} rows x {len(self.feature_names)} features | "
            f"train={sizes['train']} val={sizes['val']} test={sizes['test']}"
        )


def run(cfg: ExperimentConfig, *, verify_leakage: bool = True) -> PreparedData:
    """Execute stage 1 and persist its artifacts."""
    log_section(logger, "stage 1 — prepare data")
    seed_everything(cfg.seed)
    cfg.paths.ensure()
    timer = StageTimer()

    with timer("load_and_validate"):
        dataset = load_cycles(cfg)
        cycles = dataset.frame

    with timer("attach_target"):
        labelled, target_report = attach_target(cycles, cfg)

    with timer("build_features"):
        features, feature_report = build_features(labelled, cfg.features)

    if verify_leakage:
        with timer("verify_leakage"):
            longest = labelled.groupby("battery_id").size().idxmax()
            assert_no_leakage(labelled, cfg.features, battery_id=str(longest))

    with timer("plan_split"):
        split = make_split(features, cfg.split)
        features = features.copy()
        features["split"] = split.label_column()
        folds = walk_forward_folds(features, cfg.split)

    names = feature_columns(features)
    if not names:
        raise ValueError("Feature engineering produced no usable feature columns")

    manifest: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "environment": environment_fingerprint(),
        "config": cfg.to_dict(),
        "dataset": dataset.metadata.to_dict(),
        "validation": dataset.validation.to_dict(),
        "target": {"name": cfg.target.name, **target_report.to_dict()},
        "features": {**feature_report.to_dict(), "n_model_features": len(names)},
        "split": split.to_dict(),
        "walk_forward_folds": len(folds),
        "n_rows": len(features),
        "battery_summary": battery_summary_table(dataset, cfg).to_dict(orient="records"),
        "timings_s": timer.as_dict(),
    }

    outdir = cfg.paths.processed_dir
    write_table(features, outdir / DATASET_FILENAME)
    write_table(labelled, outdir / CYCLES_FILENAME)
    save_json(manifest, outdir / MANIFEST_FILENAME)

    prepared = PreparedData(
        frame=features,
        cycles=labelled,
        feature_names=names,
        split=split,
        folds=folds,
        manifest=manifest,
    )
    logger.info("Stage 1 complete: %s", prepared.summary())
    logger.info("Artifacts -> %s", outdir)
    return prepared


def load_prepared(cfg: ExperimentConfig) -> PreparedData:
    """Re-hydrate stage 1's output without recomputing it.

    The split is *recomputed* from the persisted ``split`` column rather than
    re-derived from config, so a later stage can never silently disagree with the
    partition the features were prepared under.
    """
    from battery_rul.features.splitting import DataSplit

    outdir = cfg.paths.processed_dir
    dataset_path = outdir / DATASET_FILENAME
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"{dataset_path} not found. Run `python scripts/prepare_data.py` first."
        )

    frame = read_table(dataset_path)
    cycles = read_table(outdir / CYCLES_FILENAME)
    manifest = {}
    manifest_path = outdir / MANIFEST_FILENAME
    if manifest_path.is_file():
        from battery_rul.utils.io import load_json

        manifest = load_json(manifest_path)

    labels = frame["split"].to_numpy()
    split_info = manifest.get("split", {})
    split = DataSplit(
        train=labels == "train",
        val=labels == "val",
        test=labels == "test",
        strategy=str(split_info.get("strategy", cfg.split.strategy)),
        train_batteries=list(split_info.get("train_batteries", [])),
        val_batteries=list(split_info.get("val_batteries", [])),
        test_batteries=list(split_info.get("test_batteries", [])),
        notes=str(split_info.get("notes", "")),
    )
    folds = walk_forward_folds(frame, cfg.split)

    return PreparedData(
        frame=frame,
        cycles=cycles,
        feature_names=feature_columns(frame),
        split=split,
        folds=folds,
        manifest=manifest,
    )
