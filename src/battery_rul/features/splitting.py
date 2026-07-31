"""Leakage-free data partitioning.

Why not ``train_test_split``
----------------------------
Random row splitting on this dataset produces R^2 ≈ 0.99 and is meaningless.
Consecutive cycles of one cell are near-duplicates, so a random split puts cycle
*k-1* in train and cycle *k* in test; the model can interpolate rather than
forecast. Worse, rolling features make neighbouring rows overlap outright. The
resulting number answers no question anyone would ask about a battery.

Three honest alternatives are implemented, each answering a different question:

``battery_holdout``
    Whole cells are held out. Answers *"given a cell we have never seen, can we
    predict its remaining life?"* — the deployment question, and the hardest
    setting. This is the default and the one the headline metrics use.

``chronological``
    Within each cell, the first *p* % of cycles train and the tail tests, with an
    optional purge gap so trailing windows cannot bleed across the boundary.
    Answers *"given a cell's history so far, can we forecast its future?"* — the
    digital-twin question that milestone 2 will build on.

``walk_forward``
    Expanding-origin cross-validation: train on cycles ``[0, t)``, test on
    ``[t + gap, t + gap + h)``, advance *t*. The honest way to estimate variance
    of a forecaster over time.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from battery_rul.config import SplitConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DataSplit", "SplitPlan", "make_split", "walk_forward_folds"]


@dataclass(slots=True)
class DataSplit:
    """Boolean row masks for one train/val/test partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str
    train_batteries: list[str] = field(default_factory=list)
    val_batteries: list[str] = field(default_factory=list)
    test_batteries: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        overlap_tv = np.logical_and(self.train, self.val).sum()
        overlap_tt = np.logical_and(self.train, self.test).sum()
        overlap_vt = np.logical_and(self.val, self.test).sum()
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Split partitions overlap (train∩val={overlap_tv}, train∩test={overlap_tt}, "
                f"val∩test={overlap_vt}). This would leak."
            )
        if not self.train.any():
            raise ValueError("Training partition is empty")
        if not self.test.any():
            raise ValueError("Test partition is empty")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "val": int(self.val.sum()),
            "test": int(self.test.sum()),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "sizes": self.sizes,
            "train_batteries": self.train_batteries,
            "val_batteries": self.val_batteries,
            "test_batteries": self.test_batteries,
            "notes": self.notes,
        }

    def label_column(self) -> np.ndarray:
        """A ``train``/``val``/``test``/``unused`` label per row."""
        labels = np.full(self.train.shape[0], "unused", dtype=object)
        labels[self.train] = "train"
        labels[self.val] = "val"
        labels[self.test] = "test"
        return labels


@dataclass(slots=True)
class SplitPlan:
    """A split plus the walk-forward folds used for cross-validated tuning."""

    split: DataSplit
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.split.to_dict(),
            "n_folds": len(self.folds),
            "fold_sizes": [
                {"train": int(tr.sum()), "test": int(te.sum())} for tr, te in self.folds
            ],
        }


# ---------------------------------------------------------------------------
def make_split(df: pd.DataFrame, cfg: SplitConfig) -> DataSplit:
    """Build the configured partition for ``df`` (must contain ``battery_id``)."""
    if df.empty:
        raise ValueError("Cannot split an empty frame")

    dispatch = {
        "battery_holdout": _battery_holdout,
        "chronological": _chronological,
        "walk_forward": _walk_forward_holdout,
    }
    split = dispatch[cfg.strategy](df, cfg)
    logger.info(
        "Split[%s]: train=%d val=%d test=%d rows | test batteries=%s",
        split.strategy,
        *split.sizes.values(),
        split.test_batteries or "(per-cell tail)",
    )
    return split


def _battery_holdout(df: pd.DataFrame, cfg: SplitConfig) -> DataSplit:
    """Hold out entire cells, chosen deterministically.

    Selection is *not* uniformly random: cells are ordered by length and the
    holdout is drawn stratified across that ordering, so the test set is not
    accidentally all-short or all-long cells. With only ~16 cells available, an
    unlucky random draw would dominate the reported metric.
    """
    batteries = sorted(df["battery_id"].unique().tolist())
    if len(batteries) < 3:
        raise ValueError(
            f"battery_holdout needs >= 3 batteries, found {len(batteries)}: {batteries}. "
            "Use split.strategy=chronological for a single-cell dataset."
        )

    lengths = df.groupby("battery_id").size().sort_values(ascending=False)
    ordered = lengths.index.astype(str).tolist()

    if cfg.test_batteries:
        test = [b for b in cfg.test_batteries if b in batteries]
        unknown = sorted(set(cfg.test_batteries) - set(batteries))
        if unknown:
            logger.warning("Configured test batteries not present: %s", unknown)
    else:
        n_test = max(int(round(len(batteries) * cfg.test_size)), 1)
        # Stride through the length-ordered list to spread the draw.
        step = max(len(ordered) // n_test, 1)
        test = [ordered[i] for i in range(0, len(ordered), step)][:n_test]

    remaining = [b for b in ordered if b not in set(test)]
    if cfg.val_batteries:
        val = [b for b in cfg.val_batteries if b in remaining]
    elif cfg.val_size > 0 and len(remaining) >= 3:
        n_val = max(int(round(len(batteries) * cfg.val_size)), 1)
        n_val = min(n_val, len(remaining) - 1)
        step = max(len(remaining) // n_val, 1)
        val = [remaining[i] for i in range(0, len(remaining), step)][:n_val]
    else:
        val = []

    train = [b for b in remaining if b not in set(val)]
    if not train:
        raise ValueError("Battery holdout left no training cells; lower test_size/val_size")

    battery_col = df["battery_id"].to_numpy()
    return DataSplit(
        train=np.isin(battery_col, train),
        val=np.isin(battery_col, val),
        test=np.isin(battery_col, test),
        strategy="battery_holdout",
        train_batteries=sorted(train),
        val_batteries=sorted(val),
        test_batteries=sorted(test),
        notes=(
            "Entire cells held out. Test cells were never seen during training, "
            "scaling or feature selection."
        ),
    )


def _chronological(df: pd.DataFrame, cfg: SplitConfig) -> DataSplit:
    """Per cell: earliest cycles train, latest cycles test, with a purge gap."""
    n = len(df)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    positions = np.arange(n)
    battery_col = df["battery_id"].to_numpy()
    cycle_col = df["cycle_index"].to_numpy()

    for battery_id in pd.unique(battery_col):
        idx = positions[battery_col == battery_id]
        idx = idx[np.argsort(cycle_col[idx], kind="stable")]
        m = idx.size
        n_test = max(int(round(m * cfg.test_size)), 1)
        n_val = int(round(m * cfg.val_size))
        n_train = m - n_test - n_val - cfg.gap_cycles
        if n_train <= 0:
            logger.warning(
                "Battery %s too short to split chronologically; all rows -> train", battery_id
            )
            train[idx] = True
            continue

        train[idx[:n_train]] = True
        cursor = n_train + cfg.gap_cycles  # purge window stays unused
        if n_val:
            val[idx[cursor : cursor + n_val]] = True
            cursor += n_val
        test[idx[cursor:]] = True

    return DataSplit(
        train=train,
        val=val,
        test=test,
        strategy="chronological",
        notes=(
            f"Per-cell temporal split with a {cfg.gap_cycles}-cycle purge gap between "
            "train and the evaluation partitions."
        ),
    )


def _walk_forward_holdout(df: pd.DataFrame, cfg: SplitConfig) -> DataSplit:
    """Final fold of the walk-forward schedule, used as the reported holdout."""
    folds = walk_forward_folds(df, cfg)
    if not folds:
        raise ValueError("walk_forward produced no folds; lower walk_forward_min_train_fraction")
    train_mask, test_mask = folds[-1]
    val_mask = np.zeros(len(df), dtype=bool)
    if len(folds) >= 2:
        prev_train, prev_test = folds[-2]
        val_mask = prev_test & ~test_mask
        train_mask = train_mask & ~val_mask
    return DataSplit(
        train=train_mask,
        val=val_mask,
        test=test_mask,
        strategy="walk_forward",
        notes=f"Final fold of a {len(folds)}-fold expanding-origin schedule.",
    )


def walk_forward_folds(df: pd.DataFrame, cfg: SplitConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-origin folds, computed per cell and unioned across cells.

    Fold *j* trains on each cell's first ``f_j`` fraction of cycles and tests on
    the next ``walk_forward_horizon`` cycles (after a ``gap_cycles`` purge).
    Because the origin expands, a fold's training set is always a superset of the
    previous fold's — which is what a deployed model retrained over time would see.
    """
    n = len(df)
    positions = np.arange(n)
    battery_col = df["battery_id"].to_numpy()
    cycle_col = df["cycle_index"].to_numpy()

    per_battery: dict[str, np.ndarray] = {}
    for battery_id in pd.unique(battery_col):
        idx = positions[battery_col == battery_id]
        per_battery[str(battery_id)] = idx[np.argsort(cycle_col[idx], kind="stable")]

    fractions = np.linspace(
        cfg.walk_forward_min_train_fraction,
        1.0,
        cfg.n_folds + 1,
    )[:-1]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fraction in fractions:
        train = np.zeros(n, dtype=bool)
        test = np.zeros(n, dtype=bool)
        for idx in per_battery.values():
            m = idx.size
            cut = int(m * fraction)
            if cut < 2 or cut >= m:
                continue
            train[idx[:cut]] = True
            start = cut + cfg.gap_cycles
            stop = min(start + cfg.walk_forward_horizon, m)
            if start < stop:
                test[idx[start:stop]] = True
        if train.any() and test.any():
            folds.append((train, test))

    logger.info(
        "Walk-forward: %d usable folds (train fractions %s)",
        len(folds),
        [round(float(f), 2) for f in fractions],
    )
    return folds


def iter_group_folds(
    df: pd.DataFrame, n_folds: int, *, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Battery-grouped K-fold — used by the Optuna objective.

    Grouping by cell is essential: a plain KFold over rows would let the tuner
    select hyperparameters that memorise individual cells.
    """
    batteries = sorted(df["battery_id"].unique().tolist())
    if len(batteries) < n_folds:
        n_folds = max(2, len(batteries))
        logger.warning("Reduced CV folds to %d (only %d batteries)", n_folds, len(batteries))

    rng = np.random.default_rng(seed)
    shuffled = list(batteries)
    rng.shuffle(shuffled)
    chunks = np.array_split(np.array(shuffled, dtype=object), n_folds)

    battery_col = df["battery_id"].to_numpy()
    for chunk in chunks:
        if chunk.size == 0:
            continue
        test = np.isin(battery_col, chunk)
        yield ~test, test
