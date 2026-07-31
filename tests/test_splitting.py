"""Temporal and battery-aware splitting — the anti-leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import SplitConfig
from battery_rul.features.splitting import (
    iter_group_folds,
    make_split,
    walk_forward_folds,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = []
    for i, n in enumerate([120, 90, 140, 80, 110, 100], start=1):
        rows.append(
            pd.DataFrame(
                {
                    "battery_id": f"B{i:04d}",
                    "cycle_index": np.arange(1, n + 1),
                    "rul_cycles": np.arange(n, 0, -1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Battery holdout
# ---------------------------------------------------------------------------
def test_battery_holdout_partitions_are_disjoint(frame: pd.DataFrame):
    split = make_split(frame, SplitConfig(strategy="battery_holdout"))
    assert not (split.train & split.val).any()
    assert not (split.train & split.test).any()
    assert not (split.val & split.test).any()


def test_battery_holdout_keeps_cells_whole(frame: pd.DataFrame):
    """The core guarantee: no cell appears in two partitions."""
    split = make_split(frame, SplitConfig(strategy="battery_holdout"))
    cells = frame["battery_id"].to_numpy()
    train_cells = set(cells[split.train])
    val_cells = set(cells[split.val])
    test_cells = set(cells[split.test])

    assert not (train_cells & test_cells)
    assert not (train_cells & val_cells)
    assert not (val_cells & test_cells)


def test_battery_holdout_is_deterministic(frame: pd.DataFrame):
    a = make_split(frame, SplitConfig(strategy="battery_holdout"))
    b = make_split(frame, SplitConfig(strategy="battery_holdout"))
    assert a.test_batteries == b.test_batteries
    assert a.train_batteries == b.train_batteries


def test_explicit_test_batteries_are_honoured(frame: pd.DataFrame):
    split = make_split(
        frame, SplitConfig(strategy="battery_holdout", test_batteries=["B0002", "B0005"])
    )
    assert split.test_batteries == ["B0002", "B0005"]
    assert "B0002" not in split.train_batteries


def test_battery_holdout_needs_enough_cells():
    tiny = pd.DataFrame({"battery_id": ["A"] * 50, "cycle_index": np.arange(1, 51)})
    with pytest.raises(ValueError, match=">= 3 batteries"):
        make_split(tiny, SplitConfig(strategy="battery_holdout"))


# ---------------------------------------------------------------------------
# Chronological
# ---------------------------------------------------------------------------
def test_chronological_test_is_strictly_after_train(frame: pd.DataFrame):
    split = make_split(frame, SplitConfig(strategy="chronological", val_size=0.0, gap_cycles=0))
    cells = frame["battery_id"].to_numpy()
    cycles = frame["cycle_index"].to_numpy()

    for cell in np.unique(cells):
        mask = cells == cell
        train_cycles = cycles[mask & split.train]
        test_cycles = cycles[mask & split.test]
        if train_cycles.size and test_cycles.size:
            assert train_cycles.max() < test_cycles.min()


def test_chronological_purge_gap_is_respected(frame: pd.DataFrame):
    gap = 7
    split = make_split(frame, SplitConfig(strategy="chronological", val_size=0.0, gap_cycles=gap))
    cells = frame["battery_id"].to_numpy()
    cycles = frame["cycle_index"].to_numpy()

    for cell in np.unique(cells):
        mask = cells == cell
        train_cycles = cycles[mask & split.train]
        test_cycles = cycles[mask & split.test]
        if train_cycles.size and test_cycles.size:
            assert test_cycles.min() - train_cycles.max() > gap


def test_chronological_covers_every_cell(frame: pd.DataFrame):
    split = make_split(frame, SplitConfig(strategy="chronological"))
    cells = frame["battery_id"].to_numpy()
    assert set(cells[split.train]) == set(cells)
    assert set(cells[split.test]) == set(cells)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def test_walk_forward_training_sets_expand(frame: pd.DataFrame):
    folds = walk_forward_folds(frame, SplitConfig(n_folds=4))
    assert len(folds) >= 2
    sizes = [int(train.sum()) for train, _ in folds]
    assert sizes == sorted(sizes)
    # Each fold's training set contains the previous one.
    for (prev_train, _), (next_train, _) in zip(folds, folds[1:], strict=False):
        assert bool((prev_train & ~next_train).sum() == 0)


def test_walk_forward_test_is_always_in_the_future(frame: pd.DataFrame):
    cfg = SplitConfig(n_folds=4, gap_cycles=2)
    cells = frame["battery_id"].to_numpy()
    cycles = frame["cycle_index"].to_numpy()

    for train, test in walk_forward_folds(frame, cfg):
        for cell in np.unique(cells):
            mask = cells == cell
            tr = cycles[mask & train]
            te = cycles[mask & test]
            if tr.size and te.size:
                assert te.min() > tr.max()


def test_walk_forward_folds_never_overlap(frame: pd.DataFrame):
    for train, test in walk_forward_folds(frame, SplitConfig(n_folds=5)):
        assert not (train & test).any()


def test_walk_forward_strategy_produces_a_split(frame: pd.DataFrame):
    split = make_split(frame, SplitConfig(strategy="walk_forward", n_folds=4))
    assert split.strategy == "walk_forward"
    assert split.sizes["train"] > 0
    assert split.sizes["test"] > 0


# ---------------------------------------------------------------------------
# Grouped CV
# ---------------------------------------------------------------------------
def test_group_folds_keep_cells_whole(frame: pd.DataFrame):
    cells = frame["battery_id"].to_numpy()
    for train, test in iter_group_folds(frame, 3, seed=1):
        assert not (train & test).any()
        assert not (set(cells[train]) & set(cells[test]))


def test_group_folds_cover_every_cell_exactly_once(frame: pd.DataFrame):
    cells = frame["battery_id"].to_numpy()
    seen: set[str] = set()
    for _, test in iter_group_folds(frame, 3, seed=1):
        fold_cells = set(cells[test])
        assert not (seen & fold_cells)
        seen |= fold_cells
    assert seen == set(cells)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def test_overlapping_masks_are_rejected():
    from battery_rul.features.splitting import DataSplit

    n = 10
    overlap = np.ones(n, dtype=bool)
    with pytest.raises(ValueError, match="overlap"):
        DataSplit(train=overlap, val=np.zeros(n, bool), test=overlap, strategy="bad")


def test_empty_test_partition_is_rejected():
    from battery_rul.features.splitting import DataSplit

    n = 10
    with pytest.raises(ValueError, match="Test partition is empty"):
        DataSplit(
            train=np.ones(n, bool), val=np.zeros(n, bool), test=np.zeros(n, bool), strategy="bad"
        )


def test_label_column_matches_masks(frame: pd.DataFrame):
    split = make_split(frame, SplitConfig(strategy="battery_holdout"))
    labels = split.label_column()
    assert (labels[split.train] == "train").all()
    assert (labels[split.test] == "test").all()
