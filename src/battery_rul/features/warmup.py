"""Warm-up policy — one definition of "the earliest cycle we can honestly score".

Two independent requirements stack up before a row is scoreable:

``features.drop_warmup_cycles``
    Rolling windows at the very start of a cell are only partially populated.
    Training discards those rows, so the model has never seen such a row and its
    behaviour there is undefined. Serving must discard them too.

sequence ``window``
    A sequence model consumes ``window`` consecutive rows *of the already
    warm-up-trimmed table*. During training the first window therefore ends at
    ``drop_warmup_cycles + window``; no shorter window was ever presented.

Milestone 1.1 exists partly because those two facts lived only in the training
code. Serving that pads a short history, or that windows the untrimmed table,
produces confident-looking predictions from inputs the model never encountered.
This module is the single source of truth both paths call, so the first
scoreable cycle is the same number in training, evaluation and the digital-twin
service, by construction rather than by discipline.

The cycle numbering here is the canonical ``cycle_index``: 1-based, gap-free,
assigned by the loader after leading rig artifacts are trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "WarmupPolicy",
    "first_scoreable_cycle",
    "scoreable_mask",
]


@dataclass(frozen=True, slots=True)
class WarmupPolicy:
    """The context requirement for one model family."""

    drop_warmup_cycles: int
    sequence_window: int | None

    @property
    def is_sequence(self) -> bool:
        return self.sequence_window is not None

    @property
    def first_scoreable_cycle(self) -> int:
        """Lowest ``cycle_index`` this family can score.

        Tabular: the first cycle surviving the warm-up trim.
        Sequence: that cycle plus ``window - 1`` further cycles of context.
        """
        first = self.drop_warmup_cycles + 1
        if self.sequence_window is None:
            return first
        return first + int(self.sequence_window) - 1

    @property
    def min_history_cycles(self) -> int:
        """Rows a caller must supply before any prediction is possible."""
        return self.first_scoreable_cycle

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "drop_warmup_cycles": self.drop_warmup_cycles,
            "sequence_window": self.sequence_window,
            "first_scoreable_cycle": self.first_scoreable_cycle,
            "min_history_cycles": self.min_history_cycles,
        }


def _policy(cfg: ExperimentConfig, *, family: str) -> WarmupPolicy:
    if family == "tabular":
        return WarmupPolicy(cfg.features.drop_warmup_cycles, None)
    if family == "sequence":
        return WarmupPolicy(cfg.features.drop_warmup_cycles, cfg.models.sequence.window)
    if family == "multitask":
        return WarmupPolicy(cfg.features.drop_warmup_cycles, cfg.multitask.window)
    raise ValueError(f"Unknown model family {family!r}; expected tabular/sequence/multitask")


def first_scoreable_cycle(cfg: ExperimentConfig, *, family: str = "tabular") -> int:
    """First ``cycle_index`` the given model family can score under ``cfg``."""
    return _policy(cfg, family=family).first_scoreable_cycle


def scoreable_mask(
    frame: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    family: str = "tabular",
    cycle_col: str = "cycle_index",
) -> np.ndarray:
    """Boolean mask of rows the family may score, by the training-time rule.

    Applied identically at evaluation and at serving. Rows outside it are
    reported as unscored rather than dropped, so coverage is always visible in
    the metric tables instead of hiding in a denominator.
    """
    threshold = first_scoreable_cycle(cfg, family=family)
    return frame[cycle_col].to_numpy(dtype=int) >= threshold
