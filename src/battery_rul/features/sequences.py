"""Sliding-window tensor construction for the sequence models.

A window ending at cycle *k* contains cycles ``[k-w+1, k]`` of a **single**
battery and is labelled with ``RUL(k)`` — the target at the window's *last*
cycle. Windows never straddle a battery boundary and never extend into the
future, so the LSTM/GRU/Transformer inputs inherit the same causality guarantee
as the tabular features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SequenceBatch", "make_sequences"]


@dataclass(slots=True)
class SequenceBatch:
    """Windowed tensors plus the metadata needed to map predictions back."""

    X: np.ndarray  # (n_windows, window, n_features)
    y: np.ndarray  # (n_windows,)
    battery_ids: np.ndarray  # (n_windows,) str
    cycle_index: np.ndarray  # (n_windows,) int — the *last* cycle in each window
    feature_names: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def window(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])

    def subset(self, mask: np.ndarray) -> SequenceBatch:
        return SequenceBatch(
            X=self.X[mask],
            y=self.y[mask],
            battery_ids=self.battery_ids[mask],
            cycle_index=self.cycle_index[mask],
            feature_names=self.feature_names,
        )


def make_sequences(
    frame: pd.DataFrame,
    values: np.ndarray,
    target: np.ndarray,
    *,
    window: int,
    stride: int = 1,
    feature_names: list[str] | None = None,
    battery_col: str = "battery_id",
    cycle_col: str = "cycle_index",
) -> SequenceBatch:
    """Build ``(n_windows, window, n_features)`` tensors from a row-aligned table.

    Parameters
    ----------
    frame:
        Row metadata (must contain ``battery_id`` and ``cycle_index``), aligned
        row-for-row with ``values`` and ``target``.
    values:
        Already-scaled feature matrix — scaling must happen *before* windowing so
        the scaler is fit on training rows only.
    window, stride:
        Window length in cycles and the step between consecutive windows.

    Notes
    -----
    Batteries shorter than ``window`` contribute no windows and are reported.
    """
    if len(frame) != len(values) or len(frame) != len(target):
        raise ValueError(
            f"Row-count mismatch: frame={len(frame)}, values={len(values)}, target={len(target)}"
        )
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    xs: list[np.ndarray] = []
    ys: list[float] = []
    bids: list[str] = []
    cycles: list[int] = []
    too_short: list[str] = []

    positions = np.arange(len(frame))
    battery_values = frame[battery_col].to_numpy()
    for battery_id in pd.unique(battery_values):
        idx = positions[battery_values == battery_id]
        if idx.size < window:
            too_short.append(str(battery_id))
            continue
        # Cycle order within a battery is guaranteed by the loader, but assert it
        # rather than trust it — a mis-ordered window is silent, subtle leakage.
        cyc = frame[cycle_col].to_numpy()[idx]
        if not np.all(np.diff(cyc) > 0):
            order = np.argsort(cyc, kind="stable")
            idx = idx[order]
            cyc = cyc[order]

        for end in range(window - 1, idx.size, stride):
            sl = idx[end - window + 1 : end + 1]
            xs.append(values[sl])
            ys.append(float(target[idx[end]]))
            bids.append(str(battery_id))
            cycles.append(int(cyc[end]))

    if too_short:
        logger.warning(
            "%d battery(ies) shorter than window=%d produced no sequences: %s",
            len(too_short),
            window,
            too_short,
        )
    if not xs:
        raise ValueError(
            f"No sequences produced with window={window}. The longest battery has "
            f"{frame.groupby(battery_col).size().max()} rows."
        )

    batch = SequenceBatch(
        X=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        battery_ids=np.asarray(bids),
        cycle_index=np.asarray(cycles, dtype=np.int32),
        feature_names=list(feature_names or []),
    )
    logger.debug(
        "Windowed %d rows -> %d sequences of shape (%d, %d)",
        len(frame),
        len(batch),
        batch.window,
        batch.n_features,
    )
    return batch
