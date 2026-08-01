r"""Target definition: Remaining Useful Life in cycles.

Formal definition
-----------------
Let :math:`\tilde Q_i(k)` be the trailing-median-smoothed discharge capacity of
cell *i* at discharge cycle *k*, and let

.. math::
    Q_{EOL} = \theta \cdot Q_{nom}

be the end-of-life capacity, where :math:`Q_{nom}` is the manufacturer's nominal
capacity (2.0 Ah for the NASA 18650 cells) and :math:`\theta` the EOL threshold
(0.70 by default, i.e. 70 % state of health — the convention in the automotive
second-life literature and the value used in most NASA-dataset papers).

The **end-of-life cycle** is the first crossing that *persists*:

.. math::
    k^{EOL}_i = \min \{\, k : \tilde Q_i(j) \le Q_{EOL}\ \ \forall\, j \in [k, k+p) \,\}

with persistence :math:`p` (default 3 cycles). The persistence requirement is not
cosmetic: lithium-ion cells exhibit *capacity recovery* after rest periods, so a
single dip below threshold is routinely followed by several cycles back above it.
Taking the first bare crossing would systematically under-estimate life.

The target is then

.. math::
    \mathrm{RUL}_i(k) = k^{EOL}_i - k , \qquad k \le k^{EOL}_i

measured in **discharge cycles**. RUL is 0 exactly at end of life.

Design decisions, stated explicitly
-----------------------------------
* **Cycles, not wall-clock time.** The NASA rig ran cells at different duty
  cycles with long idle gaps; calendar time is an artifact of lab scheduling,
  cycle count is the physically meaningful ageing clock.
* **Right-censored cells are excluded** by default
  (``target.require_eol_reached``). A cell that never degrades to 70 % SoH has an
  unknown RUL; training on a censored label teaches the model to predict the
  experiment's end date. Handling those properly needs survival methods — noted
  as future work rather than fudged.
* **Post-EOL cycles are dropped** by default. RUL below zero is not defined here;
  keeping those rows would force the model to extrapolate into a regime the
  business question never asks about.
* **Optional piecewise-linear cap** (``target.cap_at``). Early-life RUL is
  essentially unpredictable — a fresh cell looks identical whether it will last
  120 or 160 cycles. Capping the target is the standard remedy (introduced for
  the C-MAPSS benchmark) and is available but **off** by default so headline
  metrics remain comparable to the raw-RUL literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "eol_capacity_for",
    "EOL_PERSISTENCE",
    "TargetReport",
    "attach_target",
    "find_eol_cycle",
    "inverse_transform_target",
    "transform_target",
]

#: Default consecutive cycles that must stay at/below threshold for a crossing to
#: count. The live value is ``target.eol_persistence``; this constant is only the
#: pydantic default and is kept for backwards compatibility of imports.
EOL_PERSISTENCE = 3


@dataclass(slots=True)
class TargetReport:
    """What target generation did, per battery."""

    eol_cycles: dict[str, int]
    censored_batteries: list[str]
    dropped_rows: int
    n_rows: int
    rul_min: float
    rul_max: float
    rul_mean: float
    eol_capacity_ah: float

    def to_dict(self) -> dict[str, object]:
        return {
            "eol_capacity_ah": round(self.eol_capacity_ah, 4),
            "eol_cycles": self.eol_cycles,
            "censored_batteries": self.censored_batteries,
            "dropped_rows": self.dropped_rows,
            "n_rows": self.n_rows,
            "rul_min": self.rul_min,
            "rul_max": self.rul_max,
            "rul_mean": round(self.rul_mean, 3),
        }


def eol_capacity_for(group: pd.DataFrame, cfg: ExperimentConfig) -> float:
    """Absolute end-of-life capacity (Ah) for one cell.

    Under ``data.eol_reference == "nominal"`` this is a fleet-wide constant
    (``theta * Q_nom``). Under ``"initial"`` it is per cell
    (``theta * Q_i(1)``), which matters when cells do not all leave the factory
    at the same capacity.
    """
    if cfg.data.eol_reference == "nominal":
        return cfg.eol_capacity_ah
    if "reference_capacity_ah" in group.columns:
        reference = float(group["reference_capacity_ah"].iloc[0])
    else:
        reference = float(group["capacity_smooth_ah"].iloc[0])
    return reference * cfg.data.eol_threshold


def find_eol_cycle(
    group: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    persistence: int | None = None,
    capacity_col: str = "capacity_smooth_ah",
) -> int | None:
    """First *confirmed* end-of-life crossing for one battery, or ``None``.

    A crossing is confirmed only when **P complete consecutive observations** at
    or below the threshold exist, where P is ``target.eol_persistence``.

    Milestone 1.1 changed this. The previous implementation accepted a crossing
    that held merely "for every remaining observation", so a cell whose final one
    or two rows dipped below threshold was labelled as having reached end of life
    on the strength of one or two readings — exactly the transient dip the
    persistence rule exists to reject, and precisely where lithium-ion capacity
    recovery makes a single reading least trustworthy. A record that ends before
    persistence can be confirmed is **right-censored**: the honest statement is
    "we do not know when this cell reaches end of life", and ``None`` is what
    says that. ``attach_target`` then applies the configured censoring policy.

    ``group`` must be a single battery's rows, sorted by ``cycle_index``.
    """
    if group.empty:
        return None

    p = max(int(cfg.target.eol_persistence if persistence is None else persistence), 1)
    threshold = eol_capacity_for(group, cfg)
    capacity = group[capacity_col].to_numpy(dtype=float)
    cycles = group["cycle_index"].to_numpy(dtype=int)

    below = capacity <= threshold
    n = below.size
    if not below.any() or n < p:
        return None

    for idx in range(n - p + 1):
        if below[idx : idx + p].all():
            return int(cycles[idx])
    return None


def transform_target(y: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Apply the configured target transform (identity unless ``log_transform``)."""
    y = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y, 0.0, None)) if cfg.target.log_transform else y


def inverse_transform_target(y: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Invert :func:`transform_target` and re-apply the physical floor."""
    y = np.asarray(y, dtype=float)
    if cfg.target.log_transform:
        y = np.expm1(y)
    if cfg.target.clip_negative:
        y = np.clip(y, 0.0, None)
    return y


def attach_target(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, TargetReport]:
    """Add the RUL columns to a canonical cycle table.

    Columns added
    -------------
    ``rul_cycles``      int   — the supervised target (post-cap, pre-log)
    ``rul_raw_cycles``  int   — uncapped remaining cycles, kept for reporting
    ``eol_cycle``       int   — this battery's end-of-life cycle index
    ``life_fraction``   float — ``cycle_index / eol_cycle`` in [0, 1]
    ``is_censored``     bool  — battery never reached EOL
    """
    if df.empty:
        raise ValueError("Cannot attach a target to an empty frame")

    df = df.sort_values(["battery_id", "cycle_index"], kind="stable").reset_index(drop=True)

    eol_cycles: dict[str, int] = {}
    censored: list[str] = []
    for battery_id, group in df.groupby("battery_id", sort=True):
        eol = find_eol_cycle(group, cfg)
        if eol is None:
            censored.append(str(battery_id))
            # Right-censored: the best available lower bound on life is the last
            # observed cycle. Only used when require_eol_reached is disabled.
            eol_cycles[str(battery_id)] = int(group["cycle_index"].iloc[-1])
        else:
            eol_cycles[str(battery_id)] = int(eol)

    if censored:
        logger.warning(
            "%d battery(ies) never reach %.0f%% SoH: %s",
            len(censored),
            100 * cfg.data.eol_threshold,
            censored,
        )

    n_before = len(df)
    if cfg.target.require_eol_reached and censored:
        df = df.loc[~df["battery_id"].isin(censored)].reset_index(drop=True)
        logger.info("Dropped %d rows from censored batteries", n_before - len(df))
        if df.empty:
            raise ValueError(
                f"No battery reaches the end-of-life threshold "
                f"({cfg.data.eol_threshold:.0%} SoH = {cfg.eol_capacity_ah:.3f} Ah). "
                "Either the nominal capacity is wrong for this dataset or the "
                "threshold is too aggressive."
            )

    df["eol_cycle"] = df["battery_id"].map(eol_cycles).astype("int32")
    df["is_censored"] = df["battery_id"].isin(censored)
    df["rul_raw_cycles"] = (df["eol_cycle"] - df["cycle_index"]).astype("int32")

    if cfg.target.drop_post_eol:
        post_eol = df["rul_raw_cycles"] < 0
        if post_eol.any():
            logger.info("Dropped %d post-EOL cycles", int(post_eol.sum()))
            df = df.loc[~post_eol].reset_index(drop=True)

    # Cells that cross EOL almost immediately contribute a handful of rows that
    # describe a cold cell rather than a worn one; keeping them would let a few
    # unrepresentative cycles steer the loss.
    counts = df.groupby("battery_id", sort=False).size()
    too_few = counts[counts < cfg.target.min_labelled_cycles]
    if len(too_few):
        logger.info(
            "Dropped %d cell(s) with fewer than %d labelled cycles: %s",
            len(too_few),
            cfg.target.min_labelled_cycles,
            dict(too_few),
        )
        df = df.loc[~df["battery_id"].isin(too_few.index)].reset_index(drop=True)
        for battery_id in too_few.index:
            eol_cycles.pop(str(battery_id), None)
        if df.empty:
            raise ValueError(
                f"target.min_labelled_cycles={cfg.target.min_labelled_cycles} removed "
                "every cell. Lower it or relax the EOL threshold."
            )

    rul = df["rul_raw_cycles"].to_numpy(dtype=float)
    if cfg.target.clip_negative:
        rul = np.clip(rul, 0.0, None)
    if cfg.target.cap_at is not None:
        rul = np.minimum(rul, float(cfg.target.cap_at))

    df[cfg.target.name] = rul.astype("float32")
    df["life_fraction"] = (df["cycle_index"] / df["eol_cycle"].replace(0, np.nan)).astype("float32")

    report = TargetReport(
        eol_cycles={k: v for k, v in eol_cycles.items() if k in set(df["battery_id"])},
        censored_batteries=censored,
        dropped_rows=n_before - len(df),
        n_rows=len(df),
        rul_min=float(df[cfg.target.name].min()),
        rul_max=float(df[cfg.target.name].max()),
        rul_mean=float(df[cfg.target.name].mean()),
        eol_capacity_ah=cfg.eol_capacity_ah,
    )
    logger.info(
        "Target %s: n=%d, range=[%.0f, %.0f], mean=%.1f cycles over %d batteries",
        cfg.target.name,
        report.n_rows,
        report.rul_min,
        report.rul_max,
        report.rul_mean,
        len(report.eol_cycles),
    )
    return df, report
