r"""Future end-of-life risk: a derived label, and honest about it.

Definition
----------
At cycle *k* of cell *i*:

.. math::
    y^{\mathrm{risk}}_i(k) = \mathbb{1}\left[\,\mathrm{RUL}_i(k) \le H\,\right]

with horizon :math:`H` = ``risk.horizon_cycles``. In words: *will this cell reach
the configured end-of-life capacity threshold within the next H cycles?*

What this label is not
----------------------
It is **not** an observed safety failure. Nothing in the NASA dataset records a
thermal event, a venting incident, an internal short or a pack-level fault. The
label is derived arithmetically from the RUL target, which is itself derived from
a capacity threshold that the project chose. Consequently:

* a positive label means "capacity is projected to cross 70 % of reference within
  H cycles", not "this cell is about to become dangerous";
* the model cannot learn to predict safety failures, because it has never seen
  one;
* calling the output "failure risk" is a naming convention inherited from the
  prognostics literature, and every user-facing surface in this repository says
  so alongside the number.

Leakage
-------
The label at cycle *k* is a function of the *record's* end-of-life cycle, which
is established offline over the complete series. That is legitimate for a label —
supervision is always constructed with hindsight. What must never happen is a
*feature* at cycle *k* seeing beyond *k*, and that is enforced separately in
:mod:`battery_rul.features.engineering`. The two concerns are kept in separate
modules precisely so a reviewer can check them independently.

Because the label derives from the confirmed end-of-life crossing, right-censored
cells (no confirmed crossing) carry no valid risk label and are excluded by the
same ``target.require_eol_reached`` gate that governs RUL.

Class imbalance
---------------
Positives are rare by construction: only the final H cycles of each cell qualify,
so at H = 30 over ~100-cycle cells roughly a quarter of rows are positive, and at
H = 20 fewer. The handling is class weighting plus threshold tuning on
validation, never row-level oversampling — duplicating rows of a time series
breaks the temporal structure the sequence models depend on and inflates every
metric computed over the resampled set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig, RiskConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "RiskClass",
    "RiskTargetReport",
    "attach_failure_risk_target",
    "classify_risk",
    "risk_target_column",
]


class RiskClass(StrEnum):
    """Banding of the calibrated failure-risk probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def severity(self) -> int:
        return {
            RiskClass.LOW: 0,
            RiskClass.MEDIUM: 1,
            RiskClass.HIGH: 2,
            RiskClass.VERY_HIGH: 3,
            RiskClass.UNKNOWN: 4,
        }[self]


def classify_risk(probability: float | None, cfg: RiskConfig) -> RiskClass:
    """Map a probability in [0, 1] onto its risk band."""
    if probability is None or not np.isfinite(probability):
        return RiskClass.UNKNOWN
    if probability < cfg.low_max:
        return RiskClass.LOW
    if probability < cfg.medium_max:
        return RiskClass.MEDIUM
    if probability < cfg.high_max:
        return RiskClass.HIGH
    return RiskClass.VERY_HIGH


def risk_target_column(horizon: int, base_name: str = "failure_within_horizon") -> str:
    """Column name for a given horizon, so several horizons can coexist."""
    return f"{base_name}_{int(horizon)}"


@dataclass(slots=True)
class RiskTargetReport:
    """What risk target generation did, per horizon."""

    primary_horizon: int
    horizons: list[int]
    positive_rate: dict[int, float]
    positives: dict[int, int]
    n_rows: int
    excluded_censored: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": "RUL(t) <= H, derived from the capacity end-of-life threshold",
            "is_observed_safety_failure": False,
            "primary_horizon_cycles": self.primary_horizon,
            "horizons": self.horizons,
            "n_rows": self.n_rows,
            "positives": {str(k): v for k, v in self.positives.items()},
            "positive_rate": {str(k): round(v, 5) for k, v in self.positive_rate.items()},
            "excluded_right_censored_batteries": self.excluded_censored,
        }


def attach_failure_risk_target(
    df: pd.DataFrame, cfg: ExperimentConfig
) -> tuple[pd.DataFrame, RiskTargetReport]:
    """Add a binary risk label per configured horizon.

    Requires the RUL target to be attached already (``attach_target``), because
    the risk label is a thresholding of it and must not re-derive end of life
    independently — two derivations drift apart the moment either changes.

    Columns added
    -------------
    ``failure_within_horizon_{H}``  int8 — one per configured horizon
    ``failure_within_horizon``      int8 — alias for the primary horizon, the
                                    column models actually train on
    """
    if df.empty:
        raise ValueError("Cannot attach a risk target to an empty frame")

    risk_cfg = cfg.risk
    target = cfg.target.name
    if target not in df.columns:
        raise KeyError(
            f"Risk target generation needs the RUL column {target!r}. Call "
            "attach_target() first — the risk label is a thresholding of RUL, "
            "not an independent derivation."
        )

    frame = df.copy()
    horizons = sorted({risk_cfg.horizon_cycles, *risk_cfg.additional_horizons})
    rul = frame[target].to_numpy(dtype=float)

    censored: list[str] = []
    if "is_censored" in frame.columns:
        censored = sorted(
            str(b) for b in frame.loc[frame["is_censored"].astype(bool), "battery_id"].unique()
        )
        if censored:
            logger.warning(
                "%d right-censored cell(s) carry no valid risk label and are labelled " "NaN: %s",
                len(censored),
                censored,
            )

    valid = np.isfinite(rul)
    if censored:
        valid &= ~frame["battery_id"].astype(str).isin(censored).to_numpy()

    positives: dict[int, int] = {}
    positive_rate: dict[int, float] = {}
    for horizon in horizons:
        column = risk_target_column(horizon, risk_cfg.target_name)
        label = np.where(valid, (rul <= float(horizon)).astype(float), np.nan)
        frame[column] = pd.array(label, dtype="Float32")
        n_pos = int(np.nansum(label))
        n_valid = int(np.sum(valid))
        positives[horizon] = n_pos
        positive_rate[horizon] = n_pos / n_valid if n_valid else float("nan")

    primary = risk_target_column(risk_cfg.horizon_cycles, risk_cfg.target_name)
    frame[risk_cfg.target_name] = frame[primary]

    report = RiskTargetReport(
        primary_horizon=risk_cfg.horizon_cycles,
        horizons=horizons,
        positive_rate=positive_rate,
        positives=positives,
        n_rows=len(frame),
        excluded_censored=censored,
    )
    logger.info(
        "Risk target (H=%d): %d/%d positive (%.1f%%); horizons %s also attached",
        risk_cfg.horizon_cycles,
        positives[risk_cfg.horizon_cycles],
        int(np.sum(valid)),
        100 * positive_rate[risk_cfg.horizon_cycles],
        [h for h in horizons if h != risk_cfg.horizon_cycles],
    )
    return frame, report
