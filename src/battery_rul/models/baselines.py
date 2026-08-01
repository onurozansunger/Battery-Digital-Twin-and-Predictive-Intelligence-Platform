"""Interpretable baselines the learned models must beat to be worth anything.

A gradient-boosted ensemble that reports MAE 8 cycles is meaningless until you
know what a one-line rule reports on the same rows. All three baselines here are
fitted with exactly the same interface, on exactly the same partitions, and are
scored on exactly the same rows as every other model, so the comparison table
answers "did learning help?" rather than "did we train something?".

Included
--------
``cohort_median_life``
    Predict ``median(EOL cycle over the training cells) - current cycle``. Knows
    nothing about the cell's condition; it only knows how long cells in the
    training cohort tend to last. Surprisingly strong on a homogeneous cohort,
    which is exactly why it belongs in the table.

``capacity_fade_extrapolation``
    Fit a line to the cell's recent capacity trajectory and extrapolate to the
    end-of-life capacity. This is the engineer's back-of-the-envelope method and
    the honest incumbent: it uses no training data about other cells at all.

``soh_analogue``
    Nearest-analogue table lookup on state of health. The memoryless stand-in
    for a persistence baseline, which does not transfer to an unseen cell.

Regularised linear regression (``elastic_net``) already exists in
:mod:`battery_rul.models.classical` and is included in the same comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from battery_rul.models.base import BaseModel, TrainingData, register_model
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CapacityFadeExtrapolationBaseline",
    "CohortMedianLifeBaseline",
    "SOHAnalogueBaseline",
]


@register_model("cohort_median_life")
@dataclass
class CohortMedianLifeBaseline(BaseModel):
    """``median training EOL cycle - current cycle``, clipped at zero.

    Uses no features whatsoever. The target it is fitted against is already
    ``eol_cycle - cycle_index``, so the fitted quantity is recovered as
    ``median(rul + cycle)`` over training rows, which equals the median training
    end-of-life cycle without needing the label column separately.
    """

    median_eol_cycle: float = 0.0
    statistic: str = "median"

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> BaseModel:
        cycles = np.asarray(train.cycle_index, dtype=float)
        eol = np.asarray(train.y, dtype=float) + cycles
        # One value per training cell, so a long cell does not outvote a short one.
        per_cell = [
            float(np.median(eol[train.battery_ids == cell]))
            for cell in np.unique(train.battery_ids)
        ]
        self.statistic = str(self.params.get("statistic", "median"))
        reducer = np.median if self.statistic == "median" else np.mean
        self.median_eol_cycle = float(reducer(per_cell))
        self.fitted = True
        self.fit_metadata = {
            "median_eol_cycle": self.median_eol_cycle,
            "n_train_cells": len(per_cell),
            "statistic": self.statistic,
            "uses_features": False,
        }
        logger.info(
            "cohort_median_life fitted: %s training end-of-life cycle = %.1f",
            self.statistic,
            self.median_eol_cycle,
        )
        return self

    def predict(self, data: TrainingData) -> np.ndarray:
        self._check_fitted()
        cycles = np.asarray(data.cycle_index, dtype=float)
        return np.clip(self.median_eol_cycle - cycles, 0.0, None)


@register_model("capacity_fade_extrapolation")
@dataclass
class CapacityFadeExtrapolationBaseline(BaseModel):
    """Linear extrapolation of the cell's own recent capacity fade to threshold.

    At cycle *k*, fit an ordinary least-squares line to the trailing ``lookback``
    capacity readings of that cell and solve for the cycle at which the line
    reaches the end-of-life capacity. Strictly causal: only cycles ``<= k`` enter.

    ``fit`` learns only the end-of-life capacity from configuration (a constant
    under the ``nominal`` reference) plus two quantities taken from the training
    cells: a fallback for cells whose trend is flat or rising, and a ceiling.

    The ceiling matters. Near a plateau the fitted slope approaches zero, the
    extrapolated crossing runs off to thousands of cycles, and a handful of such
    rows dominate the mean error — which says nothing about the method and
    everything about dividing by a small number. An engineer doing this by hand
    would not report "this cell has 4 000 cycles left"; the ceiling is the
    longest life the training cohort actually exhibited, which is the same
    judgement expressed as a rule.
    """

    eol_capacity_ah: float = 0.0
    fallback_rul: float = 0.0
    max_rul: float = float("inf")
    lookback: int = 20

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> BaseModel:
        self.lookback = int(self.params.get("lookback", 20))
        self.eol_capacity_ah = float(self.cfg.eol_capacity_ah)
        targets = np.asarray(train.y, dtype=float)
        self.fallback_rul = float(np.median(targets))
        self.max_rul = float(np.nanmax(targets)) if targets.size else float("inf")
        self.fitted = True
        self.fit_metadata = {
            "eol_capacity_ah": self.eol_capacity_ah,
            "lookback": self.lookback,
            "fallback_rul": self.fallback_rul,
            "max_rul_from_training_cohort": self.max_rul,
            "uses_features": False,
        }
        return self

    def predict(self, data: TrainingData) -> np.ndarray:
        self._check_fitted()
        frame = data.frame
        column = "capacity_smooth_ah" if "capacity_smooth_ah" in frame.columns else "capacity_ah"
        if column not in frame.columns:
            raise KeyError(
                "capacity_fade_extrapolation needs capacity_smooth_ah or capacity_ah "
                f"on the frame; got {list(frame.columns)[:12]}"
            )

        out = np.full(len(frame), self.fallback_rul, dtype=float)
        cycles = frame["cycle_index"].to_numpy(dtype=float)
        capacity = frame[column].to_numpy(dtype=float)
        battery = frame["battery_id"].to_numpy()
        positions = np.arange(len(frame))

        for cell in np.unique(battery):
            idx = positions[battery == cell]
            idx = idx[np.argsort(cycles[idx], kind="stable")]
            for j, row in enumerate(idx):
                window = idx[max(0, j - self.lookback + 1) : j + 1]
                if window.size < 3:
                    continue
                x = cycles[window]
                y = capacity[window]
                good = np.isfinite(x) & np.isfinite(y)
                if good.sum() < 3:
                    continue
                slope, intercept = np.polyfit(x[good], y[good], 1)
                if slope >= -1e-9:  # flat or improving: no crossing to extrapolate to
                    continue
                eol_cycle = (self.eol_capacity_ah - intercept) / slope
                out[row] = float(np.clip(eol_cycle - cycles[row], 0.0, self.max_rul))
        return out


@register_model("soh_analogue")
@dataclass
class SOHAnalogueBaseline(BaseModel):
    """Nearest-analogue lookup: "what was left for training cells at this SOH?".

    The last-value/persistence idea does not transfer directly to RUL under a
    battery-holdout split: an unseen cell has no previous RUL label to persist.
    The nearest defensible analogue is a table lookup — find the training rows
    whose state of health matches this row's, and report their median remaining
    life. It is memoryless, uses one feature, and is the thing a spreadsheet
    would do.
    """

    reference_soh: np.ndarray | None = None
    reference_rul: np.ndarray | None = None
    k: int = 25

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> BaseModel:
        self.k = int(self.params.get("k", 25))
        soh = self._soh(train)
        good = np.isfinite(soh)
        order = np.argsort(soh[good], kind="stable")
        self.reference_soh = soh[good][order]
        self.reference_rul = np.asarray(train.y, dtype=float)[good][order]
        self.fitted = True
        self.fit_metadata = {"n_reference_rows": int(self.reference_soh.size), "k": self.k}
        return self

    @staticmethod
    def _soh(data: TrainingData) -> np.ndarray:
        frame = data.frame
        for column in ("soh", "capacity_smooth_ah", "capacity_ah"):
            if column in frame.columns:
                return frame[column].to_numpy(dtype=float)
        raise KeyError("soh_analogue needs an soh or capacity column on the frame")

    def predict(self, data: TrainingData) -> np.ndarray:
        self._check_fitted()
        assert self.reference_soh is not None and self.reference_rul is not None
        query = self._soh(data)
        out = np.empty(len(query), dtype=float)
        n = self.reference_soh.size
        for i, value in enumerate(query):
            if not np.isfinite(value):
                out[i] = float(np.median(self.reference_rul))
                continue
            centre = int(np.searchsorted(self.reference_soh, value))
            lo = max(0, centre - self.k // 2)
            hi = min(n, lo + self.k)
            out[i] = float(np.median(self.reference_rul[lo:hi]))
        return np.clip(out, 0.0, None)
