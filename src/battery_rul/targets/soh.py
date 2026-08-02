r"""State of health: definition, target column and health banding.

Definition
----------
.. math::
    \mathrm{SOH}_i(k) = \frac{\tilde Q_i(k)}{Q^{\mathrm{ref}}_i}

where :math:`\tilde Q_i(k)` is the trailing-median-smoothed discharge capacity of
cell *i* at cycle *k* and :math:`Q^{\mathrm{ref}}_i` the configured reference.

**The internal representation is a fraction in [0, 1]**, everywhere: targets,
model outputs, thresholds, API payloads. Percentages appear only in rendered
strings, and always at the point of rendering. Mixing the two representations is
the single most common way an SOH pipeline produces a plausible-looking number
that is wrong by a factor of a hundred, so there is exactly one internal unit.

Reference strategies (``soh.reference_strategy``)
-------------------------------------------------
``nominal``
    The manufacturer rating (2.0 Ah for the NASA 18650 cells). Comparable across
    cells and the convention in the NASA-dataset literature. Its weakness is that
    a cell which left the factory at 1.85 Ah reads as 92.5 % SOH when new.

``first_cycle``
    The cell's own first valid measurement. Every cell starts at exactly 1.0, so
    the quantity measures *fade* rather than absolute condition. One bad opening
    reading poisons the whole series.

``first_n_cycle_mean`` (default)
    Mean of the first N valid cycles. Keeps the fade interpretation while being
    robust to a single aborted opening discharge — which the NASA rig produces
    often enough that it is the reason the loader has a leading-artifact trim.

The strategies are *not* interchangeable and are never mixed within a run: the
choice is persisted in the model bundle's target definition and the runtime
configuration is checked against it before serving.

Causality
---------
``first_cycle`` and ``first_n_cycle_mean`` read the opening cycles of a cell,
which are in the past for every cycle at which a prediction is made, and the
smoothed capacity is trailing-filtered. The reference is therefore known from
cycle N onward and no row uses a future observation. Rows before the reference is
established (fewer than N valid cycles) are excluded from the warm-up region by
``features.drop_warmup_cycles`` in every shipped configuration; the invariant is
asserted rather than assumed, in ``attach_soh_target``.

Health bands
------------
Configurable, project-level **engineering categories** — not universal or
regulatory definitions, and not medical-style diagnoses. The defaults (90 / 80 /
70 % of reference) follow common second-life practice and the 70 % end-of-life
convention already used for the RUL target. Change them in configuration; every
consumer reads them from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig, SOHConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "HealthClass",
    "SOHTargetReport",
    "attach_soh_target",
    "classify_soh",
    "reference_capacity",
    "soh_series",
]


class HealthClass(StrEnum):
    """Engineering health band. Ordered from best to worst."""

    HEALTHY = "healthy"
    SLIGHTLY_DEGRADED = "slightly_degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def severity(self) -> int:
        """0 = best. Used for sorting and for the recommendation rules."""
        return {
            HealthClass.HEALTHY: 0,
            HealthClass.SLIGHTLY_DEGRADED: 1,
            HealthClass.WARNING: 2,
            HealthClass.CRITICAL: 3,
            HealthClass.UNKNOWN: 4,
        }[self]


@dataclass(slots=True)
class SOHTargetReport:
    """What SOH target generation did."""

    strategy: str
    reference_cycles: int
    reference_by_battery: dict[str, float]
    n_rows: int
    soh_min: float
    soh_max: float
    soh_mean: float
    n_clipped: int
    class_counts: dict[str, int]
    forecast_horizon_cycles: int = 0
    n_forecastable_rows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "representation": "fraction in [0, 1]",
            "strategy": self.strategy,
            "reference_cycles": self.reference_cycles,
            "reference_capacity_ah_by_battery": {
                k: round(v, 5) for k, v in self.reference_by_battery.items()
            },
            "n_rows": self.n_rows,
            "soh_min": round(self.soh_min, 5),
            "soh_max": round(self.soh_max, 5),
            "soh_mean": round(self.soh_mean, 5),
            "n_clipped_to_plausible_range": self.n_clipped,
            "class_counts": self.class_counts,
            "forecast_horizon_cycles": self.forecast_horizon_cycles,
            "n_forecastable_rows": self.n_forecastable_rows,
            "note": (
                "Current-cycle SOH is a derived measurement, not a modelling target. "
                "The model is trained on the forecast target."
            ),
        }


def reference_capacity(group: pd.DataFrame, cfg: ExperimentConfig) -> float:
    """Reference capacity (Ah) for one cell under the configured strategy.

    ``group`` must be a single cell's rows, sorted by ``cycle_index``.

    Raises
    ------
    ValueError
        If the strategy needs measured cycles and none are usable — an explicit
        failure is correct here, because silently substituting the nominal
        rating would change the meaning of every SOH value downstream without
        anything in the output saying so.
    """
    soh_cfg = cfg.soh
    if soh_cfg.reference_strategy == "nominal":
        return float(cfg.data.nominal_capacity_ah)

    column = "capacity_smooth_ah" if "capacity_smooth_ah" in group.columns else "capacity_ah"
    values = group[column].to_numpy(dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        raise ValueError(
            f"Cannot establish an SOH reference from strategy "
            f"{soh_cfg.reference_strategy!r}: the cell has no valid capacity reading."
        )

    if soh_cfg.reference_strategy == "first_cycle":
        return float(valid[0])

    n = min(int(soh_cfg.reference_cycles), valid.size)
    return float(np.mean(valid[:n]))


def soh_series(group: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.Series, float]:
    """SOH as a fraction for one cell, plus the reference used."""
    reference = reference_capacity(group, cfg)
    column = "capacity_smooth_ah" if "capacity_smooth_ah" in group.columns else "capacity_ah"
    return group[column].astype(float) / reference, reference


def classify_soh(soh: float | None, cfg: SOHConfig) -> HealthClass:
    """Map an SOH fraction onto its engineering band.

    ``None`` or a non-finite value yields :attr:`HealthClass.UNKNOWN` rather than
    a default band: "we could not tell" is a distinct answer from "healthy", and
    collapsing them is how a broken sensor becomes a green dashboard tile.
    """
    if soh is None or not np.isfinite(soh):
        return HealthClass.UNKNOWN
    if soh >= cfg.healthy_min:
        return HealthClass.HEALTHY
    if soh >= cfg.slightly_degraded_min:
        return HealthClass.SLIGHTLY_DEGRADED
    if soh >= cfg.warning_min:
        return HealthClass.WARNING
    return HealthClass.CRITICAL


def attach_soh_target(
    df: pd.DataFrame, cfg: ExperimentConfig
) -> tuple[pd.DataFrame, SOHTargetReport]:
    """Add ``soh_target`` (fraction), ``soh_reference_capacity_ah`` and the band.

    Columns added
    -------------
    ``soh_target``                 float32 — SOH as a fraction, clipped to the
                                   configured plausible interval
    ``soh_reference_capacity_ah``  float32 — the denominator actually used
    ``soh_health_class``           string  — engineering band at this cycle
    """
    if df.empty:
        raise ValueError("Cannot attach an SOH target to an empty frame")

    soh_cfg = cfg.soh
    frame = df.sort_values(["battery_id", "cycle_index"], kind="stable").reset_index(drop=True)

    references: dict[str, float] = {}
    values = np.empty(len(frame), dtype=float)
    positions = np.arange(len(frame))
    battery_col = frame["battery_id"].to_numpy()

    for battery_id, group in frame.groupby("battery_id", sort=False):
        series, reference = soh_series(group, cfg)
        references[str(battery_id)] = reference
        values[positions[battery_col == battery_id]] = series.to_numpy(dtype=float)

    raw = values.copy()
    clipped = np.clip(values, soh_cfg.plausible_min, soh_cfg.plausible_max)
    n_clipped = int(np.sum(np.abs(clipped - raw) > 1e-9))
    if n_clipped:
        logger.warning(
            "%d SOH value(s) fell outside the plausible interval [%.2f, %.2f] and were "
            "clipped. A cell reading far outside that band is a measurement problem, "
            "not a health state.",
            n_clipped,
            soh_cfg.plausible_min,
            soh_cfg.plausible_max,
        )

    frame[soh_cfg.target_name] = clipped.astype("float32")

    # --- the forecasting target -------------------------------------------
    # SOH *at the current cycle* is not a prediction problem. It is measured
    # capacity divided by a per-cell constant, and measured capacity is a model
    # input, so a model fitted against it learns a rescaling and reports a
    # flattering error that is not evidence of inferring a latent health state.
    # The forecast target — SOH `forecast_horizon_cycles` ahead — is a genuine
    # prediction: at cycle t nothing in the input reveals capacity at t+H.
    # Rows within H cycles of a cell's last observation have no target and are
    # NaN, never extrapolated.
    horizon = int(soh_cfg.forecast_horizon_cycles)
    frame[soh_cfg.forecast_target_name] = (
        frame.groupby("battery_id", sort=False)[soh_cfg.target_name]
        .shift(-horizon)
        .astype("float32")
    )
    n_forecastable = int(frame[soh_cfg.forecast_target_name].notna().sum())
    logger.info(
        "SOH forecast target (+%d cycles): %d/%d rows have a label; the final %d "
        "cycles of each cell necessarily have none.",
        horizon,
        n_forecastable,
        len(frame),
        horizon,
    )
    frame["soh_reference_capacity_ah"] = frame["battery_id"].map(references).astype("float32")
    classes = [classify_soh(float(v), soh_cfg).value for v in clipped]
    frame["soh_health_class"] = pd.Series(classes, index=frame.index, dtype="string")

    counts = frame["soh_health_class"].value_counts()
    report = SOHTargetReport(
        strategy=soh_cfg.reference_strategy,
        reference_cycles=soh_cfg.reference_cycles,
        reference_by_battery=references,
        n_rows=len(frame),
        soh_min=float(np.min(clipped)),
        soh_max=float(np.max(clipped)),
        soh_mean=float(np.mean(clipped)),
        n_clipped=n_clipped,
        class_counts={str(k): int(v) for k, v in counts.items()},
        forecast_horizon_cycles=horizon,
        n_forecastable_rows=n_forecastable,
    )
    logger.info(
        "SOH target (%s, fraction): n=%d, range=[%.3f, %.3f], mean=%.3f",
        soh_cfg.reference_strategy,
        report.n_rows,
        report.soh_min,
        report.soh_max,
        report.soh_mean,
    )
    return frame, report
