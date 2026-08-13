"""Trend analysis for cells and fleets.

Everything here is **derived**: ordinary least squares over a trailing window of
measurements, with no model involved. That is deliberate — a maintenance rule
that fires on "capacity is falling faster than 1 % per 10 cycles" should not
depend on whether a neural network loaded successfully.

Two scopes:

*Battery-level* trends (capacity fade, temperature, resistance, SOH) feed the
priority engine and the ranking score.

*Fleet-level* trends compare successive fleet snapshots — median SOH, median RUL,
critical count over time. They need a persistence layer to have something to
compare against, so they take snapshots as input rather than reading storage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from battery_rul.fleet.domain import FleetSnapshot, FleetTrendPoint

__all__ = [
    "battery_trends",
    "cycles_per_day",
    "fleet_trend_series",
    "trend_per_10_cycles",
]

#: Trailing window used for every battery-level trend, in cycles. Long enough to
#: be robust to a single noisy reading, short enough to describe the cell's
#: current behaviour rather than its whole life.
TREND_WINDOW = 20

#: Fewest finite points before a slope is reported. Two points always fit a line
#: perfectly and say nothing.
MIN_TREND_POINTS = 5


def trend_per_10_cycles(
    history: pd.DataFrame, column: str, *, relative: bool, window: int = TREND_WINDOW
) -> float | None:
    """OLS slope of ``column`` per 10 cycles over the trailing window.

    ``relative`` expresses the slope as a percentage of the window's median
    level, which is what makes resistance and capacity trends comparable across
    cells with different absolute values.

    Returns ``None`` — never 0.0 — when the trend cannot be estimated. A missing
    trend and a flat trend are different facts, and a rule that treats them the
    same will silently stop firing when a sensor drops out.
    """
    if column not in history.columns or "cycle_index" not in history.columns:
        return None
    tail = history.tail(window)
    y = pd.to_numeric(tail[column], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(tail["cycle_index"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < MIN_TREND_POINTS or len(np.unique(x[good])) < 2:
        return None
    slope = float(np.polyfit(x[good], y[good], 1)[0])
    if not relative:
        return round(slope * 10.0, 4)
    level = float(np.nanmedian(y[good]))
    if abs(level) < 1e-12:
        return None
    return round(100.0 * slope * 10.0 / level, 4)


def battery_trends(history: pd.DataFrame) -> dict[str, float | None]:
    """The four trends the fleet layer uses, in their conventional signs.

    ``fade_trend_pct_per_10`` is positive when capacity is **falling**, which is
    the direction an operator thinks in ("this cell is fading at 1.2 % per 10
    cycles"). The underlying slope is negative; flipping it here rather than at
    each call site is what keeps the threshold comparisons readable.
    """
    capacity_column = (
        "capacity_smooth_ah" if "capacity_smooth_ah" in history.columns else "capacity_ah"
    )
    capacity_trend = trend_per_10_cycles(history, capacity_column, relative=True)
    soh_trend = trend_per_10_cycles(history, "soh", relative=True)
    return {
        "soh_trend_pct_per_10": soh_trend if soh_trend is not None else capacity_trend,
        "fade_trend_pct_per_10": None if capacity_trend is None else -capacity_trend,
        "temperature_trend_c_per_10": trend_per_10_cycles(
            history, "temperature_max_c", relative=False
        ),
        "resistance_trend_pct_per_10": trend_per_10_cycles(
            history, "internal_resistance_ohm", relative=True
        ),
    }


def cycles_per_day(history: pd.DataFrame, *, min_cycles: int = 10) -> float | None:
    """Recent duty rate from timestamps, or ``None`` when it cannot be measured.

    This is the only route by which an inspection window is ever expressed in
    days. Without timestamps there is no rate, and this returns ``None`` rather
    than assuming one — a cycle-to-day conversion invented for presentation is
    a date an operator will plan around.
    """
    if "timestamp" not in history.columns or "cycle_index" not in history.columns:
        return None
    frame = history.tail(max(min_cycles, 2))
    stamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    cycles = pd.to_numeric(frame["cycle_index"], errors="coerce")
    good = stamps.notna() & cycles.notna()
    if int(good.sum()) < min_cycles:
        return None
    stamps, cycles = stamps[good], cycles[good]
    span_days = (stamps.max() - stamps.min()).total_seconds() / 86400.0
    span_cycles = float(cycles.max() - cycles.min())
    if span_days <= 0 or span_cycles <= 0:
        return None
    return round(span_cycles / span_days, 4)


def fleet_trend_series(
    snapshots: Sequence[FleetSnapshot], *, metric: str = "median_soh"
) -> list[FleetTrendPoint]:
    """Turn a history of fleet snapshots into a trend series.

    Snapshots are ordered by generation time. Each point carries the denominator
    that produced it, so a series whose median moves because 30 cells stopped
    reporting is distinguishable from one that moves because the fleet aged.
    """
    supported = {"median_soh", "median_rul", "mean_risk", "critical_count"}
    if metric not in supported:
        raise ValueError(f"Unknown fleet trend metric {metric!r}. Supported: {sorted(supported)}")

    points: list[FleetTrendPoint] = []
    for snapshot in sorted(snapshots, key=lambda s: s.generated_at_utc):
        statistics = snapshot.fleet_statistics
        value: float | None
        denominator: int
        if metric == "median_soh":
            value, denominator = statistics.soh_median, statistics.soh_denominator
        elif metric == "median_rul":
            value, denominator = statistics.rul_median, statistics.rul_denominator
        elif metric == "mean_risk":
            value, denominator = statistics.risk_mean, statistics.risk_denominator
        else:
            value = float(snapshot.maintenance_summary.critical_count)
            denominator = snapshot.maintenance_summary.denominator
        points.append(
            FleetTrendPoint(
                label=metric,
                generated_at_utc=snapshot.generated_at_utc,
                battery_count=snapshot.battery_count,
                median_soh=statistics.soh_median,
                median_rul=statistics.rul_median,
                mean_risk=statistics.risk_mean,
                critical_count=snapshot.maintenance_summary.critical_count,
                value=value,
                denominator=denominator,
            )
        )
    return points
