"""Causal feature engineering.

The leakage contract
--------------------
Every feature emitted here for battery *i* at cycle *k* is a function of

    { x_i(1), x_i(2), …, x_i(k) }

and nothing else. Concretely that means:

* all rolling windows use pandas' **trailing** semantics (``closed='right'``) and
  are computed **within** ``groupby('battery_id')`` so one cell never sees another;
* lags are strictly positive (``shift(+n)``); there is no ``shift(-n)`` anywhere;
* "trend" and "slope" features fit a line to a *trailing* window;
* cumulative features use ``cumsum``/``expanding``, never totals;
* normalisation-by-initial-value uses the value at cycle 1, which is in the past
  for every k >= 1;
* **no statistic is computed over the full series** — no ``.mean()`` broadcast
  back onto rows, no min-max scaling against a global max.

:func:`assert_no_leakage` mechanically verifies the contract by truncating a
battery's history and checking that already-computed rows are bit-identical. It
runs in the test suite and, optionally, inside the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from battery_rul.config import FeatureConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "FeatureBuildReport",
    "assert_no_leakage",
    "build_features",
    "feature_columns",
]

#: Columns that identify a row or encode the label; never used as inputs.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "dataset",
        "battery_id",
        "cycle_index",
        "timestamp",
        "eol_cycle",
        "is_censored",
        "rul_cycles",
        "rul_raw_cycles",
        "life_fraction",
        "split",
        "fold",
    }
)

#: Raw signals that leak the label almost perfectly and must not be fed to a model.
#: ``capacity_smooth_ah`` and ``soh`` *are* allowed as inputs (they are measured,
#: not derived from the label), but the EOL-derived quantities are not.
LEAKY_COLUMNS: frozenset[str] = frozenset({"eol_cycle", "rul_raw_cycles", "life_fraction"})


@dataclass(slots=True)
class FeatureBuildReport:
    n_input_columns: int = 0
    n_generated: int = 0
    n_after_pruning: int = 0
    dropped_constant: list[str] = field(default_factory=list)
    dropped_correlated: list[tuple[str, str, float]] = field(default_factory=list)
    warmup_rows_dropped: int = 0
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_input_columns": self.n_input_columns,
            "n_generated": self.n_generated,
            "n_after_pruning": self.n_after_pruning,
            "warmup_rows_dropped": self.warmup_rows_dropped,
            "dropped_constant": self.dropped_constant,
            "dropped_correlated": [
                {"kept": a, "dropped": b, "rho": round(r, 5)} for a, b, r in self.dropped_correlated
            ],
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns eligible as model inputs."""
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in LEAKY_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Individual feature families
# ---------------------------------------------------------------------------
def _trailing_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope over a trailing window, in units per cycle.

    Uses the closed-form slope of a regression on ``x = 0..w-1``:
    ``beta = cov(x, y) / var(x)``. Rolling ``cov`` with a fixed x reduces to a
    weighted sum, which pandas evaluates in one vectorised pass — orders of
    magnitude faster than ``rolling.apply(polyfit)`` on 3 000+ rows.
    """
    w = int(window)
    if w < 2:
        return pd.Series(np.nan, index=series.index)
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float((x_centered**2).sum())

    def _apply(values: np.ndarray) -> float:
        if np.isnan(values).any():
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return np.nan
            xs = x[mask]
            ys = values[mask]
            xc = xs - xs.mean()
            d = float((xc**2).sum())
            return float((xc * (ys - ys.mean())).sum() / d) if d > 0 else np.nan
        return float((x_centered * (values - values.mean())).sum() / denom)

    return series.rolling(window=w, min_periods=w).apply(_apply, raw=True)


def _build_for_battery(
    group: pd.DataFrame, cfg: FeatureConfig, signals: Sequence[str]
) -> pd.DataFrame:
    """All engineered features for one battery. ``group`` is cycle-ordered."""
    out: dict[str, pd.Series] = {}
    index = group.index

    for signal in signals:
        series = group[signal].astype(float)
        prefix = signal

        # --- rolling statistics -------------------------------------------
        for window in cfg.rolling_windows:
            min_periods = max(int(np.ceil(window * cfg.min_periods_fraction)), 1)
            roll = series.rolling(window=window, min_periods=min_periods)
            mean = roll.mean()
            std = roll.std()
            mn = roll.min()
            mx = roll.max()
            out[f"{prefix}_rmean_{window}"] = mean
            out[f"{prefix}_rstd_{window}"] = std
            out[f"{prefix}_rmin_{window}"] = mn
            out[f"{prefix}_rmax_{window}"] = mx
            out[f"{prefix}_rrange_{window}"] = mx - mn
            # Deviation of the current reading from its recent regime: a compact
            # anomaly/regime-shift signal.
            out[f"{prefix}_rdev_{window}"] = series - mean

        # --- exponentially weighted -----------------------------------------
        for halflife in cfg.ewm_halflives:
            out[f"{prefix}_ewm_{halflife}"] = series.ewm(halflife=halflife, adjust=False).mean()

        # --- lags and differences -------------------------------------------
        for lag in cfg.lags:
            shifted = series.shift(lag)
            out[f"{prefix}_lag_{lag}"] = shifted
            out[f"{prefix}_diff_{lag}"] = series - shifted
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{prefix}_pct_{lag}"] = (series - shifted) / shifted.replace(0.0, np.nan)

        # --- trailing slopes --------------------------------------------------
        for window in cfg.slope_windows:
            out[f"{prefix}_slope_{window}"] = _trailing_slope(series, window)

        # --- normalisation against the cell's own initial state ---------------
        initial = series.iloc[0]
        if np.isfinite(initial) and abs(initial) > 1e-12:
            out[f"{prefix}_ratio_to_initial"] = series / initial
            out[f"{prefix}_delta_from_initial"] = series - initial

        # --- expanding (past-only) statistics ---------------------------------
        if cfg.include_cumulative:
            expanding = series.expanding(min_periods=1)
            out[f"{prefix}_cummean"] = expanding.mean()
            out[f"{prefix}_cummin"] = expanding.min()
            out[f"{prefix}_cummax"] = expanding.max()
            out[f"{prefix}_cumstd"] = expanding.std()

    # --- cycle-position features ---------------------------------------------
    if cfg.include_cycle_index:
        cycles = group["cycle_index"].astype(float)
        out["cycle_index_f"] = cycles
        out["log_cycle_index"] = np.log1p(cycles)
        out["inv_cycle_index"] = 1.0 / cycles

    if cfg.include_cumulative and "energy_throughput_wh" in group:
        out["cum_energy_wh"] = group["energy_throughput_wh"].astype(float).cumsum()
    if cfg.include_cumulative and "discharge_duration_s" in group:
        out["cum_discharge_time_h"] = group["discharge_duration_s"].astype(float).cumsum() / 3600.0

    # --- cross-signal interactions -------------------------------------------
    if {"internal_resistance_ohm", "capacity_smooth_ah"} <= set(group.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            out["resistance_capacity_product"] = group["internal_resistance_ohm"].astype(
                float
            ) * group["capacity_smooth_ah"].astype(float)
            out["capacity_per_ohm"] = group["capacity_smooth_ah"].astype(float) / group[
                "internal_resistance_ohm"
            ].astype(float).replace(0.0, np.nan)
    if {"temperature_max_c", "ambient_temperature_c"} <= set(group.columns):
        out["temperature_excess_c"] = group["temperature_max_c"].astype(float) - group[
            "ambient_temperature_c"
        ].astype(float)
    if {"discharge_duration_s", "charge_duration_s"} <= set(group.columns):
        out["discharge_charge_time_ratio"] = group["discharge_duration_s"].astype(float) / group[
            "charge_duration_s"
        ].astype(float).replace(0.0, np.nan)

    frame = pd.DataFrame(out, index=index)
    return frame.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    keep_columns: Sequence[str] | None = None,
    prune: bool = True,
) -> tuple[pd.DataFrame, FeatureBuildReport]:
    """Generate the engineered feature matrix.

    Parameters
    ----------
    df:
        Canonical cycle table (target columns may already be attached).
    cfg:
        Feature configuration.
    keep_columns:
        Non-feature columns to carry through unchanged (ids, target, ...). By
        default every column already present is carried through.
    prune:
        Drop near-constant and near-duplicate columns. Both decisions depend on
        the data in hand, so a serving batch would prune a *different* set than
        training did and the fitted pipeline would then be handed the wrong
        columns. Inference therefore passes ``prune=False``: generation is
        deterministic, so the unpruned output is always a superset of the
        training columns, and :class:`~battery_rul.features.pipeline.FeaturePipeline`
        selects the exact set it was fitted on.

    Returns
    -------
    (frame, report)
        ``frame`` contains the carried columns, the original base signals, and
        the engineered features. Rows in the warm-up region are dropped.
    """
    if df.empty:
        raise ValueError("build_features received an empty frame")

    df = df.sort_values(["battery_id", "cycle_index"], kind="stable").reset_index(drop=True)
    report = FeatureBuildReport(n_input_columns=df.shape[1])

    signals = [s for s in cfg.base_signals if s in df.columns]
    missing = sorted(set(cfg.base_signals) - set(signals))
    if missing:
        logger.warning("Base signals absent from the table and skipped: %s", missing)
    if not signals:
        raise ValueError(
            f"None of the configured base signals exist. Configured: {cfg.base_signals}; "
            f"available numeric columns: {sorted(feature_columns(df))[:20]}"
        )

    # An explicit loop rather than ``groupby.apply``: it keeps the per-battery
    # isolation obvious to a reader (that isolation *is* the leakage guarantee)
    # and sidesteps pandas' shifting apply/include_groups semantics.
    blocks = [
        _build_for_battery(group, cfg, signals) for _, group in df.groupby("battery_id", sort=False)
    ]
    engineered = pd.concat(blocks).sort_index()
    report.n_generated = engineered.shape[1]
    logger.info(
        "Generated %d engineered features from %d base signals", report.n_generated, len(signals)
    )

    carried = list(df.columns) if keep_columns is None else list(keep_columns)
    frame = pd.concat([df[carried], engineered], axis=1)

    # --- warm-up trim ------------------------------------------------------
    if cfg.drop_warmup_cycles > 0:
        before = len(frame)
        frame = frame.loc[frame["cycle_index"] > cfg.drop_warmup_cycles].reset_index(drop=True)
        report.warmup_rows_dropped = before - len(frame)
        logger.info(
            "Dropped %d warm-up rows (cycle_index <= %d)",
            report.warmup_rows_dropped,
            cfg.drop_warmup_cycles,
        )
        if frame.empty:
            raise ValueError(
                f"features.drop_warmup_cycles={cfg.drop_warmup_cycles} removed every row. "
                "Lower it or use longer batteries."
            )

    # Any remaining NaN comes from a window that is still partially filled at the
    # start of a cell. Forward-filling within the cell is causal; the residual is
    # zero-filled (the feature is simply "not yet observable").
    feats = feature_columns(frame)
    frame[feats] = (
        frame.groupby("battery_id", sort=False)[feats].ffill().fillna(0.0).astype("float32")
    )

    if prune:
        frame, report = _prune(frame, cfg, report)
    report.feature_names = feature_columns(frame)
    report.n_after_pruning = len(report.feature_names)
    logger.info(
        "Feature matrix: %d rows x %d features (%d pruned)",
        len(frame),
        report.n_after_pruning,
        report.n_generated + len(signals) - report.n_after_pruning,
    )
    return frame, report


def _prune(
    frame: pd.DataFrame, cfg: FeatureConfig, report: FeatureBuildReport
) -> tuple[pd.DataFrame, FeatureBuildReport]:
    """Remove degenerate and near-duplicate features.

    Note this is an *unsupervised* filter — it looks only at the design matrix,
    never at the target — so running it before the train/test split is safe with
    respect to label leakage. Supervised selection lives in the fitted
    :class:`~battery_rul.features.pipeline.FeaturePipeline`, which is fit on the
    training partition only.
    """
    feats = feature_columns(frame)
    if not feats:
        return frame, report

    variances = frame[feats].var(numeric_only=True)
    constant = sorted(variances[variances <= cfg.variance_threshold].index.tolist())
    if constant:
        frame = frame.drop(columns=constant)
        report.dropped_constant = constant
        logger.info("Dropped %d near-constant features", len(constant))

    if cfg.correlation_prune_threshold is not None:
        feats = feature_columns(frame)
        corr = frame[feats].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop: list[tuple[str, str, float]] = []
        dropped: set[str] = set()
        for column in upper.columns:
            if column in dropped:
                continue
            partners = upper[column]
            for other, rho in partners[partners > cfg.correlation_prune_threshold].items():
                if other in dropped or other == column:
                    continue
                dropped.add(str(other))
                to_drop.append((column, str(other), float(rho)))
        if dropped:
            frame = frame.drop(columns=sorted(dropped))
            report.dropped_correlated = to_drop
            logger.info(
                "Dropped %d features correlated above |rho| > %.3f",
                len(dropped),
                cfg.correlation_prune_threshold,
            )
    return frame, report


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------
def assert_no_leakage(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    battery_id: str | None = None,
    truncate_at: float = 0.6,
    tolerance: float = 1e-5,
    builder: Callable[[pd.DataFrame, FeatureConfig], tuple[pd.DataFrame, object]] | None = None,
) -> None:
    """Mechanically prove the causality contract for one battery.

    Build features on the full history, then rebuild on a truncated prefix. For
    every row present in both, the feature values must agree — if any feature
    peeked at a future cycle, truncating the future would change it.

    ``builder`` exists so the guard itself can be tested: the suite passes a
    deliberately non-causal builder and asserts this function raises. A checker
    that has never been shown to fail is not evidence of anything.

    Raises
    ------
    AssertionError
        With the offending feature names and their maximum discrepancy.
    """
    builder = builder or build_features
    if battery_id is None:
        battery_id = str(df["battery_id"].iloc[0])
    subset = df.loc[df["battery_id"] == battery_id].reset_index(drop=True)
    if len(subset) < 30:
        raise ValueError(f"Need >= 30 cycles to test leakage; {battery_id} has {len(subset)}")

    cut = int(len(subset) * truncate_at)
    truncated = subset.iloc[:cut].reset_index(drop=True)

    full_feats, _ = builder(subset, cfg)
    trunc_feats, _ = builder(truncated, cfg)

    shared_rows = min(len(full_feats), len(trunc_feats))
    if shared_rows == 0:
        raise ValueError("Truncation left no comparable rows; raise truncate_at")

    shared_cols = [c for c in feature_columns(trunc_feats) if c in full_feats.columns]
    a = full_feats.iloc[:shared_rows][shared_cols].to_numpy(dtype=float)
    b = trunc_feats.iloc[:shared_rows][shared_cols].to_numpy(dtype=float)

    delta = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
    scale = np.maximum(np.abs(np.nan_to_num(a)), 1.0)
    offending = np.where((delta / scale).max(axis=0) > tolerance)[0]
    if offending.size:
        details = ", ".join(
            f"{shared_cols[i]} (max rel. delta {(delta[:, i] / scale[:, i]).max():.2e})"
            for i in offending[:10]
        )
        raise AssertionError(
            f"Temporal leakage detected in {offending.size} feature(s) for {battery_id}: {details}"
        )
    logger.info(
        "Leakage check passed for %s: %d features identical over %d shared rows",
        battery_id,
        len(shared_cols),
        shared_rows,
    )
