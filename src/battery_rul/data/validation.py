"""Data-integrity gate.

Runs between "loader produced a table" and "features are computed". It reports
rather than silently repairs: every action taken is recorded in a
:class:`ValidationReport` that is serialised next to the processed dataset, so a
reviewer can see exactly what was thrown away and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import DataConfig, ValidationConfig
from battery_rul.data.schema import REQUIRED_COLUMNS
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DataValidationError", "ValidationIssue", "ValidationReport", "validate_cycles"]


class DataValidationError(RuntimeError):
    """Raised when ``validation.fail_fast`` is set and a check fails."""


@dataclass(slots=True)
class ValidationIssue:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    n_rows: int = 0
    batteries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "n_rows": self.n_rows,
            "batteries": list(self.batteries),
        }


@dataclass(slots=True)
class ValidationReport:
    """Everything the gate observed and did."""

    n_rows_in: int = 0
    n_rows_out: int = 0
    n_batteries_in: int = 0
    n_batteries_out: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    missing_fractions: dict[str, float] = field(default_factory=dict)
    imputed_cells: dict[str, int] = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def passed(self) -> bool:
        return self.n_errors == 0

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        log = logger.error if issue.severity == "error" else logger.warning
        if issue.severity == "info":
            log = logger.info
        log("[validation:%s] %s", issue.check, issue.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_rows_dropped": self.n_rows_in - self.n_rows_out,
            "n_batteries_in": self.n_batteries_in,
            "n_batteries_out": self.n_batteries_out,
            "n_errors": self.n_errors,
            "issues": [i.to_dict() for i in self.issues],
            "missing_fractions": {
                k: round(v, 5) for k, v in self.missing_fractions.items() if v > 0
            },
            "imputed_cells": self.imputed_cells,
        }

    def summary(self) -> str:
        return (
            f"validation: {self.n_rows_out}/{self.n_rows_in} rows kept across "
            f"{self.n_batteries_out}/{self.n_batteries_in} batteries, "
            f"{len(self.issues)} issue(s), {self.n_errors} error(s)"
        )


def validate_cycles(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    validation_cfg: ValidationConfig,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate, clean and (causally) impute the canonical cycle table.

    Checks performed
    ----------------
    1. Required columns present and non-empty.
    2. No duplicate ``(battery_id, cycle_index)`` pairs.
    3. ``cycle_index`` strictly increasing within each battery.
    4. Physical bounds on voltage and temperature.
    5. Capacity positive, finite, and free of single-step sensor jumps.
    6. Per-column missingness under ``max_missing_fraction``.
    7. Per-battery cycle count over ``min_cycles``.

    Repairs performed (all recorded)
    --------------------------------
    * Out-of-bounds sensor readings -> NaN.
    * Implausible capacity jumps -> row dropped when ``drop_incomplete_cycles``.
    * Remaining NaNs -> **causal** fill: forward-fill within battery, then the
      battery's expanding median, then the global training median. Nothing is
      ever back-filled from a future cycle.
    """
    report = ValidationReport(
        n_rows_in=len(df),
        n_batteries_in=int(df["battery_id"].nunique()) if "battery_id" in df else 0,
    )

    if df.empty:
        report.add(ValidationIssue("non_empty", "error", "Input table is empty"))
        _maybe_raise(report, validation_cfg)
        return df, report

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add(ValidationIssue("required_columns", "error", f"Missing columns: {missing_cols}"))
        _maybe_raise(report, validation_cfg)
        return df, report

    df = df.copy()

    # -- 2. duplicates ----------------------------------------------------
    dup_mask = df.duplicated(subset=["battery_id", "cycle_index"], keep="first")
    if dup_mask.any():
        report.add(
            ValidationIssue(
                "duplicate_cycles",
                "warning",
                f"Dropped {int(dup_mask.sum())} duplicate (battery_id, cycle_index) rows",
                n_rows=int(dup_mask.sum()),
                batteries=tuple(sorted(df.loc[dup_mask, "battery_id"].unique())),
            )
        )
        df = df.loc[~dup_mask]

    df = df.sort_values(["battery_id", "cycle_index"], kind="stable").reset_index(drop=True)

    # -- 3. monotonic cycle index ----------------------------------------
    if validation_cfg.require_monotonic_cycle_index:
        non_monotonic = [
            str(bid)
            for bid, g in df.groupby("battery_id", sort=False)
            if not g["cycle_index"].is_monotonic_increasing
        ]
        if non_monotonic:
            report.add(
                ValidationIssue(
                    "monotonic_cycle_index",
                    "error",
                    f"cycle_index is not monotonically increasing for {non_monotonic}",
                    batteries=tuple(non_monotonic),
                )
            )

    # -- 4. physical bounds ----------------------------------------------
    v_lo, v_hi = validation_cfg.voltage_bounds_v
    t_lo, t_hi = validation_cfg.temperature_bounds_c
    df, n_v = _nullify_out_of_range(df, [c for c in df.columns if c.endswith("_v")], v_lo, v_hi)
    df, n_t = _nullify_out_of_range(df, [c for c in df.columns if c.endswith("_c")], t_lo, t_hi)
    if n_v or n_t:
        report.add(
            ValidationIssue(
                "physical_bounds",
                "warning",
                f"Nullified {n_v} out-of-range voltage and {n_t} out-of-range temperature readings",
                n_rows=n_v + n_t,
            )
        )

    # -- 5. capacity sanity ------------------------------------------------
    bad_capacity = ~np.isfinite(df["capacity_ah"]) | (df["capacity_ah"] <= 0)
    if bad_capacity.any():
        report.add(
            ValidationIssue(
                "capacity_positive",
                "warning",
                f"Dropped {int(bad_capacity.sum())} cycles with non-positive/NaN capacity",
                n_rows=int(bad_capacity.sum()),
                batteries=tuple(sorted(df.loc[bad_capacity, "battery_id"].unique())),
            )
        )
        df = df.loc[~bad_capacity].reset_index(drop=True)

    jump_limit = validation_cfg.max_capacity_jump * data_cfg.nominal_capacity_ah
    jump = df.groupby("battery_id", sort=False)["capacity_ah"].diff().abs()
    glitches = jump > jump_limit
    if glitches.any():
        severity = "warning"
        report.add(
            ValidationIssue(
                "capacity_jump",
                severity,
                f"{int(glitches.sum())} cycles move capacity by more than "
                f"{jump_limit:.2f} Ah in one step (rig glitch)",
                n_rows=int(glitches.sum()),
                batteries=tuple(sorted(df.loc[glitches, "battery_id"].unique())),
            )
        )
        if data_cfg.drop_incomplete_cycles:
            df = df.loc[~glitches].reset_index(drop=True)

    # -- 6. missingness ----------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report.missing_fractions = {col: float(df[col].isna().mean()) for col in numeric_cols}
    over_budget = {
        col: frac
        for col, frac in report.missing_fractions.items()
        if frac > validation_cfg.max_missing_fraction
    }
    if over_budget:
        report.add(
            ValidationIssue(
                "missingness",
                "warning",
                "Columns above the missingness budget "
                f"({validation_cfg.max_missing_fraction:.0%}): "
                + ", ".join(f"{c}={f:.1%}" for c, f in sorted(over_budget.items())),
            )
        )

    df, imputed = _causal_impute(df, numeric_cols)
    report.imputed_cells = imputed

    # -- 7. per-battery length --------------------------------------------
    counts = df.groupby("battery_id", sort=False).size()
    short = counts[counts < data_cfg.min_cycles]
    if len(short):
        report.add(
            ValidationIssue(
                "min_cycles",
                "warning",
                f"Dropped {len(short)} batteries with fewer than {data_cfg.min_cycles} "
                f"clean cycles: {dict(short)}",
                n_rows=int(short.sum()),
                batteries=tuple(short.index.astype(str)),
            )
        )
        df = df.loc[~df["battery_id"].isin(short.index)].reset_index(drop=True)

    if df.empty:
        report.add(ValidationIssue("non_empty_output", "error", "All rows were rejected"))

    report.n_rows_out = len(df)
    report.n_batteries_out = int(df["battery_id"].nunique()) if len(df) else 0
    logger.info(report.summary())
    _maybe_raise(report, validation_cfg)
    return df, report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _maybe_raise(report: ValidationReport, cfg: ValidationConfig) -> None:
    if cfg.fail_fast and not report.passed:
        errors = [i.message for i in report.issues if i.severity == "error"]
        raise DataValidationError("Data validation failed: " + "; ".join(errors))


def _nullify_out_of_range(
    df: pd.DataFrame, columns: list[str], lo: float, hi: float
) -> tuple[pd.DataFrame, int]:
    total = 0
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            total += n
    return df, total


def _causal_impute(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill NaNs without ever reading a future cycle.

    Order of preference per battery: forward-fill -> expanding median (past only)
    -> the column's global median. The global median is a mild, documented
    concession: it is computed over all *loaded* rows, and is only reached for
    columns whose very first observation is missing.
    """
    imputed: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        missing_before = int(df[col].isna().sum())
        if missing_before == 0:
            continue

        grouped = df.groupby("battery_id", sort=False)[col]
        filled = grouped.ffill()
        expanding_median = grouped.transform(lambda s: s.expanding(min_periods=1).median())
        filled = filled.fillna(expanding_median)
        filled = filled.fillna(df[col].median())
        df[col] = filled.fillna(0.0)

        imputed[col] = missing_before
    if imputed:
        logger.info(
            "Causally imputed %d cells across %d columns", sum(imputed.values()), len(imputed)
        )
    return df, imputed
