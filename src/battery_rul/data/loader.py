"""Ingestion orchestrator: raw files in, validated canonical table out.

This is the only module the rest of the codebase calls to *get data*. It owns
source selection, the synthetic fallback, capacity smoothing, SoH derivation,
validation and interim caching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.data import nasa as _nasa  # noqa: F401  (registers the source)
from battery_rul.data import synthetic as _synthetic  # noqa: F401  (registers the source)
from battery_rul.data.base import get_source
from battery_rul.data.schema import DatasetMetadata
from battery_rul.data.validation import ValidationReport, validate_cycles
from battery_rul.utils.io import read_table, write_table
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["CycleDataset", "load_cycles"]


@dataclass(slots=True)
class CycleDataset:
    """Validated cycle table plus its provenance."""

    frame: pd.DataFrame
    metadata: DatasetMetadata
    validation: ValidationReport

    @property
    def batteries(self) -> list[str]:
        return sorted(self.frame["battery_id"].unique().tolist())

    def battery(self, battery_id: str) -> pd.DataFrame:
        return self.frame.loc[self.frame["battery_id"] == battery_id].reset_index(drop=True)

    def describe(self) -> pd.DataFrame:
        """Per-battery summary used by EDA and the dataset card."""
        g = self.frame.groupby("battery_id", sort=True)
        out = pd.DataFrame(
            {
                "n_cycles": g.size(),
                "capacity_start_ah": g["capacity_smooth_ah"].first(),
                "capacity_end_ah": g["capacity_smooth_ah"].last(),
                "soh_start": g["soh"].first(),
                "soh_end": g["soh"].last(),
                "ambient_c_mean": g["ambient_temperature_c"].mean(),
                "temp_max_c": g["temperature_max_c"].max(),
                "resistance_start_ohm": g["internal_resistance_ohm"].first(),
                "resistance_end_ohm": g["internal_resistance_ohm"].last(),
            }
        )
        out["capacity_fade_pct"] = (
            100.0 * (out["capacity_start_ah"] - out["capacity_end_ah"]) / out["capacity_start_ah"]
        )
        return out.round(4).reset_index()


def load_cycles(cfg: ExperimentConfig, *, use_cache: bool | None = None) -> CycleDataset:
    """Load, clean and cache the canonical cycle table for ``cfg.data.source``.

    Falls back to the synthetic generator when the configured raw source is
    absent and ``data.allow_synthetic_fallback`` is set; the returned metadata
    always states which happened.
    """
    use_cache = cfg.data.cache_interim if use_cache is None else use_cache
    cache_path = cfg.paths.interim_dir / f"cycles_{cfg.data.source}.parquet"

    if use_cache and cache_path.is_file():
        logger.info("Reading cached cycle table: %s", cache_path)
        frame = read_table(cache_path)
        meta = DatasetMetadata(
            dataset=cfg.data.source,
            n_batteries=int(frame["battery_id"].nunique()),
            n_cycles=len(frame),
            batteries=tuple(sorted(frame["battery_id"].unique().tolist())),
            nominal_capacity_ah=cfg.data.nominal_capacity_ah,
            eol_threshold=cfg.data.eol_threshold,
            notes="Loaded from interim cache.",
        )
        return CycleDataset(
            frame=frame,
            metadata=meta,
            validation=ValidationReport(
                n_rows_in=len(frame),
                n_rows_out=len(frame),
                n_batteries_in=meta.n_batteries,
                n_batteries_out=meta.n_batteries,
            ),
        )

    source = get_source(cfg.data, cfg.paths.raw_dir)
    try:
        raw, meta = source.load()
    except FileNotFoundError as exc:
        if not cfg.data.allow_synthetic_fallback or cfg.data.source == "synthetic":
            raise
        logger.error("Raw source unavailable (%s)", exc)
        logger.error(
            "FALLING BACK TO SYNTHETIC DATA. Metrics from this run are NOT real "
            "results. Run `python scripts/download_data.py` to fetch the NASA dataset."
        )
        fallback_cfg = cfg.data.model_copy(update={"source": "synthetic", "batteries": None})
        raw, meta = get_source(fallback_cfg, cfg.paths.raw_dir).load()

    # Order matters: truncate the tail collapse before trimming the head, so the
    # early-life reference used by the head trim is computed on a clean record.
    frame = _truncate_at_collapse(raw, cfg)
    frame = _trim_leading_artifacts(frame, cfg)
    frame = _derive_health(frame, cfg)
    frame, report = validate_cycles(frame, data_cfg=cfg.data, validation_cfg=cfg.validation)
    frame = _apply_cohort_gates(frame, cfg, report)
    frame = _derive_health(frame, cfg)  # recompute after row/cell drops

    meta = DatasetMetadata(
        dataset=meta.dataset,
        n_batteries=int(frame["battery_id"].nunique()),
        n_cycles=len(frame),
        batteries=tuple(sorted(frame["battery_id"].unique().tolist())),
        nominal_capacity_ah=meta.nominal_capacity_ah,
        eol_threshold=meta.eol_threshold,
        source_files=meta.source_files,
        synthetic=meta.synthetic,
        notes=meta.notes,
    )

    if use_cache:
        write_table(frame, cache_path)
        logger.info("Cached cycle table -> %s", cache_path)

    return CycleDataset(frame=frame, metadata=meta, validation=report)


def _truncate_at_collapse(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """End a cell's record at a sustained capacity collapse.

    Why this exists
    ---------------
    Cells B0042-B0044 were moved from a 22 degC chamber into a 4 degC chamber at
    cycle 41. Measured capacity drops from ~1.5 Ah to ~0.07 Ah in one step and
    stays there for the remaining 67 cycles. The cells are not dead — at 4 degC the
    discharge test terminates almost immediately, so what is recorded is not a
    capacity measurement of a working cell at all.

    Left in, this is not a cosmetic problem: the end-of-life detector reads the
    collapse as a persistent threshold crossing and labels EOL at cycle 44, which
    is wrong by roughly the entire remaining life of the cell. Every RUL label for
    those cells would be wrong.

    Why the single-step jump check does not catch it
    -----------------------------------------------
    ``validation.max_capacity_jump`` inspects first differences, so it flags the
    one transition cycle and drops it — after which the series looks perfectly
    smooth at 0.07 Ah and nothing further trips. A first-difference test cannot
    see a *level shift*; it can only see the edge, and dropping the edge hides the
    evidence. This check looks at the level instead.

    The rule
    --------
    Find the first cycle whose capacity falls below ``collapse_fraction`` of the
    cell's reference capacity **and** stays below for ``collapse_persistence``
    consecutive cycles, then discard that cycle and everything after it. The
    persistence requirement means a single corrupt reading truncates nothing. The
    threshold is far below any state of health a gradually ageing cell reaches, so
    genuine degradation is never truncated.
    """
    if not cfg.data.truncate_at_collapse:
        return df

    fraction = cfg.data.collapse_fraction
    persistence = cfg.data.collapse_persistence
    kept: list[pd.DataFrame] = []
    truncated: dict[str, tuple[int, int]] = {}

    for battery_id, group in df.groupby("battery_id", sort=False):
        group = group.sort_values("cycle_index", kind="stable")
        capacity = group["capacity_ah"].to_numpy(dtype=float)

        # Reference is the cell's early-life level, robust to a bad first reading.
        reference = float(np.nanmedian(capacity[: min(10, len(capacity))]))
        if not np.isfinite(reference) or reference <= 0:
            kept.append(group)
            continue

        below = capacity < fraction * reference
        cut = None
        for idx in range(len(below)):
            if not below[idx]:
                continue
            window = below[idx : idx + persistence]
            if window.size >= min(persistence, len(below) - idx) and window.all():
                cut = idx
                break

        if cut is None:
            kept.append(group)
            continue

        truncated[str(battery_id)] = (int(group["cycle_index"].iloc[cut]), len(capacity) - cut)
        survivor = group.iloc[:cut].copy()
        if survivor.empty:
            logger.warning("%s: capacity collapses immediately; cell discarded", battery_id)
            continue
        kept.append(survivor)

    if truncated:
        logger.warning(
            "Truncated %d cell(s) at a sustained capacity collapse (regime change, "
            "not end of life): %s",
            len(truncated),
            ", ".join(f"{k} at cycle {c} (-{n} rows)" for k, (c, n) in sorted(truncated.items())),
        )

    if not kept:
        raise ValueError("Every cell collapsed immediately; check nominal_capacity_ah")
    return pd.concat(kept, ignore_index=True)


def _trim_leading_artifacts(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Drop aborted/partial discharges at the head of a cell's record.

    Several NASA cells (B0033, B0034, B0036, B0038-B0041 …) open with one to
    seven discharges whose measured capacity is a fraction of the cell's true
    beginning-of-life level — the rig was still being brought up, or the run was
    cut short. Left in place they corrupt three things at once: the
    beginning-of-life reference capacity, every ``*_ratio_to_initial`` feature,
    and the health gate below.

    The rule is deliberately conservative and one-sided: starting at cycle 1,
    drop a cycle while its capacity is below ``leading_outlier_ratio`` times the
    median of the next ``leading_outlier_lookahead`` cycles. It stops at the
    first healthy cycle, so genuine early-life fade is never removed, and it can
    only ever remove a *prefix*.
    """
    if not cfg.data.trim_leading_outliers:
        return df

    ratio = cfg.data.leading_outlier_ratio
    lookahead = cfg.data.leading_outlier_lookahead
    keep_parts: list[pd.DataFrame] = []
    trimmed: dict[str, int] = {}

    for battery_id, group in df.groupby("battery_id", sort=False):
        group = group.sort_values("cycle_index", kind="stable")
        capacity = group["capacity_ah"].to_numpy(dtype=float)
        start = 0
        while start < len(capacity) - lookahead:
            window = capacity[start + 1 : start + 1 + lookahead]
            reference = float(np.median(window))
            if reference > 0 and capacity[start] < ratio * reference:
                start += 1
            else:
                break
        if start:
            trimmed[str(battery_id)] = start
        keep = group.iloc[start:].copy()
        # Re-index so cycle_index stays 1-based and gap-free — downstream code
        # treats it as an ageing clock, not a rig serial number.
        keep["cycle_index"] = np.arange(1, len(keep) + 1, dtype="int32")
        keep_parts.append(keep)

    if trimmed:
        logger.info(
            "Trimmed leading rig artifacts: %s",
            ", ".join(f"{k}(-{v})" for k, v in sorted(trimmed.items())),
        )
    return pd.concat(keep_parts, ignore_index=True)


def _apply_cohort_gates(df: pd.DataFrame, cfg: ExperimentConfig, report) -> pd.DataFrame:
    """Exclude cells for which "remaining useful life" is not a meaningful label.

    Two gates, both physical rather than statistical:

    ``min_start_soh``
        A cell whose beginning-of-life capacity is already at or near the EOL
        threshold has no useful life to predict. Including it teaches the model
        that low capacity means "about to die", which is exactly the shortcut we
        want to avoid — the interesting signal is the *rate* of fade.

    ``min_fade_fraction``
        A cell whose capacity never falls meaningfully was stopped long before
        end of life. Its label is right-censored, and its rows would otherwise
        anchor the model at large RUL values regardless of condition.
    """
    from battery_rul.data.validation import ValidationIssue

    reference = cfg.data.nominal_capacity_ah
    keep: list[str] = []
    rejected: dict[str, str] = {}

    for battery_id, group in df.groupby("battery_id", sort=True):
        capacity = group["capacity_smooth_ah"].to_numpy(dtype=float)
        ref = reference if cfg.data.eol_reference == "nominal" else float(capacity[0])
        start_soh = float(capacity[0]) / ref if ref > 0 else 0.0
        fade = (float(capacity[0]) - float(np.min(capacity))) / ref if ref > 0 else 0.0

        if start_soh < cfg.data.min_start_soh:
            rejected[str(battery_id)] = (
                f"starts at {start_soh:.2f} SoH (< {cfg.data.min_start_soh})"
            )
        elif fade < cfg.data.min_fade_fraction:
            rejected[str(battery_id)] = (
                f"fades only {fade:.1%} (< {cfg.data.min_fade_fraction:.1%})"
            )
        else:
            keep.append(str(battery_id))

    if rejected:
        logger.info(
            "Excluded %d cell(s) failing the beginning-of-life health gates: %s",
            len(rejected),
            "; ".join(f"{k}: {v}" for k, v in sorted(rejected.items())),
        )
        report.add(
            ValidationIssue(
                "cohort_gates",
                "info",
                f"Excluded {len(rejected)} cell(s): "
                + "; ".join(f"{k} ({v})" for k, v in sorted(rejected.items())),
                batteries=tuple(sorted(rejected)),
            )
        )

    if not keep:
        raise ValueError(
            "Every cell failed the beginning-of-life health gates "
            f"(min_start_soh={cfg.data.min_start_soh}). Reasons: {rejected}"
        )
    out = df.loc[df["battery_id"].isin(keep)].reset_index(drop=True)
    report.n_rows_out = len(out)
    report.n_batteries_out = len(keep)
    return out


def _derive_health(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add ``capacity_smooth_ah`` and ``soh``.

    NASA capacity readings carry ~1-3 % measurement noise and genuine
    rest-recovery bumps. A **centred** filter would leak future cycles into the
    present, so the smoother here is strictly trailing: at cycle *k* it sees
    cycles ``k-w+1 … k`` only. The cost is a small lag; the benefit is that the
    EOL label and every capacity-derived feature stay causal.
    """
    df = df.copy()
    window = max(int(cfg.data.capacity_smoothing_window), 1)

    df["capacity_smooth_ah"] = (
        df.groupby("battery_id", sort=False)["capacity_ah"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).median())
        .astype("float32")
    )

    initial = df.groupby("battery_id", sort=False)["capacity_smooth_ah"].transform("first")
    reference = (
        pd.Series(cfg.data.nominal_capacity_ah, index=df.index)
        if cfg.data.eol_reference == "nominal"
        else initial
    )
    df["reference_capacity_ah"] = reference.astype("float32")
    df["soh"] = (df["capacity_smooth_ah"] / reference).astype("float32")
    df["capacity_fade_ah"] = (initial - df["capacity_smooth_ah"]).astype("float32")
    df["equivalent_full_cycles"] = (
        df.groupby("battery_id", sort=False)["capacity_ah"].cumsum() / cfg.data.nominal_capacity_ah
    ).astype("float32")

    if not np.isfinite(df["soh"]).all():
        logger.warning("Non-finite SoH values after derivation; check nominal_capacity_ah")
    return df


def battery_summary_table(dataset: CycleDataset, cfg: ExperimentConfig) -> pd.DataFrame:
    """Per-battery table including whether/when each cell reaches end of life."""
    from battery_rul.features.target import find_eol_cycle

    rows = []
    for battery_id, group in dataset.frame.groupby("battery_id", sort=True):
        eol = find_eol_cycle(group, cfg)
        rows.append(
            {
                "battery_id": battery_id,
                "n_cycles": len(group),
                "capacity_start_ah": round(float(group["capacity_smooth_ah"].iloc[0]), 4),
                "capacity_end_ah": round(float(group["capacity_smooth_ah"].iloc[-1]), 4),
                "soh_end": round(float(group["soh"].iloc[-1]), 4),
                "eol_cycle": eol,
                "reaches_eol": eol is not None,
                "ambient_c": round(float(group["ambient_temperature_c"].mean()), 1),
            }
        )
    return pd.DataFrame(rows)
