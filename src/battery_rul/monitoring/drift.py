"""Batch feature-drift detection.

Four numerical metrics and two categorical ones, each reporting **effect
magnitude and alert status separately**. That separation is the point: a KS test
on 40 000 rows returns p < 0.001 for a shift nobody would act on, and a PSI of
0.3 on 40 rows is noise. Reporting only "drifted: true" throws away the
information needed to tell those apart.

Edge cases are handled explicitly rather than by exception:

*constant features* — no distribution to compare; reported ``reliable=False``
*small samples* — below ``min_sample_size`` the result is reported unscored
*missing features* — present in the reference, absent in the batch, is itself a
 finding (a sensor stopped reporting), and is reported as one
*new features* — present in the batch, absent from the reference, cannot be
 tested and are listed

Multiple comparisons
--------------------
Testing 200 features at alpha = 0.05 produces about ten "significant" results
from noise alone. The p-value-bearing tests (KS, chi-square) are corrected with
Benjamini-Hochberg by default; the magnitude metrics (PSI, Wasserstein, JS) have
no p-value and are judged on thresholds, which is documented rather than
disguised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import DriftConfig, ExperimentConfig
from battery_rul.fleet.domain import MonitoringStatus
from battery_rul.monitoring.domain import FeatureDriftReport, FeatureDriftResult
from battery_rul.monitoring.reference import ReferenceDistribution
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "benjamini_hochberg",
    "chi_square_drift",
    "detect_feature_drift",
    "jensen_shannon_divergence",
    "kolmogorov_smirnov",
    "population_stability_index",
    "wasserstein",
]

#: Floor applied to a bin frequency before taking a logarithm. Without it a
#: category absent from one side sends PSI to infinity, which is a division
#: artifact rather than infinite drift.
_EPSILON = 1e-6


def _binned(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Frequencies of ``values`` under the reference's own bin edges."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(max(len(edges) - 1, 1))
    counts, _ = np.histogram(finite, bins=np.asarray(edges, dtype=float))
    total = max(counts.sum(), 1)
    return counts / total


def population_stability_index(
    reference_frequencies: Sequence[float], current_frequencies: Sequence[float]
) -> float:
    """PSI = sum((c - r) * ln(c / r)) over shared bins.

    The industry convention (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant)
    is a rule of thumb from credit scoring, not a law; the thresholds are
    configurable for exactly that reason.
    """
    reference = np.clip(np.asarray(reference_frequencies, dtype=float), _EPSILON, None)
    current = np.clip(np.asarray(current_frequencies, dtype=float), _EPSILON, None)
    if reference.shape != current.shape:
        raise ValueError(
            f"PSI needs matching bin counts, got {reference.shape} and {current.shape}"
        )
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def jensen_shannon_divergence(
    reference_frequencies: Sequence[float], current_frequencies: Sequence[float]
) -> float:
    """JS divergence in [0, ln 2], symmetric and always finite."""
    reference = np.clip(np.asarray(reference_frequencies, dtype=float), _EPSILON, None)
    current = np.clip(np.asarray(current_frequencies, dtype=float), _EPSILON, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    mixture = 0.5 * (reference + current)
    kl = lambda p, q: float(np.sum(p * np.log(p / q)))  # noqa: E731 - local, one use
    return 0.5 * kl(reference, mixture) + 0.5 * kl(current, mixture)


def kolmogorov_smirnov(
    reference_summary: dict, current_values: np.ndarray
) -> tuple[float | None, float | None]:
    """Two-sample KS statistic and p-value against the reference's quantiles.

    The reference artifact stores quantiles, not raw rows (it is JSON, and it
    would otherwise be the training set in a file). The reference sample is
    reconstructed by inverting the stored empirical CDF — which is exact at the
    stored quantiles and linear between them. That approximation is why the
    statistic is reported next to the sample sizes rather than on its own.
    """
    from scipy import stats

    quantiles = reference_summary.get("quantiles") or {}
    if not quantiles:
        return None, None
    levels = sorted(float(k[1:]) / 100.0 for k in quantiles)
    values = [float(quantiles[f"q{int(round(100 * level))}"]) for level in levels]
    grid = np.linspace(min(levels), max(levels), 200)
    reconstructed = np.interp(grid, levels, values)

    finite = current_values[np.isfinite(current_values)]
    if finite.size < 2 or reconstructed.size < 2:
        return None, None
    result = stats.ks_2samp(reconstructed, finite)
    return float(result.statistic), float(result.pvalue)


def wasserstein(reference_summary: dict, current_values: np.ndarray) -> float | None:
    """Wasserstein-1 distance, standardised by the reference's spread.

    Standardised on purpose: an unnormalised distance is in the feature's own
    units, so one threshold cannot serve a voltage and a duration in the same
    report.
    """
    from scipy import stats

    quantiles = reference_summary.get("quantiles") or {}
    if not quantiles:
        return None
    levels = sorted(float(k[1:]) / 100.0 for k in quantiles)
    values = [float(quantiles[f"q{int(round(100 * level))}"]) for level in levels]
    grid = np.linspace(min(levels), max(levels), 200)
    reconstructed = np.interp(grid, levels, values)

    finite = current_values[np.isfinite(current_values)]
    if finite.size < 2:
        return None
    distance = float(stats.wasserstein_distance(reconstructed, finite))
    scale = float(reference_summary.get("std") or 0.0)
    if scale <= 0:
        span = float(reference_summary.get("max", 0.0)) - float(reference_summary.get("min", 0.0))
        scale = span if span > 0 else 1.0
    return distance / scale


def chi_square_drift(
    reference_categories: dict[str, float],
    current_values: pd.Series,
    reference_count: int | None = None,
) -> tuple[float | None, float | None, float]:
    """Chi-square statistic, p-value and the unseen-category rate.

    A **two-sample** test over a 2xK contingency table, not a goodness-of-fit
    test against the stored proportions. The distinction is not pedantic:
    treating the reference proportions as exact ignores the reference's own
    sampling error and roughly doubles the statistic, so a fair coin measured
    twice at 500 rows each reports "drift" several times more often than the
    nominal alpha claims.

    The unseen rate is returned separately because it is the finding that
    matters operationally: a category the model never trained on is not a shift
    in proportions, it is an input the model has no basis for.
    """
    from scipy import stats

    observed = current_values.dropna().astype(str).value_counts()
    total = int(observed.sum())
    if total == 0 or not reference_categories:
        return None, None, 0.0

    unseen = sum(
        int(count) for label, count in observed.items() if label not in reference_categories
    )
    unseen_rate = unseen / total

    labels = sorted(set(reference_categories) | set(observed.index.astype(str)))
    if len(labels) < 2:
        return None, None, unseen_rate

    # Reconstruct reference counts from the stored proportions. With no recorded
    # reference size, fall back to the current size — the least-assuming choice
    # available, and one that keeps the test two-sample rather than one-sample.
    reference_n = int(reference_count or total)
    table = np.vstack(
        [
            np.array(
                [reference_categories.get(label, 0.0) * reference_n for label in labels],
                dtype=float,
            ),
            np.array([float(observed.get(label, 0)) for label in labels], dtype=float),
        ]
    )
    # A category absent from both samples makes the test undefined, not informative.
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return None, None, unseen_rate

    statistic, p_value = stats.chi2_contingency(table, correction=table.shape[1] == 2)[:2]
    return float(statistic), float(p_value), unseen_rate


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values, preserving input order.

    ``None`` entries (tests that could not run) pass through untouched and are
    excluded from the ranking, so a skipped feature neither gains nor loses
    significance from the correction.
    """
    indexed = [(index, p) for index, p in enumerate(p_values) if p is not None]
    if not indexed:
        return list(p_values)
    indexed.sort(key=lambda pair: pair[1])
    n = len(indexed)
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank in range(n, 0, -1):
        index, p = indexed[rank - 1]
        running = min(running, float(p) * n / rank)
        adjusted[index] = min(running, 1.0)
    return [adjusted.get(index) if p is not None else None for index, p in enumerate(p_values)]


def _severity(value: float | None, thresholds: tuple[float, float]) -> MonitoringStatus:
    if value is None or not np.isfinite(value):
        return MonitoringStatus.UNKNOWN
    warning, critical = thresholds
    if value >= critical:
        return MonitoringStatus.CRITICAL
    if value >= warning:
        return MonitoringStatus.WARNING
    return MonitoringStatus.OK


def _thresholds_for(metric: str, cfg: DriftConfig) -> tuple[float, float]:
    return {
        "psi": cfg.psi_thresholds,
        "ks": cfg.ks_thresholds,
        "wasserstein": cfg.wasserstein_thresholds,
        "js": cfg.js_thresholds,
        "chi2": cfg.js_thresholds,
        "unseen_rate": cfg.unseen_rate_thresholds,
    }[metric]


def detect_feature_drift(
    current: pd.DataFrame,
    reference: ReferenceDistribution,
    cfg: ExperimentConfig,
    *,
    current_window: str = "",
) -> FeatureDriftReport:
    """Compare a current feature batch against a stored reference."""
    drift_cfg = cfg.monitoring.drift
    results: list[FeatureDriftResult] = []
    warnings: list[str] = []
    skipped = 0

    reference_features = reference.feature_names[: drift_cfg.max_features]
    missing_here = [name for name in reference_features if name not in current.columns]
    if missing_here:
        warnings.append(
            f"{len(missing_here)} reference feature(s) are absent from the current "
            f"batch and could not be tested: {missing_here[:10]}. An input the model "
            "expects and no longer receives is a data-collection failure, not drift."
        )
    new_features = [
        name
        for name in current.columns
        if name not in reference.feature_stats and pd.api.types.is_numeric_dtype(current[name])
    ]
    if new_features:
        warnings.append(
            f"{len(new_features)} feature(s) in the batch are absent from the "
            f"reference and were not tested: {new_features[:10]}."
        )

    for name in reference_features:
        summary = reference.feature_stats[name]
        if name not in current.columns:
            skipped += 1
            results.append(
                FeatureDriftResult(
                    feature_name=name,
                    feature_type=summary.get("type", "numerical"),
                    drift_metric="unavailable",
                    reference_sample_size=int(summary.get("count", 0)),
                    sample_size=0,
                    reliable=False,
                    severity=MonitoringStatus.UNKNOWN,
                    warnings=["Feature is absent from the current batch."],
                )
            )
            continue

        if summary.get("type") == "categorical":
            results.extend(_categorical_results(name, summary, current[name], drift_cfg))
            continue
        results.extend(
            _numerical_results(
                name,
                summary,
                pd.to_numeric(current[name], errors="coerce").to_numpy(dtype=float),
                drift_cfg,
            )
        )

    # -- multiple-comparison control over the tests that carry a p-value ----
    if drift_cfg.multiple_comparison != "none":
        p_values = [r.p_value for r in results]
        if drift_cfg.multiple_comparison == "bonferroni":
            n = sum(1 for p in p_values if p is not None)
            adjusted = [None if p is None else min(1.0, float(p) * n) for p in p_values]
        else:
            adjusted = benjamini_hochberg(p_values)
        for result, value in zip(results, adjusted, strict=True):
            result.adjusted_p_value = None if value is None else round(float(value), 6)
            if result.p_value is not None and result.reliable:
                significant = (result.adjusted_p_value or 1.0) <= drift_cfg.alpha
                # A significant test with a tiny effect is not drift worth acting
                # on; both conditions must hold.
                result.drift_detected = bool(result.drift_detected and significant)
                if not significant and result.severity is not MonitoringStatus.UNKNOWN:
                    result.severity = MonitoringStatus.OK

    tested = [r for r in results if r.reliable]
    drifted = [r for r in tested if r.drift_detected]
    drifted_names = sorted({r.feature_name for r in drifted})
    tested_names = {r.feature_name for r in tested}
    fraction = len(drifted_names) / max(len(tested_names), 1)

    if not tested_names:
        status = MonitoringStatus.UNKNOWN
        warnings.append("No feature could be tested reliably; the drift status is UNKNOWN.")
    elif fraction >= drift_cfg.fleet_critical_fraction or any(
        r.severity is MonitoringStatus.CRITICAL for r in drifted
    ):
        status = MonitoringStatus.CRITICAL
    elif fraction >= drift_cfg.fleet_warning_fraction or drifted_names:
        status = MonitoringStatus.WARNING
    else:
        status = MonitoringStatus.OK

    return FeatureDriftReport(
        reference_id=reference.reference_id,
        reference_fingerprint=reference.fingerprint(),
        reference_partition=reference.partition,
        reference_window=f"{reference.partition} partition, {reference.n_rows} rows",
        current_window=current_window or f"{len(current)} rows",
        status=status,
        results=results,
        n_features_tested=len(tested_names),
        n_features_drifted=len(drifted_names),
        n_features_skipped=skipped,
        drifted_features=drifted_names[:50],
        multiple_comparison=drift_cfg.multiple_comparison,
        method_notes=[
            f"Numerical metrics: {', '.join(drift_cfg.numerical_metrics)}.",
            f"Categorical metrics: {', '.join(drift_cfg.categorical_metrics)}.",
            "PSI, Wasserstein and JS are magnitude metrics with no p-value; they are "
            "judged against configured thresholds only.",
            "KS and chi-square p-values are adjusted for multiple comparisons "
            f"({drift_cfg.multiple_comparison}, alpha={drift_cfg.alpha}).",
            "The reference sample for KS and Wasserstein is reconstructed from stored "
            "quantiles, not from raw training rows; both statistics are approximate.",
            f"Features with fewer than {drift_cfg.min_sample_size} current observations "
            "are reported unscored.",
        ],
        warnings=warnings,
    )


def _numerical_results(
    name: str, summary: dict, values: np.ndarray, cfg: DriftConfig
) -> list[FeatureDriftResult]:
    finite = values[np.isfinite(values)]
    reference_size = int(summary.get("count", 0))
    # Annotated: this dictionary is splatted into a Pydantic model whose fields
    # have several different types, so an inferred `dict[str, object]` makes every
    # construction below a type error.
    base: dict[str, Any] = {
        "feature_name": name,
        "feature_type": "numerical",
        "reference_sample_size": reference_size,
        "sample_size": int(finite.size),
        "reference_summary": {
            k: float(v)
            for k, v in summary.items()
            if k in ("mean", "std", "min", "max") and v is not None
        },
        "current_summary": (
            {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
            if finite.size
            else {}
        ),
    }

    if summary.get("constant"):
        return [
            FeatureDriftResult(
                **base,
                drift_metric="constant_reference",
                reliable=False,
                severity=MonitoringStatus.UNKNOWN,
                warnings=[
                    "The reference feature is constant, so no distributional "
                    "comparison is defined. Reported, not scored."
                ],
            )
        ]
    if finite.size < cfg.min_sample_size:
        return [
            FeatureDriftResult(
                **base,
                drift_metric="insufficient_sample",
                reliable=False,
                severity=MonitoringStatus.UNKNOWN,
                warnings=[
                    f"{finite.size} finite observation(s) is below the configured "
                    f"minimum of {cfg.min_sample_size}; drift statistics on this "
                    "sample would be dominated by sampling noise."
                ],
            )
        ]

    out: list[FeatureDriftResult] = []
    for metric in cfg.numerical_metrics:
        thresholds = _thresholds_for(metric, cfg)
        p_value: float | None = None
        value: float | None
        if metric == "psi":
            value = population_stability_index(
                summary["bin_frequencies"], list(_binned(finite, summary["bin_edges"]))
            )
        elif metric == "js":
            value = jensen_shannon_divergence(
                summary["bin_frequencies"], list(_binned(finite, summary["bin_edges"]))
            )
        elif metric == "ks":
            value, p_value = kolmogorov_smirnov(summary, finite)
        else:
            value = wasserstein(summary, finite)

        severity = _severity(value, thresholds)
        out.append(
            FeatureDriftResult(
                **base,
                drift_metric=metric,
                drift_value=None if value is None else round(float(value), 6),
                p_value=None if p_value is None else round(float(p_value), 8),
                threshold=thresholds[0],
                drift_detected=severity in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                severity=severity,
                reliable=value is not None,
            )
        )
    return out


def _categorical_results(
    name: str, summary: dict, values: pd.Series, cfg: DriftConfig
) -> list[FeatureDriftResult]:
    categories = summary.get("categories") or {}
    observed = values.dropna()
    base: dict[str, Any] = {
        "feature_name": name,
        "feature_type": "categorical",
        "reference_sample_size": int(summary.get("count", 0)),
        "sample_size": int(observed.size),
        "reference_summary": {},
        "current_summary": {},
    }
    if observed.size < cfg.min_sample_size or summary.get("constant"):
        return [
            FeatureDriftResult(
                **base,
                drift_metric="insufficient_sample" if observed.size else "constant_reference",
                reliable=False,
                severity=MonitoringStatus.UNKNOWN,
                warnings=[
                    "Too few observations, or a single-category reference; reported "
                    "without a drift verdict."
                ],
            )
        ]

    statistic, p_value, unseen_rate = chi_square_drift(
        categories, observed, reference_count=int(summary.get("count", 0)) or None
    )
    out: list[FeatureDriftResult] = []
    for metric in cfg.categorical_metrics:
        thresholds = _thresholds_for(metric, cfg)
        if metric == "unseen_rate":
            severity = _severity(unseen_rate, thresholds)
            out.append(
                FeatureDriftResult(
                    **base,
                    drift_metric="unseen_rate",
                    drift_value=round(float(unseen_rate), 6),
                    threshold=thresholds[0],
                    drift_detected=severity
                    in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                    severity=severity,
                    warnings=(
                        ["Categories absent from training carry no learned behaviour."]
                        if unseen_rate > 0
                        else []
                    ),
                )
            )
        elif metric == "chi2":
            detected = p_value is not None and p_value <= cfg.alpha
            out.append(
                FeatureDriftResult(
                    **base,
                    drift_metric="chi2",
                    drift_value=None if statistic is None else round(float(statistic), 6),
                    p_value=None if p_value is None else round(float(p_value), 8),
                    threshold=cfg.alpha,
                    drift_detected=bool(detected),
                    severity=MonitoringStatus.WARNING if detected else MonitoringStatus.OK,
                    reliable=statistic is not None,
                )
            )
        else:
            current_frequencies = observed.astype(str).value_counts(normalize=True)
            labels = sorted(set(categories) | set(current_frequencies.index))
            value = jensen_shannon_divergence(
                [categories.get(label, 0.0) for label in labels],
                [float(current_frequencies.get(label, 0.0)) for label in labels],
            )
            severity = _severity(value, thresholds)
            out.append(
                FeatureDriftResult(
                    **base,
                    drift_metric="js",
                    drift_value=round(float(value), 6),
                    threshold=thresholds[0],
                    drift_detected=severity
                    in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL),
                    severity=severity,
                )
            )
    return out
