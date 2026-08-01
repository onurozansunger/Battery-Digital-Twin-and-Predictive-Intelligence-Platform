"""The fitted, serialisable feature transform.

Split of responsibility:

* :mod:`battery_rul.features.engineering` is **stateless** — it derives features
  from a cell's own history and is safe to run before any split.
* :class:`FeaturePipeline` is **stateful** — it learns *every* data-dependent
  decision: the fleet-level imputation fallback, the variance filter, the
  correlation pruning, the supervised top-K selection and the scaler statistics.
  It is fit on the **training partition only** and merely applied to
  validation/test/serving. This is the second leakage guard in the repository
  (the first being causality within a cell).

Milestone 1.1 moved variance filtering and correlation pruning here from the
pre-split feature builder. Before that change, both were computed over the whole
loaded table, so the *identity of the surviving columns* was a function of the
held-out cells — a quiet but real violation of "held-out batteries were never
seen". Everything data-dependent now sits behind the evaluation boundary and is
re-fitted inside every cross-validation fold.

The fitted object is what ships as ``models/feature_pipeline.pkl`` and is what
``predict.py`` and the digital-twin service load at inference time, guaranteeing
that serving reproduces training exactly — including column order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from battery_rul.config import FeatureConfig
from battery_rul.utils.io import load_pickle, save_pickle
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FeaturePipeline"]

_SCALERS: dict[str, Any] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


class FeatureUnseenColumnsError(ValueError):
    """Raised when transform() is handed a frame missing fitted columns."""


def _prune_correlated(
    X: pd.DataFrame, threshold: float | None
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Drop one column of every near-duplicate pair, keeping the earlier column.

    Called **only** from :meth:`FeaturePipeline.fit`. The correlation matrix is a
    statistic of the rows it is computed over, so computing it before the split —
    which is what this repository used to do — let a held-out cell decide which
    column survived into the training schema.
    """
    columns = list(X.columns)
    if threshold is None or len(columns) < 2:
        return columns, []

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped: set[str] = set()
    record: list[tuple[str, str, float]] = []
    for column in upper.columns:
        if column in dropped:
            continue
        partners = upper[column]
        for other, rho in partners[partners > threshold].items():
            if other in dropped or other == column:
                continue
            dropped.add(str(other))
            record.append((str(column), str(other), float(rho)))
    if dropped:
        logger.info("Dropped %d features correlated above |rho| > %.3f", len(dropped), threshold)
    return [c for c in columns if c not in dropped], record


def _select_top_k(
    X: pd.DataFrame, y: np.ndarray, k: int, method: str
) -> tuple[list[str], dict[str, float]]:
    """Rank features against the target and keep the best ``k``.

    Called **only** from :meth:`FeaturePipeline.fit`, i.e. with training rows.
    Running this on the full dataset would be textbook selection leakage: the
    identity of the surviving columns would encode test-set information.

    ``tree_importance`` (the default) uses a small ExtraTrees ensemble. Unlike
    ``f_regression`` it captures the non-linear, saturating relationships that
    dominate here (a rolling-std feature matters near end of life and nowhere
    else), and unlike ``mutual_info`` it is stable at these sample sizes.
    """
    values = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    columns = list(X.columns)

    if method == "f_regression":
        from sklearn.feature_selection import f_regression

        stat, _ = f_regression(values, y)
        scores = np.nan_to_num(stat)
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression

        scores = mutual_info_regression(values, y, random_state=0)
    else:
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=300, max_depth=None, random_state=0, n_jobs=-1, min_samples_leaf=2
        )
        estimator.fit(values, y)
        scores = estimator.feature_importances_

    order = np.argsort(scores)[::-1][:k]
    # Restore the original column order among survivors: stable ordering makes
    # artifacts diffable across runs.
    chosen = sorted((columns[i] for i in order), key=columns.index)
    return chosen, {columns[i]: float(scores[i]) for i in order}


@dataclass
class FeaturePipeline:
    """Scale a fixed, ordered set of feature columns.

    Attributes
    ----------
    feature_names:
        The exact columns, in the exact order, the model was trained on.
    scaler:
        Fitted sklearn scaler, or ``None`` when ``cfg.scaler == "none"``.
    """

    cfg: FeatureConfig
    feature_names: list[str] = field(default_factory=list)
    scaler: Any | None = None
    fitted: bool = False
    fit_stats: dict[str, Any] = field(default_factory=dict)
    selection_scores: dict[str, float] = field(default_factory=dict)
    #: Fleet-level fallback per surviving column, learned from training rows only.
    fallback_values: dict[str, float] = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_correlated: list[tuple[str, str, float]] = field(default_factory=list)

    # -- fitting ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> FeaturePipeline:
        """Learn every data-dependent decision from *training rows only*.

        In order: fleet fallback values, variance filter, correlation pruning,
        supervised top-K selection, scaler statistics. Each stage sees only the
        rows handed in, which is what makes re-fitting inside a CV fold correct
        rather than merely conventional.
        """
        if X.empty:
            raise ValueError("Cannot fit FeaturePipeline on an empty frame")

        numeric = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if numeric.shape[1] == 0:
            raise ValueError("No numeric feature columns to fit on")
        n_candidates = int(numeric.shape[1])

        # --- 1. fleet fallback for values never observed in a cell's history ---
        self.fallback_values = self._fit_fallback(numeric)
        filled = numeric.fillna(value=self.fallback_values).fillna(0.0)

        # --- 2. variance filter (train partition only) -------------------------
        variances = filled.var(numeric_only=True)
        keep = [c for c in filled.columns if variances.get(c, 0.0) > self.cfg.variance_threshold]
        self.dropped_constant = sorted(set(filled.columns) - set(keep))
        if self.dropped_constant:
            logger.info(
                "FeaturePipeline dropped %d train-constant columns", len(self.dropped_constant)
            )

        # --- 3. correlation pruning (train partition only) ---------------------
        keep, self.dropped_correlated = _prune_correlated(
            filled[keep], self.cfg.correlation_prune_threshold
        )

        # --- 4. supervised selection (train partition only) --------------------
        if (
            self.cfg.max_features is not None
            and y is not None
            and len(keep) > self.cfg.max_features
        ):
            keep, scores = _select_top_k(
                filled[keep],
                np.asarray(y, dtype=float),
                self.cfg.max_features,
                self.cfg.selection_method,
            )
            self.selection_scores = scores
            logger.info(
                "Selected top %d/%d features by %s (train partition only)",
                len(keep),
                n_candidates,
                self.cfg.selection_method,
            )

        self.feature_names = list(keep)
        self.fallback_values = {
            k: v for k, v in self.fallback_values.items() if k in set(self.feature_names)
        }

        # --- 5. scaler ---------------------------------------------------------
        values = filled[self.feature_names].to_numpy(dtype=np.float64)
        if self.cfg.scaler == "none":
            self.scaler = None
        else:
            self.scaler = _SCALERS[self.cfg.scaler]()
            self.scaler.fit(values)

        self.fit_stats = {
            "n_rows_fit": int(len(X)),
            "n_candidate_features": n_candidates,
            "n_features": len(self.feature_names),
            "scaler": self.cfg.scaler,
            "selection_method": self.cfg.selection_method if self.cfg.max_features else None,
            "fallback_imputation": self.cfg.fallback_imputation,
            "correlation_prune_threshold": self.cfg.correlation_prune_threshold,
            "dropped_constant": self.dropped_constant,
            "n_dropped_correlated": len(self.dropped_correlated),
        }
        self.fitted = True
        logger.info(
            "FeaturePipeline fitted: %d/%d features kept, scaler=%s, on %d rows",
            len(self.feature_names),
            n_candidates,
            self.cfg.scaler,
            len(X),
        )
        return self

    def _fit_fallback(self, numeric: pd.DataFrame) -> dict[str, float]:
        """Fleet-level fallback statistic per column, from training rows only."""
        if self.cfg.fallback_imputation == "zero":
            return dict.fromkeys(numeric.columns, 0.0)
        stat = (
            numeric.median(numeric_only=True)
            if self.cfg.fallback_imputation == "median"
            else numeric.mean(numeric_only=True)
        )
        return {
            str(col): (float(value) if np.isfinite(value) else 0.0) for col, value in stat.items()
        }

    # -- application ------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transform. Column order is enforced, not assumed.

        Missing values are filled with the *persisted training* fallback, never
        with a statistic computed from the frame being transformed — otherwise a
        held-out battery would impute itself and a single-row serving request
        would impute from nothing at all.
        """
        self._check_fitted()
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise FeatureUnseenColumnsError(
                f"{len(missing)} column(s) required by the fitted pipeline are absent "
                f"from the input: {missing[:10]}{'…' if len(missing) > 10 else ''}"
            )
        frame = X[self.feature_names].replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(value=self.fallback_values).fillna(0.0)
        values = frame.to_numpy(dtype=np.float64)
        return values if self.scaler is None else self.scaler.transform(values)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform but keep a labelled DataFrame — used by SHAP and plots."""
        return pd.DataFrame(self.transform(X), columns=self.feature_names, index=X.index)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return values if self.scaler is None else self.scaler.inverse_transform(values)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        return save_pickle(self, path)

    @staticmethod
    def load(path: str | Path) -> FeaturePipeline:
        obj = load_pickle(path)
        if not isinstance(obj, FeaturePipeline):
            raise TypeError(f"{path} does not contain a FeaturePipeline (got {type(obj).__name__})")
        return obj

    # -- provenance ---------------------------------------------------------
    def schema(self) -> dict[str, Any]:
        """The exact feature contract this pipeline enforces, for the bundle."""
        self._check_fitted()
        return {
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
            "scaler": self.cfg.scaler,
            "fallback_imputation": self.cfg.fallback_imputation,
            "fallback_values": {k: round(v, 8) for k, v in self.fallback_values.items()},
            "dropped_constant": list(self.dropped_constant),
            "dropped_correlated": [
                {"kept": a, "dropped": b, "rho": round(r, 5)} for a, b, r in self.dropped_correlated
            ],
            "selection_method": self.cfg.selection_method if self.cfg.max_features else None,
            "max_features": self.cfg.max_features,
        }

    def fingerprint(self) -> str:
        """Stable hash over the ordered feature schema and preprocessing decisions."""
        import hashlib
        import json as _json

        payload = _json.dumps(
            {
                "feature_names": list(self.feature_names),
                "scaler": self.cfg.scaler,
                "fallback_imputation": self.cfg.fallback_imputation,
                "fallback_values": {
                    k: round(float(v), 6) for k, v in sorted(self.fallback_values.items())
                },
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -- misc ---------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline is not fitted. Call fit() first.")

    def __len__(self) -> int:
        return len(self.feature_names)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted else "unfitted"
        return f"FeaturePipeline({state}, n_features={len(self.feature_names)}, scaler={self.cfg.scaler})"
