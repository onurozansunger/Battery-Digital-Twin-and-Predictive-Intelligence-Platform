"""The fitted, serialisable feature transform.

Split of responsibility:

* :mod:`battery_rul.features.engineering` is **stateless** — it derives features
  from a cell's own history and is safe to run before any split.
* :class:`FeaturePipeline` is **stateful** — it learns scaler statistics and the
  final column set. It is therefore fit on the **training partition only** and
  merely applied to validation/test. This is the second leakage guard in the
  repository (the first being causality within a cell).

The fitted object is what ships as ``models/feature_pipeline.pkl`` and is what
``predict.py`` loads at inference time, guaranteeing that serving reproduces
training exactly — including column order.
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

    # -- fitting ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> FeaturePipeline:
        """Learn the column set and scaler statistics from *training rows only*."""
        if X.empty:
            raise ValueError("Cannot fit FeaturePipeline on an empty frame")

        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("No numeric feature columns to fit on")

        # Drop columns that are constant *within the training partition* — they
        # carry no learnable signal and destabilise scalers.
        variances = numeric.var(numeric_only=True)
        keep = [c for c in numeric.columns if variances.get(c, 0.0) > self.cfg.variance_threshold]
        dropped = sorted(set(numeric.columns) - set(keep))
        if dropped:
            logger.info("FeaturePipeline dropped %d train-constant columns", len(dropped))

        if (
            self.cfg.max_features is not None
            and y is not None
            and len(keep) > self.cfg.max_features
        ):
            keep, scores = _select_top_k(
                numeric[keep],
                np.asarray(y, dtype=float),
                self.cfg.max_features,
                self.cfg.selection_method,
            )
            self.selection_scores = scores
            logger.info(
                "Selected top %d/%d features by %s (train partition only)",
                len(keep),
                numeric.shape[1],
                self.cfg.selection_method,
            )

        self.feature_names = list(keep)
        values = numeric[self.feature_names].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if self.cfg.scaler == "none":
            self.scaler = None
        else:
            self.scaler = _SCALERS[self.cfg.scaler]()
            self.scaler.fit(values)

        self.fit_stats = {
            "n_rows_fit": int(len(X)),
            "n_candidate_features": int(numeric.shape[1]),
            "n_features": len(self.feature_names),
            "scaler": self.cfg.scaler,
            "selection_method": self.cfg.selection_method if self.cfg.max_features else None,
            "dropped_constant": dropped,
        }
        self.fitted = True
        logger.info(
            "FeaturePipeline fitted: %d features, scaler=%s, on %d rows",
            len(self.feature_names),
            self.cfg.scaler,
            len(X),
        )
        return self

    # -- application ------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transform. Column order is enforced, not assumed."""
        self._check_fitted()
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise FeatureUnseenColumnsError(
                f"{len(missing)} column(s) required by the fitted pipeline are absent "
                f"from the input: {missing[:10]}{'…' if len(missing) > 10 else ''}"
            )
        values = X[self.feature_names].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
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

    # -- misc ---------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline is not fitted. Call fit() first.")

    def __len__(self) -> int:
        return len(self.feature_names)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted else "unfitted"
        return f"FeaturePipeline({state}, n_features={len(self.feature_names)}, scaler={self.cfg.scaler})"
