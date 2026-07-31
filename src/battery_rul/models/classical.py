"""Tabular baselines: linear, ensemble and gradient-boosting regressors.

These are baselines in the useful sense — they are *strong*, properly configured,
and given the same features as the neural models. A deep model that cannot beat a
tuned LightGBM on 600 rows of tabular data has not earned its place, and the
comparison table in the evaluation report is written to make that visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.svm import SVR

from battery_rul.models.base import BaseModel, TrainingData, register_model
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CatBoostModel",
    "ElasticNetModel",
    "GradientBoostingModel",
    "LightGBMModel",
    "LinearRegressionModel",
    "RandomForestModel",
    "RidgeModel",
    "SVRModel",
    "XGBoostModel",
]


@dataclass
class SklearnModel(BaseModel):
    """Adapter for any estimator exposing ``fit``/``predict``."""

    estimator: Any = None
    default_params: dict[str, Any] = field(default_factory=dict)

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(
        self,
        train: TrainingData,
        val: TrainingData | None = None,  # noqa: ARG002
    ) -> SklearnModel:
        # `val` is part of the BaseModel contract but unused here: plain sklearn
        # estimators have no early stopping. Subclasses that do (XGBoost,
        # LightGBM, CatBoost) override this method and consume it.
        if train.is_empty:
            raise ValueError(f"{self.name}: empty training partition")
        self.estimator = self._build()
        self.estimator.fit(train.X, train.y)
        self.fitted = True
        self.fit_metadata = {
            "n_train_rows": len(train),
            "n_features": train.X.shape[1],
            "feature_names": list(train.feature_names),
        }
        logger.info("%s fitted on %d rows x %d features", self.name, len(train), train.X.shape[1])
        return self

    def predict(self, data: TrainingData) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.estimator.predict(data.X), dtype=float)

    def feature_importance(self) -> pd.Series | None:
        if not self.fitted:
            return None
        names = self.fit_metadata.get("feature_names", [])
        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)
        elif hasattr(self.estimator, "coef_"):
            # For linear models the absolute standardised coefficient is the
            # closest analogue; features are already scaled by FeaturePipeline.
            values = np.abs(np.ravel(self.estimator.coef_)).astype(float)
        else:
            return None
        if len(values) != len(names):
            return None
        return pd.Series(values, index=names).sort_values(ascending=False)

    def _merged(self, **defaults: Any) -> dict[str, Any]:
        merged = dict(defaults)
        merged.update(self.params)
        return merged


# ---------------------------------------------------------------------------
# Linear family
# ---------------------------------------------------------------------------
@register_model("linear_regression")
class LinearRegressionModel(SklearnModel):
    """Ordinary least squares — the sanity floor.

    Included deliberately: RUL is close to linear in cycle count within a single
    cell, so OLS looks excellent under a chronological split and poor under a
    battery holdout. The gap between those two numbers is itself a useful
    diagnostic of how much a model has actually learned about degradation.
    """

    def _build(self) -> Any:
        return LinearRegression(**self._merged())


@register_model("ridge")
class RidgeModel(SklearnModel):
    def _build(self) -> Any:
        return Ridge(**self._merged(alpha=1.0, random_state=None))


@register_model("elastic_net")
class ElasticNetModel(SklearnModel):
    def _build(self) -> Any:
        return ElasticNet(
            **self._merged(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=self.cfg.seed)
        )


@register_model("svr")
class SVRModel(SklearnModel):
    def _build(self) -> Any:
        return SVR(**self._merged(kernel="rbf", C=10.0, epsilon=2.0, gamma="scale"))


# ---------------------------------------------------------------------------
# Tree ensembles
# ---------------------------------------------------------------------------
@register_model("random_forest")
class RandomForestModel(SklearnModel):
    is_tree: ClassVar[bool] = True

    def _build(self) -> Any:
        return RandomForestRegressor(
            **self._merged(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=self.cfg.seed,
            )
        )


@register_model("gradient_boosting")
class GradientBoostingModel(SklearnModel):
    is_tree: ClassVar[bool] = True

    def _build(self) -> Any:
        return GradientBoostingRegressor(
            **self._merged(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.85,
                random_state=self.cfg.seed,
            )
        )


@register_model("xgboost")
class XGBoostModel(SklearnModel):
    """XGBoost with early stopping against the validation cells."""

    is_tree: ClassVar[bool] = True

    def _build(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            **self._merged(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.7,
                min_child_weight=3,
                reg_lambda=2.0,
                reg_alpha=0.1,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=self.cfg.seed,
                early_stopping_rounds=60,
                verbosity=0,
            )
        )

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> XGBoostModel:
        self.estimator = self._build()
        eval_set = None if val is None or val.is_empty else [(val.X, val.y)]
        if eval_set is None:
            # Without a validation partition early stopping is meaningless; drop
            # it rather than let XGBoost stop on the training set.
            self.estimator.set_params(early_stopping_rounds=None)
        self.estimator.fit(train.X, train.y, eval_set=eval_set, verbose=False)
        self.fitted = True
        best_iteration = getattr(self.estimator, "best_iteration", None)
        self.fit_metadata = {
            "n_train_rows": len(train),
            "n_features": train.X.shape[1],
            "feature_names": list(train.feature_names),
            "best_iteration": None if best_iteration is None else int(best_iteration),
        }
        results = getattr(self.estimator, "evals_result_", None) or {}
        for split, metrics in results.items():
            for metric, values in metrics.items():
                self.train_history[f"{split}_{metric}"] = [float(v) for v in values]
        logger.info(
            "%s fitted on %d rows (best_iteration=%s)", self.name, len(train), best_iteration
        )
        return self


@register_model("lightgbm")
class LightGBMModel(SklearnModel):
    is_tree: ClassVar[bool] = True

    def _build(self) -> Any:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **self._merged(
                n_estimators=1500,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=6,
                min_child_samples=10,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=2.0,
                n_jobs=-1,
                random_state=self.cfg.seed,
                verbose=-1,
            )
        )

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> LightGBMModel:
        import lightgbm as lgb

        self.estimator = self._build()
        callbacks = [lgb.log_evaluation(period=0)]
        fit_kwargs: dict[str, Any] = {}
        if val is not None and not val.is_empty:
            callbacks.append(lgb.early_stopping(stopping_rounds=80, verbose=False))
            # LightGBM >= 4.7 deprecates `eval_set` in favour of `eval_X`/`eval_y`;
            # probe the signature so the wrapper works on both generations.
            import inspect

            signature = inspect.signature(self.estimator.fit)
            if "eval_X" in signature.parameters:
                fit_kwargs.update(eval_X=val.X, eval_y=val.y)
            else:
                fit_kwargs["eval_set"] = [(val.X, val.y)]
        self.estimator.fit(train.X, train.y, callbacks=callbacks, **fit_kwargs)
        self.fitted = True
        self.fit_metadata = {
            "n_train_rows": len(train),
            "n_features": train.X.shape[1],
            "feature_names": list(train.feature_names),
            "best_iteration": int(getattr(self.estimator, "best_iteration_", 0) or 0),
        }
        for split, metrics in (getattr(self.estimator, "evals_result_", None) or {}).items():
            for metric, values in metrics.items():
                self.train_history[f"{split}_{metric}"] = [float(v) for v in values]
        return self


@register_model("catboost")
class CatBoostModel(SklearnModel):
    is_tree: ClassVar[bool] = True

    def _build(self) -> Any:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **self._merged(
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                loss_function="RMSE",
                random_seed=self.cfg.seed,
                verbose=False,
                allow_writing_files=False,
            )
        )

    def fit(self, train: TrainingData, val: TrainingData | None = None) -> CatBoostModel:
        self.estimator = self._build()
        eval_set = None if val is None or val.is_empty else (val.X, val.y)
        self.estimator.fit(
            train.X,
            train.y,
            eval_set=eval_set,
            early_stopping_rounds=80 if eval_set is not None else None,
            verbose=False,
        )
        self.fitted = True
        self.fit_metadata = {
            "n_train_rows": len(train),
            "n_features": train.X.shape[1],
            "feature_names": list(train.feature_names),
            "best_iteration": int(self.estimator.get_best_iteration() or 0),
        }
        return self
