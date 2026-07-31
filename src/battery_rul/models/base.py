"""Model abstraction and registry.

Every estimator — a scikit-learn regressor, a gradient-boosting library, or a
PyTorch sequence network — is wrapped so the training pipeline can treat them
identically:

    model = build_model("lstm", cfg)
    model.fit(train, val)          # TrainingData objects, not raw arrays
    y_hat = model.predict(test)

The wrapper owns windowing for the sequence models, so the pipeline never needs
to know whether a model is tabular or sequential.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.io import load_pickle, save_pickle
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "BaseModel",
    "TrainingData",
    "available_models",
    "build_model",
    "register_model",
]


@dataclass(slots=True)
class TrainingData:
    """One partition, carrying everything any model family might need.

    ``X`` is the *scaled* design matrix, row-aligned with ``frame`` — the
    sequence models re-window ``X`` themselves using ``frame``'s ids and cycle
    indices, which is why the metadata travels with the arrays.
    """

    X: np.ndarray
    y: np.ndarray
    frame: pd.DataFrame
    feature_names: list[str]

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y) or len(self.X) != len(self.frame):
            raise ValueError(
                f"TrainingData length mismatch: X={len(self.X)}, y={len(self.y)}, "
                f"frame={len(self.frame)}"
            )

    def __len__(self) -> int:
        return len(self.y)

    @property
    def battery_ids(self) -> np.ndarray:
        return self.frame["battery_id"].to_numpy()

    @property
    def cycle_index(self) -> np.ndarray:
        return self.frame["cycle_index"].to_numpy()

    @property
    def is_empty(self) -> bool:
        return len(self.y) == 0


_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(key: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Class decorator publishing a model under ``key``."""

    def _decorate(cls: type[BaseModel]) -> type[BaseModel]:
        normalised = key.strip().lower()
        if normalised in _REGISTRY:
            raise ValueError(f"Model {normalised!r} is already registered")
        _REGISTRY[normalised] = cls
        cls.key = normalised
        return cls

    return _decorate


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_model(name: str, cfg: ExperimentConfig, **overrides: Any) -> BaseModel:
    """Instantiate a registered model with config-supplied hyperparameters."""
    key = name.strip().lower()
    try:
        cls = _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}. Registered: {available_models()}") from exc
    params = cfg.models.params_for(key)
    params.update(overrides)
    return cls(cfg=cfg, params=params)


@dataclass
class BaseModel(ABC):
    """Common interface, persistence and bookkeeping for all estimators."""

    cfg: ExperimentConfig
    params: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False
    train_history: dict[str, list[float]] = field(default_factory=dict)
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    # Class-level metadata, NOT dataclass fields: as fields they would shadow the
    # value ``@register_model`` sets on the subclass, and every model would report
    # itself as "base".
    key: ClassVar[str] = "base"
    #: Sequence models consume windowed tensors; tabular models consume rows.
    is_sequence: ClassVar[bool] = False
    #: Whether SHAP's fast TreeExplainer applies.
    is_tree: ClassVar[bool] = False

    # -- interface --------------------------------------------------------
    @abstractmethod
    def fit(self, train: TrainingData, val: TrainingData | None = None) -> BaseModel:
        """Fit on ``train``; use ``val`` for early stopping where supported."""

    @abstractmethod
    def predict(self, data: TrainingData) -> np.ndarray:
        """Predict RUL in cycles, aligned row-for-row with ``data``.

        Sequence models cannot score the first ``window-1`` cycles of a cell;
        implementations must still return one value per input row and mark the
        unscoreable rows as NaN rather than silently dropping them.
        """

    # -- shared -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.key

    def feature_importance(self) -> pd.Series | None:
        """Native importances where the family provides them, else ``None``."""
        return None

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name} is not fitted. Call fit() first.")

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        return save_pickle(self, path)

    @staticmethod
    def load(path: str | Path) -> BaseModel:
        obj = load_pickle(path)
        if not isinstance(obj, BaseModel):
            raise TypeError(f"{path} does not contain a BaseModel (got {type(obj).__name__})")
        return obj

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "sequence" if self.is_sequence else "tabular",
            "params": {k: _jsonable(v) for k, v in self.params.items()},
            "fitted": self.fitted,
            **self.fit_metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(key={self.key!r}, fitted={self.fitted})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
