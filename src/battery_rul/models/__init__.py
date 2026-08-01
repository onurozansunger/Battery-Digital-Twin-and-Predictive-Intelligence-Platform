"""Model zoo: tabular baselines, gradient boosting and sequence networks."""

from __future__ import annotations

from battery_rul.models import baselines as _baselines  # noqa: F401  (registers models)
from battery_rul.models import classical as _classical  # noqa: F401  (registers models)
from battery_rul.models import neural as _neural  # noqa: F401  (registers models)
from battery_rul.models.base import (
    BaseModel,
    TrainingData,
    available_models,
    build_model,
    register_model,
)
from battery_rul.models.search_spaces import SEARCH_SPACES, describe_spaces, suggest_params

__all__ = [
    "SEARCH_SPACES",
    "BaseModel",
    "TrainingData",
    "available_models",
    "build_model",
    "describe_spaces",
    "register_model",
    "suggest_params",
]
