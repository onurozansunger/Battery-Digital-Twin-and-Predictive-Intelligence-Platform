"""Battery Digital Twin & Predictive Intelligence Platform.

Milestone 1: Remaining Useful Life (RUL) prediction for lithium-ion cells.

The public surface is intentionally small — pipelines are the entry point:

>>> from battery_rul import load_config
>>> from battery_rul.pipelines import prepare_data, train
>>> cfg = load_config("configs/default.yaml")
>>> dataset = prepare_data.run(cfg)
>>> results = train.run(cfg)
"""

from __future__ import annotations

# MUST come first: pre-loads LightGBM/XGBoost/CatBoost ahead of torch to avoid a
# duplicate-OpenMP segfault. See battery_rul/_compat.py for the full story.
from battery_rul._compat import preload_native_libraries

preload_native_libraries()

from battery_rul.config import ExperimentConfig, load_config, project_root  # noqa: E402

__version__ = "0.1.0"
__all__ = ["ExperimentConfig", "__version__", "load_config", "project_root"]
