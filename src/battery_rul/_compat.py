"""Environment fixes that must run before anything else is imported.

Duplicate OpenMP runtimes
-------------------------
PyTorch, LightGBM and XGBoost each ship their own copy of the OpenMP runtime.
On macOS (and on some Linux/conda installs) whichever library loads *second*
binds against the already-loaded ``libomp``, and LightGBM in particular then
segfaults — the process dies with SIGSEGV inside ``LGBMRegressor.fit``, with no
Python traceback, which is a miserable thing to debug from a stack trace that
does not exist.

Loading the boosting libraries **before** torch avoids the conflict entirely.
The commonly cited ``KMP_DUPLICATE_LIB_OK=TRUE`` escape hatch was measured on
this project's reference environment and does *not* fix it — it suppresses the
Intel runtime's guard message without making the situation safe.

This module is imported at the top of ``battery_rul/__init__.py``, so any code
that does ``import battery_rul`` gets a working process regardless of the order
its own imports happen to be in.
"""

from __future__ import annotations

import os
import sys

__all__ = ["preload_native_libraries"]

_DONE = False

#: Order matters. These must all be resident before torch's OpenMP is loaded.
_BOOSTING_LIBRARIES = ("lightgbm", "xgboost", "catboost")


def preload_native_libraries() -> list[str]:
    """Import the OpenMP-linked boosting libraries ahead of torch.

    Returns the module names that were successfully pre-loaded. Missing
    libraries are ignored — they are optional dependencies, and a user who has
    not installed CatBoost should not be stopped from running everything else.
    """
    global _DONE
    if _DONE:
        return []

    # A single OpenMP thread. This is not a performance compromise worth
    # agonising over: the reference environment has FOUR libomp copies resident
    # (scikit-learn's, torch's, LightGBM's and conda's), and with more than one
    # thread the nested parallel regions crash the process outright. The whole
    # dataset is ~600 rows x 80 features, where OpenMP parallelism buys nothing
    # measurable. Export OMP_NUM_THREADS yourself to override on a machine with a
    # single, consistent runtime.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    loaded: list[str] = []
    torch_already_loaded = "torch" in sys.modules

    for name in _BOOSTING_LIBRARIES:
        try:
            __import__(name)
            loaded.append(name)
        except ImportError:
            continue

    if torch_already_loaded and loaded:
        import warnings

        warnings.warn(
            "torch was imported before battery_rul; LightGBM/XGBoost may crash "
            "with SIGSEGV from a duplicate OpenMP runtime. Import battery_rul "
            "(or battery_rul._compat) first.",
            RuntimeWarning,
            stacklevel=2,
        )

    _DONE = True
    return loaded
