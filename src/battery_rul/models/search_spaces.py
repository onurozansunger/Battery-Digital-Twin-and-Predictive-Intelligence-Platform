"""Optuna search spaces, versioned with the code.

Keeping the spaces here rather than inside the tuning script means the exact
space a study explored is recoverable from the git revision recorded in
``metrics.json`` — a search space defined ad hoc in a notebook is not
reproducible in any meaningful sense.

Ranges are deliberately conservative. With ~270 training rows across four cells,
a wide space finds hyperparameters that fit the validation cells rather than the
degradation process; every range below is bounded to keep capacity in the region
where the model can be checked against physics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna

__all__ = ["SEARCH_SPACES", "describe_spaces", "suggest_params"]

SpaceFn = Callable[[optuna.Trial], dict[str, Any]]


def _random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.6]),
    }


def _xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 25.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 2.0, log=True),
    }


def _lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 25.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def _catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def _gradient_boosting(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
    }


def _ridge(trial: optuna.Trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 500.0, log=True)}


def _elastic_net(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


def _svr(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.1, 500.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.1, 20.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _recurrent(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 8e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "window": trial.suggest_categorical("window", [10, 15, 20, 30]),
        "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
    }


def _transformer(trial: optuna.Trial) -> dict[str, Any]:
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
    return {
        # Keep the head count compatible with the model width; Optuna samples the
        # two independently, so the model wrapper rounds d_model down if needed.
        "nhead": nhead,
        "d_model": d_model,
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "window": trial.suggest_categorical("window", [10, 15, 20, 30]),
    }


SEARCH_SPACES: dict[str, SpaceFn] = {
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "gradient_boosting": _gradient_boosting,
    "ridge": _ridge,
    "elastic_net": _elastic_net,
    "svr": _svr,
    "lstm": _recurrent,
    "gru": _recurrent,
    "transformer": _transformer,
}


def suggest_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter vector for ``model_name``."""
    key = model_name.strip().lower()
    if key not in SEARCH_SPACES:
        raise KeyError(
            f"No search space for {model_name!r}. Tunable models: {sorted(SEARCH_SPACES)}"
        )
    return SEARCH_SPACES[key](trial)


def describe_spaces() -> dict[str, list[str]]:
    """Parameter names per model — rendered into the experiment report."""
    import inspect
    import re

    pattern = re.compile(r"trial\.suggest_\w+\(\s*\"([^\"]+)\"")
    out: dict[str, list[str]] = {}
    for name, fn in SEARCH_SPACES.items():
        out[name] = sorted(set(pattern.findall(inspect.getsource(fn))))
    return out
