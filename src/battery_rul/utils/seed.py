"""Determinism helpers.

Reproducibility is a first-class requirement of this repository: the same config
and the same raw data must yield the same metrics.json. This module pins every
RNG we can reach.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np

from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["seed_everything", "torch_generator"]


def seed_everything(seed: int = 42, *, deterministic_torch: bool = True) -> int:
    """Seed ``random``, ``numpy``, ``PYTHONHASHSEED`` and (if installed) ``torch``.

    Parameters
    ----------
    seed:
        The seed to apply.
    deterministic_torch:
        Also force cuDNN into deterministic mode. Costs some throughput; worth it
        for a portfolio/benchmark repository where numbers must be citable.

    Returns
    -------
    int
        The seed, so callers can log or persist it in one expression.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        logger.debug("torch not installed; skipped torch seeding")
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Not all ops have deterministic kernels, and older torch builds lack the
        # call entirely; neither is worth failing a run over.
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def torch_generator(seed: int):  # type: ignore[no-untyped-def]
    """A seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
