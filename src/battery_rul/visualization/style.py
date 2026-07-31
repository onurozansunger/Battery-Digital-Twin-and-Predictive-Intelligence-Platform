"""Publication-quality plotting defaults.

One place defines what every figure in the repository looks like, so the figure
set reads as one document rather than a scrapbook of notebook output.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from battery_rul.config import VizConfig
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["apply_style", "battery_palette", "figure", "save_figure"]

#: Colour-blind-safe qualitative palette (Okabe–Ito), extended for larger fleets.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO)),
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
}

_APPLIED = False


def apply_style(cfg: VizConfig | None = None, *, force: bool = False) -> None:
    """Install the project matplotlib style. Idempotent."""
    global _APPLIED
    if _APPLIED and not force:
        return
    mpl.use("Agg", force=False)  # headless: pipelines write files, never windows
    plt.rcParams.update(_STYLE)
    if cfg is not None:
        plt.rcParams["savefig.dpi"] = cfg.dpi
        plt.rcParams["figure.figsize"] = list(cfg.figsize)
    _APPLIED = True


def battery_palette(battery_ids: list[str]) -> dict[str, str]:
    """Stable colour per cell — the same cell keeps its colour across all figures."""
    ordered = sorted(set(battery_ids))
    if len(ordered) <= len(OKABE_ITO):
        colours = OKABE_ITO
    else:
        cmap = plt.get_cmap("viridis")
        colours = tuple(mpl.colors.to_hex(cmap(v)) for v in np.linspace(0, 0.92, len(ordered)))
    return {battery: colours[i % len(colours)] for i, battery in enumerate(ordered)}


@contextmanager
def figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    path: str | Path | None = None,
    cfg: VizConfig | None = None,
    **kwargs,
) -> Iterator[tuple[plt.Figure, np.ndarray | plt.Axes]]:
    """Create, yield, save and close a figure — so no pipeline leaks handles."""
    apply_style(cfg)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    try:
        if title:
            fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.995)
        yield fig, axes
        fig.tight_layout()
        if path is not None:
            save_figure(fig, path, cfg)
    finally:
        plt.close(fig)


def save_figure(fig: plt.Figure, path: str | Path, cfg: VizConfig | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=(cfg.dpi if cfg else 160), bbox_inches="tight")
    logger.debug("Figure -> %s", path)
    return path
