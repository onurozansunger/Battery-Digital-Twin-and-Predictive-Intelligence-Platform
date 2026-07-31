"""Centralised logging.

One call to :func:`setup_logging` at process start; every module then uses
``get_logger(__name__)``. Rich handles the console, a plain rotating file handler
keeps a durable, grep-able record next to the run artifacts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False
_CONSOLE = Console(stderr=True)

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"

__all__ = ["get_logger", "log_section", "setup_logging"]


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger. Safe to call repeatedly (no duplicate handlers)."""
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level)

    console_handler = RichHandler(
        console=_CONSOLE,
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
        markup=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    # Third-party libraries are chatty and rarely say anything actionable.
    for noisy in ("matplotlib", "PIL", "optuna", "shap", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Auto-configures with defaults if the app forgot to."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, *, width: int = 78) -> None:
    """Visually separate pipeline stages in long console runs."""
    logger.info("")
    logger.info("=" * width)
    logger.info(title.upper())
    logger.info("=" * width)


def excepthook_to_logger(logger: logging.Logger) -> None:
    """Route uncaught exceptions through logging so the log file captures them."""

    def _hook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _hook
