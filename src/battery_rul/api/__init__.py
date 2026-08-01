"""FastAPI application for the battery digital twin.

Importing this package must not load a model, read a config file or touch the
filesystem: ``create_app`` does that, and only when called. An import with side
effects makes the module untestable and turns a broken artifact into an
ImportError somewhere unrelated.
"""

from __future__ import annotations

from battery_rul.api.app import create_app

__all__ = ["create_app"]
