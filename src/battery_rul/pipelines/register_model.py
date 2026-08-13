"""``python -m battery_rul.pipelines.register_model`` — the ``register-model`` stage.

Register a built model bundle as a registry CANDIDATE.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["register-model", *sys.argv[1:]]))
