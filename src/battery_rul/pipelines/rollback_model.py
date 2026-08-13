"""``python -m battery_rul.pipelines.rollback_model`` — the ``rollback-model`` stage.

Restore the previously live version of a model family.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["rollback-model", *sys.argv[1:]]))
