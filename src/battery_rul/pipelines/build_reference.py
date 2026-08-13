"""``python -m battery_rul.pipelines.build_reference`` — the ``build-reference`` stage.

Build the drift reference distribution from the training partition.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["build-reference", *sys.argv[1:]]))
