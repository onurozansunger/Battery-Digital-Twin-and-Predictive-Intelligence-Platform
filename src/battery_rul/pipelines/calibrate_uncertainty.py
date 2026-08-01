"""``python -m battery_rul.pipelines.calibrate_uncertainty`` — the ``calibrate-uncertainty`` stage.

A thin alias so each documented command is a real module path. The
implementation lives in :mod:`battery_rul.pipelines.milestone_2`; splitting it
across nine near-empty files would scatter one coherent pipeline for the sake of
the command names.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_2 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["calibrate-uncertainty", *sys.argv[1:]]))
