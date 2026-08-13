"""``python -m battery_rul.pipelines.run_fleet_batch`` — the ``run-fleet-batch`` stage.

Score a fleet offline and write its snapshot, ranking and plans.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["run-fleet-batch", *sys.argv[1:]]))
