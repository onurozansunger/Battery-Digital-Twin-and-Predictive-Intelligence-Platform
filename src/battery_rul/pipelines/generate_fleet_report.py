"""``python -m battery_rul.pipelines.generate_fleet_report`` — the ``generate-fleet-report`` stage.

Render the Markdown fleet report from stored snapshots.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["generate-fleet-report", *sys.argv[1:]]))
