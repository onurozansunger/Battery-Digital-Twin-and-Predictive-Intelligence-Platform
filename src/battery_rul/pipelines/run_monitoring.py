"""``python -m battery_rul.pipelines.run_monitoring`` — the ``run-monitoring`` stage.

Run data-quality, drift and performance monitoring over a fleet batch.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["run-monitoring", *sys.argv[1:]]))
