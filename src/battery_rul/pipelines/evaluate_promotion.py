"""``python -m battery_rul.pipelines.evaluate_promotion`` — the ``evaluate-promotion`` stage.

Evaluate the promotion gate for a candidate. Never promotes.

A thin alias so each documented command is a real module path; the
implementation lives in :mod:`battery_rul.pipelines.milestone_3`.
"""

from __future__ import annotations

import sys

from battery_rul.pipelines.milestone_3 import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["evaluate-promotion", *sys.argv[1:]]))
