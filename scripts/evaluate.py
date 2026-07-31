#!/usr/bin/env python3
"""Stage 3 — regenerate figures, explanations and the evaluation report.

Reloads the persisted model zoo, so the report can be rebuilt without retraining.

    python scripts/evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
