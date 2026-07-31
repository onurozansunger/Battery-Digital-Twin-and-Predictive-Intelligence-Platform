#!/usr/bin/env python3
"""Stage 1b — Optuna hyperparameter optimisation.

python scripts/tune.py --config configs/tuned.yaml
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["tune", *sys.argv[1:]]))
