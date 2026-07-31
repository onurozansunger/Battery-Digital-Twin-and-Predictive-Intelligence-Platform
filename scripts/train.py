#!/usr/bin/env python3
"""Stage 2 — fit the model zoo and select a champion.

python scripts/train.py --config configs/default.yaml
python scripts/train.py --set models.enabled='[xgboost, lightgbm]'
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
