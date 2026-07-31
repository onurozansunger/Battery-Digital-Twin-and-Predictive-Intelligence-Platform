#!/usr/bin/env python3
"""Stage 4 — score new cycle data with the persisted champion.

python scripts/predict.py                                   # held-out cells
python scripts/predict.py --input my_cycles.parquet --output out.csv
python scripts/predict.py --battery B0005
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["predict", *sys.argv[1:]]))
