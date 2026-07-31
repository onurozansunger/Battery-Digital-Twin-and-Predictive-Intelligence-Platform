#!/usr/bin/env python3
"""Run the entire pipeline: prepare -> (tune) -> train -> evaluate -> predict.

python scripts/run_pipeline.py --config configs/default.yaml
python scripts/run_pipeline.py --config configs/fast.yaml     # ~1 minute
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["all", *sys.argv[1:]]))
