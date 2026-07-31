#!/usr/bin/env python3
"""Stage 1 — build the modelling dataset from raw battery files.

    python scripts/prepare_data.py --config configs/default.yaml

Thin wrapper around ``battery_rul.pipelines.prepare_data``; all logic lives in
``src/`` so notebooks, scripts and the CLI cannot drift apart.
"""

from __future__ import annotations

import sys

from battery_rul.cli import main

if __name__ == "__main__":
    sys.exit(main(["prepare", *sys.argv[1:]]))
