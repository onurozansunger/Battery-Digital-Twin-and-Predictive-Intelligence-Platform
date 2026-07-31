"""Command-line interface.

    battery-rul prepare  --config configs/default.yaml
    battery-rul tune     --config configs/tuned.yaml
    battery-rul train    --config configs/default.yaml --set models.enabled='[xgboost]'
    battery-rul evaluate --config configs/default.yaml
    battery-rul predict  --input data/processed/cycles.parquet
    battery-rul all      --config configs/default.yaml

Every command accepts repeated ``--set key.path=value`` overrides, parsed with
YAML semantics, so any configuration field is reachable without editing a file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from battery_rul import __version__
from battery_rul.config import ExperimentConfig, load_config
from battery_rul.utils.logging import excepthook_to_logger, get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CONFIG = "configs/default.yaml"


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _load(args: argparse.Namespace) -> ExperimentConfig:
    path = Path(args.config)
    cfg = load_config(path if path.is_file() else None, overrides=_parse_overrides(args.set))
    if not path.is_file() and args.config != DEFAULT_CONFIG:
        raise SystemExit(f"Config not found: {path}")
    if not path.is_file():
        logger.warning("%s not found; using built-in defaults", path)
    cfg.paths.ensure()
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=cfg.paths.reports_dir / "pipeline.log",
        force=True,
    )
    excepthook_to_logger(logger)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battery-rul",
        description="Battery remaining-useful-life prediction platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"battery-rul {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help="Path to a YAML config.")
    common.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Dotted config override; repeatable (e.g. --set models.training.epochs=5).",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", parents=[common], help="Stage 1: build the modelling dataset.")
    sub.add_parser("tune", parents=[common], help="Stage 1b: Optuna hyperparameter search.")
    sub.add_parser("train", parents=[common], help="Stage 2: fit the model zoo.")
    sub.add_parser("evaluate", parents=[common], help="Stage 3: figures, SHAP and the report.")

    predict_parser = sub.add_parser("predict", parents=[common], help="Stage 4: score new cycles.")
    predict_parser.add_argument("--input", help="Cycle table (.parquet/.csv) to score.")
    predict_parser.add_argument("--output", help="Where to write predictions.")
    predict_parser.add_argument("--battery", action="append", help="Restrict to these cells.")

    all_parser = sub.add_parser("all", parents=[common], help="Run the whole pipeline.")
    all_parser.add_argument("--skip-tuning", action="store_true")
    all_parser.add_argument("--skip-eda", action="store_true")
    all_parser.add_argument("--skip-predict", action="store_true")
    all_parser.add_argument(
        "--no-leakage-check",
        action="store_true",
        help="Skip the causality verification (not recommended).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load(args)

    from battery_rul.pipelines import evaluate, predict, prepare_data, run_pipeline, train, tune

    if args.command == "prepare":
        prepare_data.run(cfg)
    elif args.command == "tune":
        tune.run(cfg)
    elif args.command == "train":
        train.run(cfg)
    elif args.command == "evaluate":
        evaluate.run_from_disk(cfg)
    elif args.command == "predict":
        predict.run(
            cfg,
            input_path=args.input,
            output_path=args.output,
            batteries=args.battery,
        )
    elif args.command == "all":
        run_pipeline.run(
            cfg,
            skip_tuning=args.skip_tuning,
            skip_eda=args.skip_eda,
            skip_predict=args.skip_predict,
            verify_leakage=not args.no_leakage_check,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
