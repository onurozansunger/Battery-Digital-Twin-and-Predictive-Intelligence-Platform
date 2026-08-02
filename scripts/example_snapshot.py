#!/usr/bin/env python3
"""Produce an example digital-twin snapshot from a processed cell.

Writes ``reports/milestone_2/example_snapshot.json`` and prints a short
human-readable summary. Uses the same service the API and dashboard use — this
script is a client, not a second inference path.

    python scripts/example_snapshot.py --config configs/default.yaml
    python scripts/example_snapshot.py --battery B0007 --as-of-cycle 121
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_rul.config import load_config  # noqa: E402
from battery_rul.digital_twin.service import BatteryDigitalTwinService  # noqa: E402
from battery_rul.utils.io import save_json  # noqa: E402
from battery_rul.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Columns derived from the label or from the training-time health derivation.
#: A serving client would not have them, so they are stripped before the request.
_DERIVED_PREFIXES = (
    "rul_",
    "eol_",
    "life_",
    "is_censored",
    "soh",
    "capacity_smooth",
    "reference_capacity",
    "capacity_fade",
    "equivalent_full",
    "split",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--battery", help="Cell id. Defaults to the first available.")
    parser.add_argument(
        "--as-of-cycle",
        type=int,
        help="Truncate the history here, so the twin sees only what was available then.",
    )
    parser.add_argument("--output", help="Where to write the snapshot JSON.")
    args = parser.parse_args(argv)

    path = Path(args.config)
    cfg = load_config(path if path.is_file() else None)
    setup_logging(log_file=cfg.paths.reports_dir / "pipeline.log", force=True)

    cycles_path = cfg.paths.processed_dir / "cycles.parquet"
    if not cycles_path.is_file():
        logger.error(
            "%s not found. Run `python scripts/run_pipeline.py --config %s` first.",
            cycles_path,
            args.config,
        )
        return 1

    cycles = pd.read_parquet(cycles_path)
    battery_id = args.battery or str(cycles["battery_id"].iloc[0])
    if battery_id not in set(cycles["battery_id"].astype(str)):
        logger.error(
            "Battery %s is not in the processed dataset. Available: %s",
            battery_id,
            sorted(cycles["battery_id"].astype(str).unique().tolist()),
        )
        return 1

    history = cycles.loc[cycles["battery_id"].astype(str) == battery_id].reset_index(drop=True)
    if args.as_of_cycle is not None:
        history = history.loc[history["cycle_index"] <= args.as_of_cycle]
    history = history.drop(
        columns=[c for c in history.columns if c.startswith(_DERIVED_PREFIXES)], errors="ignore"
    )

    service = BatteryDigitalTwinService.create(cfg)
    if not service.readiness()["ready"]:
        logger.error(
            "No usable model bundle. Run `python -m battery_rul.pipelines.run_milestone_2 "
            "--config %s` first. Errors: %s",
            args.config,
            service.bundles.errors,
        )
        return 1

    snapshot = service.create_snapshot(battery_id, history)
    output = (
        Path(args.output)
        if args.output
        else (cfg.paths.reports_dir / "milestone_2" / "example_snapshot.json")
    )
    save_json(snapshot.to_json_dict(), output)

    print(_summary(snapshot))
    print(f"\nFull snapshot -> {output}")
    return 0


def _summary(snapshot) -> str:
    """The worked example from the milestone brief, filled in with real values."""
    health = snapshot.health
    prediction = snapshot.prediction
    risk = snapshot.failure_risk
    interval = prediction.rul_interval
    recommendation = snapshot.recommendation

    def _pct(value: float | None) -> str:
        return "—" if value is None else f"{100 * value:.1f}%"

    lines = [
        f"Battery ID: {snapshot.battery_id}",
        f"Current cycle: {snapshot.measurement_summary.latest_cycle}  (observed)",
        f"Current SOH: {_pct(health.soh)}  ({health.provenance.value} — a measurement, "
        "not a model output)",
        "Estimated RUL: "
        + ("—" if prediction.rul_cycles is None else f"{prediction.rul_cycles:.0f} cycles")
        + "  (predicted)",
    ]
    if health.soh_forecast is not None:
        lines.append(
            f"SOH forecast (+{health.soh_forecast_horizon_cycles} cycles): "
            f"{_pct(health.soh_forecast)}  (predicted, class {health.soh_forecast_class})"
        )
    if interval:
        lines.append(
            f"RUL interval: {interval.lower_bound:.0f}–{interval.upper_bound:.0f} cycles "
            f"({100 * interval.interval_coverage_target:.0f}% target coverage, "
            f"{interval.uncertainty_method}; prediction interval, not a confidence interval)"
        )
    lines += [
        f"Failure risk within {risk.horizon_cycles} cycles: {_pct(risk.probability)}"
        f"  ({'calibrated' if risk.is_calibrated else 'UNCALIBRATED'}"
        + (", EXPERIMENTAL — withheld from the recommendation" if risk.is_experimental else "")
        + ")",
        f"Health class: {health.health_class}",
        f"Risk class: {risk.risk_class}",
        f"Data quality: {snapshot.data_quality.quality_class} "
        f"(score {snapshot.data_quality.quality_score:.2f})",
    ]

    if snapshot.explanation and snapshot.explanation.drivers:
        lines.append("Main degradation factors (model attributions, not causal claims):")
        for driver in snapshot.explanation.drivers[:3]:
            lines.append(f"  - {driver.display_name} ({driver.contribution_direction})")

    lines += [
        f"Recommendation: {recommendation.title}  [{recommendation.action_code}, "
        f"priority {recommendation.priority}]",
        f"  {recommendation.explanation}",
    ]
    if recommendation.suggested_window_cycles:
        low, high = recommendation.suggested_window_cycles
        lines.append(f"  Suggested window: {low}–{high} cycles")
    if snapshot.warnings:
        lines.append(f"Warnings: {len(snapshot.warnings)}")
        for warning in snapshot.warnings:
            lines.append(f"  - {warning}")
    lines.append(f"\n{snapshot.disclaimer}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
