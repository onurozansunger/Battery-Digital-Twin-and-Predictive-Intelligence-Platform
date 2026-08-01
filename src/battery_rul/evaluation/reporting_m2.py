"""Milestone 2 evaluation report.

Written from the metrics payload the pipelines actually produced — never from
remembered numbers. Every table in the rendered report is a projection of
``reports/milestone_2/metrics.json``, so a reader can diff the two.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.io import atomic_write_text
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["write_milestone_2_report"]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return "—" if value != value else f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    frame = pd.DataFrame(rows)
    present = [c for c in columns if c in frame.columns]
    if not present:
        present = list(frame.columns)
    header = "| " + " | ".join(present) + " |"
    divider = "| " + " | ".join("---" for _ in present) + " |"
    body = [
        "| " + " | ".join(_fmt(row.get(c)) for c in present) + " |"
        for row in frame[present].to_dict(orient="records")
    ]
    return "\n".join([header, divider, *body]) + "\n"


def write_milestone_2_report(cfg: ExperimentConfig, payload: dict[str, Any]) -> str:
    """Render ``reports/milestone_2/evaluation_report.md`` and return its path."""
    rul = payload.get("rul", {})
    soh = payload.get("soh", {})
    risk = payload.get("risk", {})
    multitask = payload.get("multitask", {})
    targets = payload.get("targets", {})
    environment = payload.get("environment", {})

    lines: list[str] = []
    add = lines.append

    add("# Milestone 2 — Battery Digital Twin: evaluation report\n")
    add(
        f"Generated at {environment.get('generated_at', 'unknown')} from git revision "
        f"`{environment.get('git_revision', 'unknown')}`.\n"
    )
    add(
        "> Every number below was produced by the pipeline run that wrote "
        "`reports/milestone_2/metrics.json`. Nothing here is carried over from an "
        "earlier run.\n"
    )

    # --- definitions ------------------------------------------------------
    add("\n## Definitions in force\n")
    add(
        f"- **End of life**: smoothed capacity at or below "
        f"{cfg.data.eol_threshold:.0%} of the {cfg.data.eol_reference} reference for "
        f"{cfg.target.eol_persistence} complete consecutive cycles.\n"
        f"- **State of health**: fraction in [0, 1], reference strategy "
        f"`{cfg.soh.reference_strategy}` over {cfg.soh.reference_cycles} cycle(s).\n"
        f"- **Failure risk**: RUL(t) ≤ {cfg.risk.horizon_cycles} cycles. A *derived* "
        "label from the capacity threshold — not an observed safety failure.\n"
        f"- **Uncertainty**: {cfg.uncertainty.method} at "
        f"{cfg.uncertainty.coverage:.0%} target coverage. Prediction intervals, not "
        "confidence intervals.\n"
        f"- **Probability calibration**: {cfg.calibration.method}, fitted on "
        "out-of-fold predictions over non-test cells only.\n"
    )

    # --- targets ----------------------------------------------------------
    if targets:
        add("\n## Target generation\n")
        soh_report = targets.get("soh", {})
        risk_report = targets.get("risk", {})
        add(
            f"SOH ({soh_report.get('strategy')}): {soh_report.get('n_rows')} rows, "
            f"range [{_fmt(soh_report.get('soh_min'), 3)}, "
            f"{_fmt(soh_report.get('soh_max'), 3)}], mean "
            f"{_fmt(soh_report.get('soh_mean'), 3)}.\n"
        )
        rates = risk_report.get("positive_rate", {})
        add(
            "\nRisk label positive rate by horizon: "
            + ", ".join(f"H={k}: {_fmt(v, 3)}" for k, v in rates.items())
            + "\n"
        )

    # --- RUL --------------------------------------------------------------
    add("\n## Remaining useful life and prediction intervals\n")
    if rul:
        oof = rul.get("out_of_fold_coverage", {})
        add(f"Deployed family: `{rul.get('model')}`.\n")
        add(
            f"\nOut-of-fold empirical coverage: **{_fmt(oof.get('empirical_coverage'), 3)}** "
            f"against a {cfg.uncertainty.coverage:.0%} target, mean interval width "
            f"{_fmt(oof.get('mean_interval_width'), 2)} cycles over {oof.get('n')} rows.\n"
        )
        test = rul.get("test_coverage", {})
        if test:
            add(
                f"\nHeld-out test coverage: **{_fmt(test.get('empirical_coverage'), 3)}**, "
                f"mean width {_fmt(test.get('mean_interval_width'), 2)} cycles over "
                f"{test.get('n')} rows.\n"
            )
        by_stage = oof.get("by_life_stage", {})
        if by_stage:
            add("\n### Coverage by life stage\n")
            add(
                _table(
                    [{"life_stage": stage, **values} for stage, values in by_stage.items()],
                    ["life_stage", "n", "empirical_coverage", "mean_interval_width"],
                )
            )
        by_battery = oof.get("by_battery_id", {})
        if by_battery:
            add("\n### Coverage by battery\n")
            add(
                _table(
                    [{"battery_id": cell, **values} for cell, values in by_battery.items()],
                    ["battery_id", "n", "empirical_coverage", "mean_interval_width"],
                )
            )
    else:
        add("_Not produced in this run._\n")

    # --- SOH --------------------------------------------------------------
    add("\n## State of health\n")
    if soh:
        add(f"Selected model: `{soh.get('selected_model')}` (chosen on validation).\n")
        add("\n### Out-of-fold metrics (non-test cells)\n")
        add(
            _table(
                [soh.get("out_of_fold_metrics", {})],
                ["n", "mae", "rmse", "r2", "max_absolute_error"],
            )
        )
        if soh.get("test_metrics"):
            add("\n### Held-out test metrics\n")
            add(_table([soh["test_metrics"]], ["n", "mae", "rmse", "r2", "max_error"]))
        if soh.get("per_battery"):
            add("\n### Per battery\n")
            add(_table(soh["per_battery"], ["battery_id", "partition", "n", "mae", "rmse", "r2"]))
    else:
        add("_Not produced in this run._\n")

    # --- risk -------------------------------------------------------------
    add("\n## Failure risk\n")
    if risk:
        calibration = risk.get("calibration", {})
        add(
            f"Model: `{risk.get('model')}`, horizon {risk.get('horizon_cycles')} cycles, "
            f"decision threshold {_fmt(risk.get('threshold'), 3)} "
            f"(tuned on out-of-fold non-test rows, objective "
            f"`{cfg.risk.threshold_objective}`).\n"
        )
        add(
            f"\nCalibration ({calibration.get('method')}) on "
            f"{calibration.get('n_calibration_rows')} rows: "
            f"Brier {_fmt((calibration.get('metrics_before') or {}).get('brier'))} → "
            f"{_fmt((calibration.get('metrics_after') or {}).get('brier'))}, "
            f"ECE {_fmt((calibration.get('metrics_before') or {}).get('ece'))} → "
            f"{_fmt((calibration.get('metrics_after') or {}).get('ece'))}.\n"
        )
        add(
            "\n> **Read the AUCs against the cycle-index baseline, not against 1.0.** "
            "The label is `RUL ≤ H`, so within a single cell the positives are exactly "
            "the last H cycles and *cycle index alone* ranks them perfectly. Any AUC "
            "on a single-cell partition is degenerate; the `*_cycle_index_baseline` "
            "columns are what carry information.\n"
        )
        add(
            "\n> The post-calibration Brier and ECE on out-of-fold rows are "
            "**in-sample** — the calibrator was fitted on those rows, so its ECE there "
            "is often exactly zero. The test-partition figures are the out-of-sample "
            "calibration evidence.\n"
        )
        add("\n### Out-of-fold, before and after calibration\n")
        add(
            _table(
                [
                    {"variant": "raw", **risk.get("out_of_fold_raw", {})},
                    {"variant": "calibrated", **risk.get("out_of_fold_calibrated", {})},
                ],
                [
                    "variant",
                    "n",
                    "n_positive",
                    "pr_auc",
                    "pr_auc_cycle_index_baseline",
                    "roc_auc",
                    "roc_auc_cycle_index_baseline",
                    "beats_cycle_index_baseline",
                    "precision",
                    "recall",
                    "f1",
                    "brier",
                    "ece",
                ],
            )
        )
        if risk.get("test_calibrated"):
            add("\n### Held-out test, calibrated\n")
            add(
                _table(
                    [risk["test_calibrated"]],
                    [
                        "n",
                        "n_test_cells",
                        "n_positive",
                        "pr_auc",
                        "pr_auc_cycle_index_baseline",
                        "roc_auc",
                        "roc_auc_cycle_index_baseline",
                        "beats_cycle_index_baseline",
                        "precision",
                        "recall",
                        "f1",
                        "brier",
                        "ece",
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "true_negative",
                    ],
                )
            )
    else:
        add("_Not produced in this run._\n")

    # --- multi-task -------------------------------------------------------
    add("\n## Multi-task model versus independent models\n")
    if multitask and not multitask.get("skipped"):
        fit = multitask.get("fit", {})
        add(
            f"Encoder `{fit.get('encoder')}`, window {fit.get('window')}, "
            f"{fit.get('n_parameters')} parameters, best epoch {fit.get('best_epoch')}.\n"
        )
        rows = []
        for partition, entry in (multitask.get("partitions") or {}).items():
            rows.append(
                {
                    "partition": partition,
                    "n_rows": entry.get("n_rows"),
                    "n_scored": entry.get("n_scored"),
                    "coverage": entry.get("coverage"),
                    "rul_mae": (entry.get("rul") or {}).get("mae"),
                    "soh_mae": (entry.get("soh") or {}).get("mae"),
                    "risk_pr_auc": (entry.get("risk") or {}).get("pr_auc"),
                    "risk_pr_auc_cycle_baseline": (entry.get("risk") or {}).get(
                        "pr_auc_cycle_index_baseline"
                    ),
                    "n_cells": entry.get("n_cells"),
                }
            )
        add(
            _table(
                rows,
                [
                    "partition",
                    "n_rows",
                    "n_scored",
                    "coverage",
                    "rul_mae",
                    "soh_mae",
                    "risk_pr_auc",
                ],
            )
        )
        add(
            "\nA multi-task risk PR-AUC near 1.0 on a one-cell partition is not "
            "evidence of a good classifier: cycle index alone achieves it, for the "
            "reason given in the risk section above. Compare the two columns.\n"
        )
        add(
            "\nCoverage below 1.0 is the sequence warm-up: the first "
            f"{cfg.multitask.window - 1} scoreable cycles of each cell have no full "
            "window. Those rows are reported as unscored rather than dropped from the "
            "denominator, so this table is comparable with the independent models "
            "only on the rows both can score.\n"
        )
    else:
        add("_Not produced in this run._\n")

    add("\n## Limitations\n")
    add(
        "See `docs/MILESTONE_2_LIMITATIONS.md`. In short: a five-cell laboratory "
        "cohort, one chemistry, one duty cycle, a derived rather than observed "
        "failure label, and conformal coverage that assumes an exchangeability "
        "between calibration and served cells that physical cells only "
        "approximately satisfy.\n"
    )

    path = cfg.paths.reports_dir / "milestone_2" / "evaluation_report.md"
    atomic_write_text(path, "\n".join(lines))
    logger.info("Milestone 2 report -> %s", path)
    return str(path)
