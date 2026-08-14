"""Render README section 8 from the committed evaluation artifacts.

The pipeline owns the numbers.  This script makes the README a projection of
those artifacts instead of a second, hand-maintained source of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "reports" / "metrics.json"
CV_BY_BATTERY = ROOT / "reports" / "cross_validation_by_battery.csv"
MODEL_COMPARISON = ROOT / "reports" / "model_comparison.csv"
MODEL_COMPARISON_COMMON = ROOT / "reports" / "model_comparison_common_rows.csv"
START = "<!-- BEGIN AUTO-GENERATED RESULTS -->"
END = "<!-- END AUTO-GENERATED RESULTS -->"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _name(value: str) -> str:
    names = {
        "catboost": "CatBoost",
        "gru": "GRU",
        "lightgbm": "LightGBM",
        "linear_regression": "Linear Regression",
        "lstm": "LSTM",
        "random_forest": "Random Forest",
        "ridge": "Ridge",
        "transformer": "Transformer",
        "xgboost": "XGBoost",
    }
    return names.get(value, value.replace("_", " ").title())


def _f(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    number = float(value)
    return f"{number:+.{digits}f}".replace("-", "−")


def _pct(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f} %"


def render() -> str:
    metrics: dict[str, Any] = json.loads(METRICS.read_text(encoding="utf-8"))
    cv_rows = _rows(CV_BY_BATTERY)
    comparison = _rows(MODEL_COMPARISON)
    common = _rows(MODEL_COMPARISON_COMMON)

    champion = str(metrics["champion"])
    champion_name = _name(champion)
    cv = metrics["cross_validation"]
    if cv["model"] != champion:
        raise ValueError(
            "reports/metrics.json disagrees with itself: champion "
            f"{champion!r}, cross-validation model {cv['model']!r}"
        )
    pooled = cv["pooled"]
    fold_mae = [float(row["mae"]) for row in cv_rows]
    split = metrics["split"]
    train = "/".join(split["train_batteries"])
    val = "/".join(split["val_batteries"])
    test = "/".join(split["test_batteries"])
    selection_metric = str(metrics["selection_metric"]).upper()
    validation_rmse = metrics["validation"][champion]["metrics"]["rmse"]

    lines = [
        START,
        "",
        "### 8.1 The headline: leave-one-battery-out cross-validation",
        "",
        f"Each of the {len(cv_rows)} cells is held out in turn, the feature pipeline is re-fit",
        "inside every fold, and out-of-fold predictions are pooled. The model selected",
        f"on the validation partition is **{champion_name}**.",
        "",
        "| | MAE | RMSE | R² | Bias | within 10 cycles |",
        "|---|---|---|---|---|---|",
        (
            f"| **Pooled ({int(pooled['n'])} rows, {int(pooled['n_folds'])} folds)** "
            f"| **{_f(pooled['mae'])}** | **{_f(pooled['rmse'])}** "
            f"| **{_f(pooled['r2'], 3)}** | {_signed(pooled['bias'])} "
            f"| {_pct(pooled['within_10_cycles'])} |"
        ),
        "",
        "| Held-out cell | n | MAE | RMSE | R² | Bias |",
        "|---|---|---|---|---|---|",
    ]
    for row in cv_rows:
        lines.append(
            f"| {row['battery_id']} | {int(float(row['n']))} | {_f(row['mae'])} "
            f"| {_f(row['rmse'])} | {_f(row['r2'], 3)} | {_signed(row['bias'])} |"
        )
    lines += [
        "",
        (
            f"Per-fold MAE spans {_f(min(fold_mae), 1)}–{_f(max(fold_mae), 1)} cycles "
            f"(σ = {_f(pooled['mae_across_folds_std'])}). **That spread is the honest"
        ),
        "uncertainty on this model-level estimate**—more so than a bootstrap over rows,",
        "because rows within a cell are strongly correlated. The repository's primary",
        "claim remains the nested LOBO estimate in §2, which also includes model selection.",
        "",
        "### 8.2 The single holdout, and why it is not the headline",
        "",
        f"Train on {train}, validate on {val}, test on {test}. {champion_name} was selected",
        f"by validation {selection_metric} ({_f(validation_rmse)} cycles); the ranking below is",
        "the untouched test result and therefore must not be used to re-select the model.",
        "",
        "| Rank | Model | n | MAE | RMSE | R² | Bias | α-λ (20 %) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison:
        model = _name(row["model"])
        if row["model"] == champion:
            model = f"**{model} (validation-selected champion)**"
        lines.append(
            f"| {int(float(row['rank']))} | {model} | {int(float(row['n']))} "
            f"| {_f(row['mae'])} | {_f(row['rmse'])} | {_f(row['r2'], 3)} "
            f"| {_signed(row['bias'])} | {_f(row['alpha_lambda'], 3)} |"
        )
    test_best = comparison[0]
    lines += [
        "",
        "![Model comparison](figures/results/model_comparison.png)",
        "",
        (
            f"**{champion_name} is the frozen validation-selected champion, while "
            f"{_name(test_best['model'])} ranks first by test RMSE.** This disagreement is"
        ),
        "exactly why the one-cell test partition is not used for model selection and why",
        "the nested cross-validated procedure in §2 is the repository's headline estimate.",
        "",
        "### 8.3 Like-for-like",
        "",
        "Sequence models cannot score a cell's first 19 cycles, and those early rows are",
        "often the hardest. The table below restricts every model to the rows all models can",
        "score, so input coverage cannot silently change the ranking.",
        "",
        "| Rank | Model | n | MAE | RMSE | R² | Bias | α-λ (20 %) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in common:
        model = _name(row["model"])
        if row["model"] == champion:
            model = f"**{model} (validation-selected champion)**"
        lines.append(
            f"| {int(float(row['rank']))} | {model} | {int(float(row['n']))} "
            f"| {_f(row['mae'])} | {_f(row['rmse'])} | {_f(row['r2'], 3)} "
            f"| {_signed(row['bias'])} | {_f(row['alpha_lambda'], 3)} |"
        )
    best_common = common[0]
    champion_common = next(row for row in common if row["model"] == champion)
    lines += [
        "",
        (
            f"On the {int(float(best_common['n']))} common rows, {_name(best_common['model'])} "
            f"has the lowest RMSE ({_f(best_common['rmse'])}); {champion_name} records "
            f"{_f(champion_common['rmse'])}. This is a diagnostic comparison, not a second"
        ),
        "selection step.",
        "",
        "_Generated from `reports/metrics.json`, `cross_validation_by_battery.csv`,",
        "`model_comparison.csv`, and `model_comparison_common_rows.csv` by",
        "`scripts/update_readme_results.py`._",
        "",
        END,
    ]
    return "\n".join(lines)


def update(*, check: bool) -> bool:
    current = README.read_text(encoding="utf-8")
    if START not in current or END not in current:
        raise ValueError(f"{README} is missing the generated-results markers")
    prefix, rest = current.split(START, 1)
    _, suffix = rest.split(END, 1)
    expected = prefix + render() + suffix
    if current == expected:
        return False
    if check:
        raise SystemExit(
            "README result tables are stale; run `python scripts/update_readme_results.py`."
        )
    README.write_text(expected, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of updating")
    args = parser.parse_args()
    changed = update(check=args.check)
    if not args.check:
        print("updated README result tables" if changed else "README result tables already current")


if __name__ == "__main__":
    main()
