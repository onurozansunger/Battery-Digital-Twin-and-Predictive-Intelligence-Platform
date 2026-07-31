"""Stage 4 — inference on new cycle data.

This is the serving path. It deliberately reuses the *same* engineering and
transform code as training, loading the persisted ``feature_pipeline.pkl`` rather
than rebuilding one — training/serving skew is the most common way a working
model quietly stops working.

Input contract
--------------
A canonical cycle table (see :mod:`battery_rul.data.schema`) for one or more
cells, ordered by cycle. Raw NASA ``.mat`` files can be converted with
:func:`battery_rul.data.load_cycles`. The target column is optional: when it is
present the output includes the error, when it is absent only predictions are
returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.features.engineering import build_features
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.target import inverse_transform_target
from battery_rul.models.base import BaseModel, TrainingData
from battery_rul.utils.io import read_table, write_table
from battery_rul.utils.logging import get_logger, log_section

logger = get_logger(__name__)

__all__ = ["PredictionResult", "RULPredictor", "run"]


@dataclass(slots=True)
class PredictionResult:
    """Per-cycle predictions plus a per-cell roll-up."""

    predictions: pd.DataFrame
    per_battery: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "n_rows": len(self.predictions),
            "n_batteries": int(self.predictions["battery_id"].nunique()),
            "per_battery": self.per_battery.to_dict(orient="records"),
        }


class RULPredictor:
    """Loaded champion model + feature pipeline, ready to score new cycles."""

    def __init__(self, model: BaseModel, pipeline: FeaturePipeline, cfg: ExperimentConfig) -> None:
        self.model = model
        self.pipeline = pipeline
        self.cfg = cfg

    @classmethod
    def from_artifacts(
        cls,
        cfg: ExperimentConfig,
        *,
        model_path: str | Path | None = None,
        pipeline_path: str | Path | None = None,
    ) -> RULPredictor:
        model_path = Path(model_path or cfg.paths.models_dir / "trained_model.pkl")
        pipeline_path = Path(pipeline_path or cfg.paths.models_dir / "feature_pipeline.pkl")
        model = BaseModel.load(model_path)
        pipeline = FeaturePipeline.load(pipeline_path)
        logger.info(
            "Loaded %s with %d features from %s", model.name, len(pipeline), model_path.parent
        )
        return cls(model, pipeline, cfg)

    # -- inference -------------------------------------------------------
    def predict(self, cycles: pd.DataFrame) -> PredictionResult:
        """Score a canonical cycle table.

        The warm-up trim applied during training is **not** re-applied here: at
        serving time we want a prediction for every cycle we can produce one for.
        Rows whose rolling windows are not yet populated get partially-filled
        features (forward-filled, as in training) and are flagged via
        ``is_warmup`` so a consumer can discount them.
        """
        if cycles.empty:
            raise ValueError("predict() received an empty cycle table")

        required = {"battery_id", "cycle_index"}
        missing = required - set(cycles.columns)
        if missing:
            raise KeyError(f"Input is missing required columns: {sorted(missing)}")

        # prune=False is essential, not an optimisation: pruning thresholds are
        # evaluated against whatever rows are in hand, so a serving batch would
        # drop a different set of columns than training did. Generating the full
        # (unpruned) set guarantees a superset, and the fitted pipeline picks the
        # exact columns and order it was trained on.
        serving_cfg = self.cfg.features.model_copy(update={"drop_warmup_cycles": 0})
        features, _ = build_features(cycles, serving_cfg, prune=False)

        X = self.pipeline.transform(features)
        data = TrainingData(
            X=X,
            y=np.zeros(len(features)),
            frame=features,
            feature_names=self.pipeline.feature_names,
        )
        raw = self.model.predict(data)
        rul = inverse_transform_target(raw, self.cfg)

        out = pd.DataFrame(
            {
                "battery_id": features["battery_id"].to_numpy(),
                "cycle_index": features["cycle_index"].to_numpy(),
                "predicted_rul_cycles": rul,
            }
        )
        out["is_warmup"] = out["cycle_index"] <= self.cfg.features.drop_warmup_cycles
        out["predicted_eol_cycle"] = out["cycle_index"] + out["predicted_rul_cycles"]
        for column in ("capacity_ah", "soh", self.cfg.target.name):
            if column in features.columns:
                out[column] = features[column].to_numpy()

        target = self.cfg.target.name
        if target in out.columns:
            out = out.rename(columns={target: "true_rul_cycles"})
            out["error_cycles"] = out["predicted_rul_cycles"] - out["true_rul_cycles"]
            out["abs_error_cycles"] = out["error_cycles"].abs()

        scored = out.dropna(subset=["predicted_rul_cycles"])
        logger.info(
            "Predicted RUL for %d/%d cycles across %d cell(s)",
            len(scored),
            len(out),
            out["battery_id"].nunique(),
        )
        return PredictionResult(predictions=out, per_battery=self._roll_up(out))

    @staticmethod
    def _roll_up(out: pd.DataFrame) -> pd.DataFrame:
        """Latest-cycle view per cell — the answer a fleet operator actually wants."""
        rows = []
        for battery_id, group in out.groupby("battery_id", sort=True):
            scored = group.dropna(subset=["predicted_rul_cycles"])
            if scored.empty:
                rows.append({"battery_id": battery_id, "status": "insufficient history"})
                continue
            last = scored.sort_values("cycle_index").iloc[-1]
            row = {
                "battery_id": battery_id,
                "last_cycle": int(last["cycle_index"]),
                "predicted_rul_cycles": round(float(last["predicted_rul_cycles"]), 1),
                "predicted_eol_cycle": round(float(last["predicted_eol_cycle"]), 1),
                "n_scored_cycles": int(len(scored)),
            }
            if "soh" in scored.columns and pd.notna(last.get("soh")):
                row["current_soh"] = round(float(last["soh"]), 4)
            if "true_rul_cycles" in scored.columns:
                row["true_rul_cycles"] = round(float(last["true_rul_cycles"]), 1)
                row["mae_cycles"] = round(float(scored["abs_error_cycles"].mean()), 2)
            rows.append(row)
        return pd.DataFrame(rows)


def run(
    cfg: ExperimentConfig,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    batteries: list[str] | None = None,
) -> PredictionResult:
    """CLI entry point: read cycles, score them, write predictions."""
    log_section(logger, "stage 4 — predict")

    if input_path is not None:
        cycles = read_table(input_path)
        logger.info("Read %d cycles from %s", len(cycles), input_path)
    else:
        # Default to the held-out test cells of the prepared dataset — a
        # meaningful demonstration rather than scoring the training data.
        from battery_rul.pipelines.prepare_data import load_prepared

        prepared = load_prepared(cfg)
        test_cells = prepared.split.test_batteries or sorted(
            prepared.frame.loc[prepared.split.test, "battery_id"].unique().tolist()
        )
        cycles = prepared.cycles[prepared.cycles["battery_id"].isin(test_cells)].copy()
        logger.info("No --input given; scoring held-out cells %s", test_cells)

    if batteries:
        cycles = cycles[cycles["battery_id"].isin(batteries)].copy()
        if cycles.empty:
            raise ValueError(f"No rows for requested batteries: {batteries}")

    predictor = RULPredictor.from_artifacts(cfg)
    result = predictor.predict(cycles)

    output_path = Path(output_path or cfg.paths.reports_dir / "predictions.csv")
    write_table(result.predictions, output_path)
    write_table(result.per_battery, output_path.with_name("predictions_by_battery.csv"))
    logger.info("Predictions -> %s", output_path)
    logger.info("\n%s", result.per_battery.to_string(index=False))
    return result
