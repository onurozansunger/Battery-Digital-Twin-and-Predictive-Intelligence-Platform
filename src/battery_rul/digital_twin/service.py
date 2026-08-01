"""The digital-twin service — the one place inference happens.

Everything user-facing goes through here: the FastAPI application, the Streamlit
dashboard, the example-snapshot script. None of them loads a model file, builds a
feature, or applies a threshold on its own. That is not tidiness for its own
sake — duplicated inference logic is how a dashboard ends up disagreeing with the
API it is supposed to be a view of, and how a serving path quietly drifts away
from the training path.

Pipeline
--------
1. validate the supplied history (schema, ordering, duplicates, size)
2. derive health with the *training* derivation (trailing smoothing, SOH)
3. generate features with the *training* generator (stateless, causal)
4. assess data quality — and stop here if it is INSUFFICIENT
5. transform with the *fitted* preprocessing artifact from the bundle
6. check the warm-up policy: is this cycle scoreable at all?
7. predict RUL / SOH / risk
8. calibrate the risk probability
9. attach a conformal prediction interval to RUL
10. attribute the prediction to features
11. apply the deterministic recommendation rules
12. assemble the snapshot, tagging every value's provenance

Steps 2, 3 and 5 are literally the training functions, not re-implementations.
Step 6 is the training warm-up policy read from
:mod:`battery_rul.features.warmup`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.calibration.probability import ProbabilityCalibrator
from battery_rul.config import ExperimentConfig
from battery_rul.data.loader import derive_health
from battery_rul.data.schema import REQUIRED_COLUMNS, coerce_schema
from battery_rul.digital_twin.domain import (
    SNAPSHOT_SCHEMA_VERSION,
    BatteryExplanation,
    BatteryHealthState,
    BatteryIdentity,
    BatteryMeasurements,
    BatteryPrediction,
    BatteryRecommendation,
    BatteryRiskAssessment,
    BatteryTwinSnapshot,
    BatteryUncertainty,
    DataQualityAssessment,
    Provenance,
    TwinMetadata,
)
from battery_rul.digital_twin.quality import assess_data_quality
from battery_rul.explainability.drivers import DriverExplainer
from battery_rul.features.engineering import build_features
from battery_rul.features.warmup import WarmupPolicy
from battery_rul.models.base import TrainingData
from battery_rul.models.bundle import ArtifactCompatibilityError, ModelBundle, load_bundle
from battery_rul.recommendations.engine import RecommendationEngine, RecommendationInputs
from battery_rul.targets.risk import classify_risk
from battery_rul.targets.soh import classify_soh
from battery_rul.uncertainty.conformal import ConformalIntervalEstimator
from battery_rul.utils.io import environment_fingerprint
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "BatteryDigitalTwinService",
    "InsufficientDataError",
    "InvalidHistoryError",
    "ModelsUnavailableError",
]


class InvalidHistoryError(ValueError):
    """The supplied history cannot be interpreted as a cell's cycle record."""


class InsufficientDataError(ValueError):
    """The history is well-formed but too thin to support a prediction."""


class ModelsUnavailableError(RuntimeError):
    """No usable model bundle is present; the service is not ready to serve."""


@dataclass
class _Bundles:
    """The loaded artifacts, or an explanation of why each is missing."""

    rul: ModelBundle | None = None
    soh: ModelBundle | None = None
    risk: ModelBundle | None = None
    multitask: Any | None = None
    multitask_preprocessing: Any | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def any_available(self) -> bool:
        return any((self.rul, self.soh, self.risk, self.multitask))


@dataclass
class BatteryDigitalTwinService:
    """Orchestrates one cell's history into a :class:`BatteryTwinSnapshot`.

    Constructed with an :class:`ExperimentConfig`; bundles are loaded eagerly at
    construction so a broken artifact surfaces at startup rather than on the
    first request. Nothing is loaded at *import* time — an import must never
    touch the filesystem or a GPU.
    """

    cfg: ExperimentConfig
    bundles: _Bundles = field(default_factory=_Bundles)
    recommender: RecommendationEngine | None = None
    _loaded: bool = False

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, cfg: ExperimentConfig, *, strict: bool = False) -> BatteryDigitalTwinService:
        """Build a service and load its artifacts.

        ``strict`` raises when no bundle can be loaded. The API's readiness probe
        uses it; the dashboard's demo mode does not, so it can still render a
        data view with a clear "models unavailable" banner rather than a stack
        trace.
        """
        service = cls(cfg=cfg, recommender=RecommendationEngine(cfg=cfg))
        service.load_artifacts()
        if strict and not service.bundles.any_available:
            raise ModelsUnavailableError(
                "No model bundle could be loaded. Run "
                "`python -m battery_rul.pipelines.run_milestone_2` first. Errors: "
                f"{service.bundles.errors}"
            )
        return service

    def load_artifacts(self) -> None:
        """Load every configured bundle, recording rather than raising failures."""
        artifacts = self.cfg.artifacts
        for name, path in (
            ("rul", artifacts.rul_dir),
            ("soh", artifacts.soh_dir),
            ("risk", artifacts.risk_dir),
        ):
            try:
                setattr(self.bundles, name, load_bundle(path, self.cfg))
            except (FileNotFoundError, ArtifactCompatibilityError) as exc:
                self.bundles.errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("Bundle %s unavailable: %s", name, exc)

        multitask_path = Path(artifacts.multitask_dir) / "model.pkl"
        if multitask_path.is_file():
            try:
                from battery_rul.models.multitask import MultiTaskSequenceModel

                self.bundles.multitask = MultiTaskSequenceModel.load(multitask_path, self.cfg)
                preprocessing = Path(artifacts.multitask_dir) / "preprocessing.pkl"
                if preprocessing.is_file():
                    from battery_rul.features.pipeline import FeaturePipeline

                    self.bundles.multitask_preprocessing = FeaturePipeline.load(preprocessing)
            except Exception as exc:  # noqa: BLE001 - record, do not crash the service
                self.bundles.errors["multitask"] = f"{type(exc).__name__}: {exc}"
                logger.warning("Multi-task model unavailable: %s", exc)
        else:
            self.bundles.errors["multitask"] = f"not found: {multitask_path}"

        if self.recommender is None:
            self.recommender = RecommendationEngine(cfg=self.cfg)
        self._loaded = True

    # -- health / metadata ---------------------------------------------------
    def health_check(self) -> dict[str, Any]:
        """Liveness: the process is up and the service object is constructed."""
        return {"status": "ok", "service": "battery-digital-twin", "loaded": self._loaded}

    def readiness(self) -> dict[str, Any]:
        """Readiness: can this process actually answer a prediction request?"""
        available = {
            "rul": self.bundles.rul is not None,
            "soh": self.bundles.soh is not None,
            "risk": self.bundles.risk is not None,
            "multitask": self.bundles.multitask is not None,
        }
        ready = self.bundles.rul is not None or self.bundles.multitask is not None
        return {
            "ready": ready,
            "bundles": available,
            "errors": self.bundles.errors,
            "reason": (
                None
                if ready
                else "No RUL-capable artifact is loaded; prediction endpoints will fail."
            ),
        }

    def get_model_metadata(self) -> dict[str, Any]:
        """Everything a consumer needs to interpret this service's outputs."""
        out: dict[str, Any] = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "environment": environment_fingerprint(),
            "end_of_life_definition": {
                "threshold_fraction_of_reference": self.cfg.data.eol_threshold,
                "reference": self.cfg.data.eol_reference,
                "persistence_cycles": self.cfg.target.eol_persistence,
                "capacity_ah": self.cfg.eol_capacity_ah,
            },
            "soh_definition": {
                "representation": "fraction in [0, 1]",
                "reference_strategy": self.cfg.soh.reference_strategy,
                "reference_cycles": self.cfg.soh.reference_cycles,
                "bands": {
                    "healthy_min": self.cfg.soh.healthy_min,
                    "slightly_degraded_min": self.cfg.soh.slightly_degraded_min,
                    "warning_min": self.cfg.soh.warning_min,
                },
            },
            "risk_definition": {
                "horizon_cycles": self.cfg.risk.horizon_cycles,
                "is_observed_safety_failure": False,
            },
            "warmup_policy": self._policy().to_dict(),
            "bundles": {},
            "errors": self.bundles.errors,
        }
        for name in ("rul", "soh", "risk"):
            bundle = getattr(self.bundles, name)
            if bundle is not None:
                out["bundles"][name] = bundle.describe()
        if self.bundles.multitask is not None:
            out["bundles"]["multitask"] = self.bundles.multitask.fit_metadata
        return out

    def _policy(self) -> WarmupPolicy:
        if self.bundles.multitask is not None:
            return WarmupPolicy(
                self.cfg.features.drop_warmup_cycles, self.bundles.multitask.mt.window
            )
        return WarmupPolicy(self.cfg.features.drop_warmup_cycles, None)

    # -- input handling -------------------------------------------------------
    def _prepare(self, battery_id: str, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate, order and feature-engineer a supplied history.

        Returns ``(health_frame, feature_frame)``. ``feature_frame`` may be empty
        when the history is shorter than the warm-up trim; the caller reports
        that as an unscoreable cell rather than treating it as an error.
        """
        if history is None or len(history) == 0:
            raise InvalidHistoryError("History is empty; at least one cycle is required.")
        if len(history) > self.cfg.service.max_history_cycles:
            raise InvalidHistoryError(
                f"History has {len(history)} rows; the configured maximum is "
                f"{self.cfg.service.max_history_cycles}."
            )

        frame = history.copy()
        frame["battery_id"] = battery_id
        if "dataset" not in frame.columns:
            frame["dataset"] = self.cfg.data.source

        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise InvalidHistoryError(
                f"History is missing required column(s): {missing}. The canonical "
                "cycle schema is documented in battery_rul.data.schema."
            )

        if frame["cycle_index"].isna().any():
            raise InvalidHistoryError("cycle_index contains missing values.")
        if (pd.to_numeric(frame["cycle_index"], errors="coerce") < 0).any():
            raise InvalidHistoryError("cycle_index must be non-negative.")

        duplicated = frame.duplicated(subset=["cycle_index"], keep=False)
        if duplicated.any():
            raise InvalidHistoryError(
                f"{int(duplicated.sum())} row(s) share a cycle_index. A cycle record "
                "must have one row per cycle; deduplicate before submitting."
            )

        frame = coerce_schema(frame)
        frame = frame.sort_values("cycle_index", kind="stable").reset_index(drop=True)
        health = derive_health(frame, self.cfg)

        try:
            features, _ = build_features(health, self.cfg.features)
        except ValueError as exc:
            logger.info("Feature generation produced no usable rows for %s: %s", battery_id, exc)
            features = health.iloc[0:0].copy()
        return health, features

    # -- public prediction API -------------------------------------------------
    def predict_rul(self, battery_id: str, history: pd.DataFrame) -> BatteryPrediction:
        """RUL for the latest cycle, with a conformal interval when available."""
        snapshot = self.create_snapshot(battery_id, history, explain=False)
        return snapshot.prediction

    def predict_soh(self, battery_id: str, history: pd.DataFrame) -> BatteryHealthState:
        """State of health for the latest cycle."""
        snapshot = self.create_snapshot(battery_id, history, explain=False)
        return snapshot.health

    def predict_failure_risk(self, battery_id: str, history: pd.DataFrame) -> BatteryRiskAssessment:
        """Calibrated probability of reaching end of life within the horizon."""
        snapshot = self.create_snapshot(battery_id, history, explain=False)
        return snapshot.failure_risk

    def explain_prediction(self, battery_id: str, history: pd.DataFrame) -> BatteryExplanation:
        """Local feature attributions for the latest cycle."""
        snapshot = self.create_snapshot(battery_id, history, explain=True)
        return snapshot.explanation or BatteryExplanation(method="unavailable", drivers=[])

    # -- the snapshot -----------------------------------------------------------
    def create_snapshot(
        self,
        battery_id: str,
        history: pd.DataFrame,
        *,
        explain: bool = True,
    ) -> BatteryTwinSnapshot:
        """Full digital-twin view of ``battery_id`` at its latest supplied cycle."""
        health, features = self._prepare(battery_id, history)
        warnings: list[str] = []

        policy = self._policy()
        latest_cycle = int(health["cycle_index"].iloc[-1])
        required = self._required_features()

        quality = assess_data_quality(
            health,
            self.cfg,
            required_features=required,
            feature_frame=features if len(features) else None,
            reference_ranges=self._reference_ranges(),
            min_history_cycles=policy.min_history_cycles,
        )

        identity = BatteryIdentity(
            battery_id=battery_id,
            dataset=str(health["dataset"].iloc[-1]) if "dataset" in health else None,
            chemistry="LCO 18650 (NASA cohort)" if self.cfg.data.source == "nasa" else None,
            nominal_capacity_ah=self.cfg.data.nominal_capacity_ah,
        )
        measurements = self._measurements(health)

        scoreable = (
            len(features) > 0
            and latest_cycle >= policy.first_scoreable_cycle
            and quality.quality_class != "INSUFFICIENT"
        )
        unscoreable_reason: str | None = None
        if quality.quality_class == "INSUFFICIENT":
            unscoreable_reason = "Data quality is INSUFFICIENT; no prediction was produced."
        elif latest_cycle < policy.first_scoreable_cycle:
            unscoreable_reason = (
                f"Cycle {latest_cycle} is before the first scoreable cycle "
                f"({policy.first_scoreable_cycle}) under the training warm-up policy: "
                f"{policy.drop_warmup_cycles} warm-up cycle(s)"
                + (f" plus a {policy.sequence_window}-cycle window." if policy.is_sequence else ".")
            )
        elif len(features) == 0:
            unscoreable_reason = "Feature generation produced no rows for this history."

        if unscoreable_reason:
            warnings.append(unscoreable_reason)

        # --- measured SOH is always available; it is derived, not predicted ----
        soh_measured = self._measured_soh(health)

        prediction = BatteryPrediction(
            rul_cycles=None,
            is_scoreable=bool(scoreable),
            unscoreable_reason=unscoreable_reason,
        )
        risk = BatteryRiskAssessment(horizon_cycles=self.cfg.risk.horizon_cycles)
        health_state = BatteryHealthState(
            soh=soh_measured,
            soh_measured=soh_measured,
            health_class=classify_soh(soh_measured, self.cfg.soh).value,  # type: ignore[arg-type]
            capacity_fade_percent=self._fade_percent(health),
            provenance=Provenance.DERIVED,
        )
        explanation: BatteryExplanation | None = None

        if scoreable:
            outputs = self._infer(features)
            warnings.extend(outputs["warnings"])
            prediction, risk, health_state, explanation = self._assemble_outputs(
                outputs,
                features=features,
                soh_measured=soh_measured,
                explain=explain,
            )
            prediction.is_scoreable = True

        recommendation = self._recommend(prediction, risk, health_state, quality, health, scoreable)
        metadata = self._metadata(policy, health)

        if not self.bundles.any_available:
            warnings.append(
                "No model bundle is loaded; only measured and derived quantities are "
                "reported. Predicted fields are null."
            )

        return BatteryTwinSnapshot(
            battery_id=battery_id,
            identity=identity,
            measurement_summary=measurements,
            health=health_state,
            prediction=prediction,
            failure_risk=risk,
            explanation=explanation,
            recommendation=recommendation,
            data_quality=quality,
            metadata=metadata,
            warnings=warnings,
        )

    # -- inference ---------------------------------------------------------------
    def _infer(self, features: pd.DataFrame) -> dict[str, Any]:
        """Run whichever artifacts are available on the final row."""
        out: dict[str, Any] = {"warnings": []}
        last = len(features) - 1

        if self.bundles.rul is not None:
            bundle = self.bundles.rul
            data = self._training_data(features, bundle)
            values = np.asarray(bundle.model.predict(data), dtype=float)
            raw = float(values[last]) if np.isfinite(values[last]) else None
            out["rul_raw"] = raw
            # Physical post-processing, recorded as distinct from the raw output.
            out["rul"] = None if raw is None else float(max(raw, 0.0))
            out["rul_model"] = bundle.metadata.model_name
            out["rul_bundle"] = bundle
            out["rul_scaled"] = data.X
            if raw is None:
                out["warnings"].append(
                    "The RUL model could not score the latest cycle (insufficient "
                    "window); the reported interval and recommendation reflect that."
                )

        if self.bundles.soh is not None:
            bundle = self.bundles.soh
            data = self._training_data(features, bundle)
            values = np.asarray(bundle.model.predict(data), dtype=float)
            raw = float(values[last]) if np.isfinite(values[last]) else None
            out["soh_pred"] = (
                None
                if raw is None
                else float(np.clip(raw, self.cfg.soh.plausible_min, self.cfg.soh.plausible_max))
            )
            out["soh_model"] = bundle.metadata.model_name

        if self.bundles.risk is not None:
            bundle = self.bundles.risk
            data = self._training_data(features, bundle)
            estimator = getattr(bundle.model, "estimator", bundle.model)
            if hasattr(estimator, "predict_proba"):
                probabilities = np.asarray(estimator.predict_proba(data.X), dtype=float)[:, 1]
            else:
                probabilities = np.asarray(bundle.model.predict(data), dtype=float)
            raw = float(np.clip(probabilities[last], 0.0, 1.0))
            out["risk_raw"] = raw
            out["risk_model"] = bundle.metadata.model_name
            out["risk_bundle"] = bundle
            out["risk_scaled"] = data.X

            calibrator = bundle.calibrator
            if isinstance(calibrator, ProbabilityCalibrator) and calibrator.fitted:
                out["risk"] = float(calibrator.transform(np.array([raw]))[0])
                out["risk_calibrated"] = True
                out["risk_calibration_method"] = calibrator.method
                out["risk_threshold"] = float(
                    self.cfg.risk.threshold
                    if self.cfg.risk.threshold is not None
                    else calibrator.threshold
                )
            else:
                out["risk"] = raw
                out["risk_calibrated"] = False
                out["risk_threshold"] = self.cfg.risk.threshold
                out["warnings"].append(
                    "The risk probability is uncalibrated: no calibration artifact was "
                    "found in the risk bundle. Treat it as a ranking score, not a "
                    "probability."
                )

        if self.bundles.multitask is not None and self.bundles.multitask_preprocessing is not None:
            pipeline = self.bundles.multitask_preprocessing
            values = pipeline.transform(features[pipeline.feature_names])
            multitask = self.bundles.multitask.predict(features, values)
            if bool(multitask.scoreable[last]):
                out["multitask"] = {
                    "rul": float(multitask.rul[last]),
                    "soh": float(multitask.soh[last]),
                    "risk": float(multitask.risk_probability[last]),
                }
                out.setdefault("rul", out["multitask"]["rul"])
                out.setdefault("rul_model", "multitask")
                out.setdefault("soh_pred", out["multitask"]["soh"])
                out.setdefault("soh_model", "multitask")
                out.setdefault("risk", out["multitask"]["risk"])
                out.setdefault("risk_raw", out["multitask"]["risk"])
                out.setdefault("risk_model", "multitask")
                out.setdefault("risk_calibrated", False)
            else:
                out["warnings"].append(
                    "The multi-task model could not score the latest cycle: the "
                    "history is shorter than its training window."
                )
        return out

    def _assemble_outputs(
        self,
        outputs: dict[str, Any],
        *,
        features: pd.DataFrame,
        soh_measured: float | None,
        explain: bool,
    ) -> tuple[
        BatteryPrediction, BatteryRiskAssessment, BatteryHealthState, BatteryExplanation | None
    ]:
        last = len(features) - 1
        # The conformal stage variable is measured state of health. Life fraction
        # would need the end-of-life cycle — a label — so a served cell could
        # never supply it and the conditioning would silently fall back to one
        # global quantile precisely where the interval matters most.
        life_fraction = soh_measured

        # --- RUL + interval ---------------------------------------------------
        rul_value = outputs.get("rul")
        interval: BatteryUncertainty | None = None
        bundle = outputs.get("rul_bundle")
        estimator = getattr(bundle, "uncertainty", None) if bundle is not None else None
        if (
            rul_value is not None
            and isinstance(estimator, ConformalIntervalEstimator)
            and estimator.fitted
        ):
            conformal = estimator.interval(rul_value, life_fraction=life_fraction)
            interval = BatteryUncertainty(
                point_estimate=conformal.point_estimate,
                lower_bound=conformal.lower_bound,
                upper_bound=conformal.upper_bound,
                interval_coverage_target=conformal.interval_coverage_target,
                uncertainty_method=conformal.uncertainty_method,
                calibration_sample_size=conformal.calibration_sample_size,
                life_stage=conformal.life_stage,
                note=conformal.note,
            )
            rul_value = conformal.point_estimate

        prediction = BatteryPrediction(
            rul_cycles=None if rul_value is None else round(float(rul_value), 3),
            rul_interval=interval,
            model_name=outputs.get("rul_model"),
            is_scoreable=rul_value is not None,
            unscoreable_reason=(
                None
                if rul_value is not None
                else "No RUL artifact produced a value for the latest cycle."
            ),
        )

        # --- risk --------------------------------------------------------------
        probability = outputs.get("risk")
        threshold = outputs.get("risk_threshold")
        risk = BatteryRiskAssessment(
            horizon_cycles=self.cfg.risk.horizon_cycles,
            probability=None if probability is None else round(float(probability), 5),
            probability_raw=(
                None if outputs.get("risk_raw") is None else round(float(outputs["risk_raw"]), 5)
            ),
            is_calibrated=bool(outputs.get("risk_calibrated", False)),
            calibration_method=outputs.get("risk_calibration_method"),
            decision_threshold=None if threshold is None else float(threshold),
            exceeds_threshold=(
                None
                if (probability is None or threshold is None)
                else bool(probability >= threshold)
            ),
            risk_class=classify_risk(probability, self.cfg.risk).value,  # type: ignore[arg-type]
        )

        # --- SOH ---------------------------------------------------------------
        soh_predicted = outputs.get("soh_pred")
        chosen = soh_predicted if soh_predicted is not None else soh_measured
        health_state = BatteryHealthState(
            soh=None if chosen is None else round(float(chosen), 5),
            soh_measured=None if soh_measured is None else round(float(soh_measured), 5),
            health_class=classify_soh(chosen, self.cfg.soh).value,  # type: ignore[arg-type]
            capacity_fade_percent=None,
            provenance=Provenance.PREDICTED if soh_predicted is not None else Provenance.DERIVED,
        )

        # --- explanation --------------------------------------------------------
        explanation: BatteryExplanation | None = None
        if explain:
            explanation = self._explain(outputs, features, last)
        return prediction, risk, health_state, explanation

    def _explain(
        self, outputs: dict[str, Any], features: pd.DataFrame, last: int
    ) -> BatteryExplanation | None:
        bundle = outputs.get("risk_bundle") or outputs.get("rul_bundle")
        if bundle is None:
            return None
        task = "risk" if outputs.get("risk_bundle") is not None else "rul"
        scaled = outputs.get("risk_scaled" if task == "risk" else "rul_scaled")
        if scaled is None:
            return None

        explainer = DriverExplainer(
            feature_names=list(bundle.preprocessing.feature_names),
            reference_values=dict(bundle.preprocessing.fallback_values),
            top_k=min(self.cfg.explainability.top_k_features, 8),
        )
        estimator = getattr(bundle.model, "estimator", bundle.model)

        def _predict(matrix: np.ndarray) -> np.ndarray:
            if task == "risk" and hasattr(estimator, "predict_proba"):
                return np.asarray(estimator.predict_proba(matrix), dtype=float)[:, 1]
            frame = features.iloc[[last] * len(matrix)].reset_index(drop=True)
            data = TrainingData(
                X=matrix,
                y=np.zeros(len(matrix)),
                frame=frame,
                feature_names=list(bundle.preprocessing.feature_names),
            )
            return np.asarray(bundle.model.predict(data), dtype=float)

        raw_values = {
            name: float(features.iloc[last][name])
            for name in bundle.preprocessing.feature_names
            if name in features.columns and np.isfinite(features.iloc[last][name])
        }
        try:
            return explainer.explain(
                predict_fn=_predict,
                scaled_row=scaled[last],
                raw_values=raw_values,
                task=task,
                model=bundle.model,
            )
        except Exception as exc:  # noqa: BLE001 - explanation must never sink a prediction
            logger.warning("Explanation failed: %s", exc)
            return BatteryExplanation(method="unavailable", drivers=[])

    # -- assembly helpers ------------------------------------------------------
    def _training_data(self, features: pd.DataFrame, bundle: ModelBundle) -> TrainingData:
        """Project a feature frame through a bundle's fitted preprocessing."""
        return TrainingData(
            X=bundle.preprocessing.transform(features),
            y=np.zeros(len(features)),
            frame=features,
            feature_names=list(bundle.preprocessing.feature_names),
        )

    def _required_features(self) -> list[str]:
        for bundle in (self.bundles.rul, self.bundles.soh, self.bundles.risk):
            if bundle is not None:
                return list(bundle.preprocessing.feature_names)
        if self.bundles.multitask_preprocessing is not None:
            return list(self.bundles.multitask_preprocessing.feature_names)
        return []

    def _reference_ranges(self) -> dict[str, tuple[float, float]] | None:
        """Training-time plausible ranges, when the bundle recorded them."""
        for bundle in (self.bundles.rul, self.bundles.risk, self.bundles.soh):
            if bundle is None:
                continue
            ranges = bundle.metadata.thresholds.get("feature_reference_ranges")
            if isinstance(ranges, dict) and ranges:
                return {k: (float(v[0]), float(v[1])) for k, v in ranges.items()}
        return None

    def _measurements(self, health: pd.DataFrame) -> BatteryMeasurements:
        last = health.iloc[-1]

        def _value(column: str) -> float | None:
            if column not in health.columns:
                return None
            value = last[column]
            return float(value) if pd.notna(value) and np.isfinite(value) else None

        available = [
            c
            for c in self.cfg.features.base_signals
            if c in health.columns and health[c].notna().any()
        ]
        missing = [c for c in self.cfg.features.base_signals if c not in available]
        return BatteryMeasurements(
            latest_cycle=int(last["cycle_index"]),
            n_cycles_supplied=int(len(health)),
            first_cycle=int(health["cycle_index"].iloc[0]),
            measured_capacity_ah=_value("capacity_ah"),
            reference_capacity_ah=_value("reference_capacity_ah"),
            voltage_mean_v=_value("voltage_mean_v"),
            voltage_min_v=_value("voltage_min_v"),
            current_mean_a=_value("current_mean_a"),
            temperature_mean_c=_value("temperature_mean_c"),
            temperature_max_c=_value("temperature_max_c"),
            internal_resistance_ohm=_value("internal_resistance_ohm"),
            charge_duration_s=_value("charge_duration_s"),
            discharge_duration_s=_value("discharge_duration_s"),
            energy_throughput_wh=_value("energy_throughput_wh"),
            available_signals=available,
            missing_signals=missing,
        )

    def _measured_soh(self, health: pd.DataFrame) -> float | None:
        """SOH from the latest measured capacity, under the configured reference."""
        from battery_rul.targets.soh import reference_capacity

        try:
            reference = reference_capacity(health, self.cfg)
        except ValueError:
            return None
        column = "capacity_smooth_ah" if "capacity_smooth_ah" in health.columns else "capacity_ah"
        value = health[column].iloc[-1]
        if pd.isna(value) or reference <= 0:
            return None
        return float(
            np.clip(
                float(value) / reference, self.cfg.soh.plausible_min, self.cfg.soh.plausible_max
            )
        )

    def _fade_percent(self, health: pd.DataFrame) -> float | None:
        column = "capacity_smooth_ah" if "capacity_smooth_ah" in health.columns else "capacity_ah"
        values = pd.to_numeric(health[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2 or values[0] <= 0:
            return None
        return round(float(100.0 * (values[0] - values[-1]) / values[0]), 3)

    def _trend_per_10(self, health: pd.DataFrame, column: str, *, relative: bool) -> float | None:
        """Trailing OLS trend per 10 cycles — deterministic, no model involved."""
        if column not in health.columns:
            return None
        window = health.tail(20)
        y = pd.to_numeric(window[column], errors="coerce").to_numpy(dtype=float)
        x = window["cycle_index"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 5:
            return None
        slope = float(np.polyfit(x[good], y[good], 1)[0])
        if not relative:
            return round(slope * 10.0, 4)
        level = float(np.nanmedian(y[good]))
        if abs(level) < 1e-12:
            return None
        return round(100.0 * slope * 10.0 / level, 4)

    def _recommend(
        self,
        prediction: BatteryPrediction,
        risk: BatteryRiskAssessment,
        health_state: BatteryHealthState,
        quality: DataQualityAssessment,
        health: pd.DataFrame,
        scoreable: bool,
    ) -> BatteryRecommendation:
        assert self.recommender is not None
        inputs = RecommendationInputs(
            rul_point=prediction.rul_cycles,
            rul_lower_bound=(
                prediction.rul_interval.lower_bound if prediction.rul_interval else None
            ),
            soh=health_state.soh,
            health_class=health_state.health_class,
            risk_probability=risk.probability,
            risk_class=risk.risk_class,
            quality_class=quality.quality_class,
            temperature_trend_c_per_10=self._trend_per_10(
                health, "temperature_max_c", relative=False
            ),
            resistance_trend_pct_per_10=self._trend_per_10(
                health, "internal_resistance_ohm", relative=True
            ),
            fade_trend_pct_per_10=(
                None
                if (trend := self._trend_per_10(health, "capacity_smooth_ah", relative=True))
                is None
                else -trend  # fade is a positive number when capacity is falling
            ),
            is_scoreable=scoreable and prediction.rul_cycles is not None,
        )
        return self.recommender.recommend(inputs)

    def _metadata(self, policy: WarmupPolicy, health: pd.DataFrame) -> TwinMetadata:
        bundle = self.bundles.rul or self.bundles.risk or self.bundles.soh
        environment = environment_fingerprint()
        return TwinMetadata(
            snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
            model_version=bundle.metadata.model_version if bundle else None,
            model_name=bundle.metadata.model_name if bundle else None,
            feature_pipeline_fingerprint=(
                bundle.metadata.preprocessing_fingerprint if bundle else None
            ),
            data_version=bundle.metadata.dataset_fingerprint if bundle else None,
            git_revision=environment.get("git_revision"),
            end_of_life_definition={
                "threshold_fraction_of_reference": self.cfg.data.eol_threshold,
                "reference": self.cfg.data.eol_reference,
                "persistence_cycles": self.cfg.target.eol_persistence,
            },
            risk_horizon_cycles=self.cfg.risk.horizon_cycles,
            uncertainty_method=(
                "split_conformal"
                if (bundle and getattr(bundle, "uncertainty", None) is not None)
                else None
            ),
            calibration_method=(
                self.bundles.risk.calibrator.method
                if (self.bundles.risk and self.bundles.risk.calibrator is not None)
                else None
            ),
            soh_definition={
                "representation": "fraction in [0, 1]",
                "reference_strategy": self.cfg.soh.reference_strategy,
                "reference_cycles": self.cfg.soh.reference_cycles,
            },
            input_cycle_range=(
                int(health["cycle_index"].iloc[0]),
                int(health["cycle_index"].iloc[-1]),
            ),
            first_scoreable_cycle=policy.first_scoreable_cycle,
            warmup_policy=policy.to_dict(),
        )
