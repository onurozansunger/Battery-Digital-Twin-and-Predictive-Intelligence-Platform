"""The model-promotion gate.

A candidate is compared against the current production model on the axes that
matter operationally, and every check reports its own verdict. Three outcomes:

``APPROVED``         every required check passed
``REQUIRES_REVIEW``  nothing failed outright, but something could not be checked
                     — a missing metric, no production model to compare against,
                     an unverifiable artifact
``REJECTED``         a required check failed

``REQUIRES_REVIEW`` exists because the alternative is worse in both directions:
treating "we could not measure this" as a pass promotes blind, and treating it as
a failure makes the first model of a family unpromotable.

What the gate does not do
-------------------------
Promote. It returns a decision; a human (or an explicitly configured pipeline
with ``allow_auto_promotion``) acts on it. A gate that promotes on green turns
whichever metric is easiest to move into a deploy button.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.registry.store import (
    FileModelRegistry,
    ModelStage,
    RegisteredModel,
    bundle_checksum,
)
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "GateResult",
    "PromotionDecision",
    "PromotionGate",
    "PromotionReport",
    "evaluate_promotion",
]


class PromotionDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


@dataclass
class GateResult:
    """One check's verdict."""

    name: str
    passed: bool | None
    detail: str
    candidate_value: float | None = None
    production_value: float | None = None
    threshold: float | None = None
    required: bool = True

    @property
    def status(self) -> str:
        if self.passed is None:
            return "UNKNOWN"
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "detail": self.detail,
            "candidate_value": self.candidate_value,
            "production_value": self.production_value,
            "threshold": self.threshold,
        }


@dataclass
class PromotionReport:
    """The full decision, suitable for persisting next to the registry."""

    decision: PromotionDecision
    model_name: str
    model_version: str
    generated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    compared_with: str | None = None
    results: list[GateResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "generated_at_utc": self.generated_at_utc,
            "compared_with": self.compared_with,
            "checks": [r.to_dict() for r in self.results],
            "reasons": self.reasons,
            "notes": self.notes,
        }


def _metric(payload: dict[str, Any] | None, *path: str) -> float | None:
    """Read a nested metric, returning ``None`` rather than raising."""
    cursor: Any = payload or {}
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, int | float) and np.isfinite(float(cursor)):
        return float(cursor)
    return None


@dataclass
class PromotionGate:
    """Evaluates a candidate against configured gates and production."""

    cfg: ExperimentConfig

    def evaluate(
        self,
        candidate: RegisteredModel,
        production: RegisteredModel | None,
        *,
        smoke_test_passed: bool | None = None,
        contract_tests_passed: bool | None = None,
        unit_tests_passed: bool | None = None,
        leakage_check_passed: bool | None = None,
        candidate_latency_ms: float | None = None,
        production_latency_ms: float | None = None,
    ) -> PromotionReport:
        policy = self.cfg.registry.promotion
        results: list[GateResult] = []
        notes: list[str] = []

        # -- structural checks ------------------------------------------------
        results.append(self._validation_status(candidate, policy))
        results.append(self._checksum(candidate, policy))
        results.append(self._metadata(candidate))
        results.append(self._feature_schema(candidate, production, policy))

        for name, value, required in (
            ("unit_tests_passed", unit_tests_passed, policy.require_tests_passed),
            ("contract_tests_passed", contract_tests_passed, policy.require_contract_tests),
            ("inference_smoke_test", smoke_test_passed, policy.require_inference_smoke_test),
            ("leakage_check", leakage_check_passed, policy.require_leakage_check),
        ):
            results.append(
                GateResult(
                    name=name,
                    passed=value,
                    detail=(
                        "not reported to the gate — supply the evidence explicitly"
                        if value is None
                        else ("passed" if value else "failed")
                    ),
                    required=required,
                )
            )

        # -- metric comparisons ------------------------------------------------
        if production is None:
            notes.append(
                "No PRODUCTION model exists for this family, so every comparative gate "
                "is evaluated against configured absolute floors only. A first "
                "promotion is a judgement call, not a regression test."
            )
        results.extend(self._metric_gates(candidate, production, policy))
        results.append(self._coverage(candidate, policy))
        results.append(self._latency(candidate_latency_ms, production_latency_ms, policy))

        required_failures = [r for r in results if r.required and r.passed is False]
        unknowns = [r for r in results if r.required and r.passed is None]

        if required_failures:
            decision = PromotionDecision.REJECTED
            reasons = [f"{r.name}: {r.detail}" for r in required_failures]
        elif unknowns:
            decision = PromotionDecision.REQUIRES_REVIEW
            reasons = [f"{r.name}: {r.detail}" for r in unknowns]
        else:
            decision = PromotionDecision.APPROVED
            reasons = ["Every required gate passed."]

        if decision is PromotionDecision.APPROVED and not policy.allow_auto_promotion:
            notes.append(
                "APPROVED is a recommendation. registry.promotion.allow_auto_promotion "
                "is false, so promotion remains an explicit human action."
            )

        return PromotionReport(
            decision=decision,
            model_name=candidate.model_name,
            model_version=candidate.model_version,
            compared_with=production.key if production else None,
            results=results,
            reasons=reasons,
            notes=notes,
        )

    # -- individual gates ---------------------------------------------------
    def _validation_status(self, candidate: RegisteredModel, policy: Any) -> GateResult:
        ok = candidate.validation_status.upper() in ("VALIDATED", "PASSED")
        return GateResult(
            name="validation_status",
            passed=ok,
            detail=f"validation_status={candidate.validation_status}",
            required=policy.require_validation_status,
        )

    def _checksum(self, candidate: RegisteredModel, policy: Any) -> GateResult:
        from battery_rul.registry.store import _absolute_path

        directory = _absolute_path(candidate.bundle_path, self.cfg)
        if not directory.is_dir():
            return GateResult(
                name="artifact_checksum",
                passed=False,
                detail=f"bundle directory is missing: {candidate.bundle_path}",
                required=policy.require_artifact_checksum,
            )
        actual = bundle_checksum(directory)
        ok = actual == candidate.artifact_checksum
        return GateResult(
            name="artifact_checksum",
            passed=ok,
            detail=(
                "matches the registered checksum"
                if ok
                else f"registered {candidate.artifact_checksum[:12]}…, on disk {actual[:12]}…"
            ),
            required=policy.require_artifact_checksum,
        )

    def _metadata(self, candidate: RegisteredModel) -> GateResult:
        missing = [
            name
            for name in ("dataset_fingerprint", "feature_schema_fingerprint", "task")
            if not getattr(candidate, name)
        ]
        return GateResult(
            name="required_metadata",
            passed=not missing,
            detail="complete" if not missing else f"missing: {missing}",
        )

    def _feature_schema(
        self, candidate: RegisteredModel, production: RegisteredModel | None, policy: Any
    ) -> GateResult:
        if production is None:
            return GateResult(
                name="feature_schema_compatible",
                passed=bool(candidate.feature_schema_fingerprint),
                detail=(
                    "no production model to compare against; the candidate's own schema "
                    "fingerprint is present"
                    if candidate.feature_schema_fingerprint
                    else "the candidate has no feature-schema fingerprint"
                ),
                required=policy.require_feature_schema_compatible,
            )
        same = candidate.feature_schema_fingerprint == production.feature_schema_fingerprint
        return GateResult(
            name="feature_schema_compatible",
            passed=True,
            detail=(
                "identical to production"
                if same
                else (
                    f"differs from production ({candidate.n_features} vs "
                    f"{production.n_features} features). A changed feature schema is "
                    "allowed but every consumer pinned to the old one must be checked."
                )
            ),
            required=policy.require_feature_schema_compatible,
        )

    def _metric_gates(
        self, candidate: RegisteredModel, production: RegisteredModel | None, policy: Any
    ) -> list[GateResult]:
        out: list[GateResult] = []

        # RUL MAE — lower is better, relative regression tolerated.
        candidate_mae = _metric(candidate.metrics, "out_of_fold", "mae")
        production_mae = _metric(production.metrics, "out_of_fold", "mae") if production else None
        if candidate_mae is None:
            out.append(
                GateResult(
                    name="rul_mae",
                    passed=None,
                    detail="the candidate reports no out-of-fold RUL MAE",
                )
            )
        elif production_mae is None:
            floor = policy.max_absolute_rul_mae
            out.append(
                GateResult(
                    name="rul_mae",
                    passed=None if floor is None else candidate_mae <= floor,
                    detail=(
                        f"MAE {candidate_mae:.3f}; no production baseline"
                        + ("" if floor is None else f", absolute floor {floor}")
                    ),
                    candidate_value=candidate_mae,
                    threshold=floor,
                )
            )
        else:
            limit = production_mae * (1.0 + policy.max_rul_mae_regression)
            out.append(
                GateResult(
                    name="rul_mae",
                    passed=candidate_mae <= limit,
                    detail=(
                        f"MAE {candidate_mae:.3f} vs production {production_mae:.3f}; "
                        f"tolerated up to {limit:.3f} "
                        f"(+{policy.max_rul_mae_regression:.0%})"
                    ),
                    candidate_value=candidate_mae,
                    production_value=production_mae,
                    threshold=limit,
                )
            )

        # SOH MAE
        candidate_soh = _metric(candidate.metrics, "out_of_fold", "soh_mae") or _metric(
            candidate.metrics, "soh", "mae"
        )
        production_soh = (
            (
                _metric(production.metrics, "out_of_fold", "soh_mae")
                or _metric(production.metrics, "soh", "mae")
            )
            if production
            else None
        )
        if candidate_soh is not None and production_soh is not None:
            limit = production_soh * (1.0 + policy.max_soh_mae_regression)
            out.append(
                GateResult(
                    name="soh_mae",
                    passed=candidate_soh <= limit,
                    detail=f"SOH MAE {candidate_soh:.4f} vs production {production_soh:.4f}",
                    candidate_value=candidate_soh,
                    production_value=production_soh,
                    threshold=limit,
                    required=False,
                )
            )

        # Risk: PR-AUC (higher is better) and Brier (lower is better).
        candidate_pr = _metric(candidate.calibration_metrics, "pr_auc")
        production_pr = _metric(production.calibration_metrics, "pr_auc") if production else None
        if candidate_pr is not None and production_pr is not None:
            floor = production_pr - policy.max_pr_auc_regression
            out.append(
                GateResult(
                    name="risk_pr_auc",
                    passed=candidate_pr >= floor,
                    detail=(
                        f"PR-AUC {candidate_pr:.3f} vs production {production_pr:.3f}; "
                        f"floor {floor:.3f}"
                    ),
                    candidate_value=candidate_pr,
                    production_value=production_pr,
                    threshold=floor,
                    required=False,
                )
            )
        candidate_brier = _metric(candidate.calibration_metrics, "brier_score")
        production_brier = (
            _metric(production.calibration_metrics, "brier_score") if production else None
        )
        if candidate_brier is not None and production_brier is not None:
            limit = production_brier * (1.0 + policy.max_brier_regression)
            out.append(
                GateResult(
                    name="risk_brier",
                    passed=candidate_brier <= limit,
                    detail=(f"Brier {candidate_brier:.4f} vs production {production_brier:.4f}"),
                    candidate_value=candidate_brier,
                    production_value=production_brier,
                    threshold=limit,
                    required=False,
                )
            )
        return out

    def _coverage(self, candidate: RegisteredModel, policy: Any) -> GateResult:
        coverage = _metric(candidate.uncertainty_metrics, "empirical_coverage") or _metric(
            candidate.metrics, "out_of_fold_coverage", "empirical_coverage"
        )
        if coverage is None:
            return GateResult(
                name="interval_coverage",
                passed=None,
                detail="the candidate reports no empirical interval coverage",
                threshold=policy.min_interval_coverage,
            )
        return GateResult(
            name="interval_coverage",
            passed=coverage >= policy.min_interval_coverage,
            detail=(
                f"empirical coverage {coverage:.3f}, minimum " f"{policy.min_interval_coverage:.3f}"
            ),
            candidate_value=coverage,
            threshold=policy.min_interval_coverage,
        )

    def _latency(
        self, candidate_ms: float | None, production_ms: float | None, policy: Any
    ) -> GateResult:
        if candidate_ms is None:
            return GateResult(
                name="inference_latency",
                passed=None,
                detail="no candidate latency measurement was supplied",
                required=False,
            )
        if production_ms is None or production_ms <= 0:
            return GateResult(
                name="inference_latency",
                passed=True,
                detail=f"{candidate_ms:.1f} ms; no production baseline to compare against",
                candidate_value=candidate_ms,
                required=False,
            )
        limit = production_ms * policy.max_latency_regression_ratio
        return GateResult(
            name="inference_latency",
            passed=candidate_ms <= limit,
            detail=(
                f"{candidate_ms:.1f} ms vs production {production_ms:.1f} ms; limit "
                f"{limit:.1f} ms ({policy.max_latency_regression_ratio}x)"
            ),
            candidate_value=candidate_ms,
            production_value=production_ms,
            threshold=limit,
            required=False,
        )


def evaluate_promotion(
    cfg: ExperimentConfig,
    model_name: str,
    model_version: str,
    **evidence: Any,
) -> PromotionReport:
    """Look the candidate up in the registry and evaluate it against production."""
    registry = FileModelRegistry(cfg=cfg)
    candidate = registry.get(model_name, model_version)
    if candidate is None:
        raise ValueError(f"{model_name}:{model_version} is not registered.")
    production = next(
        (
            e
            for e in registry.list_models(model_name=model_name, stage=ModelStage.PRODUCTION)
            if e.model_version != model_version
        ),
        None,
    )
    return PromotionGate(cfg=cfg).evaluate(candidate, production, **evidence)
