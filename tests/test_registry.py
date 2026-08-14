"""Model registry, promotion gate and rollback.

Built on fixture bundles. The numbers in them are fixture values and no test
asserts a quality figure; what is asserted is the *governance*: one production
version at a time, checksums verified before a promotion, illegal transitions
refused, and a rollback that restores the version that was actually live.
"""

from __future__ import annotations

import json

import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.registry.promotion import PromotionDecision, PromotionGate
from battery_rul.registry.store import (
    FileModelRegistry,
    ModelStage,
    RegisteredModel,
    RegistryError,
    bundle_checksum,
)


@pytest.fixture
def registry_cfg(tmp_path) -> ExperimentConfig:
    return load_config(overrides={"paths.root": str(tmp_path)})


def _write_bundle(root, name: str = "fixture", *, metrics: dict | None = None):
    """A minimal but structurally valid bundle directory."""
    directory = root / "bundles" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.pkl").write_bytes(b"fixture-model")
    (directory / "preprocessing.pkl").write_bytes(b"fixture-preprocessing")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "fixture",
                "model_version": "1.0.0",
                "task": "rul_regression",
                "schema_version": "2.0",
                "data_fingerprint": "abc123",
                "preprocessing_fingerprint": "def456",
                "dataset_fingerprint": "ghi789",
                "feature_names": ["a", "b", "c"],
                "metrics": metrics
                or {
                    "out_of_fold": {"mae": 10.0},
                    "out_of_fold_coverage": {"empirical_coverage": 0.91},
                    "out_of_fold_calibrated": {"pr_auc": 0.7, "brier_score": 0.12},
                },
            }
        )
    )
    return directory


def _register(cfg, registry, root, *, version: str, name: str = "fam", **kwargs):
    bundle = _write_bundle(root, f"{name}-{version}", metrics=kwargs.pop("metrics", None))
    return registry.register(
        model_name=name,
        model_version=version,
        bundle_path=bundle,
        validation_status=kwargs.pop("validation_status", "VALIDATED"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registration_records_the_bundle_contract(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")

    assert entry.stage is ModelStage.CANDIDATE
    assert entry.artifact_checksum
    assert entry.dataset_fingerprint == "ghi789"
    assert entry.feature_schema_fingerprint == "def456"
    assert entry.n_features == 3


def test_bundle_paths_are_stored_relative_to_the_project_root(registry_cfg, tmp_path):
    """A registry file must not carry a developer's home directory."""
    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")
    assert not entry.bundle_path.startswith("/")
    assert str(tmp_path) not in registry.path.read_text()


def test_registering_a_directory_that_is_not_a_bundle_is_refused(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RegistryError, match="metadata.json"):
        registry.register(model_name="x", model_version="1", bundle_path=empty)


def test_re_registering_a_version_needs_an_explicit_overwrite(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    with pytest.raises(RegistryError, match="already registered"):
        _register(registry_cfg, registry, tmp_path, version="1.0.0")


def test_the_checksum_changes_when_the_artifact_changes(tmp_path):
    bundle = _write_bundle(tmp_path, "one")
    before = bundle_checksum(bundle)
    (bundle / "model.pkl").write_bytes(b"different-model")
    assert bundle_checksum(bundle) != before


def test_a_missing_optional_artifact_changes_the_checksum(tmp_path):
    bundle = _write_bundle(tmp_path, "two")
    before = bundle_checksum(bundle)
    (bundle / "calibration.pkl").write_bytes(b"calibrator")
    assert bundle_checksum(bundle) != before


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------
def test_promotion_moves_a_candidate_to_production(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    entry = registry.promote("fam", "1.0.0", by="alice", reason="first release")

    assert entry.stage is ModelStage.PRODUCTION
    assert entry.promoted_by == "alice"
    assert entry.promoted_at_utc
    assert registry.production_model("fam").model_version == "1.0.0"


def test_only_one_version_is_in_production_at_a_time(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    _register(registry_cfg, registry, tmp_path, version="2.0.0")
    registry.promote("fam", "1.0.0", by="alice")
    registry.promote("fam", "2.0.0", by="bob")

    production = registry.list_models(model_name="fam", stage=ModelStage.PRODUCTION)
    assert len(production) == 1
    assert production[0].model_version == "2.0.0"
    assert registry.get("fam", "1.0.0").stage is ModelStage.ARCHIVED


def test_only_one_model_per_serving_task_is_in_production(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0", name="family-a")
    _register(registry_cfg, registry, tmp_path, version="1.0.0", name="family-b")

    registry.promote("family-a", "1.0.0", by="alice")
    registry.promote("family-b", "1.0.0", by="bob")

    production = registry.production_model(task="rul_regression")
    assert production is not None and production.model_name == "family-b"
    assert registry.get("family-a", "1.0.0").stage is ModelStage.ARCHIVED


def test_serving_resolves_and_rechecks_the_promoted_bundle(registry_cfg, tmp_path):
    from battery_rul.digital_twin.service import BatteryDigitalTwinService

    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")
    registry.promote("fam", "1.0.0", by="alice")

    service = BatteryDigitalTwinService(cfg=registry_cfg)
    path, resolved = service._serving_bundle_path(  # noqa: SLF001 - serving boundary test
        "rul_regression", registry_cfg.artifacts.rul_dir
    )
    assert path == (tmp_path / entry.bundle_path).resolve()
    assert resolved is not None and resolved.key == "fam:1.0.0"

    (path / "model.pkl").write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="Checksum mismatch"):
        service._serving_bundle_path(  # noqa: SLF001 - serving boundary test
            "rul_regression", registry_cfg.artifacts.rul_dir
        )


def test_an_illegal_transition_is_refused(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    registry.promote("fam", "1.0.0", by="alice")
    with pytest.raises(RegistryError, match="Illegal transition"):
        registry.transition("fam", "1.0.0", ModelStage.REJECTED, by="alice")


def test_promotion_verifies_the_artifact_checksum(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")
    bundle = tmp_path / entry.bundle_path
    (bundle / "model.pkl").write_bytes(b"tampered")

    with pytest.raises(RegistryError, match="Checksum mismatch"):
        registry.promote("fam", "1.0.0", by="alice")


def test_promotion_refuses_a_missing_bundle(registry_cfg, tmp_path):
    import shutil

    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")
    shutil.rmtree(tmp_path / entry.bundle_path)

    with pytest.raises(RegistryError, match="missing"):
        registry.promote("fam", "1.0.0", by="alice")


def test_verify_reports_a_tampered_artifact_without_raising(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    entry = _register(registry_cfg, registry, tmp_path, version="1.0.0")
    (tmp_path / entry.bundle_path / "model.pkl").write_bytes(b"tampered")

    result = registry.verify("fam", "1.0.0")
    assert result["verified"] is False
    assert result["registered_checksum"] != result["actual_checksum"]


def test_every_transition_is_recorded_with_an_author(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    registry.promote("fam", "1.0.0", by="alice", reason="ship it")

    history = registry.history()
    assert any(h["action"] == "register" for h in history)
    promotion = next(h for h in history if h.get("to_stage") == "PRODUCTION")
    assert promotion["by"] == "alice"
    assert promotion["reason"] == "ship it"


def test_a_read_only_deployment_cannot_modify_the_registry(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    registry_cfg.deployment.read_only = True
    with pytest.raises(RegistryError, match="read_only"):
        registry.promote("fam", "1.0.0", by="alice")


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------
def test_rollback_restores_the_previously_live_version(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    _register(registry_cfg, registry, tmp_path, version="2.0.0")
    registry.promote("fam", "1.0.0", by="alice")
    registry.promote("fam", "2.0.0", by="bob")

    restored = registry.rollback("fam", by="carol", reason="2.0.0 regressed")

    assert restored.model_version == "1.0.0"
    assert restored.stage is ModelStage.PRODUCTION
    assert registry.get("fam", "2.0.0").stage is ModelStage.ARCHIVED
    assert len(registry.list_models(model_name="fam", stage=ModelStage.PRODUCTION)) == 1


def test_rollback_without_a_previous_release_is_refused(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    registry.promote("fam", "1.0.0", by="alice")
    with pytest.raises(RegistryError, match="nothing to roll back"):
        registry.rollback("fam", by="alice")


def test_rollback_leaves_a_valid_production_model(registry_cfg, tmp_path):
    registry = FileModelRegistry(cfg=registry_cfg)
    _register(registry_cfg, registry, tmp_path, version="1.0.0")
    _register(registry_cfg, registry, tmp_path, version="2.0.0")
    registry.promote("fam", "1.0.0", by="alice")
    registry.promote("fam", "2.0.0", by="bob")
    registry.rollback("fam", by="carol")

    production = registry.production_model("fam")
    assert production is not None
    assert registry.verify("fam", production.model_version)["verified"] is True


# ---------------------------------------------------------------------------
# The promotion gate
# ---------------------------------------------------------------------------
def _candidate(**overrides) -> RegisteredModel:
    base = {
        "model_name": "fam",
        "model_version": "2.0.0",
        "validation_status": "VALIDATED",
        "dataset_fingerprint": "ghi789",
        "feature_schema_fingerprint": "def456",
        "task": "rul_regression",
        "metrics": {"out_of_fold": {"mae": 10.0}},
        "uncertainty_metrics": {
            "empirical_coverage": 0.91,
            "by_battery_id": {
                "B0005": {"empirical_coverage": 0.90},
                "B0006": {"empirical_coverage": 0.92},
            },
        },
        "calibration_metrics": {"pr_auc": 0.7, "brier_score": 0.12},
    }
    return RegisteredModel(**{**base, **overrides})


def _gate(cfg, candidate, production=None, **evidence):
    return PromotionGate(cfg=cfg).evaluate(
        candidate,
        production,
        smoke_test_passed=evidence.pop("smoke", True),
        contract_tests_passed=evidence.pop("contract", True),
        unit_tests_passed=evidence.pop("unit", True),
        leakage_check_passed=evidence.pop("leakage", True),
        **evidence,
    )


def test_a_missing_artifact_rejects_the_candidate(registry_cfg):
    report = _gate(registry_cfg, _candidate(bundle_path="nowhere"))
    assert report.decision is PromotionDecision.REJECTED
    assert any("checksum" in reason for reason in report.reasons)


def test_an_unvalidated_candidate_is_rejected(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        validation_status="UNVALIDATED",
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
    )
    report = _gate(registry_cfg, candidate)
    assert report.decision is PromotionDecision.REJECTED
    assert any("validation_status" in reason for reason in report.reasons)


def test_a_first_promotion_with_no_baseline_requires_review(registry_cfg, tmp_path):
    """With nothing to regress against and no absolute floor configured, the
    honest verdict is "a human should look", not "approved"."""
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.REQUIRES_REVIEW
    assert any("No PRODUCTION model" in note for note in report.notes)
    assert any("rul_mae" in reason for reason in report.reasons)


def test_an_absolute_floor_lets_a_first_candidate_be_approved(registry_cfg, tmp_path):
    registry_cfg.registry.promotion.max_absolute_rul_mae = 15.0
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.APPROVED


def test_a_candidate_below_the_absolute_floor_is_rejected(registry_cfg, tmp_path):
    registry_cfg.registry.promotion.max_absolute_rul_mae = 5.0
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.REJECTED


def test_missing_test_evidence_requires_review_rather_than_passing(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate, smoke=None)

    assert report.decision is PromotionDecision.REQUIRES_REVIEW
    assert any("smoke" in reason for reason in report.reasons)


def test_a_failed_test_rejects(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate, unit=False)
    assert report.decision is PromotionDecision.REJECTED


def test_insufficient_interval_coverage_rejects(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
        uncertainty_metrics={"empirical_coverage": 0.5},
    )
    report = _gate(registry_cfg, candidate)
    assert report.decision is PromotionDecision.REJECTED
    assert any("coverage" in reason for reason in report.reasons)


def test_marginal_coverage_cannot_hide_a_failing_cell(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
        uncertainty_metrics={
            "empirical_coverage": 0.917,
            "by_battery_id": {
                "B0005": {"empirical_coverage": 0.95},
                "B0033": {"empirical_coverage": 0.703},
            },
        },
    )

    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.REJECTED
    check = next(r for r in report.results if r.name == "worst_cell_interval_coverage")
    assert check.passed is False
    assert check.candidate_value == pytest.approx(0.703)
    assert "B0033" in check.detail


def test_missing_per_cell_coverage_requires_review(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
        uncertainty_metrics={"empirical_coverage": 0.91},
    )

    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.REQUIRES_REVIEW
    assert any("worst_cell_interval_coverage" in reason for reason in report.reasons)


def test_a_regressed_candidate_is_rejected_against_production(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
        metrics={"out_of_fold": {"mae": 20.0}},
    )
    production = _candidate(model_version="1.0.0", metrics={"out_of_fold": {"mae": 10.0}})
    report = _gate(registry_cfg, candidate, production)

    assert report.decision is PromotionDecision.REJECTED
    assert any("rul_mae" in reason for reason in report.reasons)


def test_a_small_regression_within_tolerance_is_accepted(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(
        bundle_path=str(bundle),
        artifact_checksum=bundle_checksum(bundle),
        metrics={"out_of_fold": {"mae": 10.4}},
    )
    production = _candidate(model_version="1.0.0", metrics={"out_of_fold": {"mae": 10.0}})
    report = _gate(registry_cfg, candidate, production)
    assert report.decision is PromotionDecision.APPROVED


def test_the_gate_never_promotes_by_itself(registry_cfg, tmp_path):
    registry_cfg.registry.promotion.max_absolute_rul_mae = 15.0
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate)

    assert report.decision is PromotionDecision.APPROVED
    assert registry_cfg.registry.promotion.allow_auto_promotion is False
    assert any("human action" in note for note in report.notes)


def test_every_check_reports_its_own_verdict(registry_cfg, tmp_path):
    bundle = _write_bundle(tmp_path, "cand")
    candidate = _candidate(bundle_path=str(bundle), artifact_checksum=bundle_checksum(bundle))
    report = _gate(registry_cfg, candidate)

    names = {check.name for check in report.results}
    assert {
        "validation_status",
        "artifact_checksum",
        "interval_coverage",
        "worst_cell_interval_coverage",
    } <= names
    assert all(check.status in ("PASS", "FAIL", "UNKNOWN") for check in report.results)
    assert all(check.detail for check in report.results)
