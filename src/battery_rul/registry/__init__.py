"""Model registry and promotion.

A registry answers one operational question that a directory of pickles cannot:
*which model produced this decision, and who decided it should be live?*

``store``      the registry itself — entries, stages, checksums, transitions
``promotion``  the gate a candidate must pass before it may reach PRODUCTION

Two rules run through both:

* **Promotion is explicit.** No metric promotes a model on its own, and the gate
  does not promote — it returns a verdict a person acts on.
* **Artifacts are verified, not trusted.** Every entry carries a checksum over
  its bundle files, and promotion re-verifies it. A registry entry pointing at a
  bundle that has since been overwritten is worse than no registry.
"""

from __future__ import annotations

from battery_rul.registry.promotion import (
    GateResult,
    PromotionDecision,
    PromotionGate,
    evaluate_promotion,
)
from battery_rul.registry.store import (
    REGISTRY_SCHEMA_VERSION,
    FileModelRegistry,
    ModelStage,
    RegisteredModel,
    RegistryError,
    bundle_checksum,
)

__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "FileModelRegistry",
    "GateResult",
    "ModelStage",
    "PromotionDecision",
    "PromotionGate",
    "RegisteredModel",
    "RegistryError",
    "bundle_checksum",
    "evaluate_promotion",
]
