"""Supervised targets for the digital twin.

Remaining useful life lives in :mod:`battery_rul.features.target` (Milestone 1)
and is re-exported here so all three targets can be imported from one place.

* ``rul_cycles``               — cycles until the confirmed end-of-life crossing
* ``soh_target``               — state of health as a fraction in [0, 1]
* ``failure_within_horizon``   — will this cell reach end of life within H cycles?

All three are attached by :func:`attach_all_targets`, which enforces the shared
invariant: a target at cycle *k* may depend on the future of the *record*, since
labels are constructed offline, but a *feature* at cycle *k* never may. Keeping
the two apart in separate modules is what makes that reviewable.
"""

from __future__ import annotations

from battery_rul.features.target import (
    attach_target,
    find_eol_cycle,
    inverse_transform_target,
    transform_target,
)
from battery_rul.targets.risk import RiskTargetReport, attach_failure_risk_target
from battery_rul.targets.soh import (
    SOHTargetReport,
    attach_soh_target,
    classify_soh,
    reference_capacity,
)

__all__ = [
    "RiskTargetReport",
    "SOHTargetReport",
    "attach_all_targets",
    "attach_failure_risk_target",
    "attach_soh_target",
    "attach_target",
    "classify_soh",
    "find_eol_cycle",
    "inverse_transform_target",
    "reference_capacity",
    "transform_target",
]


def attach_all_targets(df, cfg):
    """Attach RUL, SOH and failure-risk targets in dependency order.

    Returns ``(frame, {"rul": ..., "soh": ..., "risk": ...})``.
    """
    frame, rul_report = attach_target(df, cfg)
    frame, soh_report = attach_soh_target(frame, cfg)
    frame, risk_report = attach_failure_risk_target(frame, cfg)
    return frame, {"rul": rul_report, "soh": soh_report, "risk": risk_report}
