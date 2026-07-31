"""Feature engineering, target construction, splitting and windowing."""

from __future__ import annotations

from battery_rul.features.engineering import (
    NON_FEATURE_COLUMNS,
    FeatureBuildReport,
    assert_no_leakage,
    build_features,
    feature_columns,
)
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.features.sequences import SequenceBatch, make_sequences
from battery_rul.features.splitting import (
    DataSplit,
    SplitPlan,
    iter_group_folds,
    make_split,
    walk_forward_folds,
)
from battery_rul.features.target import (
    TargetReport,
    attach_target,
    find_eol_cycle,
    inverse_transform_target,
    transform_target,
)

__all__ = [
    "NON_FEATURE_COLUMNS",
    "DataSplit",
    "FeatureBuildReport",
    "FeaturePipeline",
    "SequenceBatch",
    "SplitPlan",
    "TargetReport",
    "assert_no_leakage",
    "attach_target",
    "build_features",
    "feature_columns",
    "find_eol_cycle",
    "inverse_transform_target",
    "iter_group_folds",
    "make_sequences",
    "make_split",
    "transform_target",
    "walk_forward_folds",
]
