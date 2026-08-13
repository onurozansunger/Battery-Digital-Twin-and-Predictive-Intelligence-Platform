"""Fleet ingestion: validation, partial failure and boundary enforcement.

The property under test throughout is that ingestion **never silently loses a
cell**. Every battery that goes in comes back with a status, and a rejection
carries the reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.data.synthetic import make_synthetic_cycles
from battery_rul.fleet.domain import ProcessingStatus
from battery_rul.fleet.ingestion import (
    FleetIngestionError,
    FleetIngestor,
    frame_fingerprint,
    resolve_within,
)


def _fleet_frame(n: int = 3, cycles: int = 60) -> pd.DataFrame:
    frames = []
    for index in range(n):
        frame = make_synthetic_cycles(f"C{index:03d}", n_cycles=cycles, seed=index)
        frame["dataset"] = "synthetic"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_a_multi_battery_frame_splits_into_one_history_per_cell(cfg: ExperimentConfig):
    result, histories = FleetIngestor(cfg=cfg).from_frame("F1", _fleet_frame(3))
    assert result.accepted_count == 3
    assert result.failed_count == 0
    assert {h.battery_id for h in histories} == {"C000", "C001", "C002"}
    assert all(h.history["battery_id"].nunique() == 1 for h in histories)


def test_every_submitted_battery_gets_a_record(cfg: ExperimentConfig):
    frame = _fleet_frame(3)
    frame.loc[frame["battery_id"] == "C001", "cycle_index"] = 1  # duplicate cycles
    result, histories = FleetIngestor(cfg=cfg).from_frame("F1", frame)

    assert len(result.records) == 3, "a record per submitted cell, whatever the outcome"
    assert {r.battery_id for r in result.records} == {"C000", "C001", "C002"}
    assert len(histories) == 2, "the malformed cell is rejected, the others survive"


def test_ingestion_is_deterministic(cfg: ExperimentConfig):
    frame = _fleet_frame(2)
    first, _ = FleetIngestor(cfg=cfg).from_frame("F1", frame)
    second, _ = FleetIngestor(cfg=cfg).from_frame("F1", frame)
    assert first.data_fingerprint == second.data_fingerprint


def test_fingerprint_changes_with_the_data(cfg: ExperimentConfig):
    frame = _fleet_frame(2)
    other = frame.copy()
    other.loc[0, "capacity_ah"] = float(other.loc[0, "capacity_ah"]) * 0.5
    assert frame_fingerprint(frame) != frame_fingerprint(other)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
def test_duplicate_cycles_are_rejected_with_a_reason(cfg: ExperimentConfig):
    frame = _fleet_frame(1)
    frame.loc[:, "cycle_index"] = 1
    result, histories = FleetIngestor(cfg=cfg).from_frame("F1", frame)

    record = result.records[0]
    assert record.status is ProcessingStatus.FAILED
    assert any("cycle_index" in error for error in record.errors)
    assert histories == []


def test_missing_required_columns_are_rejected(cfg: ExperimentConfig):
    frame = _fleet_frame(1).drop(columns=["capacity_ah"])
    result, _ = FleetIngestor(cfg=cfg).from_frame("F1", frame)
    assert result.records[0].status is ProcessingStatus.FAILED
    assert "capacity_ah" in result.records[0].errors[0]


def test_an_all_nan_capacity_series_is_rejected(cfg: ExperimentConfig):
    frame = _fleet_frame(1)
    frame["capacity_ah"] = np.nan
    result, _ = FleetIngestor(cfg=cfg).from_frame("F1", frame)
    assert result.records[0].status is ProcessingStatus.FAILED


def test_a_battery_shorter_than_the_minimum_is_rejected(cfg: ExperimentConfig):
    cfg.fleet.min_cycles_per_battery = 10
    frame = _fleet_frame(1, cycles=40).head(4)
    result, _ = FleetIngestor(cfg=cfg).from_frame("F1", frame)
    assert result.records[0].status is ProcessingStatus.FAILED


def test_unordered_cycles_are_a_warning_not_a_rejection(cfg: ExperimentConfig):
    """Sorting is a repair; deduplication would be a guess."""
    frame = _fleet_frame(1).sample(frac=1.0, random_state=0).reset_index(drop=True)
    result, histories = FleetIngestor(cfg=cfg).from_frame("F1", frame)

    assert result.records[0].status is ProcessingStatus.SUCCESS
    assert any("sorted" in w for w in result.records[0].warnings)
    assert histories[0].history["cycle_index"].is_monotonic_increasing


def test_a_frame_without_battery_id_is_refused(cfg: ExperimentConfig):
    frame = _fleet_frame(1).drop(columns=["battery_id"])
    with pytest.raises(FleetIngestionError, match="battery_id"):
        FleetIngestor(cfg=cfg).from_frame("F1", frame)


def test_an_empty_frame_is_refused(cfg: ExperimentConfig):
    with pytest.raises(FleetIngestionError, match="empty"):
        FleetIngestor(cfg=cfg).from_frame("F1", pd.DataFrame())


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
def test_batteries_beyond_the_batch_limit_are_reported_not_dropped(cfg: ExperimentConfig):
    cfg.fleet.max_batteries_per_batch = 2
    result, histories = FleetIngestor(cfg=cfg).from_frame("F1", _fleet_frame(4))

    assert len(histories) == 2
    assert len(result.records) == 4
    overflow = [r for r in result.records if r.status is ProcessingStatus.FAILED]
    assert len(overflow) == 2
    assert all("limit" in r.errors[0] for r in overflow)
    assert result.warnings, "the truncation is stated at fleet level as well"


def test_duplicate_battery_ids_in_a_request_are_rejected(cfg: ExperimentConfig):
    frame = _fleet_frame(1)
    result, _ = FleetIngestor(cfg=cfg).from_records(
        "F1", [("C000", frame), ("C000", frame)], source="test"
    )
    assert sum(1 for r in result.records if r.status is ProcessingStatus.FAILED) == 1
    assert "Duplicate" in result.records[1].errors[0]


def test_the_online_request_limit_is_separate_from_the_batch_limit(cfg: ExperimentConfig):
    cfg.fleet.max_batteries_per_request = 1
    frame = _fleet_frame(1)
    result, histories = FleetIngestor(cfg=cfg).from_records(
        "F1", [("A", frame), ("B", frame.copy())], source="api"
    )
    assert len(histories) == 1
    assert "online limit" in result.records[1].errors[0]


# ---------------------------------------------------------------------------
# Files and directories
# ---------------------------------------------------------------------------
def test_a_directory_of_per_battery_files_is_ingested(cfg: ExperimentConfig, tmp_path):
    directory = tmp_path / "fleet"
    directory.mkdir()
    for index in range(3):
        make_synthetic_cycles(f"D{index}", n_cycles=40, seed=index).to_parquet(
            directory / f"D{index}.parquet"
        )
    result, histories = FleetIngestor(cfg=cfg).from_directory("F1", directory)
    assert result.accepted_count == 3
    assert {h.battery_id for h in histories} == {"D0", "D1", "D2"}


def test_an_unsupported_file_type_is_refused_by_name(cfg: ExperimentConfig, tmp_path):
    path = tmp_path / "fleet.pickle"
    path.write_bytes(b"not a table")
    with pytest.raises(FleetIngestionError, match="Unsupported"):
        FleetIngestor(cfg=cfg).from_file("F1", path)


def test_an_oversized_file_is_refused(cfg: ExperimentConfig, tmp_path):
    cfg.fleet.max_upload_bytes = 1024
    path = tmp_path / "fleet.csv"
    _fleet_frame(3, cycles=200).to_csv(path, index=False)
    with pytest.raises(FleetIngestionError, match="limit"):
        FleetIngestor(cfg=cfg).from_file("F1", path)


def test_a_per_battery_file_with_several_cells_is_rejected(cfg: ExperimentConfig, tmp_path):
    directory = tmp_path / "fleet"
    directory.mkdir()
    _fleet_frame(2, cycles=30).to_parquet(directory / "mixed.parquet")
    result, histories = FleetIngestor(cfg=cfg).from_directory("F1", directory)
    assert histories == []
    assert "exactly one cell" in result.records[0].errors[0]


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def test_path_traversal_is_refused(tmp_path):
    base = tmp_path / "allowed"
    base.mkdir()
    (base / "inside.csv").write_text("a,b\n1,2\n")

    assert resolve_within(base, "inside.csv").name == "inside.csv"
    with pytest.raises(FleetIngestionError, match="outside"):
        resolve_within(base, "../escape.csv")
    with pytest.raises(FleetIngestionError, match="outside"):
        resolve_within(base, "/etc/passwd")


def test_a_battery_id_with_a_path_separator_is_rejected(cfg: ExperimentConfig):
    frame = _fleet_frame(1)
    result, histories = FleetIngestor(cfg=cfg).from_records(
        "F1", [("../../etc/passwd", frame)], source="api"
    )
    assert histories == []
    assert "path separators" in result.records[0].errors[0]


# ---------------------------------------------------------------------------
# Label hygiene
# ---------------------------------------------------------------------------
def test_the_processed_cycle_source_strips_label_columns(m3_platform):
    """A fleet history is input; handing the twin its own answer is meaningless."""
    cfg, _ = m3_platform
    _, histories = FleetIngestor(cfg=cfg).from_processed_cycles("F1")
    columns = set(histories[0].history.columns)
    for label in ("rul_cycles", "eol_cycle", "soh_target", "split", "failure_within_horizon"):
        assert label not in columns
