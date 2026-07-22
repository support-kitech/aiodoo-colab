"""Unit tests for checkpoint discovery / artifact browsing (read-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from artifacts import (
    CheckpointInfo,
    browse_training_artifacts,
    discover_checkpoints,
    is_resumable,
    latest_checkpoint,
    summarize_artifacts,
)
from config import load_config
from workspace import Workspace, ensure_workspace_layout


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    config = load_config(
        drive_mount_root=mount,
        auto_mount_drive=False,
        model_cache_root=tmp_path / "aiodoo-model-cache",
    )
    ws = Workspace.from_config(config)
    ensure_workspace_layout(ws)
    return ws


def test_discover_checkpoints_empty_dir_returns_empty(workspace: Workspace) -> None:
    checkpoints_dir = workspace.checkpoints_root("coding")
    assert discover_checkpoints(checkpoints_dir) == ()
    assert latest_checkpoint(checkpoints_dir) is None


def test_discover_checkpoints_sorts_by_step(workspace: Workspace) -> None:
    checkpoints_dir = workspace.checkpoints_root("coding")
    for step in (500, 100, 1000):
        d = checkpoints_dir / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text("{}", encoding="utf-8")

    checkpoints = discover_checkpoints(checkpoints_dir)
    assert [c.step for c in checkpoints] == [100, 500, 1000]

    latest = latest_checkpoint(checkpoints_dir)
    assert latest is not None
    assert latest.step == 1000


def test_discover_checkpoints_ignores_non_matching_dirs(workspace: Workspace) -> None:
    checkpoints_dir = workspace.checkpoints_root("coding")
    (checkpoints_dir / "checkpoint-100").mkdir(parents=True)
    (checkpoints_dir / "checkpoint-100" / "f.bin").write_bytes(b"x")
    (checkpoints_dir / "scratch").mkdir(parents=True)
    (checkpoints_dir / ".hidden").mkdir(parents=True)
    (checkpoints_dir / "checkpoint-abc").mkdir(parents=True)

    checkpoints = discover_checkpoints(checkpoints_dir)
    assert [c.step for c in checkpoints] == [100]


def test_is_resumable_requires_nonempty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "checkpoint-100"
    empty_dir.mkdir()
    assert is_resumable(CheckpointInfo(step=100, path=empty_dir)) is False

    (empty_dir / "f.bin").write_bytes(b"x")
    assert is_resumable(CheckpointInfo(step=100, path=empty_dir)) is True


def test_browse_training_artifacts_reports_publication_state(workspace: Workspace) -> None:
    artifacts = browse_training_artifacts(workspace, "coding")
    assert artifacts.adapter_published is False
    assert artifacts.checkpoints == ()
    assert artifacts.resumable is False

    adapter_dir = workspace.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "artifact.json").write_text("{}", encoding="utf-8")

    checkpoints_dir = workspace.checkpoints_root("coding")
    ckpt = checkpoints_dir / "checkpoint-200"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")

    artifacts = browse_training_artifacts(workspace, "coding")
    assert artifacts.adapter_published is True
    assert artifacts.merged_published is False
    assert len(artifacts.checkpoints) == 1
    assert artifacts.latest_checkpoint is not None
    assert artifacts.latest_checkpoint.step == 200
    assert artifacts.resumable is True

    summary = summarize_artifacts(artifacts)
    assert summary["adapter_published"] is True
    assert summary["resumable"] is True
    assert summary["latest_checkpoint_step"] == 200


def test_browse_training_artifacts_accepts_legacy_id(workspace: Workspace) -> None:
    artifacts = browse_training_artifacts(workspace, "EXP-0001")
    assert artifacts.training_id == "coding"
