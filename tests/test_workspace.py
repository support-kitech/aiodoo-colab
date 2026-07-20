"""Unit tests for AIODOO workspace layout (local filesystem)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from constants import REQUIRED_MODELS_SUBDIRS, REQUIRED_TOP_LEVEL_DIRS
from exceptions import WorkspaceError
from workspace import Workspace, ensure_workspace_layout, prepare_workspace


@pytest.fixture
def mounted_drive(tmp_path: Path) -> Path:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    return mount


@pytest.fixture
def workspace_config(mounted_drive: Path, tmp_path: Path) -> object:
    return load_config(
        drive_mount_root=mounted_drive,
        auto_mount_drive=False,
        model_cache_root=tmp_path / "aiodoo-model-cache",
    )


def test_workspace_from_config_paths(workspace_config: object) -> None:
    ws = Workspace.from_config(workspace_config)
    assert ws.root == workspace_config.drive_mount_root / "AIODOO"
    assert ws.datasets.name == "datasets"
    assert ws.models.name == "models"
    assert ws.experiments.name == "experiments"
    assert ws.logs.name == "logs"
    assert ws.training.name == "training"
    assert ws.training_repository.name == "aiodoo-training"
    assert ws.model_cache == workspace_config.model_cache_root
    assert ws.adapters == ws.models / "adapters"
    assert ws.merged == ws.models / "merged"
    assert ws.exports == ws.models / "exports"
    assert ws.model_registry == ws.models / "registry"
    assert ws.model_registry_storage == ws.models / "registry_storage"
    assert ws.training_cache == ws.training / "cache"
    assert ws.checkpoints_root("coding") == ws.training / "cache" / "coding" / "checkpoints"
    # Accepts legacy EXP ids too, normalizing to the semantic training id.
    assert ws.checkpoints_root("EXP-0001") == ws.checkpoints_root("coding")


def test_ensure_workspace_layout_creates_directories(workspace_config: object) -> None:
    ws = Workspace.from_config(workspace_config)
    ensure_workspace_layout(ws)

    for name in REQUIRED_TOP_LEVEL_DIRS:
        assert (ws.root / name).is_dir()

    for name in REQUIRED_MODELS_SUBDIRS:
        assert (ws.models / name).is_dir()

    assert ws.model_cache.is_dir()


def test_colab_default_model_cache_is_local_ssd() -> None:
    from constants import DEFAULT_MODEL_CACHE_ROOT

    assert load_config().model_cache_root == DEFAULT_MODEL_CACHE_ROOT
    assert DEFAULT_MODEL_CACHE_ROOT == Path("/content/aiodoo-model-cache")


def test_ensure_workspace_layout_does_not_remove_existing_files(
    workspace_config: object,
) -> None:
    ws = Workspace.from_config(workspace_config)
    ws.datasets.mkdir(parents=True)
    existing = ws.datasets / "v1.0.0" / "data.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"keep": true}', encoding="utf-8")

    ensure_workspace_layout(ws)
    assert existing.read_text(encoding="utf-8") == '{"keep": true}'


def test_prepare_workspace_end_to_end(workspace_config: object) -> None:
    ws = prepare_workspace(workspace_config, mount=False)
    assert ws.root.is_dir()
    assert ws.training_repository.parent.is_dir()


def test_ensure_workspace_layout_raises_when_file_blocks_directory(
    workspace_config: object,
) -> None:
    ws = Workspace.from_config(workspace_config)
    ws.root.mkdir(parents=True)
    ws.logs.write_text("not a dir", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="non-directory"):
        ensure_workspace_layout(ws)
