"""Unit tests for generic Hugging Face model store (mocked — no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config import load_config
from exceptions import ModelDownloadError, ModelNotFoundError, ModelVerificationError
from models import ModelStore, deterministic_model_dirname
from workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    config = load_config(drive_mount_root=mount, auto_mount_drive=False)
    ws = Workspace.from_config(config)
    (ws.models / "base").mkdir(parents=True)
    return ws


def _write_valid_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_deterministic_model_dirname_preserves_org() -> None:
    assert deterministic_model_dirname("Qwen/Qwen3-8B") == "Qwen__Qwen3-8B"
    assert deterministic_model_dirname("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B") == (
        "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"
    )
    assert deterministic_model_dirname("Standalone-Model") == "Standalone-Model"


def test_local_path_under_models_base(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Example-Model")
    assert store.local_path() == workspace.models / "base" / "acme__Example-Model"


def test_exists_false_when_missing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Missing")
    assert store.exists() is False


def test_exists_true_when_verified(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Ready")
    _write_valid_model(store.local_path())
    assert store.exists() is True


def test_verify_raises_not_found(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Gone")
    with pytest.raises(ModelNotFoundError):
        store.verify()


def test_verify_raises_when_config_missing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Bad")
    store.local_path().mkdir(parents=True)
    (store.local_path() / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ModelVerificationError, match="config.json"):
        store.verify()


def test_verify_raises_when_weights_missing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/NoWeights")
    store.local_path().mkdir(parents=True)
    (store.local_path() / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelVerificationError, match="weight"):
        store.verify()


def test_ensure_reuses_existing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Reuse")
    _write_valid_model(store.local_path())
    with patch.object(ModelStore, "download") as download:
        path = store.ensure()
    download.assert_not_called()
    assert path == store.local_path()


def test_ensure_downloads_when_missing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Need")

    def fake_download(*, force_download: bool = False) -> Path:
        del force_download
        _write_valid_model(store.local_path())
        return store.local_path()

    with patch.object(ModelStore, "download", side_effect=fake_download) as download:
        path = store.ensure()
    download.assert_called_once()
    assert path == store.local_path()


def test_download_calls_snapshot_download(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Fetch")

    def fake_snapshot(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "acme/Fetch"
        assert kwargs["local_dir"] == str(store.local_path())
        _write_valid_model(store.local_path())
        return str(store.local_path())

    import sys
    from types import ModuleType

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        path = store.download()
    assert path == store.local_path()
    store.verify()


def test_download_raises_when_hub_missing(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/NoHub")
    real_import = __import__

    def fake_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no hub")
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ModelDownloadError, match="huggingface_hub"):
            store.download()


def test_remove_deletes_directory(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/DeleteMe")
    _write_valid_model(store.local_path())
    store.remove()
    assert not store.local_path().exists()


def test_force_download_removes_and_redownloads(workspace: Workspace) -> None:
    store = ModelStore(workspace=workspace, model_id="acme/Force")
    _write_valid_model(store.local_path())
    old_token = store.local_path() / "old.txt"
    old_token.write_text("stale", encoding="utf-8")

    def fake_snapshot(**kwargs: object) -> str:
        assert kwargs.get("force_download") is True
        _write_valid_model(store.local_path())
        return str(store.local_path())

    import sys
    from types import ModuleType

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        path = store.ensure(force_download=True)
    assert path == store.local_path()
    assert not old_token.exists()
