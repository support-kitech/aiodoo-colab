"""Unit tests for model packaging orchestration (real aiodoo_model, no ML)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_config
from exceptions import PackagingIntegrationError
from experiments import ExperimentStore
from packaging import (
    ModelRegistry,
    latest_release_id,
    materialize_release,
    publish_adapter,
    resolve_release,
    summarize_packaging,
)
from trainer import TrainingResult
from workspace import Workspace, ensure_workspace_layout

pytest.importorskip("aiodoo_model", reason="aiodoo-model is not installed")


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Workspace, ExperimentStore]:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    config = load_config(
        drive_mount_root=mount,
        auto_mount_drive=False,
        model_cache_root=tmp_path / "aiodoo-model-cache",
    )
    ws = Workspace.from_config(config)
    ensure_workspace_layout(ws)
    return ws, ExperimentStore(workspace=ws)


def _write_experiment(ws: Workspace, experiment_id: str = "coding") -> None:
    config_dir = ws.experiments / experiment_id / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text("dataset_version: v1.0.0\n", encoding="utf-8")
    (config_dir / "model.yaml").write_text("base_model: Qwen/Qwen3-8B\n", encoding="utf-8")
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("metrics: []\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("merge_adapter: false\n", encoding="utf-8")
    (ws.datasets / "v1.0.0").mkdir(parents=True, exist_ok=True)


def _publishable_adapter_dir(ws: Workspace, adapter_id: str = "aiodoo-coding") -> Path:
    adapter_dir = ws.adapters / adapter_id
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_type": "adapter",
                "protocol_major": 1,
                "capability_id": "coding",
                "adapter_type": "lora",
                "created_at": "2026-07-20T00:00:00Z",
                "supported_odoo_versions": [17, 18, 19],
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.bin").write_bytes(b"fake-weights")
    return adapter_dir


def _make_result(ws: Workspace, *, success: bool, adapter_dir: Path) -> TrainingResult:
    return TrainingResult(
        success=success,
        exit_code=0 if success else 1,
        adapter_path=adapter_dir,
        checkpoint_path=ws.checkpoints_root("coding"),
        logs_path=ws.experiments / "coding" / "logs",
        metrics_path=ws.experiments / "coding" / "metrics",
        duration_seconds=1.0,
        message="ok" if success else "boom",
    )


def test_publish_adapter_rejects_failed_training_run(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    adapter_dir = ws.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)
    failed = _make_result(ws, success=False, adapter_dir=adapter_dir)
    registry = ModelRegistry.from_workspace(ws)

    with pytest.raises(PackagingIntegrationError, match="Refusing to package"):
        publish_adapter(registry, experiment, failed)


def test_publish_adapter_rejects_missing_artifact_json(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    adapter_dir = ws.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)  # no artifact.json written
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)
    registry = ModelRegistry.from_workspace(ws)

    with pytest.raises(PackagingIntegrationError, match="artifact.json"):
        publish_adapter(registry, experiment, result, drive_sync_timeout=1.0)


def test_publish_adapter_end_to_end_and_idempotent(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    adapter_dir = _publishable_adapter_dir(ws)
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)
    registry = ModelRegistry.from_workspace(ws)

    first = publish_adapter(registry, experiment, result, version="0.1.0")
    assert first.already_published is False
    assert first.artifact_id == "aiodoo-coding-0.1.0"
    assert first.storage_uri is not None

    second = publish_adapter(registry, experiment, result, version="0.1.0")
    assert second.already_published is True
    assert second.artifact_id == first.artifact_id

    summary = summarize_packaging(first)
    assert summary["artifact_id"] == "aiodoo-coding-0.1.0"
    assert summary["already_published"] is False


def test_publish_adapter_auto_increments_version(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    adapter_dir = _publishable_adapter_dir(ws)
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)
    registry = ModelRegistry.from_workspace(ws)

    first = publish_adapter(registry, experiment, result)
    second = publish_adapter(registry, experiment, result)
    assert first.version != second.version
    assert second.already_published is False


def test_resolve_and_materialize_release(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    adapter_dir = _publishable_adapter_dir(ws)
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)
    registry = ModelRegistry.from_workspace(ws)

    published = publish_adapter(registry, experiment, result, version="0.1.0")
    release_id = latest_release_id(registry, published.family_id)
    assert release_id == published.release_id

    resolved = resolve_release(registry, release_id)
    assert resolved.primary_artifact_id == published.artifact_id

    destination = ws.model_cache / "materialized"
    materialized = materialize_release(registry, release_id, destination_root=destination)
    assert len(materialized.artifacts) == 1
    assert Path(materialized.artifacts[0].local_path).is_dir()


def test_latest_release_id_returns_none_when_unpublished(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, _store = prepared
    registry = ModelRegistry.from_workspace(ws)
    assert latest_release_id(registry, "aiodoo-coding") is None
