"""Unit tests for validation orchestration (real aiodoo_validation, no ML)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from exceptions import ValidationIntegrationError
from experiments import ExperimentStore
from trainer import TrainingResult, build_training_context
from validation import (
    resolve_validation_refs,
    run_validation,
    summarize_validation,
)
from workspace import Workspace, ensure_workspace_layout

pytest.importorskip("aiodoo_validation", reason="aiodoo-validation sibling checkout not available")


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
    model_dir = ws.model_cache / "Qwen__Qwen3-8B"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")


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


def test_resolve_validation_refs_uses_training_paths(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    adapter_dir = ws.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)

    base_ref, adapter_ref, merged_ref = resolve_validation_refs(context, result)
    assert base_ref == str(context.model_path)
    assert adapter_ref == str(adapter_dir)
    assert merged_ref is None


def test_run_validation_rejects_failed_training_run(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    adapter_dir = ws.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)
    failed = _make_result(ws, success=False, adapter_dir=adapter_dir)

    with pytest.raises(ValidationIntegrationError, match="Refusing to validate"):
        run_validation(context, failed)


def test_run_validation_rejects_unsupported_profile(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    config_dir = ws.experiments / "repair" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text("dataset_version: v1\n", encoding="utf-8")
    (config_dir / "model.yaml").write_text("base_model: Qwen/Qwen3-8B\n", encoding="utf-8")
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("metrics: []\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("merge_adapter: false\n", encoding="utf-8")
    (ws.datasets / "v1").mkdir(parents=True)
    model_dir = ws.model_cache / "Qwen__Qwen3-8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    experiment = store.load("repair")
    context = build_training_context(ws, experiment)
    adapter_dir = ws.adapters / "aiodoo-repair"
    adapter_dir.mkdir(parents=True)
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)

    with pytest.raises(ValidationIntegrationError, match="no request builder"):
        run_validation(context, result)


def test_run_validation_runs_and_summarizes(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("coding")
    context = build_training_context(ws, experiment)
    adapter_dir = ws.adapters / "aiodoo-coding"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "artifact.json").write_text("{}", encoding="utf-8")
    result = _make_result(ws, success=True, adapter_dir=adapter_dir)

    outcome = run_validation(context, result, execution_tier="standard")
    assert outcome.training_id == "coding"
    # "standard" execution tier never certifies (aiodoo-validation invariant) —
    # this only proves the call reached ValidationService, not a scoring outcome.
    assert outcome.certified is False

    summary = summarize_validation(outcome)
    assert summary["training_id"] == "coding"
    assert "errors" in summary
    assert "warnings" in summary
