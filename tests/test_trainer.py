"""Unit tests for training orchestration (mocked invocation — no ML)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config import load_config
from exceptions import ExperimentValidationError, LauncherError
from experiments import ExperimentStore
from trainer import (
    TrainingResult,
    build_training_context,
    run_training,
    summarize_result,
)
from workspace import Workspace, ensure_workspace_layout


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


def _write_experiment(ws: Workspace, experiment_id: str = "EXP-0001") -> Path:
    exp = ws.experiments / experiment_id
    config_dir = exp / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text(
        "dataset_version: v1.0.0\ndataset_root: datasets/v1.0.0\n",
        encoding="utf-8",
    )
    (config_dir / "model.yaml").write_text(
        "base_model: Qwen/Qwen3-8B\n",
        encoding="utf-8",
    )
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("metrics: []\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("merge_adapter: false\n", encoding="utf-8")
    (ws.datasets / "v1.0.0").mkdir(parents=True, exist_ok=True)
    model_dir = ws.model_cache / "Qwen__Qwen3-8B"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")
    return exp


def test_build_training_context_resolves_paths(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    _write_experiment(ws)
    experiment = store.load("EXP-0001")
    context = build_training_context(ws, experiment)

    assert context.model_path == ws.model_cache / "Qwen__Qwen3-8B"
    assert context.dataset_path == ws.datasets / "v1.0.0"
    assert context.adapter_output == ws.models / "adapters" / "EXP-0001"
    assert context.logs_output == ws.logs / "EXP-0001"
    assert context.training_repository == ws.training_repository
    assert context.training_config_path.name == "training.yaml"

def test_run_training_invokes_public_entrypoint(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    # Fake public entrypoint
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")

    experiment = store.load("EXP-0001")
    context = build_training_context(ws, experiment)

    completed = type("R", (), {"returncode": 0})()

    with patch("trainer.subprocess.run", return_value=completed) as run_mock:
        result = run_training(context)

    assert result.success is True
    assert result.exit_code == 0
    assert result.adapter_path == context.adapter_output
    assert context.adapter_output.is_dir()
    assert context.logs_output.is_dir()
    args = run_mock.call_args.args[0]
    assert args[1].endswith("train.py")
    assert "--config" in args


def test_run_training_reports_failure_exit_code(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    _write_experiment(ws)
    (ws.training_repository).mkdir(parents=True)
    (ws.training_repository / "train.py").write_text("# fake\n", encoding="utf-8")
    experiment = store.load("EXP-0001")
    context = build_training_context(ws, experiment)

    with patch("trainer.subprocess.run", return_value=type("R", (), {"returncode": 7})()):
        result = run_training(context)

    assert result.success is False
    assert result.exit_code == 7


def test_run_training_missing_entrypoint(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    _write_experiment(ws)
    ws.training_repository.mkdir(parents=True)
    experiment = store.load("EXP-0001")
    context = build_training_context(ws, experiment)
    result = run_training(context)
    assert result.success is False
    assert "entrypoint missing" in result.message


def test_summarize_result() -> None:
    result = TrainingResult(
        success=True,
        exit_code=0,
        adapter_path=Path("/a"),
        checkpoint_path=Path("/c"),
        logs_path=Path("/l"),
        metrics_path=Path("/m"),
        duration_seconds=1.5,
        message="ok",
    )
    summary = summarize_result(result)
    assert summary["success"] is True
    assert summary["duration_seconds"] == 1.5


def test_missing_model_id_raises(prepared: tuple[Workspace, ExperimentStore]) -> None:
    ws, store = prepared
    exp = ws.experiments / "EXP-0001"
    config_dir = exp / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text("dataset_version: v1\n", encoding="utf-8")
    (config_dir / "model.yaml").write_text("tokenizer: x\n", encoding="utf-8")
    (config_dir / "training.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (config_dir / "evaluation.yaml").write_text("{}\n", encoding="utf-8")
    (config_dir / "export.yaml").write_text("{}\n", encoding="utf-8")
    (ws.datasets / "v1").mkdir(parents=True)

    experiment = store.load("EXP-0001")
    with pytest.raises(LauncherError, match="no model id"):
        build_training_context(ws, experiment)


def test_invalid_experiment_rejected_on_validate(
    prepared: tuple[Workspace, ExperimentStore],
) -> None:
    ws, store = prepared
    bad = ws.experiments / "not-an-exp"
    (bad / "config").mkdir(parents=True)
    with pytest.raises(ExperimentValidationError, match="Invalid experiment id"):
        store.validate("not-an-exp")
