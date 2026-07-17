"""Unit tests for training discovery and YAML loading (no training)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from exceptions import ExperimentNotFoundError, ExperimentValidationError
from experiments import ExperimentStore
from workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    config = load_config(drive_mount_root=mount, auto_mount_drive=False)
    ws = Workspace.from_config(config)
    ws.experiments.mkdir(parents=True)
    return ws


def _write_experiment(root: Path, experiment_id: str) -> Path:
    exp = root / experiment_id
    config_dir = exp / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "dataset.yaml").write_text(
        "dataset_version: v1.0.0\ndataset_root: /data\n",
        encoding="utf-8",
    )
    (config_dir / "model.yaml").write_text(
        "base_model: org/Example-7B\ntokenizer: org/Example-7B\n",
        encoding="utf-8",
    )
    (config_dir / "training.yaml").write_text(
        "epochs: 3\nlearning_rate: 0.0002\n",
        encoding="utf-8",
    )
    (config_dir / "evaluation.yaml").write_text(
        "evaluation_dataset: eval\nsave_predictions: true\n",
        encoding="utf-8",
    )
    (config_dir / "export.yaml").write_text(
        "export_directory: export\nmerge_adapter: false\n",
        encoding="utf-8",
    )
    return exp


def test_discover_returns_sorted_ids(workspace: Workspace) -> None:
    _write_experiment(workspace.experiments, "planner")
    _write_experiment(workspace.experiments, "coding")
    _write_experiment(workspace.experiments, "scratch-run")
    store = ExperimentStore(workspace=workspace)
    assert store.discover() == ["coding", "planner"]


def test_exists(workspace: Workspace) -> None:
    store = ExperimentStore(workspace=workspace)
    assert store.exists("coding") is False
    assert store.exists("bad-name") is False
    _write_experiment(workspace.experiments, "coding")
    assert store.exists("coding") is True
    # Legacy EXP id still resolves when Drive has the semantic folder.
    assert store.exists("EXP-0001") is True


def test_validate_raises_not_found(workspace: Workspace) -> None:
    store = ExperimentStore(workspace=workspace)
    with pytest.raises(ExperimentNotFoundError):
        store.validate("planner")


def test_validate_rejects_invalid_id(workspace: Workspace) -> None:
    store = ExperimentStore(workspace=workspace)
    with pytest.raises(ExperimentValidationError, match="Invalid training id"):
        store.validate("EXP-BAD")


def test_validate_raises_when_config_missing(workspace: Workspace) -> None:
    exp = workspace.experiments / "context"
    (exp / "config").mkdir(parents=True)
    store = ExperimentStore(workspace=workspace)
    with pytest.raises(ExperimentValidationError, match="dataset.yaml"):
        store.validate("context")


def test_validate_falls_back_to_canonical_when_drive_incomplete(
    workspace: Workspace,
) -> None:
    # Incomplete Drive output leftover from a prior training run.
    drive = workspace.experiments / "coding"
    (drive / "config").mkdir(parents=True)
    (drive / "summary.json").write_text('{"success": true}\n', encoding="utf-8")

    canonical = workspace.training_repository / "configs" / "training" / "coding"
    canonical.mkdir(parents=True)
    for name, body in (
        ("dataset.yaml", "dataset_version: v1.0.0\n"),
        ("model.yaml", "base_model: Qwen/Qwen3-8B\n"),
        ("training.yaml", "epochs: 1\n"),
        ("evaluation.yaml", "metrics: []\n"),
        ("export.yaml", "merge_adapter: false\n"),
    ):
        (canonical / name).write_text(body, encoding="utf-8")

    store = ExperimentStore(workspace=workspace)
    root = store.validate("coding")
    assert root == canonical
    experiment = store.load("EXP-0001")  # legacy ref normalizes
    assert experiment.experiment_id == "coding"
    assert experiment.model_id == "Qwen/Qwen3-8B"


def test_load_returns_typed_experiment(workspace: Workspace) -> None:
    _write_experiment(workspace.experiments, "coding")
    store = ExperimentStore(workspace=workspace)
    experiment = store.load("coding")

    assert experiment.experiment_id == "coding"
    assert experiment.training_id == "coding"
    assert experiment.dataset_version == "v1.0.0"
    assert experiment.model_id == "org/Example-7B"
    assert experiment.training_configuration["epochs"] == 3
    assert experiment.evaluation_configuration["save_predictions"] is True
    assert experiment.export_configuration["merge_adapter"] is False
    assert experiment.config().model.data["base_model"] == "org/Example-7B"


def test_model_id_resolves_nested_identifier(workspace: Workspace) -> None:
    """model.yaml nests identity under ``model.identifier`` (training schema)."""
    _write_experiment(workspace.experiments, "coding")
    model_yaml = workspace.experiments / "coding" / "config" / "model.yaml"
    model_yaml.write_text(
        "model:\n  identifier: Qwen/Qwen3-8B\n  family: qwen\n",
        encoding="utf-8",
    )
    store = ExperimentStore(workspace=workspace)
    experiment = store.load("coding")
    assert experiment.model_id == "Qwen/Qwen3-8B"


def test_load_invalid_yaml_raises(workspace: Workspace) -> None:
    exp = _write_experiment(workspace.experiments, "repair")
    (exp / "config" / "model.yaml").write_text(": not: valid: yaml: [[\n", encoding="utf-8")
    store = ExperimentStore(workspace=workspace)
    with pytest.raises(ExperimentValidationError, match="Invalid YAML"):
        store.load("repair")
