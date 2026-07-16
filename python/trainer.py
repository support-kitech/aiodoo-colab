"""Orchestrate aiodoo-training execution (no ML / training logic here)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constants import (
    MODELS_ADAPTERS_DIR_NAME,
    MODELS_EXPORTS_DIR_NAME,
    MODELS_MERGED_DIR_NAME,
    TRAINING_PUBLIC_ENTRYPOINT,
)
from exceptions import LauncherError
from experiments import Experiment
from models import ModelStore, deterministic_model_dirname
from workspace import Workspace

logger = logging.getLogger("aiodoo_colab")

_TRAIN_TIMEOUT_SECONDS: int = 60 * 60 * 24  # 24h safety cap for long Colab runs


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Resolved paths and references for one orchestrated training run."""

    workspace: Workspace
    experiment: Experiment
    model_path: Path
    dataset_path: Path
    training_repository: Path
    adapter_output: Path
    merged_output: Path
    export_output: Path
    logs_output: Path
    checkpoints_output: Path
    metrics_output: Path
    # Primary config path passed to aiodoo-training's public CLI.
    training_config_path: Path


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Execution metadata only (no training metrics)."""

    success: bool
    exit_code: int
    adapter_path: Path
    checkpoint_path: Path
    logs_path: Path
    metrics_path: Path
    duration_seconds: float
    message: str


def _resolve_dataset_path(workspace: Workspace, experiment: Experiment) -> Path:
    """
    Resolve dataset path from experiment config without loading data.

    Preference order:
    1. ``dataset_root`` if present and absolute / existing
    2. ``workspace.datasets / dataset_version`` when version is a string
    """
    data = experiment.configs.dataset.data
    root = data.get("dataset_root")
    logger.info("root: %s", root)
    if isinstance(root, str) and root.strip():
        path = Path(root)
        if path.is_absolute():
            return path
        candidate = workspace.datasets / path
        if candidate.exists():
            return candidate

    version = experiment.dataset_version
    if isinstance(version, str) and version.strip():
        return workspace.datasets / version

    raise LauncherError(
        f"Experiment {experiment.experiment_id!r} has no resolvable dataset path "
        "(expected dataset_root or dataset_version in dataset.yaml)."
    )


def _resolve_model_path(workspace: Workspace, experiment: Experiment) -> Path:
    model_id = experiment.model_id
    if not isinstance(model_id, str) or not model_id.strip():
        raise LauncherError(
            f"Experiment {experiment.experiment_id!r} has no model id "
            "(expected base_model / model_id / model in model.yaml)."
        )
    return ModelStore(workspace=workspace, model_id=model_id).local_path()


def _output_paths(workspace: Workspace, experiment_id: str) -> dict[str, Path]:
    adapters = workspace.models / MODELS_ADAPTERS_DIR_NAME / experiment_id
    return {
        "adapter_output": adapters,
        "merged_output": workspace.models / MODELS_MERGED_DIR_NAME / experiment_id,
        "export_output": workspace.models / MODELS_EXPORTS_DIR_NAME / experiment_id,
        "logs_output": workspace.logs / experiment_id,
        "checkpoints_output": adapters / "checkpoints",
        "metrics_output": workspace.logs / experiment_id / "metrics",
    }


def _resolve_training_config_path(workspace: Workspace, experiment: Experiment) -> Path:
    """
    Prefer canonical aiodoo-training production experiment.yaml when present.

    Fallback order:
    1. ``training/.../configs/experiments/production/<EXP>/experiment.yaml``
    2. Drive ``experiments/<EXP>/experiment.yaml``
    3. Drive ``experiments/<EXP>/config/training.yaml``
    """
    canonical = (
        workspace.training_repository
        / "configs"
        / "experiments"
        / "production"
        / experiment.experiment_id
        / "experiment.yaml"
    )
    if canonical.is_file():
        return canonical
    composed = experiment.root / "experiment.yaml"
    if composed.is_file():
        return composed
    return experiment.config_dir / "training.yaml"


def build_training_context(
    workspace: Workspace,
    experiment: Experiment,
    *,
    model_path: Path | None = None,
) -> TrainingContext:
    """
    Build a ``TrainingContext`` from workspace + loaded experiment.

    Does not download models or mutate experiment directories.
    """
    resolved_model = model_path or _resolve_model_path(workspace, experiment)
    dataset_path = _resolve_dataset_path(workspace, experiment)
    logger.info("dataset_path: %s", dataset_path)
    outputs = _output_paths(workspace, experiment.experiment_id)
    logger.info("outputs: %s", outputs)
    config_path = _resolve_training_config_path(workspace, experiment)
    logger.info("config_path: %s", config_path)
    context = TrainingContext(
        workspace=workspace,
        experiment=experiment,
        model_path=resolved_model,
        dataset_path=dataset_path,
        training_repository=workspace.training_repository,
        training_config_path=config_path,
        **outputs,
    )

    logger.info("Training context built for experiment=%s", experiment.experiment_id)
    logger.info("  dataset_path=%s", context.dataset_path)
    logger.info("  model_path=%s", context.model_path)
    logger.info("  training_repository=%s", context.training_repository)
    logger.info("  training_config_path=%s", context.training_config_path)
    logger.info("  adapter_output=%s", context.adapter_output)
    logger.info("  logs_output=%s", context.logs_output)
    return context


def _ensure_output_directories(context: TrainingContext) -> None:
    """Create output directories under models/ and logs/ (experiments stay read-only)."""
    for path in (
        context.adapter_output,
        context.merged_output,
        context.export_output,
        context.logs_output,
        context.checkpoints_output,
        context.metrics_output,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _invoke_aiodoo_training(context: TrainingContext) -> int:
    """
    Invoke the public aiodoo-training entrypoint ``train.py``.

    Passes resolved workspace paths via ``AIODOO_COLAB_*`` environment hints
    consumed by aiodoo-training's public orchestrator. Does not import private
    ``aiodoo_training`` modules.
    """
    entry = context.training_repository / TRAINING_PUBLIC_ENTRYPOINT
    if not entry.is_file():
        raise LauncherError(
            f"Public training entrypoint missing: {entry}. "
            "Ensure aiodoo-training was cloned under the workspace training/ directory."
        )

    if not context.training_config_path.is_file():
        raise LauncherError(f"Training config not found: {context.training_config_path}")

    env_hints: dict[str, str] = {
        "AIODOO_COLAB_MODEL_PATH": str(context.model_path),
        "AIODOO_COLAB_DATASET_PATH": str(context.dataset_path),
        "AIODOO_COLAB_ADAPTER_OUTPUT": str(context.adapter_output),
        "AIODOO_COLAB_LOGS_OUTPUT": str(context.logs_output),
        "AIODOO_COLAB_CHECKPOINTS_OUTPUT": str(context.checkpoints_output),
        "AIODOO_COLAB_METRICS_OUTPUT": str(context.metrics_output),
        "AIODOO_COLAB_MERGED_OUTPUT": str(context.merged_output),
        "AIODOO_COLAB_EXPORT_OUTPUT": str(context.export_output),
    }

    command = [
        sys.executable,
        str(entry),
        "--config",
        str(context.training_config_path),
    ]
    logger.info("Invoking aiodoo-training: %s", " ".join(command))
    logger.info("Working directory: %s", context.training_repository)

    env = {**os.environ, **env_hints}
    try:
        completed = subprocess.run(
            command,
            cwd=context.training_repository,
            env=env,
            check=False,
            timeout=_TRAIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise LauncherError("aiodoo-training invocation timed out.") from exc
    except OSError as exc:
        raise LauncherError(f"Failed to invoke aiodoo-training: {exc}") from exc

    return int(completed.returncode)


def run_training(context: TrainingContext) -> TrainingResult:
    """
    Orchestrate one training run via aiodoo-training's public entrypoint.

    Creates output directories under ``models/`` and ``logs/`` only.
    Experiments remain read-only.
    """
    logger.info("Training start experiment=%s", context.experiment.experiment_id)
    logger.info(
        "model_dirname=%s",
        deterministic_model_dirname(str(context.experiment.model_id))
        if isinstance(context.experiment.model_id, str)
        else "<unknown>",
    )

    _ensure_output_directories(context)
    started = time.perf_counter()
    try:
        exit_code = _invoke_aiodoo_training(context)
    except LauncherError as exc:
        duration = time.perf_counter() - started
        logger.error("Training finish (failed orchestration): %s", exc)
        return TrainingResult(
            success=False,
            exit_code=1,
            adapter_path=context.adapter_output,
            checkpoint_path=context.checkpoints_output,
            logs_path=context.logs_output,
            metrics_path=context.metrics_output,
            duration_seconds=duration,
            message=str(exc),
        )

    duration = time.perf_counter() - started
    success = exit_code == 0
    message = (
        "Training entrypoint completed successfully"
        if success
        else (f"Training entrypoint exited with code {exit_code}")
    )
    logger.info(
        "Training finish success=%s exit_code=%s duration=%.2fs",
        success,
        exit_code,
        duration,
    )
    return TrainingResult(
        success=success,
        exit_code=exit_code,
        adapter_path=context.adapter_output,
        checkpoint_path=context.checkpoints_output,
        logs_path=context.logs_output,
        metrics_path=context.metrics_output,
        duration_seconds=duration,
        message=message,
    )


def summarize_result(result: TrainingResult) -> dict[str, Any]:
    """Plain dict summary suitable for notebook printing."""
    return {
        "success": result.success,
        "exit_code": result.exit_code,
        "adapter_path": str(result.adapter_path),
        "checkpoint_path": str(result.checkpoint_path),
        "logs_path": str(result.logs_path),
        "metrics_path": str(result.metrics_path),
        "duration_seconds": result.duration_seconds,
        "message": result.message,
    }


__all__ = [
    "TrainingContext",
    "TrainingResult",
    "build_training_context",
    "run_training",
    "summarize_result",
]
