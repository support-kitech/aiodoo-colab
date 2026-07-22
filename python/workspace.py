"""AIODOO workspace layout on Google Drive (plus local model cache)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import ColabConfig
from constants import (
    CHECKPOINTS_DIR_NAME,
    DATASETS_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    LOGS_DIR_NAME,
    MODELS_ADAPTERS_DIR_NAME,
    MODELS_DIR_NAME,
    MODELS_EXPORTS_DIR_NAME,
    MODELS_MERGED_DIR_NAME,
    MODELS_REGISTRY_DIR_NAME,
    MODELS_REGISTRY_STORAGE_DIR_NAME,
    REQUIRED_MODELS_SUBDIRS,
    REQUIRED_TOP_LEVEL_DIRS,
    TRAINING_CACHE_DIR_NAME,
    TRAINING_DIR_NAME,
    TRAINING_REPOSITORY_NAME,
)
from drive import mount_google_drive, verify_drive_mounted
from exceptions import WorkspaceError
from naming import normalize_training_id

logger = logging.getLogger("aiodoo_colab")


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved AIODOO workspace paths (Drive artifacts + local model cache)."""

    drive_mount_root: Path
    root: Path
    datasets: Path
    models: Path
    experiments: Path
    logs: Path
    training: Path
    # Colab local SSD (or override) for Hugging Face base models only.
    model_cache: Path

    @property
    def training_repository(self) -> Path:
        """Path to the cloned ``aiodoo-training`` repository."""
        return self.training / TRAINING_REPOSITORY_NAME

    @property
    def adapters(self) -> Path:
        """Drive path for LoRA / QLoRA adapters (persistent)."""
        return self.models / MODELS_ADAPTERS_DIR_NAME

    @property
    def merged(self) -> Path:
        """Drive path for merged base+adapter models (persistent)."""
        return self.models / MODELS_MERGED_DIR_NAME

    @property
    def exports(self) -> Path:
        """Drive path for export bundles (persistent)."""
        return self.models / MODELS_EXPORTS_DIR_NAME

    @property
    def model_registry(self) -> Path:
        """Drive root for aiodoo-model's ``FileBackedRegistry`` (identity index)."""
        return self.models / MODELS_REGISTRY_DIR_NAME

    @property
    def model_registry_storage(self) -> Path:
        """Drive root for aiodoo-model's ``StorageManager`` (published bytes)."""
        return self.models / MODELS_REGISTRY_STORAGE_DIR_NAME

    @property
    def training_cache(self) -> Path:
        """Root for ephemeral per-training runtime cache (checkpoints, resume configs)."""
        return self.training / TRAINING_CACHE_DIR_NAME

    def checkpoints_root(self, training_id: str) -> Path:
        """
        Canonical checkpoint directory for one training id (persistent).

        Matches aiodoo-training's ``ArtifactOutputLayout.adapter_checkpoints_dir``
        exactly (``training/cache/<training_id>/checkpoints/``) — this is the
        single source of truth other modules (``trainer``, ``artifacts``) must
        use instead of re-deriving the path locally.
        """
        return self.training_cache / normalize_training_id(training_id) / CHECKPOINTS_DIR_NAME

    @classmethod
    def from_config(cls, config: ColabConfig) -> Workspace:
        """Build workspace paths from configuration (does not create directories)."""
        root = config.aiodoo_root
        return cls(
            drive_mount_root=config.drive_mount_root,
            root=root,
            datasets=root / DATASETS_DIR_NAME,
            models=root / MODELS_DIR_NAME,
            experiments=root / EXPERIMENTS_DIR_NAME,
            logs=root / LOGS_DIR_NAME,
            training=root / TRAINING_DIR_NAME,
            model_cache=config.model_cache_root,
        )

    def required_top_level_paths(self) -> tuple[Path, ...]:
        return (
            self.datasets,
            self.models,
            self.experiments,
            self.logs,
            self.training,
        )

    def required_models_subpaths(self) -> tuple[Path, ...]:
        return tuple(self.models / name for name in REQUIRED_MODELS_SUBDIRS)


def _ensure_directory(path: Path) -> None:
    """Create ``path`` if missing; never modify existing directory contents."""
    if path.exists():
        if not path.is_dir():
            raise WorkspaceError(f"Expected directory but found non-directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Created directory: %s", path)


def ensure_workspace_layout(workspace: Workspace) -> Workspace:
    """
    Validate and create missing workspace directories.

    Creates top-level ``datasets``, ``models``, ``experiments``, ``logs``,
    ``training`` and required
    ``models/{base,adapters,merged,exports,registry,registry_storage}`` on
    Drive, plus the local ``model_cache`` directory for Hugging Face base
    models. Does not modify existing dataset or experiment contents.
    """
    if not workspace.root.parent.exists():
        raise WorkspaceError(f"Drive mount parent missing for workspace root: {workspace.root}")

    _ensure_directory(workspace.root)

    for path in workspace.required_top_level_paths():
        if path.name not in REQUIRED_TOP_LEVEL_DIRS:
            raise WorkspaceError(f"Unexpected top-level directory name: {path.name}")
        _ensure_directory(path)

    for path in workspace.required_models_subpaths():
        _ensure_directory(path)

    _ensure_directory(workspace.model_cache)

    logger.info("Workspace layout verified at %s", workspace.root)
    logger.info("Local HF model cache at %s", workspace.model_cache)
    return workspace


def prepare_workspace(
    config: ColabConfig,
    *,
    mount: bool = True,
    force_remount: bool = False,
) -> Workspace:
    """
    Mount Drive (optional), locate AIODOO workspace, validate and create layout.

    Returns a ``Workspace`` with all standard paths resolved.
    """
    if mount:
        mount_google_drive(config, force_remount=force_remount)
    else:
        verify_drive_mounted(config)

    workspace = Workspace.from_config(config)
    logger.info("Located AIODOO workspace at %s", workspace.root)
    return ensure_workspace_layout(workspace)


__all__ = [
    "Workspace",
    "ensure_workspace_layout",
    "prepare_workspace",
]
