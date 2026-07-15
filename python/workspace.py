"""AIODOO workspace layout on Google Drive."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import ColabConfig
from constants import (
    DATASETS_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    LOGS_DIR_NAME,
    MODELS_DIR_NAME,
    REQUIRED_MODELS_SUBDIRS,
    REQUIRED_TOP_LEVEL_DIRS,
    TRAINING_DIR_NAME,
    TRAINING_REPOSITORY_NAME,
)
from drive import mount_google_drive, verify_drive_mounted
from exceptions import WorkspaceError

logger = logging.getLogger("aiodoo_colab")


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved AIODOO workspace paths on Google Drive."""

    drive_mount_root: Path
    root: Path
    datasets: Path
    models: Path
    experiments: Path
    logs: Path
    training: Path

    @property
    def training_repository(self) -> Path:
        """Path to the cloned ``aiodoo-training`` repository."""
        return self.training / TRAINING_REPOSITORY_NAME

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
    ``training`` and required ``models/{base,adapters,merged,exports}`` only.
    Does not modify existing dataset or experiment contents.
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

    logger.info("Workspace layout verified at %s", workspace.root)
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
