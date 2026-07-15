"""Centralized configuration for aiodoo-colab (no hardcoded paths in business logic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import (
    AIODOO_ROOT_RELATIVE,
    DEFAULT_DRIVE_MOUNT_RELATIVE,
    TRAINING_REPOSITORY_URL,
)


@dataclass(frozen=True, slots=True)
class ColabConfig:
    """Immutable runtime configuration for Drive workspace and repository management."""

    # Root where Google Drive is mounted (Colab default: /content/drive/MyDrive).
    drive_mount_root: Path

    # Relative path from drive_mount_root to AIODOO workspace.
    aiodoo_root_relative: Path

    # Remote URL for the frozen training framework repository.
    training_repository_url: str

    # Default git branch when none is specified for clone / checkout.
    default_branch: str = "main"

    # When True, attempt ``google.colab.drive.mount`` when not already mounted.
    auto_mount_drive: bool = True

    @property
    def aiodoo_root(self) -> Path:
        """Absolute path to the AIODOO workspace on Drive."""
        return self.drive_mount_root / self.aiodoo_root_relative

    @classmethod
    def colab_default(cls) -> ColabConfig:
        """Configuration matching the standard Google Colab + Drive layout."""
        return cls(
            drive_mount_root=Path("/content") / DEFAULT_DRIVE_MOUNT_RELATIVE,
            aiodoo_root_relative=AIODOO_ROOT_RELATIVE,
            training_repository_url=TRAINING_REPOSITORY_URL,
        )


def load_config(
    *,
    drive_mount_root: Path | None = None,
    training_repository_url: str | None = None,
    default_branch: str = "main",
    auto_mount_drive: bool = True,
) -> ColabConfig:
    """
    Build configuration, optionally overriding mount root or repository URL.

    Used by tests and local runs; production Colab uses ``colab_default()``
    unless overrides are passed explicitly.
    """
    base = ColabConfig.colab_default()
    return ColabConfig(
        drive_mount_root=drive_mount_root or base.drive_mount_root,
        aiodoo_root_relative=base.aiodoo_root_relative,
        training_repository_url=training_repository_url or base.training_repository_url,
        default_branch=default_branch,
        auto_mount_drive=auto_mount_drive,
    )


__all__ = ["ColabConfig", "load_config"]
