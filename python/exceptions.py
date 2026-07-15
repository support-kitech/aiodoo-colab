"""Shared exception types for aiodoo-colab."""

from __future__ import annotations


class AiodooColabError(Exception):
    """Base error for all aiodoo-colab failures."""


class DriveError(AiodooColabError):
    """Google Drive mount or path failures."""


class DriveMountError(DriveError):
    """Drive could not be mounted or verified."""


class WorkspaceError(AiodooColabError):
    """Workspace layout verification or path resolution failures."""


class RepositoryError(AiodooColabError):
    """aiodoo-training clone / update failures."""


class RepositoryCloneError(RepositoryError):
    """Git clone of aiodoo-training failed."""


class RepositoryUpdateError(RepositoryError):
    """Git pull / update of aiodoo-training failed."""


class RepositoryCheckoutError(RepositoryError):
    """Git checkout of branch / tag / commit failed."""


class ModelCacheError(AiodooColabError):
    """Model download or cache directory failures."""


class ModelNotFoundError(ModelCacheError):
    """Local model path does not exist."""


class ModelDownloadError(ModelCacheError):
    """Hugging Face model download failed."""


class ModelVerificationError(ModelCacheError):
    """Downloaded model failed verification."""


class ExperimentError(AiodooColabError):
    """Experiment configuration location or load failures."""


class ExperimentNotFoundError(ExperimentError):
    """Requested experiment directory does not exist."""


class ExperimentValidationError(ExperimentError):
    """Experiment directory or config files failed validation."""


class LauncherError(AiodooColabError):
    """Failures while invoking aiodoo-training entrypoints."""


__all__ = [
    "AiodooColabError",
    "DriveError",
    "DriveMountError",
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentValidationError",
    "LauncherError",
    "ModelCacheError",
    "ModelDownloadError",
    "ModelNotFoundError",
    "ModelVerificationError",
    "RepositoryCheckoutError",
    "RepositoryCloneError",
    "RepositoryError",
    "RepositoryUpdateError",
    "WorkspaceError",
]
