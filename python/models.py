"""Generic Hugging Face base-model availability helpers (no training logic)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from constants import (
    MODEL_REQUIRED_FILES,
    MODEL_WEIGHT_SUFFIXES,
)
from exceptions import (
    ModelDownloadError,
    ModelNotFoundError,
    ModelVerificationError,
)
from workspace import Workspace

logger = logging.getLogger("aiodoo_colab")


def deterministic_model_dirname(model_id: str) -> str:
    """
    Derive a collision-safe local folder name from a Hugging Face model id.

    Replaces ``/`` with ``__`` so org and name are preserved:

    - ``deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`` → ``deepseek-ai__DeepSeek-Coder-V2-Lite-Instruct``
    - ``deepseek-ai/deepseek-vl2`` → ``deepseek-ai__deepseek-vl2``
    - ``Standalone-Model`` → ``Standalone-Model``
    """
    cleaned = model_id.strip().rstrip("/")
    if not cleaned:
        raise ModelDownloadError("model_id must be a non-empty string.")
    # Reject path traversal / empty segments after normalization.
    for part in cleaned.split("/"):
        if not part or part in {".", ".."}:
            raise ModelDownloadError(f"Invalid model_id for local path: {model_id!r}")
    name = cleaned.replace("/", "__")
    if not name or name in {".", ".."}:
        raise ModelDownloadError(f"Invalid model_id for local path: {model_id!r}")
    return name


@dataclass(frozen=True, slots=True)
class ModelStore:
    """Ensure a Hugging Face model is available under ``workspace.model_cache``."""

    workspace: Workspace
    model_id: str

    @property
    def local_dirname(self) -> str:
        return deterministic_model_dirname(self.model_id)

    def local_path(self) -> Path:
        """Absolute path where the model snapshot is (or will be) stored."""
        return self.workspace.model_cache / self.local_dirname

    def exists(self) -> bool:
        """Return True when the local model directory exists and verifies."""
        path = self.local_path()
        if not path.is_dir():
            return False
        try:
            self.verify()
        except ModelVerificationError:
            return False
        return True

    def verify(self) -> None:
        """
        Verify required model files exist under ``local_path()``.

        Raises ``ModelNotFoundError`` / ``ModelVerificationError``.
        """
        path = self.local_path()
        if not path.exists():
            raise ModelNotFoundError(f"Model directory does not exist: {path}")
        if not path.is_dir():
            raise ModelVerificationError(f"Model path is not a directory: {path}")

        for required in MODEL_REQUIRED_FILES:
            required_path = path / required
            if not required_path.is_file():
                raise ModelVerificationError(
                    f"Model verification failed: missing {required!r} in {path}"
                )

        has_weights = any(
            candidate.is_file() and candidate.suffix in MODEL_WEIGHT_SUFFIXES
            for candidate in path.rglob("*")
        )
        if not has_weights:
            raise ModelVerificationError(
                f"Model verification failed: no weight files {MODEL_WEIGHT_SUFFIXES} under {path}"
            )

        logger.info("Model verified at %s (model_id=%s)", path, self.model_id)

    def download(self, *, force_download: bool = False) -> Path:
        """
        Download the model from Hugging Face into ``local_path()``.

        Uses ``huggingface_hub.snapshot_download``. Does not use git / git-lfs.
        """
        target = self.local_path()
        target.parent.mkdir(parents=True, exist_ok=True)

        if force_download and target.exists():
            logger.info("Force download: removing existing model at %s", target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        logger.info(
            "Downloading model %s → %s (force_download=%s)",
            self.model_id,
            target,
            force_download,
        )
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise ModelDownloadError(
                "huggingface_hub is required for model download. "
                "Install it in the Colab / runtime environment."
            ) from exc

        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(target),
                force_download=force_download,
            )
        except Exception as exc:  # noqa: BLE001 — surface HF client failures
            raise ModelDownloadError(f"Failed to download model {self.model_id!r}: {exc}") from exc

        logger.info("Model download completed: %s", target)
        self.verify()
        return target

    def remove(self) -> None:
        """Remove the local model directory if present."""
        path = self.local_path()
        if not path.exists():
            logger.info("Model remove skipped; path does not exist: %s", path)
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info("Model removed: %s", path)

    def ensure(self, *, force_download: bool = False) -> Path:
        """
        Ensure the model is available locally.

        Reuses an existing verified snapshot unless ``force_download`` is True.
        If a directory exists but fails verification (e.g. Colab killed mid-download
        after disk full), remove it and download again — same for Qwen or DeepSeek.
        """
        path = self.local_path()
        if not force_download and self.exists():
            logger.info("Reusing existing model at %s (model_id=%s)", path, self.model_id)
            return path

        if force_download:
            logger.info("Force download requested for model_id=%s", self.model_id)
        elif path.exists():
            logger.warning(
                "Incomplete model cache at %s; removing before re-download (model_id=%s)",
                path,
                self.model_id,
            )
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            logger.info("Model missing or incomplete at %s; downloading", path)

        return self.download(force_download=force_download)


__all__ = ["ModelStore", "deterministic_model_dirname"]
