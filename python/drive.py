"""Google Drive mount and verification for Google Colab."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

from config import ColabConfig
from constants import GOOGLE_DRIVE_ROOT_NAME
from exceptions import DriveMountError, DriveSyncError

logger = logging.getLogger("aiodoo_colab")

# Google Drive's FUSE mount is eventually consistent: a file written by a
# just-exited subprocess (e.g. aiodoo-training's train.py) can take a moment
# to become visible through the mount point. Callers that must read a
# Drive-persisted path immediately after another process wrote it (packaging,
# validation) should poll via ``wait_for_path`` instead of checking once and
# failing closed on a false negative.
DEFAULT_DRIVE_SYNC_TIMEOUT_SECONDS: float = 30.0
DEFAULT_DRIVE_SYNC_POLL_SECONDS: float = 0.5


def is_colab_environment() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _default_colab_mount_root() -> Path:
    return Path("/content") / "drive" / GOOGLE_DRIVE_ROOT_NAME


def resolve_drive_mount_root(config: ColabConfig) -> Path:
    """Return the configured Drive mount root path."""
    return config.drive_mount_root


def is_drive_mounted(mount_root: Path) -> bool:
    """
    Return True when ``mount_root`` exists and looks like a mounted Drive folder.

    Heuristic: directory exists and is readable. Colab mounts are directories
    under ``/content/drive/MyDrive``.
    """
    try:
        return mount_root.is_dir() and mount_root.exists()
    except OSError:
        return False


def mount_google_drive(
    config: ColabConfig,
    *,
    force_remount: bool = False,
) -> Path:
    """
    Mount Google Drive when in Colab and verify the mount point.

    Outside Colab, returns ``config.drive_mount_root`` if already present;
    otherwise raises ``DriveMountError``.
    """
    mount_root = resolve_drive_mount_root(config)

    if is_drive_mounted(mount_root) and not force_remount:
        logger.info("Drive already mounted at %s", mount_root)
        return mount_root

    if not config.auto_mount_drive:
        if not is_drive_mounted(mount_root):
            raise DriveMountError(
                f"Drive is not mounted at {mount_root} and auto_mount_drive is disabled."
            )
        return mount_root

    if is_colab_environment():
        try:
            drive_mod = importlib.import_module("google.colab.drive")
            mount_fn = getattr(drive_mod, "mount", None)
            if mount_fn is None:
                raise DriveMountError("google.colab.drive.mount is unavailable.")
            logger.info("Mounting Google Drive (force_remount=%s)", force_remount)
            mount_fn("/content/drive", force_remount=force_remount)
        except DriveMountError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface Colab mount failures
            raise DriveMountError(f"Failed to mount Google Drive: {exc}") from exc
    elif not is_drive_mounted(mount_root):
        raise DriveMountError(
            f"Not in Colab and Drive is not mounted at {mount_root}. "
            "Provide an existing drive_mount_root for local testing."
        )

    if not is_drive_mounted(mount_root):
        raise DriveMountError(f"Drive mount verification failed at {mount_root}.")

    logger.info("Drive mounted at %s", mount_root)
    return mount_root


def verify_drive_mounted(config: ColabConfig) -> Path:
    """Ensure Drive is mounted; raise ``DriveMountError`` if not."""
    mount_root = resolve_drive_mount_root(config)
    if not is_drive_mounted(mount_root):
        raise DriveMountError(f"Google Drive is not mounted at {mount_root}.")
    logger.info("Drive mount verified at %s", mount_root)
    return mount_root


def wait_for_path(
    path: Path,
    *,
    timeout: float = DEFAULT_DRIVE_SYNC_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_DRIVE_SYNC_POLL_SECONDS,
) -> bool:
    """
    Poll for ``path`` to become visible, tolerating Google Drive FUSE sync lag.

    Returns True as soon as ``path.exists()`` is true; returns False (never
    raises) once ``timeout`` elapses without the path appearing — callers
    decide whether a missing path after the wait is a hard failure. A single
    ``path.exists()`` check right after a subprocess exits is not reliable on
    Drive-mounted paths, so this must be used before treating a fresh
    Drive-written path as absent.
    """
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if path.exists():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(poll_interval, 0.0))


def require_path_synced(
    path: Path,
    *,
    timeout: float = DEFAULT_DRIVE_SYNC_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_DRIVE_SYNC_POLL_SECONDS,
    description: str | None = None,
) -> None:
    """
    Fail-closed variant of ``wait_for_path``: raises ``DriveSyncError`` instead
    of returning False when ``path`` does not appear within ``timeout``.

    Use this at call sites (packaging, validation) where a missing path after
    the sync window is an error, not a caller-decided outcome.
    """
    if wait_for_path(path, timeout=timeout, poll_interval=poll_interval):
        return
    label = description or str(path)
    raise DriveSyncError(f"{label} did not become visible on Drive within {timeout}s: {path}")


__all__ = [
    "is_colab_environment",
    "is_drive_mounted",
    "mount_google_drive",
    "require_path_synced",
    "resolve_drive_mount_root",
    "verify_drive_mounted",
    "wait_for_path",
]
