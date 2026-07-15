"""Unit tests for Google Drive integration (mocked — no Colab required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import load_config
from drive import (
    is_colab_environment,
    is_drive_mounted,
    mount_google_drive,
    verify_drive_mounted,
)
from exceptions import DriveMountError


@pytest.fixture
def drive_config(tmp_path: Path) -> object:
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    return load_config(drive_mount_root=mount, auto_mount_drive=False)


def test_is_drive_mounted_true_when_directory_exists(drive_config: object) -> None:
    assert is_drive_mounted(drive_config.drive_mount_root) is True


def test_is_drive_mounted_false_when_missing(tmp_path: Path) -> None:
    assert is_drive_mounted(tmp_path / "missing") is False


def test_verify_drive_mounted_succeeds(drive_config: object) -> None:
    root = verify_drive_mounted(drive_config)
    assert root == drive_config.drive_mount_root


def test_verify_drive_mounted_raises_when_missing(tmp_path: Path) -> None:
    config = load_config(drive_mount_root=tmp_path / "nope", auto_mount_drive=False)
    with pytest.raises(DriveMountError, match="not mounted"):
        verify_drive_mounted(config)


def test_mount_google_drive_uses_existing_mount(drive_config: object) -> None:
    root = mount_google_drive(drive_config)
    assert root == drive_config.drive_mount_root


def test_mount_google_drive_raises_when_not_colab_and_missing(tmp_path: Path) -> None:
    config = load_config(drive_mount_root=tmp_path / "nope", auto_mount_drive=True)
    with patch("drive.is_colab_environment", return_value=False):
        with pytest.raises(DriveMountError, match="Not in Colab"):
            mount_google_drive(config)


def test_mount_google_drive_calls_colab_mount(tmp_path: Path) -> None:
    mount_root = tmp_path / "MyDrive"
    config = load_config(drive_mount_root=mount_root, auto_mount_drive=True)

    def _fake_mount(path: str, *, force_remount: bool = False) -> None:
        del force_remount, path
        mount_root.mkdir(parents=True, exist_ok=True)

    fake_drive = MagicMock()
    fake_drive.mount = _fake_mount

    with patch("drive.is_colab_environment", return_value=True):
        with patch("drive.is_drive_mounted", side_effect=[False, True]):
            with patch("importlib.import_module", return_value=fake_drive):
                root = mount_google_drive(config)
    assert root == mount_root


def test_is_colab_environment_false_outside_colab() -> None:
    assert is_colab_environment() is False
