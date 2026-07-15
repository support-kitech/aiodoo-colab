"""Unit tests for aiodoo-training repository management (mocked git)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from config import load_config
from exceptions import (
    RepositoryCheckoutError,
    RepositoryCloneError,
    RepositoryError,
    RepositoryUpdateError,
)
from repository import TrainingRepository
from workspace import Workspace


@pytest.fixture
def repo(tmp_path: Path) -> TrainingRepository:
    mount = tmp_path / "MyDrive"
    config = load_config(drive_mount_root=mount, auto_mount_drive=False)
    ws = Workspace.from_config(config)
    ws.training.mkdir(parents=True)
    return TrainingRepository.from_workspace(ws, config)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> CompletedProcess[str]:
    return CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr=stderr)


def _seed_repo(repo: TrainingRepository) -> None:
    repo.path.mkdir(parents=True, exist_ok=True)
    (repo.path / ".git").mkdir(exist_ok=True)
    for marker in ("train.py", "pyproject.toml", "README.md"):
        (repo.path / marker).write_text("x", encoding="utf-8")


def _git_side_effect(
    repo: TrainingRepository,
    recorded: list[list[str]] | None = None,
):
    def fake_run_git(
        args: list[str], *, cwd: Path | None = None, check: bool = True
    ) -> CompletedProcess[str]:
        del cwd, check
        if recorded is not None:
            recorded.append(args)
        if args[:2] == ["git", "clone"] or args[0] == "clone":
            _seed_repo(repo)
            return _completed(stdout="cloned")
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return _completed(stdout="true\n")
        if args == ["rev-parse", "HEAD"]:
            return _completed(stdout="abc123deadbeef\n")
        if args[:2] == ["remote", "get-url"] or args == ["remote", "get-url", "origin"]:
            return _completed(stdout=repo.remote_url)
        if args[0] == "pull":
            return _completed()
        if args[0] == "checkout":
            return _completed()
        return _completed()

    return fake_run_git


def test_exists_false_when_missing(repo: TrainingRepository) -> None:
    assert repo.exists() is False


def test_exists_true_when_git_present(repo: TrainingRepository) -> None:
    _seed_repo(repo)
    assert repo.exists() is True


def test_clone_invokes_git_and_verifies(repo: TrainingRepository) -> None:
    recorded: list[list[str]] = []
    with patch.object(TrainingRepository, "_run_git", side_effect=_git_side_effect(repo, recorded)):
        repo.clone()
    assert any(args and args[0] == "clone" for args in recorded)
    assert repo.exists()


def test_clone_raises_when_path_is_non_git_file(repo: TrainingRepository) -> None:
    repo.path.parent.mkdir(parents=True, exist_ok=True)
    repo.path.write_text("blocked", encoding="utf-8")

    with pytest.raises(RepositoryCloneError, match="not a git repository"):
        repo.clone()


def test_update_raises_when_missing(repo: TrainingRepository) -> None:
    with pytest.raises(RepositoryUpdateError, match="missing"):
        repo.update()


def test_update_invokes_git_pull(repo: TrainingRepository) -> None:
    _seed_repo(repo)
    recorded: list[list[str]] = []
    with patch.object(TrainingRepository, "_run_git", side_effect=_git_side_effect(repo, recorded)):
        repo.update()
    assert ["pull", "--ff-only"] in recorded


def test_checkout_branch(repo: TrainingRepository) -> None:
    _seed_repo(repo)
    recorded: list[list[str]] = []
    with patch.object(TrainingRepository, "_run_git", side_effect=_git_side_effect(repo, recorded)):
        repo.checkout(branch="develop")
    assert ["checkout", "develop"] in recorded


def test_checkout_requires_exactly_one_target(repo: TrainingRepository) -> None:
    _seed_repo(repo)
    with pytest.raises(RepositoryCheckoutError, match="Exactly one"):
        repo.checkout(branch="main", tag="v1")


def test_verify_raises_when_marker_missing(repo: TrainingRepository) -> None:
    repo.path.mkdir(parents=True)
    (repo.path / ".git").mkdir()
    with pytest.raises(RepositoryError, match="missing marker"):
        repo.verify()


def test_verify_requires_usable_work_tree(repo: TrainingRepository) -> None:
    _seed_repo(repo)

    def fake_run_git(
        args: list[str], *, cwd: Path | None = None, check: bool = True
    ) -> CompletedProcess[str]:
        del cwd, check
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return _completed(stdout="false\n")
        return _completed(stdout="x")

    with patch.object(TrainingRepository, "_run_git", side_effect=fake_run_git):
        with pytest.raises(RepositoryError, match="usable git work tree"):
            repo.verify()
