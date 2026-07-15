"""Clone and manage the aiodoo-training repository on Drive."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import ColabConfig
from constants import TRAINING_REPOSITORY_MARKERS
from exceptions import (
    RepositoryCheckoutError,
    RepositoryCloneError,
    RepositoryError,
    RepositoryUpdateError,
)
from workspace import Workspace

logger = logging.getLogger("aiodoo_colab")

_GIT_TIMEOUT_SECONDS: int = 600


@dataclass(frozen=True, slots=True)
class TrainingRepository:
    """Manage the frozen ``aiodoo-training`` git checkout under the workspace."""

    path: Path
    remote_url: str
    default_branch: str

    @classmethod
    def from_workspace(cls, workspace: Workspace, config: ColabConfig) -> TrainingRepository:
        return cls(
            path=workspace.training_repository,
            remote_url=config.training_repository_url,
            default_branch=config.default_branch,
        )

    def exists(self) -> bool:
        """Return True when the repository directory contains a ``.git`` folder."""
        git_dir = self.path / ".git"
        return self.path.is_dir() and git_dir.is_dir()

    def _run_git(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        work_dir = cwd or self.path
        command = ["git", *args]
        logger.info("Running git command in %s: %s", work_dir, " ".join(command))
        try:
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepositoryError(f"Git command timed out: {' '.join(command)}") from exc
        except OSError as exc:
            raise RepositoryError(f"Failed to execute git: {exc}") from exc

        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            raise RepositoryError(f"Git command failed: {' '.join(command)} — {detail}")
        return result

    def verify(self) -> None:
        """
        Verify repository integrity and usability.

        Checks:
        - repository directory exists
        - ``.git`` directory exists
        - required repository markers exist
        - git reports a usable work tree with a resolvable ``HEAD``

        Raises ``RepositoryError`` when verification fails.
        """
        if not self.path.exists():
            raise RepositoryError(f"Repository directory does not exist: {self.path}")
        if not self.path.is_dir():
            raise RepositoryError(f"Repository path is not a directory: {self.path}")

        git_dir = self.path / ".git"
        if not git_dir.exists():
            raise RepositoryError(f"Repository missing .git at {self.path}")
        if not git_dir.is_dir():
            raise RepositoryError(f"Repository .git is not a directory at {self.path}")

        for marker in TRAINING_REPOSITORY_MARKERS:
            marker_path = self.path / marker
            if not marker_path.is_file():
                raise RepositoryError(
                    f"Repository verification failed: missing marker {marker!r} at {self.path}"
                )

        try:
            inside = self._run_git(["rev-parse", "--is-inside-work-tree"], check=True)
            if (inside.stdout or "").strip().lower() != "true":
                raise RepositoryError(f"Repository is not a usable git work tree at {self.path}")
            head = self._run_git(["rev-parse", "HEAD"], check=True)
            head_sha = (head.stdout or "").strip()
            if not head_sha:
                raise RepositoryError(f"Repository HEAD could not be resolved at {self.path}")
        except RepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RepositoryError(
                f"Repository usability check failed at {self.path}: {exc}"
            ) from exc

        remote = self._run_git(["remote", "get-url", "origin"], check=True)
        origin = (remote.stdout or "").strip()
        if self.remote_url.rstrip("/") not in origin and origin not in self.remote_url:
            logger.warning(
                "Repository origin %r does not exactly match configured %r",
                origin,
                self.remote_url,
            )

        logger.info("Repository verified at %s (HEAD=%s)", self.path, head_sha[:12])

    def clone(self, *, branch: str | None = None) -> None:
        """
        Clone ``aiodoo-training`` into ``self.path``.

        Raises ``RepositoryCloneError`` when clone fails or path already exists.
        """
        if self.path.exists():
            if not self.path.is_dir():
                raise RepositoryCloneError(
                    f"Cannot clone: {self.path} exists but is not a git repository."
                )
            if any(self.path.iterdir()):
                if self.exists():
                    logger.info("Repository already exists at %s", self.path)
                    return
                raise RepositoryCloneError(
                    f"Cannot clone: {self.path} exists but is not a git repository."
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        target_branch = branch or self.default_branch
        args = [
            "clone",
            "--branch",
            target_branch,
            self.remote_url,
            str(self.path),
        ]
        try:
            self._run_git(args, cwd=self.path.parent, check=True)
        except RepositoryError as exc:
            raise RepositoryCloneError(str(exc)) from exc

        logger.info("Repository cloned to %s (branch=%s)", self.path, target_branch)
        self.verify()

    def update(self) -> None:
        """
        Run ``git pull`` on an existing repository.

        Raises ``RepositoryUpdateError`` when update fails.
        """
        if not self.exists():
            raise RepositoryUpdateError(f"Cannot update: repository missing at {self.path}")
        try:
            self._run_git(["pull", "--ff-only"], check=True)
        except RepositoryError as exc:
            raise RepositoryUpdateError(str(exc)) from exc
        logger.info("Repository updated at %s", self.path)
        self.verify()

    def checkout(
        self,
        *,
        branch: str | None = None,
        tag: str | None = None,
        commit: str | None = None,
    ) -> None:
        """
        Checkout branch, tag, or commit (exactly one should be provided).

        Raises ``RepositoryCheckoutError`` on failure.
        """
        if not self.exists():
            raise RepositoryCheckoutError(f"Cannot checkout: repository missing at {self.path}")

        targets = [value for value in (branch, tag, commit) if value is not None]
        if len(targets) != 1:
            raise RepositoryCheckoutError(
                "Exactly one of branch, tag, or commit must be provided for checkout."
            )

        try:
            if branch is not None:
                self._run_git(["checkout", branch], check=True)
                logger.info("Checked out branch %r at %s", branch, self.path)
            elif tag is not None:
                self._run_git(["checkout", tag], check=True)
                logger.info("Checked out tag %r at %s", tag, self.path)
            else:
                assert commit is not None
                self._run_git(["checkout", commit], check=True)
                logger.info("Checked out commit %r at %s", commit, self.path)
        except RepositoryError as exc:
            raise RepositoryCheckoutError(str(exc)) from exc

        self.verify()


__all__ = ["TrainingRepository"]
