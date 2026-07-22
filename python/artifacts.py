"""Read-only checkpoint / artifact discovery (no training or packaging logic).

Every path here is derived from ``Workspace`` / ``naming`` — the single
canonical layout authority (mirroring aiodoo-training's
``ArtifactOutputLayout``, see ``workspace.Workspace.checkpoints_root``). This
module never writes, validates checkpoint *contents*, or decides whether a
checkpoint is safe to resume from beyond a lightweight non-empty check;
correctness of resume itself is entirely aiodoo-training's
``ResumeCoordinator`` (via ``trainer.run_training(..., resume_from=...)``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from constants import CHECKPOINT_DIR_PREFIX
from naming import adapter_product_id, normalize_training_id
from workspace import Workspace

logger = logging.getLogger("aiodoo_colab")


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """One discovered ``checkpoint-<step>`` directory (metadata only)."""

    step: int
    path: Path

    @property
    def is_nonempty(self) -> bool:
        """True when the checkpoint directory contains at least one file."""
        return self.path.is_dir() and any(self.path.iterdir())


def _parse_checkpoint_step(name: str) -> int | None:
    """Parse the step number from a ``checkpoint-<step>`` directory name."""
    if not name.startswith(CHECKPOINT_DIR_PREFIX):
        return None
    raw = name[len(CHECKPOINT_DIR_PREFIX) :]
    try:
        return int(raw)
    except ValueError:
        return None


def discover_checkpoints(checkpoints_dir: Path) -> tuple[CheckpointInfo, ...]:
    """Return all discovered checkpoints under ``checkpoints_dir``, oldest first."""
    if not checkpoints_dir.is_dir():
        return ()
    found: list[CheckpointInfo] = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        step = _parse_checkpoint_step(child.name)
        if step is not None:
            found.append(CheckpointInfo(step=step, path=child))
    found.sort(key=lambda info: info.step)
    return tuple(found)


def latest_checkpoint(checkpoints_dir: Path) -> CheckpointInfo | None:
    """Return the highest-step checkpoint under ``checkpoints_dir``, or None."""
    checkpoints = discover_checkpoints(checkpoints_dir)
    return checkpoints[-1] if checkpoints else None


def is_resumable(checkpoint: CheckpointInfo) -> bool:
    """
    Lightweight, read-only resumability heuristic.

    Only checks the checkpoint directory exists and is non-empty. Deep
    validation (RNG state, optimizer state, model fingerprint match) is
    exclusively aiodoo-training's ``ResumeCoordinator.load_and_validate`` —
    Colab never re-implements that check, it only decides *whether to try*.
    """
    return checkpoint.path.is_dir() and checkpoint.is_nonempty


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Snapshot of everything discoverable on Drive for one training id."""

    training_id: str
    adapter_dir: Path
    adapter_published: bool
    merged_dir: Path
    merged_published: bool
    export_dir: Path
    export_published: bool
    checkpoints_dir: Path
    checkpoints: tuple[CheckpointInfo, ...]
    logs_dir: Path
    log_files: tuple[Path, ...]
    metrics_dir: Path
    metric_files: tuple[Path, ...]

    @property
    def latest_checkpoint(self) -> CheckpointInfo | None:
        return self.checkpoints[-1] if self.checkpoints else None

    @property
    def resumable(self) -> bool:
        latest = self.latest_checkpoint
        return latest is not None and is_resumable(latest)


def browse_training_artifacts(workspace: Workspace, training_id: str) -> TrainingArtifacts:
    """
    Discover every artifact Drive location for ``training_id`` (read-only).

    Never interprets artifact *contents* beyond "does artifact.json exist"
    (published) — that shape is aiodoo-training's Artifact Contract, not
    something this repository parses or validates.
    """
    tid = normalize_training_id(training_id)
    adapter_id = adapter_product_id(tid)

    adapter_dir = workspace.adapters / adapter_id
    merged_dir = workspace.merged / adapter_id
    export_dir = workspace.exports / adapter_id
    checkpoints_dir = workspace.checkpoints_root(tid)
    logs_dir = workspace.experiments / tid / "logs"
    metrics_dir = workspace.experiments / tid / "metrics"

    artifacts = TrainingArtifacts(
        training_id=tid,
        adapter_dir=adapter_dir,
        adapter_published=(adapter_dir / "artifact.json").is_file(),
        merged_dir=merged_dir,
        merged_published=(merged_dir / "artifact.json").is_file(),
        export_dir=export_dir,
        export_published=export_dir.is_dir() and any(export_dir.iterdir()),
        checkpoints_dir=checkpoints_dir,
        checkpoints=discover_checkpoints(checkpoints_dir),
        logs_dir=logs_dir,
        log_files=tuple(sorted(logs_dir.glob("*"))) if logs_dir.is_dir() else (),
        metrics_dir=metrics_dir,
        metric_files=tuple(sorted(metrics_dir.glob("*"))) if metrics_dir.is_dir() else (),
    )
    logger.info(
        "Artifact browse training_id=%s adapter_published=%s checkpoints=%d resumable=%s",
        tid,
        artifacts.adapter_published,
        len(artifacts.checkpoints),
        artifacts.resumable,
    )
    return artifacts


def summarize_artifacts(artifacts: TrainingArtifacts) -> dict[str, object]:
    """Plain dict summary suitable for notebook printing."""
    latest = artifacts.latest_checkpoint
    return {
        "training_id": artifacts.training_id,
        "adapter_published": artifacts.adapter_published,
        "adapter_dir": str(artifacts.adapter_dir),
        "merged_published": artifacts.merged_published,
        "export_published": artifacts.export_published,
        "checkpoint_count": len(artifacts.checkpoints),
        "latest_checkpoint_step": latest.step if latest else None,
        "latest_checkpoint_path": str(latest.path) if latest else None,
        "resumable": artifacts.resumable,
        "log_file_count": len(artifacts.log_files),
        "metric_file_count": len(artifacts.metric_files),
    }


__all__ = [
    "CheckpointInfo",
    "TrainingArtifacts",
    "browse_training_artifacts",
    "discover_checkpoints",
    "is_resumable",
    "latest_checkpoint",
    "summarize_artifacts",
]
