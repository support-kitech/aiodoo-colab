"""Compatibility aliases for training orchestration (see ``trainer``)."""

from __future__ import annotations

from trainer import (
    TrainingContext,
    TrainingResult,
    build_training_context,
    prepare_resume_config,
    resolve_resume_checkpoint,
    run_training,
    summarize_result,
)

__all__ = [
    "TrainingContext",
    "TrainingResult",
    "build_training_context",
    "prepare_resume_config",
    "resolve_resume_checkpoint",
    "run_training",
    "summarize_result",
]
