"""Evaluation orchestration via aiodoo-validation (no validation logic here).

Every request is built and every run is executed through
``aiodoo_validation.api`` exclusively — this module never scores, certifies,
or reports on a model itself. See the "Colab notebook" integration example in
``aiodoo-validation/docs/integration.md``.

``aiodoo_validation`` is a dependency-free application-layout package (see
its README: "no PyPI wheel packaging"); it is imported directly assuming the
Colab runtime has it on ``sys.path`` (its own ``colab_integration_hints()``
documents ``pip install -e ../aiodoo-validation`` for a source checkout
alongside this repository). This module never clones or vendors it — Colab
orchestrates *calls*, not the validation framework's lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from exceptions import ValidationIntegrationError
from naming import normalize_training_id
from trainer import TrainingContext, TrainingResult

if TYPE_CHECKING:
    # ValidationRunResult is not re-exported from aiodoo_validation.api (its
    # stable public surface, see that package's __init__.py) — only
    # ValidationService.validate()'s return type. Importing the concrete type
    # from its domain module here is for static typing only; this module
    # still calls exclusively through the api.ValidationService.validate()
    # entry point at runtime.
    from aiodoo_validation.domain.result import ValidationRunResult

logger = logging.getLogger("aiodoo_colab")

DEFAULT_EXECUTION_TIER: str = "standard"
DEFAULT_ODOO_VERSIONS: tuple[int, ...] = (17, 18, 19)

# Training ids with a request builder in aiodoo_validation.api. "repair" is a
# validation profile (aiodoo_validation.api.list_profiles() includes it) but
# the API ships no build_repair_request — a gap in aiodoo-validation itself,
# not something Colab may invent a workaround for (see ECOSYSTEM_ADOPTION.md
# Section 5). "context" is not a validation profile at all (mirrors the same
# documented gap in aiodoo-training's own CONTRACT_ADOPTION.md Section 7).
_SUPPORTED_PROFILE_BUILDERS: tuple[str, ...] = (
    "coding",
    "planner",
    "approval",
    "conversation",
    "evaluation",
    "execution",
)


def _import_validation_api() -> Any:
    try:
        from aiodoo_validation import api  # noqa: PLC0415
    except ImportError as exc:
        raise ValidationIntegrationError(
            "aiodoo_validation is not importable. Install it alongside "
            "aiodoo-colab, e.g. `pip install -e ../aiodoo-validation`, or add "
            "its repository root to sys.path before calling run_validation()."
        ) from exc
    return api


def _profile_builder(api: Any, training_id: str) -> Any:
    if training_id not in _SUPPORTED_PROFILE_BUILDERS:
        raise ValidationIntegrationError(
            f"aiodoo-validation has no request builder for training id {training_id!r}. "
            f"Supported: {', '.join(_SUPPORTED_PROFILE_BUILDERS)}."
        )
    builder_name = f"build_{training_id}_request"
    builder = getattr(api, builder_name, None)
    if builder is None:  # pragma: no cover - guarded by _SUPPORTED_PROFILE_BUILDERS above
        raise ValidationIntegrationError(f"aiodoo_validation.api has no {builder_name!r}.")
    return builder


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Colab-facing summary of one aiodoo-validation run (no re-scoring)."""

    training_id: str
    successful: bool
    certified: bool
    run_result: ValidationRunResult


def resolve_validation_refs(
    context: TrainingContext,
    result: TrainingResult,
) -> tuple[str, str, str | None]:
    """
    Resolve ``(base_model_ref, adapter_ref, merged_model_ref)`` from a
    completed training run's own context/result paths — never re-derives
    them independently, so validation always inspects exactly what training
    produced.
    """
    base_model_ref = str(context.model_path)
    adapter_ref = str(result.adapter_path)
    merged_model_ref = str(context.merged_output) if context.merged_output.is_dir() else None
    return base_model_ref, adapter_ref, merged_model_ref


def run_validation(
    context: TrainingContext,
    result: TrainingResult,
    *,
    execution_tier: str = DEFAULT_EXECUTION_TIER,
    odoo_versions: tuple[int, ...] | str = DEFAULT_ODOO_VERSIONS,
    run_id: str | None = None,
) -> ValidationOutcome:
    """
    Run one aiodoo-validation evaluation for a completed training run.

    Fails closed before ever calling into aiodoo-validation when the
    training run itself did not succeed — there is nothing meaningful to
    certify. Once invoked, ``ValidationService.validate()`` never raises for
    validation *outcomes* (a failed/uncertified run is a legitimate result,
    not an integration error) — only environment/argument problems raise
    ``ValidationIntegrationError`` here.
    """
    if not result.success:
        raise ValidationIntegrationError(
            f"Refusing to validate a failed training run for "
            f"{context.experiment.experiment_id!r} (exit_code={result.exit_code}): "
            f"{result.message}"
        )

    training_id = normalize_training_id(context.experiment.experiment_id)
    api = _import_validation_api()
    builder = _profile_builder(api, training_id)

    base_model_ref, adapter_ref, merged_model_ref = resolve_validation_refs(context, result)
    request = builder(
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref,
        merged_model_ref=merged_model_ref,
        execution_tier=execution_tier,
        odoo_versions=odoo_versions,
        run_id=run_id,
    )

    logger.info(
        "Validation start training_id=%s execution_tier=%s adapter_ref=%s",
        training_id,
        execution_tier,
        adapter_ref,
    )
    service = api.ValidationService.create_default()
    run_result = service.validate(request)

    outcome = ValidationOutcome(
        training_id=training_id,
        successful=bool(api.is_successful(run_result)),
        certified=bool(api.is_certified(run_result)),
        run_result=run_result,
    )
    logger.info(
        "Validation finish training_id=%s successful=%s certified=%s",
        training_id,
        outcome.successful,
        outcome.certified,
    )
    return outcome


def summarize_validation(outcome: ValidationOutcome) -> dict[str, object]:
    """Plain dict summary suitable for notebook printing."""
    run_context = outcome.run_result.run_context
    return {
        "training_id": outcome.training_id,
        "successful": outcome.successful,
        "certified": outcome.certified,
        "exit_status": str(outcome.run_result.exit_status),
        "message": outcome.run_result.message,
        "errors": list(run_context.errors),
        "warnings": list(run_context.warnings),
    }


__all__ = [
    "DEFAULT_EXECUTION_TIER",
    "DEFAULT_ODOO_VERSIONS",
    "ValidationOutcome",
    "resolve_validation_refs",
    "run_validation",
    "summarize_validation",
]
