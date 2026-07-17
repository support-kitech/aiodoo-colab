"""Colab public training identifiers (mirrors aiodoo_training.naming).

Numeric EXP ids are accepted only for migration and normalize to semantic ids.
"""

from __future__ import annotations

from typing import Final

TRAINING_IDS: Final[tuple[str, ...]] = (
    "coding",
    "planner",
    "context",
    "conversation",
    "repair",
    "execution",
    "approval",
    "evaluation",
)

_TRAINING_ID_SET: Final[frozenset[str]] = frozenset(TRAINING_IDS)

ADAPTER_PRODUCT_PREFIX: Final[str] = "aiodoo-"

LEGACY_INTERNAL_ID_TO_TRAINING_ID: Final[dict[str, str]] = {
    "EXP-0001": "coding",
}

TRAINING_ID_TO_INTERNAL_ID: Final[dict[str, str]] = {
    training_id: internal_id
    for internal_id, training_id in LEGACY_INTERNAL_ID_TO_TRAINING_ID.items()
}

TRAINING_CONFIG_ROOT: Final[str] = "configs/training"
LEGACY_EXPERIMENT_CONFIG_ROOT: Final[str] = "configs/experiments/production"

# Public training id pattern (semantic) + legacy EXP for migration acceptance.
TRAINING_ID_PATTERN: Final[str] = (
    r"^(?:coding|planner|context|conversation|repair|execution|approval|evaluation|"
    r"EXP-\d{4})$"
)


def is_training_id(value: str) -> bool:
    return value in _TRAINING_ID_SET


def normalize_training_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("training id must be a non-empty string")
    if is_training_id(raw):
        return raw
    if raw in LEGACY_INTERNAL_ID_TO_TRAINING_ID:
        return LEGACY_INTERNAL_ID_TO_TRAINING_ID[raw]
    if raw.startswith("EXP-"):
        raise ValueError(
            f"Unknown legacy internal id {raw!r}; known: "
            f"{sorted(LEGACY_INTERNAL_ID_TO_TRAINING_ID)}"
        )
    raise ValueError(f"Unknown training id {raw!r}; expected one of {list(TRAINING_IDS)}")


def adapter_product_id(training_id: str) -> str:
    return f"{ADAPTER_PRODUCT_PREFIX}{normalize_training_id(training_id)}"


def stage_display_name(training_id: str) -> str:
    return normalize_training_id(training_id).replace("_", " ").title()


def is_acceptable_training_ref(value: str) -> bool:
    """True for semantic training ids or known legacy EXP ids."""
    try:
        normalize_training_id(value)
        return True
    except ValueError:
        return False


__all__ = [
    "ADAPTER_PRODUCT_PREFIX",
    "LEGACY_EXPERIMENT_CONFIG_ROOT",
    "LEGACY_INTERNAL_ID_TO_TRAINING_ID",
    "TRAINING_CONFIG_ROOT",
    "TRAINING_IDS",
    "TRAINING_ID_PATTERN",
    "TRAINING_ID_TO_INTERNAL_ID",
    "adapter_product_id",
    "is_acceptable_training_ref",
    "is_training_id",
    "normalize_training_id",
    "stage_display_name",
]
