"""Workspace path constants for Google Drive layouts."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Google Drive mount (Colab default)
# ---------------------------------------------------------------------------

GOOGLE_DRIVE_ROOT_NAME: str = "MyDrive/colab_notebooks"

# Typical Colab mount root: /content/drive/MyDrive
DEFAULT_DRIVE_MOUNT_RELATIVE: Path = Path("drive") / GOOGLE_DRIVE_ROOT_NAME

# ---------------------------------------------------------------------------
# AIODOO workspace tree (relative to Drive mount root)
# ---------------------------------------------------------------------------

AIODOO_ROOT_NAME: str = "AIODOO"

# Top-level directories (validated / auto-created).
DATASETS_DIR_NAME: str = "datasets"
MODELS_DIR_NAME: str = "models"
EXPERIMENTS_DIR_NAME: str = "experiments"
LOGS_DIR_NAME: str = "logs"
TRAINING_DIR_NAME: str = "training"

# Required subdirectories under models/ (created when missing).
# ``base`` remains for Drive layout compatibility; Hugging Face base models are
# cached on the Colab local SSD via ``DEFAULT_MODEL_CACHE_ROOT`` / ``Workspace.model_cache``.
MODELS_BASE_DIR_NAME: str = "base"
MODELS_ADAPTERS_DIR_NAME: str = "adapters"
MODELS_MERGED_DIR_NAME: str = "merged"
MODELS_EXPORTS_DIR_NAME: str = "exports"

REQUIRED_TOP_LEVEL_DIRS: tuple[str, ...] = (
    DATASETS_DIR_NAME,
    MODELS_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    LOGS_DIR_NAME,
    TRAINING_DIR_NAME,
)

REQUIRED_MODELS_SUBDIRS: tuple[str, ...] = (
    MODELS_BASE_DIR_NAME,
    MODELS_ADAPTERS_DIR_NAME,
    MODELS_MERGED_DIR_NAME,
    MODELS_EXPORTS_DIR_NAME,
)

# Composed relative Path fragments (relative to Drive mount root).
AIODOO_ROOT_RELATIVE: Path = Path(AIODOO_ROOT_NAME)
DATASETS_RELATIVE: Path = AIODOO_ROOT_RELATIVE / DATASETS_DIR_NAME
MODELS_RELATIVE: Path = AIODOO_ROOT_RELATIVE / MODELS_DIR_NAME
EXPERIMENTS_RELATIVE: Path = AIODOO_ROOT_RELATIVE / EXPERIMENTS_DIR_NAME
LOGS_RELATIVE: Path = AIODOO_ROOT_RELATIVE / LOGS_DIR_NAME
TRAINING_RELATIVE: Path = AIODOO_ROOT_RELATIVE / TRAINING_DIR_NAME

# Upstream training repository (cloned into training/).
TRAINING_REPOSITORY_NAME: str = "aiodoo-training"
TRAINING_REPOSITORY_URL: str = "https://github.com/support-kitech/aiodoo-training.git"

# Markers used to verify a cloned training repository.
TRAINING_REPOSITORY_MARKERS: tuple[str, ...] = (
    "train.py",
    "pyproject.toml",
    "README.md",
)

# ---------------------------------------------------------------------------
# Hugging Face base model cache (Colab local SSD — not Google Drive)
# ---------------------------------------------------------------------------

# Default Colab runtime path for HF base-model snapshots.
# Adapters / checkpoints / merged / exports remain under AIODOO/models/ on Drive.
DEFAULT_MODEL_CACHE_ROOT: Path = Path("/content/aiodoo-model-cache")

# Required marker files for a usable Hugging Face model snapshot.
MODEL_REQUIRED_FILES: tuple[str, ...] = ("config.json",)

# At least one weight file matching these suffixes must exist.
MODEL_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".pt",
    ".gguf",
)

# ---------------------------------------------------------------------------
# Experiment configuration (under experiments/<id>/config/)
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG_DIR_NAME: str = "config"
EXPERIMENT_CONFIG_FILES: tuple[str, ...] = (
    "dataset.yaml",
    "model.yaml",
    "training.yaml",
    "evaluation.yaml",
    "export.yaml",
)

# Accepted public training ids (semantic) plus legacy EXP-NNNN for migration.
# Prefer TRAINING_ID_PATTERN from naming.py for validation.
EXPERIMENT_ID_PATTERN: str = (
    r"^(?:coding|planner|context|conversation|repair|execution|approval|evaluation|"
    r"EXP-\d{4})$"
)
# Alias — public surface uses training ids.
TRAINING_ID_PATTERN: str = EXPERIMENT_ID_PATTERN

# Public root entrypoint inside aiodoo-training (application layout).
TRAINING_PUBLIC_ENTRYPOINT: str = "train.py"

__all__ = [
    "AIODOO_ROOT_NAME",
    "AIODOO_ROOT_RELATIVE",
    "DATASETS_DIR_NAME",
    "DATASETS_RELATIVE",
    "DEFAULT_DRIVE_MOUNT_RELATIVE",
    "DEFAULT_MODEL_CACHE_ROOT",
    "EXPERIMENTS_DIR_NAME",
    "EXPERIMENTS_RELATIVE",
    "GOOGLE_DRIVE_ROOT_NAME",
    "LOGS_DIR_NAME",
    "LOGS_RELATIVE",
    "EXPERIMENT_CONFIG_DIR_NAME",
    "EXPERIMENT_CONFIG_FILES",
    "EXPERIMENT_ID_PATTERN",
    "TRAINING_ID_PATTERN",
    "MODELS_ADAPTERS_DIR_NAME",
    "MODELS_BASE_DIR_NAME",
    "MODELS_DIR_NAME",
    "MODELS_EXPORTS_DIR_NAME",
    "MODELS_MERGED_DIR_NAME",
    "MODELS_RELATIVE",
    "MODEL_REQUIRED_FILES",
    "MODEL_WEIGHT_SUFFIXES",
    "REQUIRED_MODELS_SUBDIRS",
    "REQUIRED_TOP_LEVEL_DIRS",
    "TRAINING_DIR_NAME",
    "TRAINING_PUBLIC_ENTRYPOINT",
    "TRAINING_RELATIVE",
    "TRAINING_REPOSITORY_MARKERS",
    "TRAINING_REPOSITORY_NAME",
    "TRAINING_REPOSITORY_URL",
]
