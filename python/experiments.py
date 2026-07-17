"""Experiment discovery and configuration loading (no training logic)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constants import EXPERIMENT_CONFIG_DIR_NAME, EXPERIMENT_CONFIG_FILES, EXPERIMENT_ID_PATTERN
from exceptions import ExperimentNotFoundError, ExperimentValidationError
from workspace import Workspace

logger = logging.getLogger("aiodoo_colab")

_EXPERIMENT_ID_RE = re.compile(EXPERIMENT_ID_PATTERN)


def is_valid_experiment_id(experiment_id: str) -> bool:
    """Return True when ``experiment_id`` matches ``EXP-0001`` style naming."""
    return _EXPERIMENT_ID_RE.fullmatch(experiment_id) is not None


def require_valid_experiment_id(experiment_id: str) -> None:
    if not is_valid_experiment_id(experiment_id):
        raise ExperimentValidationError(
            f"Invalid experiment id {experiment_id!r}; expected pattern {EXPERIMENT_ID_PATTERN}"
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Raw dataset.yaml contents (values not interpreted)."""

    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Raw model.yaml contents (values not interpreted)."""

    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Raw training.yaml contents (values not interpreted)."""

    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Raw evaluation.yaml contents (values not interpreted)."""

    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Raw export.yaml contents (values not interpreted)."""

    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExperimentConfigs:
    """All typed config fragments for one experiment."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    export: ExportConfig


@dataclass(frozen=True, slots=True)
class Experiment:
    """Resolved experiment with paths and loaded configuration."""

    experiment_id: str
    root: Path
    config_dir: Path
    configs: ExperimentConfigs

    @property
    def dataset_version(self) -> Any:
        """Expose ``dataset_version`` from dataset.yaml when present."""
        return self.configs.dataset.data.get("dataset_version")

    @property
    def model_id(self) -> Any:
        """Expose model identity from model.yaml when present (common keys)."""
        data = self.configs.model.data
        for key in ("base_model", "model_id", "identifier"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = data.get("model")
        if isinstance(nested, dict):
            for key in ("identifier", "base_model", "model_id"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(nested, str) and nested.strip():
            return nested
        return None

    @property
    def training_configuration(self) -> dict[str, Any]:
        return dict(self.configs.training.data)

    @property
    def evaluation_configuration(self) -> dict[str, Any]:
        return dict(self.configs.evaluation.data)

    @property
    def export_configuration(self) -> dict[str, Any]:
        return dict(self.configs.export.data)

    def config(self) -> ExperimentConfigs:
        """Return the loaded configuration object."""
        return self.configs


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ExperimentValidationError(
            "PyYAML is required to load experiment configuration. "
            "Install it in the Colab / runtime environment."
        ) from exc

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentValidationError(f"Cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentValidationError(f"Invalid YAML in {path}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExperimentValidationError(f"Config root must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _config_dir_for_experiment_root(root: Path) -> Path | None:
    """
    Accept either Colab layout (``config/*.yaml``) or flat aiodoo-training layout.

    Returns the directory that holds dataset/model/training/evaluation/export YAML.
    """
    nested = root / EXPERIMENT_CONFIG_DIR_NAME
    if nested.is_dir() and all((nested / name).is_file() for name in EXPERIMENT_CONFIG_FILES):
        return nested
    if all((root / name).is_file() for name in EXPERIMENT_CONFIG_FILES):
        return root
    return None


def _canonical_production_experiment(workspace: Workspace, experiment_id: str) -> Path | None:
    """Resolve EXP from cloned aiodoo-training production configs when present."""
    candidate = (
        workspace.training_repository / "configs" / "experiments" / "production" / experiment_id
    )
    if candidate.is_dir() and _config_dir_for_experiment_root(candidate) is not None:
        return candidate
    return None


@dataclass(frozen=True, slots=True)
class ExperimentStore:
    """Discover and load experiments under ``workspace.experiments``."""

    workspace: Workspace

    def discover(self) -> list[str]:
        """Return sorted experiment ids from Drive and canonical training configs."""
        names: set[str] = set()
        root = self.workspace.experiments
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_dir() and is_valid_experiment_id(entry.name):
                    names.add(entry.name)
                elif entry.is_dir():
                    logger.info("Ignoring invalid experiment directory name: %s", entry.name)

        production = self.workspace.training_repository / "configs" / "experiments" / "production"
        if production.is_dir():
            for entry in production.iterdir():
                if (
                    entry.is_dir()
                    and is_valid_experiment_id(entry.name)
                    and _config_dir_for_experiment_root(entry) is not None
                ):
                    names.add(entry.name)

        ordered = sorted(names)
        logger.info("Discovered %d experiment(s)", len(ordered))
        return ordered

    def exists(self, experiment_id: str) -> bool:
        if not is_valid_experiment_id(experiment_id):
            return False
        drive = self.workspace.experiments / experiment_id
        if drive.is_dir() and _config_dir_for_experiment_root(drive) is not None:
            return True
        return _canonical_production_experiment(self.workspace, experiment_id) is not None

    def validate(self, experiment_id: str) -> Path:
        """
        Validate experiment and return its root path.

        Preference order:
        1. Drive ``experiments/EXP-NNNN`` (config/ or flat layout)
        2. Canonical ``aiodoo-training/configs/experiments/production/EXP-NNNN``
        """
        require_valid_experiment_id(experiment_id)
        drive = self.workspace.experiments / experiment_id
        if drive.exists():
            if not drive.is_dir():
                raise ExperimentValidationError(f"Experiment path is not a directory: {drive}")
            if _config_dir_for_experiment_root(drive) is not None:
                logger.info("Experiment validated (Drive): %s", experiment_id)
                return drive
            # Incomplete Drive output dirs (summary/config snapshot only) must not
            # block loading canonical aiodoo-training production configs.
            logger.warning(
                "Drive experiment %s is missing required config files; "
                "falling back to aiodoo-training production configs.",
                experiment_id,
            )

        canonical = _canonical_production_experiment(self.workspace, experiment_id)
        if canonical is not None:
            logger.info("Experiment validated (aiodoo-training canonical): %s", experiment_id)
            return canonical

        if drive.exists():
            raise ExperimentValidationError(
                f"Experiment {experiment_id!r} at {drive} is missing required "
                f"config files ({', '.join(EXPERIMENT_CONFIG_FILES)}) and no "
                "canonical aiodoo-training production configs were found."
            )

        raise ExperimentNotFoundError(
            f"Experiment not found: {experiment_id} "
            f"(checked Drive {drive} and aiodoo-training production configs)."
        )

    def load(self, experiment_id: str) -> Experiment:
        """Validate and load all YAML configuration fragments."""
        root = self.validate(experiment_id)
        config_dir = _config_dir_for_experiment_root(root)
        if config_dir is None:
            raise ExperimentValidationError(f"Experiment configs missing under {root}")

        dataset = DatasetConfig(data=_load_yaml(config_dir / "dataset.yaml"))
        model = ModelConfig(data=_load_yaml(config_dir / "model.yaml"))
        training = TrainingConfig(data=_load_yaml(config_dir / "training.yaml"))
        evaluation = EvaluationConfig(data=_load_yaml(config_dir / "evaluation.yaml"))
        export = ExportConfig(data=_load_yaml(config_dir / "export.yaml"))

        experiment = Experiment(
            experiment_id=experiment_id,
            root=root,
            config_dir=config_dir,
            configs=ExperimentConfigs(
                dataset=dataset,
                model=model,
                training=training,
                evaluation=evaluation,
                export=export,
            ),
        )
        logger.info("Experiment loaded: %s (root=%s)", experiment_id, root)
        return experiment


__all__ = [
    "DatasetConfig",
    "EvaluationConfig",
    "Experiment",
    "ExperimentConfigs",
    "ExperimentStore",
    "ExportConfig",
    "ModelConfig",
    "TrainingConfig",
    "is_valid_experiment_id",
    "require_valid_experiment_id",
]
