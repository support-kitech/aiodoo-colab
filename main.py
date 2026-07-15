#!/usr/bin/env python3
"""aiodoo-colab entrypoint — thin orchestration only (no training logic).

Run from the repository root after ensuring ``python/`` is on ``PYTHONPATH``
(the Colab notebook is responsible for path setup; do not use ``pip install``).
"""

from __future__ import annotations

from colab_logging import configure_logging, get_logger
from config import load_config
from repository import TrainingRepository
from workspace import prepare_workspace


def main(argv: list[str] | None = None) -> int:
    """
    Phase 1–2 orchestration (model / experiment phases are APIs — invoke from
    notebooks when needed):

    initialize logging → load configuration → prepare Drive workspace →
    ensure training repository → exit

    Does not download models, load experiments, or invoke training.
    """
    del argv  # reserved for future CLI flags

    configure_logging()
    logger = get_logger()

    config = load_config()
    logger.info("Configuration loaded (drive root=%s)", config.drive_mount_root)

    workspace = prepare_workspace(config)
    logger.info("Workspace ready at %s", workspace.root)

    repo = TrainingRepository.from_workspace(workspace, config)
    if repo.exists():
        logger.info("Training repository exists at %s", repo.path)
        repo.update()
    else:
        repo.clone()
    repo.verify()

    logger.info("aiodoo-colab Phase 1+2 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
