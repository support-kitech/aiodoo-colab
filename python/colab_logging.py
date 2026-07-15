"""Logging configuration helpers (no side effects on import)."""

from __future__ import annotations

import logging
from typing import Final

DEFAULT_LOGGER_NAME: Final[str] = "aiodoo_colab"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return a named logger without installing handlers."""
    return logging.getLogger(name)


def configure_logging(
    *,
    level: int = logging.INFO,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """
    Idempotent logging setup for Colab / CLI entrypoints.

    Phase 0: configures a stream handler when none are present.
    Does not run at import time.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


__all__ = [
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOGGER_NAME",
    "configure_logging",
    "get_logger",
]
