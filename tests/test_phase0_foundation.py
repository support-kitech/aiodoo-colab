"""Phase 0 smoke tests for application-layout importability."""

from __future__ import annotations

import constants
from colab_logging import get_logger
from exceptions import AiodooColabError
from version import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_workspace_constants_are_paths() -> None:
    assert constants.AIODOO_ROOT_RELATIVE.parts == ("AIODOO",)
    assert constants.TRAINING_RELATIVE.name == "training"
    assert constants.TRAINING_REPOSITORY_URL.endswith(".git")


def test_base_exception_is_exception() -> None:
    assert issubclass(AiodooColabError, Exception)


def test_get_logger_has_expected_name() -> None:
    logger = get_logger()
    assert logger.name == "aiodoo_colab"
