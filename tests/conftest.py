"""Shared pytest fixtures. Everything here runs offline."""

import logging
from pathlib import Path

import pytest

from lecturelens.config import Settings, load_settings
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.logging_utils import PACKAGE_LOGGER

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPTS_PATH = ROOT / "prompts" / "prompts.yaml"


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """Undo setup_logging() after each test so handlers never point at closed test streams."""
    yield
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.fixture
def settings() -> Settings:
    """Settings from the real config file, isolated from the process env and .env."""
    return load_settings(CONFIG_PATH, environ={}, dotenv_path=None)


@pytest.fixture(scope="session")
def registry() -> PromptRegistry:
    """PromptRegistry over the real prompts file."""
    return PromptRegistry(PROMPTS_PATH)
