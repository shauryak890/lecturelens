"""Shared pytest fixtures. Everything here runs offline."""

from pathlib import Path

import pytest

from lecturelens.config import Settings, load_settings
from lecturelens.llm.prompts import PromptRegistry

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPTS_PATH = ROOT / "prompts" / "prompts.yaml"


@pytest.fixture
def settings() -> Settings:
    """Settings from the real config file, isolated from the process env and .env."""
    return load_settings(CONFIG_PATH, environ={}, dotenv_path=None)


@pytest.fixture(scope="session")
def registry() -> PromptRegistry:
    """PromptRegistry over the real prompts file."""
    return PromptRegistry(PROMPTS_PATH)
