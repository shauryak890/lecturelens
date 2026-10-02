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
SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture(autouse=True)
def _isolated_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI tests load the real config: keep their log files out of the project's logs/."""
    monkeypatch.setenv("LECTURELENS__APP__LOG_DIR", str(tmp_path / "logs"))


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


@pytest.fixture
def tiny_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Settings whose index and cache live in a temp dir."""
    return settings.with_overrides(
        {
            "app.index_dir": str(tmp_path / "index"),
            "app.log_dir": str(tmp_path / "logs"),
            "llm.cache.path": str(tmp_path / "cache.sqlite"),
        }
    )


@pytest.fixture
def tiny_index(tiny_settings: Settings):  # returns an Indexer (imported lazily)
    """The three sample notes indexed with HashEmbedder + WhitespaceCounter (no downloads)."""
    from lecturelens.indexing.indexer import Indexer

    from .fakes import HashEmbedder, WhitespaceCounter

    indexer = Indexer(tiny_settings, HashEmbedder(dim=128), WhitespaceCounter())
    indexer.ingest(SAMPLE_DIR)
    return indexer
