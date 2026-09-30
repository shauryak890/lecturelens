"""Tests for config.py: loading, env overrides, validation, snapshot and API key handling."""

import json
from pathlib import Path

import pytest
import yaml

from lecturelens.config import Settings, load_settings
from lecturelens.errors import ConfigError

from .conftest import CONFIG_PATH


def _write_config(tmp_path: Path, **section_updates: dict) -> Path:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for section, updates in section_updates.items():
        data[section].update(updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_yaml_loads_into_settings(settings: Settings) -> None:
    assert settings.app.name == "LectureLens"
    assert settings.retrieval.mode == "hybrid"
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.weights == {"dense": 1.0, "bm25": 1.0}
    assert settings.retrieval.bm25.k1 == 1.5
    assert settings.ingestion.chunk_size_tokens == 400
    assert settings.llm.cache.enabled is True
    assert settings.app.index_dir == Path("data/index")


def test_ablations_expose_dotted_overrides(settings: Settings) -> None:
    names = [a.name for a in settings.eval.ablations]
    assert names == ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"]
    assert settings.eval.ablations[0].overrides == {
        "retrieval.mode": "bm25",
        "retrieval.rerank": False,
    }


def test_env_override_string() -> None:
    env = {"LECTURELENS__LLM__MODEL": "gemini-test-flash"}
    s = load_settings(CONFIG_PATH, environ=env, dotenv_path=None)
    assert s.llm.model == "gemini-test-flash"


def test_env_override_is_typed_and_nested() -> None:
    env = {
        "LECTURELENS__RETRIEVAL__RERANK": "false",
        "LECTURELENS__RETRIEVAL__FINAL_K": "7",
        "LECTURELENS__RETRIEVAL__BM25__K1": "1.2",
        "LECTURELENS__LLM__CACHE__ENABLED": "no",
    }
    s = load_settings(CONFIG_PATH, environ=env, dotenv_path=None)
    assert s.retrieval.rerank is False
    assert s.retrieval.final_k == 7
    assert s.retrieval.bm25.k1 == 1.2
    assert s.llm.cache.enabled is False


def test_unrelated_env_vars_ignored() -> None:
    s = load_settings(CONFIG_PATH, environ={"PATH": "x", "LECTURELENS_X": "1"}, dotenv_path=None)
    assert s.llm.model


def test_env_override_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match="retrieval.no_such_key"):
        load_settings(
            CONFIG_PATH, environ={"LECTURELENS__RETRIEVAL__NO_SUCH_KEY": "1"}, dotenv_path=None
        )


def test_env_override_unknown_section_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown config section"):
        load_settings(CONFIG_PATH, environ={"LECTURELENS__NOPE__KEY": "1"}, dotenv_path=None)


def test_invalid_mode_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, retrieval={"mode": "semantic"})
    with pytest.raises(ConfigError, match="retrieval.mode"):
        load_settings(path, environ={}, dotenv_path=None)


def test_invalid_mode_via_env_rejected() -> None:
    with pytest.raises(ConfigError, match="retrieval.mode"):
        load_settings(
            CONFIG_PATH, environ={"LECTURELENS__RETRIEVAL__MODE": "semantic"}, dotenv_path=None
        )


def test_overlap_must_be_smaller_than_chunk(tmp_path: Path) -> None:
    path = _write_config(tmp_path, ingestion={"chunk_overlap_tokens": 400})
    with pytest.raises(ConfigError, match="chunk_overlap_tokens"):
        load_settings(path, environ={}, dotenv_path=None)


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, llm={"modle": "typo"})
    with pytest.raises(ConfigError, match="llm.modle"):
        load_settings(path, environ={}, dotenv_path=None)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.yaml", environ={}, dotenv_path=None)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("app: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_settings(path, environ={}, dotenv_path=None)


def test_with_overrides_returns_new_validated_settings(settings: Settings) -> None:
    changed = settings.with_overrides({"retrieval.mode": "bm25", "retrieval.rerank": False})
    assert changed.retrieval.mode == "bm25"
    assert changed.retrieval.rerank is False
    assert settings.retrieval.mode == "hybrid"  # original untouched
    with pytest.raises(ConfigError):
        settings.with_overrides({"retrieval.mode": "nope"})


def test_snapshot_is_json_and_excludes_secrets(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "AIza-super-secret-test-value"
    monkeypatch.setenv(settings.llm.api_key_env, secret)
    snap = settings.snapshot()
    dumped = json.dumps(snap)
    assert secret not in dumped
    assert snap["llm"]["api_key_env"] == "GEMINI_API_KEY"  # only the variable *name*
    assert snap["retrieval"]["mode"] == "hybrid"


def test_require_api_key_missing_raises_friendly_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY is not set"):
        settings.require_api_key(environ={})
    with pytest.raises(ConfigError):
        settings.require_api_key(environ={"GEMINI_API_KEY": "   "})


def test_require_api_key_present(settings: Settings) -> None:
    assert settings.require_api_key(environ={"GEMINI_API_KEY": "k"}) == "k"


def test_dotenv_is_loaded_without_overriding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LECTURELENS__LLM__MODEL=from-dotenv\n", encoding="utf-8")
    # setenv then delenv so monkeypatch removes whatever load_dotenv sets during the test
    monkeypatch.setenv("LECTURELENS__LLM__MODEL", "placeholder")
    monkeypatch.delenv("LECTURELENS__LLM__MODEL")
    s = load_settings(CONFIG_PATH, dotenv_path=dotenv)
    assert s.llm.model == "from-dotenv"
    monkeypatch.setenv("LECTURELENS__LLM__MODEL", "from-env")
    s = load_settings(CONFIG_PATH, dotenv_path=dotenv)
    assert s.llm.model == "from-env"
