"""Offline tests for llm.client.list_models and the `models` CLI command (SDK is faked)."""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from typer.testing import CliRunner

from lecturelens.cli import app
from lecturelens.config import Settings
from lecturelens.errors import LLMError
from lecturelens.llm import client as client_mod

from .conftest import CONFIG_PATH


def _model(name: str, actions: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"models/{name}",
        display_name=name.title(),
        input_token_limit=1000,
        output_token_limit=100,
        supported_actions=actions,
    )


class _FakeModels:
    def __init__(self, result):
        self._result = result

    def list(self):
        if isinstance(self._result, Exception):
            raise self._result
        return iter(self._result)


def _patch_sdk(monkeypatch: pytest.MonkeyPatch, result) -> None:
    fake = SimpleNamespace(models=_FakeModels(result))
    monkeypatch.setattr(client_mod, "_gemini_client", lambda cfg, key: fake)


def test_list_models_strips_prefix_filters_and_sorts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sdk(
        monkeypatch,
        [
            _model("gemini-z-flash", ["generateContent"]),
            _model("text-embedding", ["embedContent"]),
            _model("gemini-a-lite", ["countTokens", "generateContent"]),
        ],
    )
    found = client_mod.list_models(settings.llm, "k")
    assert [m.model_id for m in found] == ["gemini-a-lite", "gemini-z-flash"]
    everything = client_mod.list_models(settings.llm, "k", generate_only=False)
    assert len(everything) == 3


def test_list_models_auth_error_is_friendly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sdk(monkeypatch, genai_errors.ClientError(403, {"error": {"message": "denied"}}))
    with pytest.raises(LLMError, match="Check GEMINI_API_KEY"):
        client_mod.list_models(settings.llm, "bad-key")


def test_models_command_lists_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_sdk(monkeypatch, [_model("gemini-a-lite", ["generateContent"])])
    result = CliRunner().invoke(app, ["--config", str(CONFIG_PATH), "models"])
    assert result.exit_code == 0, result.output
    assert "gemini-a-lite" in result.output
    assert "test-key" not in result.output
    assert "is not in this list" in result.output  # configured placeholder IDs are flagged


def test_models_command_without_key_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    result = CliRunner().invoke(app, ["--config", str(CONFIG_PATH), "models"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
