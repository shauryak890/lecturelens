"""Offline tests for llm.client.list_models and the `models` CLI command (SDK is faked)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from typer.testing import CliRunner

from lecturelens import cli as cli_mod
from lecturelens.cli import app
from lecturelens.config import Settings
from lecturelens.errors import LLMError
from lecturelens.llm import client as client_mod
from lecturelens.logging_utils import APP_LOG_NAME, setup_logging

from .conftest import CONFIG_PATH


def _model(name: str, actions: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"models/{name}",
        display_name=name.title(),
        input_token_limit=1000,
        output_token_limit=100,
        supported_actions=actions,
    )


class _Transport:
    closed = False


class _FakeModels:
    def __init__(self, result, transport: _Transport):
        self._result = result
        self._transport = transport

    def list(self):
        if self._transport.closed:  # same failure as httpx once genai.Client is closed
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        if isinstance(self._result, Exception):
            raise self._result
        return iter(self._result)


class _FakeClient:
    """Mimics genai.Client's lifecycle: `models` shares a transport the client closes on GC."""

    def __init__(self, result, transport: _Transport):
        self._transport = transport
        self.models = _FakeModels(result, transport)

    def close(self) -> None:
        self._transport.closed = True

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _patch_sdk(monkeypatch: pytest.MonkeyPatch, result) -> list[_Transport]:
    """Patch the SDK factory; returns the transports of the clients created, in order."""
    transports: list[_Transport] = []

    def factory(cfg, key) -> _FakeClient:
        transports.append(_Transport())
        return _FakeClient(result, transports[-1])

    monkeypatch.setattr(client_mod, "_gemini_client", factory)
    return transports


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


def test_list_models_keeps_client_open_during_call_and_closes_after(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an unreferenced genai.Client is closed by GC before the request is sent."""
    transports = _patch_sdk(monkeypatch, [_model("gemini-a-lite", ["generateContent"])])
    assert [m.model_id for m in client_mod.list_models(settings.llm, "k")] == ["gemini-a-lite"]
    assert len(transports) == 1 and transports[0].closed


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


def test_run_turns_unexpected_errors_into_friendly_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def boom(**kwargs) -> None:
        setup_logging("INFO", tmp_path)  # as the real CLI callback does before a command
        raise RuntimeError("client has been closed")

    monkeypatch.setattr(cli_mod, "app", boom)
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.run()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Unexpected error" in err and "client has been closed" in err
    assert "Traceback" not in err  # console stays clean ...
    assert "Traceback" in (tmp_path / APP_LOG_NAME).read_text(encoding="utf-8")  # ... log has it
