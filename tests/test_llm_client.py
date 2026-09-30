"""Tests for llm.client.LLMClient: real retry/cache/repair/fallback logic, fake backend."""

import json
from pathlib import Path

import pytest

from lecturelens.config import Settings
from lecturelens.errors import LLMBlockedError, LLMError, LLMOutputError
from lecturelens.llm.cache import LLMCache
from lecturelens.llm.client import LLMClient, RateLimiter, RawResponse, TransientLLMError
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.logging_utils import LLM_CALL_LOG_NAME, LLMCallLogger
from lecturelens.schemas import CondensedQuestion

from .conftest import PROMPTS_PATH
from .fakes import FakeBackend

GOOD = RawResponse(
    text='{"standalone_question": "What is BPE?"}',
    prompt_tokens=50,
    output_tokens=12,
    finish_reason="STOP",
)
BAD_JSON = RawResponse(
    text='{"standalone_question": ', prompt_tokens=50, output_tokens=5, finish_reason="STOP"
)
RATE_LIMITED = TransientLLMError("Gemini API 429: RESOURCE_EXHAUSTED")


class Harness:
    """An LLMClient wired to a FakeBackend, a temp cache and log, and a recording sleep."""

    def __init__(
        self,
        settings: Settings,
        registry: PromptRegistry,
        tmp_path: Path,
        script: list,
        cache: bool = True,
    ) -> None:
        self.backend = FakeBackend(script)
        self.sleeps: list[float] = []
        self.log_dir = tmp_path / "logs"
        self.client = LLMClient(
            settings.llm,
            registry,
            self.backend,
            cache=LLMCache(tmp_path / "cache.sqlite") if cache else None,
            call_logger=LLMCallLogger(self.log_dir),
            limiter=RateLimiter(10**6),
            sleep=self.sleeps.append,
        )

    def condense(self, question: str = "and its merges?"):
        return self.client.run_task("condense_question", history="Student: BPE?", question=question)

    def log(self) -> list[dict]:
        path = self.log_dir / LLM_CALL_LOG_NAME
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def make(settings: Settings, registry: PromptRegistry, tmp_path: Path):
    return lambda script, **kw: Harness(settings, registry, tmp_path, script, **kw)


def test_success_returns_validated_object_and_usage(make) -> None:
    h = make([GOOD])
    result, usage = h.condense()
    assert result == CondensedQuestion(standalone_question="What is BPE?")
    assert (usage.prompt_tokens, usage.output_tokens, usage.api_calls) == (50, 12, 1)
    sent = h.backend.calls[0]
    assert sent["schema"] is CondensedQuestion and sent["temperature"] == 0.0
    assert sent["max_output_tokens"] == 128  # from prompts.yaml
    assert sent["system"] is None  # condense_question has no system prompt


def test_retries_on_429_then_succeeds(make) -> None:
    h = make([RATE_LIMITED, RATE_LIMITED, GOOD])
    result, usage = h.condense()
    assert result.standalone_question == "What is BPE?"
    assert usage.api_calls == 3
    assert len(h.sleeps) == 2 and h.sleeps[1] >= h.sleeps[0] - 1  # exponential backoff (+jitter)
    assert h.log()[-1]["attempts"] == 3 and h.log()[-1]["status"] == "ok"


def test_fallback_model_used_after_max_retries(make, settings: Settings) -> None:
    retries = settings.llm.max_retries
    h = make([RATE_LIMITED] * retries + [GOOD])
    result, _ = h.condense()
    assert result.standalone_question == "What is BPE?"
    models = [c["model"] for c in h.backend.calls]
    assert models == [settings.llm.model] * retries + [settings.llm.fallback_model]
    record = h.log()[-1]
    assert record["fallback_used"] and record["model"] == settings.llm.fallback_model


def test_error_when_primary_and_fallback_fail(make, settings: Settings) -> None:
    h = make([RATE_LIMITED] * (settings.llm.max_retries + 1))
    with pytest.raises(LLMError, match="fallback"):
        h.condense()
    assert h.log()[-1]["status"] == "error"


def test_permanent_error_is_not_retried(make) -> None:
    h = make([LLMError("API key rejected (400). Check GEMINI_API_KEY in .env")])
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        h.condense()
    assert len(h.backend.calls) == 1 and h.sleeps == []


def test_invalid_json_triggers_one_repair_call(make) -> None:
    h = make([BAD_JSON, GOOD])
    result, usage = h.condense()
    assert result.standalone_question == "What is BPE?"
    assert usage.api_calls == 2 and usage.prompt_tokens == 100
    repair = h.backend.calls[1]
    assert repair["schema"] is CondensedQuestion  # repair keeps the original schema
    assert '{"standalone_question": ' in repair["user"]  # raw text is sent back
    assert "standalone_question" in repair["user"] and "Validation error" in repair["user"]
    assert [r["task"] for r in h.log()] == ["repair_json", "condense_question"]


def test_second_invalid_output_raises(make) -> None:
    h = make([BAD_JSON, BAD_JSON])
    with pytest.raises(LLMOutputError, match="after repair"):
        h.condense()
    assert len(h.backend.calls) == 2  # exactly one repair attempt


def test_cache_hit_avoids_second_call(make) -> None:
    h = make([GOOD])
    first, _ = h.condense()
    second, usage = h.condense()  # script is empty now: a second API call would fail
    assert first == second and usage.cache_hits == 1 and usage.api_calls == 0
    assert h.log()[-1]["status"] == "cache_hit"


def test_high_temperature_tasks_are_not_cached(make, registry: PromptRegistry) -> None:
    quiz = '{"topic": "BPE", "questions": []}'
    h = make([RawResponse(text=quiz), RawResponse(text=quiz)])
    for _ in range(2):  # quiz has temperature 0.7 > max_cacheable_temperature 0.5
        h.client.run_task("quiz", context="c", n=1, topic="BPE", difficulty="easy")
    assert len(h.backend.calls) == 2
    assert registry.task("quiz").temperature > 0.5


def test_prompt_version_change_invalidates_cache(make, tmp_path: Path) -> None:
    h = make([GOOD, GOOD])
    h.condense()
    h.client.registry = PromptRegistry(PROMPTS_PATH)  # own copy: the fixture is session-wide
    h.client.registry.version = "9.9.9"
    h.condense()
    assert len(h.backend.calls) == 2


def test_empty_response_raises_blocked(make) -> None:
    h = make([RawResponse(text=None, finish_reason="SAFETY")])
    with pytest.raises(LLMBlockedError, match="SAFETY"):
        h.condense()
    assert h.log()[-1]["status"] == "blocked"


def test_truncated_empty_output_is_an_output_error(make) -> None:
    h = make([RawResponse(text="", finish_reason="MAX_TOKENS")])
    with pytest.raises(LLMOutputError, match="thinking_token_allowance"):
        h.condense()


def test_call_log_has_metadata_but_no_secrets(make, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-secret-value")
    h = make([GOOD])
    h.condense()
    line = (h.log_dir / LLM_CALL_LOG_NAME).read_text(encoding="utf-8")
    assert "AIza-secret-value" not in line and "What is BPE" not in line
    record = h.log()[0]
    assert set(record) == {
        "ts",
        "task",
        "model",
        "prompt_version",
        "prompt_tokens",
        "output_tokens",
        "latency_ms",
        "cache_hit",
        "attempts",
        "fallback_used",
        "status",
    }
    assert record["prompt_version"] == h.client.registry.version


def test_rate_limiter_blocks_until_window_frees() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, window_s=60.0, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    now[0] = 10.0
    limiter.acquire()
    now[0] = 20.0
    limiter.acquire()  # third call within 60 s: waits until the first one expires at t=60
    assert sleeps == [pytest.approx(40.0)]
    assert now[0] == pytest.approx(60.0)
