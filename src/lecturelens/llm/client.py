"""LLM provider access (SPEC 6.6, 8). All google-genai SDK calls live in this module.

``LLMClient.generate_json`` does: rate limit -> cache lookup -> API call with JSON schema ->
parse -> Pydantic validation. Transient errors (429, 5xx, timeouts) are retried with
exponential backoff and jitter (tenacity); after the last retry the fallback model is tried
once. Output that fails validation gets exactly one repair call. Every call is written to
``logs/llm_calls.jsonl``.

The provider is behind the small :class:`LLMBackend` protocol, so tests inject a fake backend
and the real retry/cache/repair logic still runs offline.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from lecturelens.config import LLMCfg, Settings
from lecturelens.errors import ConfigError, LLMBlockedError, LLMError, LLMOutputError
from lecturelens.llm.cache import LLMCache, cache_key
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.logging_utils import LLMCallLogger, utc_timestamp
from lecturelens.schemas import LLMCallRecord, Usage

logger = logging.getLogger(__name__)

GENERATE_ACTION = "generateContent"
MODEL_NAME_PREFIX = "models/"
AUTH_ERROR_CODES = frozenset({401, 403})
RATE_LIMIT_CODE = 429
SERVER_ERROR_MIN = 500
BAD_REQUEST = 400
NOT_FOUND = 404
RATE_WINDOW_S = 60.0  # requests_per_minute is enforced over any sliding 60 s window
MS_PER_S = 1000
TRUNCATED = "MAX_TOKENS"
REPAIR_TASK = "repair_json"


class TransientLLMError(LLMError):
    """A retryable provider failure: rate limit, server error or timeout."""


# ---------------------------------------------------------------- backend


@dataclass(frozen=True)
class RawResponse:
    """What a backend returns for one request."""

    text: str | None
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None


class LLMBackend:
    """Interface of a provider backend (one HTTP request per ``generate`` call)."""

    def generate(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_output_tokens: int | None,
    ) -> RawResponse:
        """Send one request and return the raw response.

        Raises:
            TransientLLMError: For retryable failures.
            LLMError: For permanent failures (bad key, unknown model, bad request).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release network resources."""


class GeminiBackend(LLMBackend):
    """google-genai ``models.generate_content`` with JSON-schema structured output."""

    def __init__(self, cfg: LLMCfg, api_key: str) -> None:
        """Create the SDK client (kept for the backend's lifetime; see DECISIONS.md).

        Args:
            cfg: LLM settings (timeout, thinking level and allowance).
            api_key: Gemini API key. Never logged.
        """
        self._cfg = cfg
        self._client = genai.Client(
            api_key=api_key, http_options={"timeout": int(cfg.timeout_s * MS_PER_S)}
        )
        self._no_thinking: set[str] = set()  # models that rejected the thinking level

    def _config(
        self,
        model: str,
        system: str | None,
        schema: type[BaseModel],
        temperature: float,
        max_output_tokens: int | None,
    ) -> genai_types.GenerateContentConfig:
        thinking = None
        if self._cfg.thinking_level and model not in self._no_thinking:
            thinking = genai_types.ThinkingConfig(thinking_level=self._cfg.thinking_level.upper())
        # thinking tokens are billed as output: give them room on top of the task's budget
        budget = (
            None
            if max_output_tokens is None
            else max_output_tokens + (self._cfg.thinking_token_allowance)
        )
        return genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=budget,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            thinking_config=thinking,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_output_tokens: int | None,
    ) -> RawResponse:
        """Send one ``generate_content`` request."""
        config = self._config(model, system, schema, temperature, max_output_tokens)
        try:
            response = self._client.models.generate_content(
                model=model, contents=user, config=config
            )
        except genai_errors.APIError as exc:
            if self._thinking_rejected(exc, config):
                logger.info("%s rejected thinking level; retrying without it", model)
                self._no_thinking.add(model)
                return self.generate(
                    model=model,
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            raise _translate_api_error(exc, model, self._cfg.api_key_env) from exc
        except httpx.TimeoutException as exc:
            raise TransientLLMError(
                f"Gemini request timed out after {self._cfg.timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Network error talking to Gemini: {exc}") from exc
        return _to_raw(response)

    @staticmethod
    def _thinking_rejected(
        exc: genai_errors.APIError, config: genai_types.GenerateContentConfig
    ) -> bool:
        return (
            exc.code == BAD_REQUEST
            and config.thinking_config is not None
            and "thinking" in (exc.message or "").lower()
        )

    def close(self) -> None:
        """Close the SDK's HTTP client."""
        self._client.close()


def _translate_api_error(exc: genai_errors.APIError, model: str, key_env: str) -> LLMError:
    code, message = exc.code, exc.message or str(exc)
    if code == RATE_LIMIT_CODE or (code and code >= SERVER_ERROR_MIN):
        return TransientLLMError(f"Gemini API {code}: {message}")
    if code in AUTH_ERROR_CODES or "api key" in message.lower():
        return LLMError(f"API key rejected ({code}). Check {key_env} in .env")
    if code == NOT_FOUND:
        return LLMError(f"Model {model!r} not found. Run `python -m lecturelens models`.")
    return LLMError(f"Gemini API {code}: {message}")


def _to_raw(response: genai_types.GenerateContentResponse) -> RawResponse:
    usage = response.usage_metadata
    candidates = response.candidates or []
    finish = candidates[0].finish_reason if candidates else None
    if finish is None and response.prompt_feedback and response.prompt_feedback.block_reason:
        finish = response.prompt_feedback.block_reason
    return RawResponse(
        text=response.text if candidates else None,
        prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
        if usage
        else 0,
        finish_reason=getattr(finish, "name", None) or (str(finish) if finish else None),
    )


# ---------------------------------------------------------------- rate limiter


class RateLimiter:
    """Client-side limiter: at most ``max_calls`` requests in any sliding 60 s window.

    Stricter than a token bucket (which can allow 2x the rate across a window boundary), so
    bursts never exceed the free-tier requests-per-minute limit.
    """

    def __init__(
        self,
        max_calls: int,
        window_s: float = RATE_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the limiter.

        Args:
            max_calls: Requests allowed per window.
            window_s: Window length in seconds.
            clock: Monotonic clock (injectable for tests).
            sleep: Sleep function (injectable for tests).
        """
        self.max_calls, self.window_s = max_calls, window_s
        self._clock, self._sleep = clock, sleep
        self._sent: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one more request is allowed, then record it."""
        with self._lock:
            while True:
                now = self._clock()
                while self._sent and self._sent[0] <= now - self.window_s:
                    self._sent.popleft()
                if len(self._sent) < self.max_calls:
                    self._sent.append(now)
                    return
                wait = self._sent[0] + self.window_s - now
                logger.info("Rate limiter: waiting %.1fs", wait)
                self._sleep(wait)


# ---------------------------------------------------------------- client


class LLMClient:
    """Structured-output LLM calls with rate limiting, retries, fallback, cache and repair."""

    def __init__(
        self,
        cfg: LLMCfg,
        registry: PromptRegistry,
        backend: LLMBackend,
        *,
        cache: LLMCache | None = None,
        call_logger: LLMCallLogger | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the client.

        Args:
            cfg: LLM settings.
            registry: Prompt registry (task templates and the prompt version).
            backend: Provider backend.
            cache: Response cache; ``None`` disables caching.
            call_logger: Writer for ``llm_calls.jsonl``; ``None`` disables it.
            limiter: Rate limiter; defaults to ``cfg.requests_per_minute``.
            sleep: Sleep used between retries (injectable for tests).
        """
        self.cfg = cfg
        self.registry = registry
        self.backend = backend
        self.cache = cache
        self.call_logger = call_logger
        self.limiter = limiter or RateLimiter(cfg.requests_per_minute)
        self._sleep = sleep

    def run_task(
        self,
        task: str,
        *,
        model: str | None = None,
        schema: type[BaseModel] | None = None,
        **variables: object,
    ) -> tuple[BaseModel, Usage]:
        """Render prompt-file task ``task`` with ``variables`` and call :meth:`generate_json`.

        Temperature, token limit, system prompt and output schema come from the prompt file.

        Args:
            task: Task name in prompts.yaml.
            model: Override the model (e.g. ``cfg.judge_model``).
            schema: Output schema for tasks that do not declare one (``repair_json``).
            **variables: Template variables.
        """
        user, spec = self.registry.render(task, **variables)
        return self.generate_json(
            task=task,
            system=self.registry.system_for(task),
            user=user,
            schema=schema or spec.response_model,
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
            model=model,
        )

    def generate_json(
        self,
        *,
        task: str,
        system: str | None,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_output_tokens: int | None = None,
        model: str | None = None,
        allow_repair: bool = True,
    ) -> tuple[BaseModel, Usage]:
        """Get a validated ``schema`` instance from the LLM.

        Returns:
            The parsed object and the usage of every request made (including repair).

        Raises:
            LLMBlockedError: The model returned no text (safety block or empty response).
            LLMOutputError: The output was invalid even after one repair call.
            LLMError: Permanent API failure, or transient failures persisted after all retries
                and the fallback model.
        """
        model = model or self.cfg.model
        key = self._cache_key(model, system, user, temperature, schema)
        cached = self._from_cache(key, schema, task, model)
        if cached is not None:
            return cached, Usage(cache_hits=1)

        record = LLMCallRecord(
            ts=utc_timestamp(), task=task, model=model, prompt_version=self.registry.version
        )
        started = time.perf_counter()
        try:
            raw, record = self._call(record, system, user, schema, temperature, max_output_tokens)
            usage = Usage(
                prompt_tokens=raw.prompt_tokens,
                output_tokens=raw.output_tokens,
                api_calls=record.attempts,
            )
            result, usage = self._validate(raw, schema, task, record.model, usage, allow_repair)
        except LLMError as exc:
            self._log(record, started, status=_status_of(exc))
            raise
        self._log(
            record.model_copy(
                update={"prompt_tokens": usage.prompt_tokens, "output_tokens": usage.output_tokens}
            ),
            started,
        )
        if key is not None and self.cache is not None:
            self.cache.put(key, result.model_dump_json())
        return result, usage

    # -------------------------------------------------------------- internals

    def _cache_key(
        self,
        model: str,
        system: str | None,
        user: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> str | None:
        if self.cache is None or temperature > self.cfg.cache.max_cacheable_temperature:
            return None
        return cache_key(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            schema_name=schema.__name__,
            prompt_version=self.registry.version,
        )

    def _from_cache(
        self, key: str | None, schema: type[BaseModel], task: str, model: str
    ) -> BaseModel | None:
        if key is None or self.cache is None:
            return None
        text = self.cache.get(key)
        if text is None:
            return None
        try:
            result = schema.model_validate_json(text)
        except ValidationError:  # stale entry from an older schema: ignore it
            return None
        self._log(
            LLMCallRecord(
                ts=utc_timestamp(),
                task=task,
                model=model,
                prompt_version=self.registry.version,
                cache_hit=True,
                attempts=0,
            ),
            time.perf_counter(),
            status="cache_hit",
        )
        return result

    def _call(
        self,
        record: LLMCallRecord,
        system: str | None,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_output_tokens: int | None,
    ) -> tuple[RawResponse, LLMCallRecord]:
        """Primary model with retries, then the fallback model once."""

        def send(model: str) -> RawResponse:
            self.limiter.acquire()
            return self.backend.generate(
                model=model,
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        attempts = 0
        retrying = Retrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential_jitter(initial=self.cfg.backoff_base_s),
            retry=retry_if_exception_type(TransientLLMError),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    attempts += 1
                    raw = send(record.model)
            return raw, record.model_copy(update={"attempts": attempts})
        except TransientLLMError as exc:
            fallback = self.cfg.fallback_model
            record = record.model_copy(update={"attempts": attempts})
            if not fallback or fallback == record.model:
                raise LLMError(
                    f"{record.model} unavailable after {attempts} attempts: {exc}"
                ) from exc
            logger.warning(
                "%s failed %d times (%s); trying fallback %s", record.model, attempts, exc, fallback
            )
            record = record.model_copy(
                update={"model": fallback, "fallback_used": True, "attempts": attempts + 1}
            )
            try:
                return send(fallback), record
            except TransientLLMError as fallback_exc:
                raise LLMError(
                    f"Both {self.cfg.model} and fallback {fallback} failed: {fallback_exc}"
                ) from fallback_exc

    def _validate(
        self,
        raw: RawResponse,
        schema: type[BaseModel],
        task: str,
        model: str,
        usage: Usage,
        allow_repair: bool,
    ) -> tuple[BaseModel, Usage]:
        if not raw.text or not raw.text.strip():
            if raw.finish_reason == TRUNCATED:
                raise LLMOutputError(
                    f"{task}: output truncated before any text "
                    "(raise llm.thinking_token_allowance or max_output_tokens)"
                )
            raise LLMBlockedError(f"{task}: empty response (finish_reason={raw.finish_reason})")
        try:
            return schema.model_validate_json(raw.text), usage
        except ValidationError as exc:
            if not allow_repair:
                raise LLMOutputError(f"{task}: invalid output after repair: {_brief(exc)}") from exc
            logger.warning("%s: invalid JSON (%s), attempting one repair", task, _brief(exc))
            repaired, repair_usage = self._repair(raw.text, exc, schema, model)
            return repaired, usage + repair_usage

    def _repair(
        self, raw_text: str, error: ValidationError, schema: type[BaseModel], model: str
    ) -> tuple[BaseModel, Usage]:
        """One repair call: send the validation error and the raw text back to the model."""
        user, spec = self.registry.render(REPAIR_TASK, error=_brief(error), raw=raw_text)
        return self.generate_json(
            task=REPAIR_TASK,
            system=self.registry.system_for(REPAIR_TASK),
            user=user,
            schema=schema,
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
            model=model,
            allow_repair=False,
        )

    def _log(self, record: LLMCallRecord, started: float, status: str = "ok") -> None:
        if self.call_logger is None:
            return
        latency = (time.perf_counter() - started) * MS_PER_S
        self.call_logger.log(
            record.model_copy(update={"latency_ms": round(latency, 1), "status": status})
        )

    def close(self) -> None:
        """Release the backend's network resources."""
        self.backend.close()


def _brief(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()
    )


def _status_of(exc: LLMError) -> str:
    if isinstance(exc, LLMBlockedError):
        return "blocked"
    if isinstance(exc, LLMOutputError):
        return "invalid_output"
    return "error"


def build_llm_client(settings: Settings, registry: PromptRegistry) -> LLMClient:
    """Create the configured client: backend, cache and call log (reads the API key).

    Raises:
        ConfigError: If the API key is missing or the provider is not implemented.
    """
    cfg = settings.llm
    if cfg.provider != "gemini":
        raise ConfigError(f"llm.provider {cfg.provider!r} is not implemented; use 'gemini'")
    backend = GeminiBackend(cfg, settings.require_api_key())
    cache = LLMCache(cfg.cache.path) if cfg.cache.enabled else None
    return LLMClient(
        cfg, registry, backend, cache=cache, call_logger=LLMCallLogger(settings.app.log_dir)
    )


# ---------------------------------------------------------------- model listing (P0)


class ModelInfo(BaseModel):
    """A model available to the configured API key."""

    model_id: str  # the value to put in config.yaml, e.g. "gemini-2.5-flash"
    display_name: str = ""
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    supported_actions: list[str] = []

    @property
    def can_generate(self) -> bool:
        """Whether the model supports text generation (generateContent)."""
        return GENERATE_ACTION in self.supported_actions


def _gemini_client(cfg: LLMCfg, api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key, http_options={"timeout": int(cfg.timeout_s * MS_PER_S)})


def list_models(cfg: LLMCfg, api_key: str, *, generate_only: bool = True) -> list[ModelInfo]:
    """List the models available to ``api_key``, sorted by ID.

    Args:
        cfg: LLM settings (provider and timeout).
        api_key: The provider API key. Never logged.
        generate_only: Keep only models that support text generation.

    Returns:
        Available models.

    Raises:
        LLMError: On unsupported provider, invalid key, or API/network failure.
    """
    if cfg.provider != "gemini":
        raise LLMError(
            f"Listing models is only supported for provider 'gemini', not {cfg.provider!r}"
        )
    try:
        # Keep the Client referenced for the whole call: genai.Client closes its HTTP connection
        # when garbage-collected, so `_gemini_client(...).models.list()` fails intermittently.
        with _gemini_client(cfg, api_key) as client:
            raw_models = list(client.models.list())
    except genai_errors.APIError as exc:
        if exc.code in AUTH_ERROR_CODES:
            raise LLMError(
                f"API key rejected ({exc.code}). Check {cfg.api_key_env} in .env"
            ) from exc
        raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except httpx.HTTPError as exc:  # timeouts and connection failures
        raise LLMError(f"Could not reach the Gemini API: {exc}") from exc

    models = [
        ModelInfo(
            model_id=(m.name or "").removeprefix(MODEL_NAME_PREFIX),
            display_name=m.display_name or "",
            input_token_limit=m.input_token_limit,
            output_token_limit=m.output_token_limit,
            supported_actions=list(m.supported_actions or []),
        )
        for m in raw_models
    ]
    logger.info("Listed %d models from Gemini", len(models))
    if generate_only:
        models = [m for m in models if m.can_generate]
    return sorted(models, key=lambda m: m.model_id)
