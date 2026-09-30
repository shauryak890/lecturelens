"""LLM provider access. All google-genai SDK calls live in this module.

P0 provides :func:`list_models` for the ``models`` CLI command. The full ``LLMClient``
(rate limiter, retries, fallback, cache, repair) is added in phase P2.
"""

import logging

import httpx
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

from lecturelens.config import LLMCfg
from lecturelens.errors import LLMError

logger = logging.getLogger(__name__)

GENERATE_ACTION = "generateContent"
MODEL_NAME_PREFIX = "models/"
AUTH_ERROR_CODES = frozenset({401, 403})


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
    timeout_ms = int(cfg.timeout_s * 1000)
    return genai.Client(api_key=api_key, http_options={"timeout": timeout_ms})


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
