"""Load config/config.yaml plus environment overrides into typed :class:`Settings`.

Precedence (highest first): ``LECTURELENS__SECTION__KEY`` environment variables, then the YAML
file. A ``.env`` file is loaded into the environment first (without overriding variables that
are already set). Secrets are never stored on ``Settings``: the API key is read from the
environment only when an LLM command asks for it via :meth:`Settings.require_api_key`.
"""

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lecturelens.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_DOTENV_PATH = Path(".env")
ENV_PREFIX = "LECTURELENS__"
ENV_SEPARATOR = "__"

RetrievalMode = Literal["dense", "bm25", "hybrid"]
AnswerMode = Literal["concise", "detailed", "eli5"]


class _Cfg(BaseModel):
    """Base for config sections: unknown keys are errors, so typos fail fast."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AppCfg(_Cfg):
    """``app`` section: names and directories."""

    name: str
    data_dir: Path
    sample_dir: Path
    index_dir: Path
    log_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LLMCacheCfg(_Cfg):
    """``llm.cache`` section: SQLite response cache."""

    enabled: bool
    path: Path
    max_cacheable_temperature: float = Field(ge=0)


class LLMCfg(_Cfg):
    """``llm`` section: provider, models and robustness settings."""

    provider: Literal["gemini", "openai_compatible"]
    api_key_env: str = Field(min_length=1)
    base_url: str | None
    model: str = Field(min_length=1)
    fallback_model: str | None
    judge_model: str = Field(min_length=1)
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=1)
    backoff_base_s: float = Field(gt=0)
    requests_per_minute: int = Field(gt=0)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None
    thinking_token_allowance: int = Field(ge=0)
    cache: LLMCacheCfg


class PromptsCfg(_Cfg):
    """``prompts`` section: location of the prompt file."""

    path: Path


class EmbeddingCfg(_Cfg):
    """``embeddings`` section: local or Gemini embedder settings."""

    provider: Literal["local", "gemini"]
    model: str
    query_instruction: str
    batch_size: int = Field(gt=0)
    normalize: bool
    device: str
    gemini_model: str
    gemini_dim: int = Field(gt=0)


class IngestionCfg(_Cfg):
    """``ingestion`` section: file types, cleaning and chunking."""

    extensions: list[str]
    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)
    min_chunk_tokens: int = Field(ge=0)
    min_page_chars: int = Field(ge=0)
    header_footer_threshold: float = Field(gt=0, le=1)
    header_footer_min_pages: int = Field(ge=2)
    header_footer_substring_min_chars: int = Field(ge=1)
    use_ocr: bool
    add_context_header: bool

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "IngestionCfg":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


class BM25Cfg(_Cfg):
    """``retrieval.bm25`` section."""

    k1: float = Field(ge=0)
    b: float = Field(ge=0, le=1)
    stemming: bool


class RetrievalCfg(_Cfg):
    """``retrieval`` section: hybrid search, fusion, reranking and context budget."""

    mode: RetrievalMode
    top_k_dense: int = Field(gt=0)
    top_k_bm25: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    weights: dict[Literal["dense", "bm25"], float]
    rerank: bool
    reranker_model: str
    rerank_candidates: int = Field(gt=0)
    final_k: int = Field(gt=0)
    min_rerank_score: float
    max_context_tokens: int = Field(gt=0)
    bm25: BM25Cfg


class GenerationCfg(_Cfg):
    """``generation`` section: answer style and follow-up handling."""

    default_mode: AnswerMode
    max_words: int = Field(gt=0)
    max_words_detailed: int = Field(gt=0)
    history_turns: int = Field(ge=0)
    repair_on_invalid_citations: bool


class QuizCfg(_Cfg):
    """``quiz`` section."""

    default_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    default_difficulty: Literal["easy", "medium", "hard", "mixed"]
    context_chunks: int = Field(gt=0)
    seed: int
    dedup_similarity: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _default_within_max(self) -> "QuizCfg":
        if self.default_n > self.max_n:
            raise ValueError("quiz.default_n must not exceed quiz.max_n")
        return self


class SummaryCfg(_Cfg):
    """``summary`` section."""

    n_points: int = Field(gt=0)
    n_terms: int = Field(gt=0)
    context_chunks: int = Field(gt=0)


class FlashcardsCfg(_Cfg):
    """``flashcards`` section."""

    default_n: int = Field(gt=0)
    context_chunks: int = Field(gt=0)


class AblationCfg(BaseModel):
    """One ablation: a ``name`` plus dotted-key overrides such as ``retrieval.mode: bm25``."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str

    @property
    def overrides(self) -> dict[str, Any]:
        """Return the dotted-key overrides (everything except ``name``)."""
        return dict(self.model_extra or {})


class EvalCfg(_Cfg):
    """``eval`` section: dataset, metrics and ablations."""

    dataset: Path
    output_dir: Path
    k_values: list[int]
    run_generation_metrics: bool
    ablations: list[AblationCfg]
    chunk_sweep: list[int]
    minimal_prompt_questions: int = Field(ge=0)
    report_examples: int = Field(ge=0)


class Settings(_Cfg):
    """All settings, mirroring the sections of config/config.yaml."""

    app: AppCfg
    llm: LLMCfg
    prompts: PromptsCfg
    embeddings: EmbeddingCfg
    ingestion: IngestionCfg
    retrieval: RetrievalCfg
    generation: GenerationCfg
    quiz: QuizCfg
    summary: SummaryCfg
    flashcards: FlashcardsCfg
    eval: EvalCfg

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the settings for eval reports.

        Settings never hold secrets (only the *name* of the API key variable), so the snapshot
        is safe to write to disk and commit.
        """
        return self.model_dump(mode="json")

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Settings":
        """Return new settings with dotted-key overrides applied and re-validated.

        Args:
            overrides: Mapping such as ``{"retrieval.mode": "bm25", "retrieval.rerank": False}``.

        Returns:
            A validated copy; ``self`` is unchanged.

        Raises:
            ConfigError: If a key is unknown or a value is invalid.
        """
        data = self.model_dump()
        for dotted, value in overrides.items():
            _set_nested(data, dotted.split("."), value)
        return _validate(data, source="overrides")

    def require_api_key(self, environ: Mapping[str, str] | None = None) -> str:
        """Return the LLM API key from the environment, failing with a friendly message.

        Called only by commands that talk to the LLM, so tests and ingestion work without a key.

        Args:
            environ: Environment mapping to read; defaults to ``os.environ``.

        Returns:
            The API key. Callers must never log or print it.

        Raises:
            ConfigError: If the variable named by ``llm.api_key_env`` is unset or empty.
        """
        env = os.environ if environ is None else environ
        key = env.get(self.llm.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"{self.llm.api_key_env} is not set. Copy .env.example to .env and add your key "
                "(get one at aistudio.google.com)."
            )
        return key


def load_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = DEFAULT_DOTENV_PATH,
) -> Settings:
    """Load YAML settings, overlay ``LECTURELENS__*`` env vars and validate.

    Args:
        path: Path to config.yaml.
        environ: Environment mapping for overrides; defaults to ``os.environ``.
        dotenv_path: ``.env`` file to load into ``os.environ`` first (existing variables win).
            Pass ``None`` to skip, e.g. in tests.

    Returns:
        Validated, immutable settings.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or fails validation.
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)
    data = _read_yaml(Path(path))
    _apply_env_overrides(data, os.environ if environ is None else environ)
    return _validate(data, source=str(path))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping of sections")
    return data


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Overlay ``LECTURELENS__A__B=value`` onto ``data['a']['b']`` (values parsed as YAML)."""
    for name, raw in environ.items():
        if not name.upper().startswith(ENV_PREFIX):
            continue
        keys = [part.lower() for part in name[len(ENV_PREFIX) :].split(ENV_SEPARATOR)]
        if not all(keys):
            raise ConfigError(f"Malformed override variable {name!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        _set_nested(data, keys, value)


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            raise ConfigError(f"Unknown config section {'.'.join(keys)!r}")
        node = child
    node[keys[-1]] = copy.deepcopy(value)


def _validate(data: dict[str, Any], *, source: str) -> Settings:
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration ({source}): {problems}") from exc
