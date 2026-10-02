"""Custom exception hierarchy for LectureLens.

Every error raised deliberately by the package derives from :class:`LectureLensError`, so the
CLI and UI can catch one type and show a friendly message instead of a stack trace.
"""


class LectureLensError(Exception):
    """Base class for all LectureLens errors."""


class ConfigError(LectureLensError):
    """Invalid or missing configuration (config.yaml, env overrides, API key)."""


class PromptError(LectureLensError):
    """Invalid prompt file, unknown task, or missing template variable."""


class IngestionError(LectureLensError):
    """A file could not be discovered, read or parsed."""


class IndexMismatchError(LectureLensError):
    """The on-disk index was built with a different embedding model or dimension."""


class NoContentError(LectureLensError):
    """Nothing in the indexed material matches the requested topic or document."""


class LLMError(LectureLensError):
    """An LLM API call failed (auth, rate limit after retries, network, safety block)."""


class LLMOutputError(LLMError):
    """The LLM returned output that could not be parsed or validated, even after repair."""


class LLMBlockedError(LLMError):
    """The LLM returned no usable text (safety block, recitation, or an empty response)."""
