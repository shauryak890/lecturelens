"""Structured logging setup and the per-call LLM log (logs/llm_calls.jsonl).

Two safeguards keep secrets out of logs: :class:`SecretRedactingFilter` masks the API key if it
ever appears in a log message, and :class:`LLMCallLogger` only writes the fields of
:class:`~lecturelens.schemas.LLMCallRecord`, which has no field for keys or document text.
"""

import logging
import os
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from lecturelens.schemas import LLMCallRecord

PACKAGE_LOGGER = "lecturelens"
APP_LOG_NAME = "lecturelens.log"
LLM_CALL_LOG_NAME = "llm_calls.jsonl"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
REDACTED = "***REDACTED***"


class SecretRedactingFilter(logging.Filter):
    """Replace the values of the given environment variables with a placeholder."""

    def __init__(self, secret_env_vars: Iterable[str]) -> None:
        """Create the filter.

        Args:
            secret_env_vars: Names of env vars whose values must never be logged.
        """
        super().__init__()
        self._secret_env_vars = tuple(secret_env_vars)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets in the formatted message; always keeps the record."""
        secrets = [v for name in self._secret_env_vars if (v := os.environ.get(name))]
        if secrets:
            message = record.getMessage()
            for secret in secrets:
                message = message.replace(secret, REDACTED)
            record.msg, record.args = message, None
        return True


class ConsoleFormatter(logging.Formatter):
    """Formatter that drops tracebacks: the console gets one line, the log file gets the rest."""

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` without its exception/stack information."""
        record = logging.makeLogRecord(record.__dict__)
        record.exc_info = record.exc_text = record.stack_info = None
        return super().format(record)


def setup_logging(level: str, log_dir: Path, secret_env_vars: Iterable[str] = ()) -> None:
    """Configure the package logger to write to stderr and ``log_dir/lecturelens.log``.

    The stderr handler shows warnings and errors as single lines (no tracebacks); the file
    handler records everything at ``level``, tracebacks included.

    Safe to call more than once (handlers are replaced, not duplicated).

    Args:
        level: Log level name, e.g. ``"INFO"``.
        log_dir: Directory for the log file; created if missing.
        secret_env_vars: Env var names whose values are redacted from every record.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    redactor = SecretRedactingFilter(secret_env_vars)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)  # keep the console quiet; details go to the file
    stream_handler.setFormatter(ConsoleFormatter(LOG_FORMAT))
    file_handler = logging.FileHandler(log_dir / APP_LOG_NAME, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    for handler in (stream_handler, file_handler):
        handler.addFilter(redactor)
        logger.addHandler(handler)


def utc_timestamp() -> str:
    """Return the current UTC time as ISO-8601 with a trailing ``Z`` (second precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LLMCallLogger:
    """Append one JSON line per LLM call to ``log_dir/llm_calls.jsonl`` (thread-safe)."""

    def __init__(self, log_dir: Path) -> None:
        """Create the logger.

        Args:
            log_dir: Directory for the jsonl file; created on first write.
        """
        self.path = log_dir / LLM_CALL_LOG_NAME
        self._lock = threading.Lock()

    def log(self, record: LLMCallRecord) -> None:
        """Append ``record`` as a single JSON line."""
        line = record.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
