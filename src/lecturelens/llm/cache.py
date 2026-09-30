"""SQLite cache of validated LLM responses (SPEC 8.4).

Keys are the SHA-256 of everything that determines the output: model, system prompt, user
prompt, temperature, schema name and prompt-file version. Re-running the evaluation or
re-asking a demo question then costs no API quota. Only low-temperature calls are cached
(the client decides), because high-temperature tasks such as quizzes should vary.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from lecturelens.logging_utils import utc_timestamp


def cache_key(
    *,
    model: str,
    system: str | None,
    user: str,
    temperature: float,
    schema_name: str,
    prompt_version: str,
) -> str:
    """Return the cache key for one LLM request."""
    payload = json.dumps(
        [model, system, user, temperature, schema_name, prompt_version], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Key -> JSON text store in a single SQLite file (safe to share between threads)."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the cache database.

        Args:
            path: SQLite file, e.g. ``data/cache/llm_cache.sqlite``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses "
                "(key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def get(self, key: str) -> str | None:
        """Return the cached response text for ``key``, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, response: str) -> None:
        """Store (or replace) the response for ``key``."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, response, created_at) VALUES (?, ?, ?)",
                (key, response, utc_timestamp()),
            )

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
