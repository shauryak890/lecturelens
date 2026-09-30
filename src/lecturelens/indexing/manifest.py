"""Index manifest: which files are indexed (by content hash) and with which settings.

The manifest makes ingestion incremental (unchanged files are skipped) and safe: vectors from
different embedding models are not comparable, so a model or chunking change requires
``ingest --rebuild`` instead of silently mixing incompatible chunks.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from lecturelens.errors import IndexMismatchError

logger = logging.getLogger(__name__)

REBUILD_HINT = "Run `python -m lecturelens ingest --rebuild` to re-index everything."


class FileEntry(BaseModel):
    """One indexed file."""

    hash: str
    doc_id: str
    file_name: str
    n_pages: int
    n_chunks: int
    n_tokens: int
    indexed_at: str


class Manifest(BaseModel):
    """Contents of ``data/index/manifest.json``."""

    embedding_model: str | None = None
    dim: int | None = None
    chunking: dict[str, Any] = {}
    files: dict[str, FileEntry] = {}  # key: resolved file path (POSIX style)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load the manifest, or return an empty one if the file does not exist.

        Raises:
            IndexMismatchError: If the file exists but is unreadable.
        """
        if not path.is_file():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            raise IndexMismatchError(f"Corrupt index manifest {path}. {REBUILD_HINT}") from exc

    def save(self, path: Path) -> None:
        """Write the manifest atomically (temp file, then rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def check_compatible(self, embedding_model: str, chunking: dict[str, Any]) -> None:
        """Raise if the existing index was built with another embedding model or chunking.

        Needs no loaded model, so it runs before anything heavy. The vector dimension is
        checked separately by :meth:`check_dim` once the model is loaded. An empty manifest
        (fresh index) is always compatible.

        Raises:
            IndexMismatchError: With a hint to run ``ingest --rebuild``.
        """
        if not self.files:
            return
        if self.embedding_model != embedding_model:
            raise IndexMismatchError(
                f"The index was built with {self.embedding_model} but config uses "
                f"{embedding_model}. Embeddings from different models are not comparable. "
                f"{REBUILD_HINT}"
            )
        if self.chunking != chunking:
            raise IndexMismatchError(
                f"Chunking settings changed ({self.chunking} -> {chunking}). {REBUILD_HINT}"
            )

    def check_dim(self, dim: int) -> None:
        """Record the vector dimension on first use; raise if it differs from the index's.

        Raises:
            IndexMismatchError: If vectors of another dimension are already indexed.
        """
        if self.dim is None or not self.files:
            self.dim = dim
        elif self.dim != dim:
            raise IndexMismatchError(
                f"The index holds {self.dim}-dimensional vectors but {self.embedding_model} "
                f"produces {dim}. {REBUILD_HINT}"
            )

    def set_settings(self, embedding_model: str, chunking: dict[str, Any]) -> None:
        """Record the embedding model and chunking settings the index is built with."""
        self.embedding_model, self.chunking = embedding_model, dict(chunking)

    def find_doc(self, doc_id: str) -> str | None:
        """Return the file key already indexed with ``doc_id`` (identical content), if any."""
        return next((key for key, e in self.files.items() if e.doc_id == doc_id), None)

    @property
    def n_chunks(self) -> int:
        """Total chunks across files."""
        return sum(e.n_chunks for e in self.files.values())

    @property
    def n_tokens(self) -> int:
        """Total chunk tokens across files."""
        return sum(e.n_tokens for e in self.files.values())
