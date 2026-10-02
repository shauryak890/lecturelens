"""ChromaDB wrapper for dense retrieval.

Embeddings are always passed in explicitly (Chroma's built-in embedder is disabled), so the
same :class:`~lecturelens.indexing.embedder.Embedder` is used for indexing and querying.
Chroma stores the user-facing chunk text and citation metadata alongside each vector.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lecturelens.schemas import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"
DISTANCE_SPACE = "cosine"  # Chroma returns distance = 1 - cosine similarity


class VectorStore:
    """Persistent ChromaDB collection of chunk vectors plus metadata."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the store at ``path``.

        Args:
            path: Directory for the Chroma database.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open_collection()

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(
            COLLECTION_NAME,
            configuration={"hnsw": {"space": DISTANCE_SPACE}},
            embedding_function=None,
        )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Insert or replace chunks with their vectors (batched to Chroma's size limit)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        batch = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=np.asarray(vectors[start : start + batch], dtype=np.float32),
                documents=[c.text for c in part],
                metadatas=[_to_metadata(c) for c in part],
            )

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, cosine_similarity)`` pairs, most similar first.

        Args:
            vector: Query embedding.
            k: Number of results.
            where: Optional Chroma metadata filter, e.g. ``{"doc_id": {"$in": [...]}}``.
        """
        n = min(k, self.count())
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=n,
            where=where,
            include=["distances"],
        )
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def get(self, ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id, in the order requested (unknown ids are skipped)."""
        if not ids:
            return []
        found = self._fetch(ids=ids)
        return [found[i] for i in ids if i in found]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, sorted by id (used to rebuild the BM25 index)."""
        found = self._fetch()
        return [found[i] for i in sorted(found)]

    def _fetch(self, ids: list[str] | None = None) -> dict[str, Chunk]:
        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            chunk_id: _from_record(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        }

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks of document ``doc_id`` in reading order (page, then chunk index)."""
        result = self._collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [
            _from_record(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]
        return sorted(chunks, key=lambda c: (c.page, c.chunk_index))

    def delete_doc(self, doc_id: str) -> None:
        """Remove every chunk of document ``doc_id``."""
        self._collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        """Delete all chunks (used by ``ingest --rebuild``)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._open_collection()

    def count(self) -> int:
        """Number of stored chunks."""
        return int(self._collection.count())


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump(exclude={"chunk_id", "text"})


def _from_record(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, **meta)
