"""Offline stand-ins for heavy models (SPEC 13): no downloads, deterministic output."""

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar unit vectors."""

    def __init__(self, dim: int = 64, model_name: str = "hash-embedder") -> None:
        self.dim = dim
        self.model_name = model_name

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "little") % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class WhitespaceCounter:
    """Token counter where every whitespace-separated word is one token."""

    def count(self, text: str) -> int:
        return len(text.split())
