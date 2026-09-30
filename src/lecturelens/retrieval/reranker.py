"""Cross-encoder reranking (SPEC 7.6).

A bi-encoder (the embedder) encodes query and chunk separately: fast but coarse. A cross-encoder
reads the pair together and outputs a relevance score: slower but much more precise, so it only
rescores the top fused candidates. Scores are raw logits (higher = more relevant).
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (default ms-marco-MiniLM-L-6-v2), loaded lazily."""

    def __init__(self, model_name: str, device: str) -> None:
        """Create the reranker; the model loads on the first :meth:`score` call.

        Args:
            model_name: Hugging Face cross-encoder id.
            device: Torch device, e.g. ``"cpu"``.
        """
        self.model_name = model_name
        self._device = device
        self._model: Any = None

    def _load(self) -> Any:
        from lecturelens.indexing.embedder import quiet_hf_libraries

        quiet_hf_libraries()
        from sentence_transformers import CrossEncoder  # heavy: imports torch

        model = CrossEncoder(self.model_name, device=self._device)
        logger.info("Loaded reranker %s on %s", self.model_name, self._device)
        return model

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage against ``query``."""
        if not passages:
            return []
        if self._model is None:
            self._model = self._load()
        scores = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
