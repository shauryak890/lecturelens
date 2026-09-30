"""Embedders: the ``Embedder`` protocol and the local bge-small implementation.

``LocalEmbedder`` runs sentence-transformers on the CPU, so indexing costs no API quota
(NFR-1). bge-small is asymmetric: queries get an instruction prefix, documents do not.
"""

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from lecturelens.config import EmbeddingCfg
from lecturelens.errors import ConfigError

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into L2-normalised float32 vectors of dimension ``dim``."""

    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages; returns an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query; returns an array of shape ``(dim,)``."""
        ...


def quiet_hf_libraries() -> None:
    """Hide Hugging Face progress bars and info/warning chatter (errors still surface)."""
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging as tf_logging

    disable_progress_bars()
    hub_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()
    tf_logging.set_verbosity_error()


class LocalEmbedder:
    """sentence-transformers embedder (default ``BAAI/bge-small-en-v1.5``)."""

    def __init__(self, cfg: EmbeddingCfg) -> None:
        """Load the model (downloaded once, then read from the local Hugging Face cache).

        Args:
            cfg: Embedding settings: model id, device, batch size, query instruction.
        """
        quiet_hf_libraries()
        from sentence_transformers import SentenceTransformer  # heavy: imports torch

        self._cfg = cfg
        self._model = SentenceTransformer(cfg.model, device=cfg.device)
        self.model_name = cfg.model
        # renamed in sentence-transformers 6; keep the old name as a fallback
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        logger.info("Loaded embedder %s (dim=%d) on %s", self.model_name, self.dim, cfg.device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages in batches of ``cfg.batch_size`` (no instruction prefix)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with bge's retrieval instruction prefix."""
        return self._encode([self._cfg.query_instruction + text])[0]


class LazyEmbedder:
    """Embedder proxy that loads the real model on first use.

    ``model_name`` is known up front, so index compatibility can be checked (and unchanged
    files skipped) without paying the ~15 s cost of importing torch and loading the model.
    """

    def __init__(self, model_name: str, loader: Callable[[], Embedder]) -> None:
        """Create the proxy.

        Args:
            model_name: Name of the model ``loader`` will produce.
            loader: Zero-argument factory for the real embedder.
        """
        self.model_name = model_name
        self._loader = loader
        self._inner: Embedder | None = None

    @property
    def loaded(self) -> bool:
        """Whether the real model has been loaded."""
        return self._inner is not None

    @property
    def inner(self) -> Embedder:
        """The real embedder, loaded on first access."""
        if self._inner is None:
            self._inner = self._loader()
        return self._inner

    @property
    def dim(self) -> int:
        """Vector dimension (loads the model)."""
        return self.inner.dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages with the real model."""
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the real model."""
        return self.inner.embed_query(text)


def build_embedder(cfg: EmbeddingCfg) -> Embedder:
    """Create the embedder selected by ``embeddings.provider`` (loaded lazily on first use).

    Raises:
        ConfigError: For providers not implemented yet.
    """
    if cfg.provider == "local":
        return LazyEmbedder(cfg.model, lambda: LocalEmbedder(cfg))
    raise ConfigError(
        f"embeddings.provider {cfg.provider!r} is not implemented; use 'local' (see DECISIONS.md)"
    )
