"""One factory for everything the CLI and the Streamlit app need (SPEC 14: no global state
except cached singletons created in one factory).

Each component is built on first use, so opening the Library works without an API key and
without loading any model. The embedder and vector store are shared between indexing and
retrieval, so the embedding model loads once and search sees new documents immediately.
"""

import logging
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

from lecturelens.config import Settings
from lecturelens.indexing.embedder import Embedder, build_embedder
from lecturelens.indexing.indexer import (
    Indexer,
    IndexPaths,
    IndexStats,
    IngestReport,
    build_indexer,
    index_stats,
    load_or_build_bm25,
)
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.llm.client import LLMClient, build_llm_client
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.rag.pipeline import RAGPipeline, build_retriever
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.tools.flashcards import FlashcardGenerator
from lecturelens.tools.quiz import QuizGenerator
from lecturelens.tools.summarizer import Summarizer

logger = logging.getLogger(__name__)


class Services:
    """Lazily built, shared LectureLens components for one :class:`Settings`."""

    def __init__(self, settings: Settings) -> None:
        """Create the container; nothing heavy is built until first accessed."""
        self.settings = settings

    # ------------------------------------------------------------ building blocks

    @cached_property
    def registry(self) -> PromptRegistry:
        """The prompt file."""
        return PromptRegistry(Path(self.settings.prompts.path))

    @cached_property
    def embedder(self) -> Embedder:
        """The (lazily loaded) embedding model, shared by indexing and retrieval."""
        return build_embedder(self.settings.embeddings)

    @cached_property
    def store(self) -> VectorStore:
        """The Chroma vector store."""
        return VectorStore(IndexPaths(self.settings.app.index_dir).chroma)

    @cached_property
    def indexer(self) -> Indexer:
        """Ingestion into the shared store."""
        return build_indexer(self.settings, self.embedder, self.store)

    @cached_property
    def retriever(self) -> HybridRetriever:
        """Hybrid retrieval over the shared store."""
        return build_retriever(self.settings, self.embedder, self.store)

    @cached_property
    def llm(self) -> LLMClient:
        """The LLM client. Raises ConfigError on first use if the API key is missing."""
        return build_llm_client(self.settings, self.registry)

    # ------------------------------------------------------------ features

    @cached_property
    def pipeline(self) -> RAGPipeline:
        """Grounded question answering."""
        return RAGPipeline(self.settings, self.retriever, self.llm, self.registry)

    @cached_property
    def quiz(self) -> QuizGenerator:
        """Quiz generator."""
        return QuizGenerator(self.settings, self.retriever, self.llm, self.registry)

    @cached_property
    def summarizer(self) -> Summarizer:
        """Summary generator."""
        return Summarizer(self.settings, self.retriever, self.llm, self.registry)

    @cached_property
    def flashcards(self) -> FlashcardGenerator:
        """Flashcard generator."""
        return FlashcardGenerator(self.settings, self.retriever, self.llm, self.registry)

    # ------------------------------------------------------------ index management

    def stats(self) -> IndexStats:
        """Index summary (no model loading)."""
        return index_stats(self.settings, self.store)

    def ingest(
        self, folder: Path, *, rebuild: bool = False, on_file: Callable[[Path], None] | None = None
    ) -> IngestReport:
        """Index a folder, then make retrieval see the new chunks."""
        report = self.indexer.ingest(folder, rebuild=rebuild, on_file=on_file)
        self._refresh_bm25()
        return report

    def delete_document(self, doc_id: str) -> str:
        """Remove a document from the index; returns its file name."""
        name = self.indexer.delete_document(doc_id)
        self._refresh_bm25()
        return name

    def _refresh_bm25(self) -> None:
        # Chroma is shared and always current; the in-memory BM25 index must be reloaded.
        if "retriever" in self.__dict__:
            self.retriever.bm25 = load_or_build_bm25(self.settings, self.store)
