"""RAGPipeline.ask(): condense -> retrieve -> generate -> validate citations (SPEC 3.2, 6.7).

Quota-saving short-circuits: no condense call on the first turn, and no answer call at all
when the index is empty or retrieval finds nothing relevant (the fixed "not found" message is
returned instead).
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lecturelens.config import AnswerMode, RetrievalMode, Settings
from lecturelens.errors import IndexMismatchError, LLMBlockedError
from lecturelens.indexing.embedder import build_embedder
from lecturelens.indexing.indexer import IndexPaths, load_or_build_bm25
from lecturelens.indexing.manifest import REBUILD_HINT, Manifest
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.llm.client import LLMClient, build_llm_client
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.rag import citations
from lecturelens.rag.context import build_context
from lecturelens.retrieval.reranker import CrossEncoderReranker
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import (
    AnswerResponse,
    AskResult,
    ChatTurn,
    CondensedQuestion,
    RetrievedChunk,
    Usage,
)

logger = logging.getLogger(__name__)

ANSWER_TASK = "answer"
MS_PER_S = 1000


@contextmanager
def stopwatch(timings: dict[str, float], name: str) -> Iterator[None]:
    """Record the wall time of the ``with`` block as ``timings[f"{name}_ms"]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[f"{name}_ms"] = (time.perf_counter() - start) * MS_PER_S


class RAGPipeline:
    """Question answering over the indexed course material."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMClient,
        registry: PromptRegistry,
    ) -> None:
        """Create the pipeline from its parts (see :func:`build_pipeline`)."""
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.registry = registry

    def ask(
        self,
        question: str,
        history: list[ChatTurn] | None = None,
        mode: AnswerMode | None = None,
        retrieval_mode: RetrievalMode | None = None,
        doc_filter: list[str] | None = None,
        rerank: bool | None = None,
    ) -> AskResult:
        """Answer ``question`` from the course material with ``[S#]`` citations.

        Args:
            question: The student's question (may be a follow-up).
            history: Earlier turns, oldest first; used to condense follow-ups.
            mode: Answer style: ``concise``, ``detailed`` or ``eli5``.
            retrieval_mode: Override ``retrieval.mode`` for this question.
            doc_filter: Only search these ``doc_id`` values.
            rerank: Override ``retrieval.rerank`` for this question.

        Returns:
            The validated answer, the sources behind ``[S1]..[Sn]``, timings and usage.

        Raises:
            LLMError: If the LLM fails permanently (after retries, fallback and repair).
        """
        timings: dict[str, float] = {}
        usage = Usage()
        started = time.perf_counter()

        with stopwatch(timings, "condense"):
            standalone, usage = self._condense(question, history, usage)
        if self.retriever.is_empty():
            return self._result(
                question, standalone, self._fixed("empty_index"), [], timings, usage, started
            )
        with stopwatch(timings, "retrieve"):
            chunks = self.retriever.retrieve(
                standalone, mode=retrieval_mode, doc_filter=doc_filter, rerank=rerank
            )
        if not chunks:  # nothing relevant: abstain without spending an LLM call
            return self._result(
                question, standalone, self._fixed("not_found"), [], timings, usage, started
            )

        context, sources = build_context(
            chunks, self.settings.retrieval.max_context_tokens, self.registry
        )
        with stopwatch(timings, "generate"):
            response, usage = self._answer(
                standalone,
                context,
                len(sources),
                mode or self.settings.generation.default_mode,
                usage,
            )
        return self._result(question, standalone, response, sources, timings, usage, started)

    # -------------------------------------------------------------- steps

    def _condense(
        self, question: str, history: list[ChatTurn] | None, usage: Usage
    ) -> tuple[str, Usage]:
        """Rewrite a follow-up into a standalone question (skipped on the first turn)."""
        turns = (history or [])[-self.settings.generation.history_turns :]
        if not turns or not self.settings.generation.history_turns:
            return question, usage
        separator = self.registry.format("history_separator")
        rendered = separator.join(
            self.registry.format("history_turn", question=t.question, answer=t.answer)
            for t in turns
        )
        result, call_usage = self.llm.run_task(
            "condense_question", history=rendered, question=question
        )
        assert isinstance(result, CondensedQuestion)
        standalone = result.standalone_question.strip() or question
        logger.info("Condensed follow-up %r -> %r", question, standalone)
        return standalone, usage + call_usage

    def _answer(
        self, question: str, context: str, n_sources: int, mode: AnswerMode, usage: Usage
    ) -> tuple[AnswerResponse, Usage]:
        """Generate the answer, validate its citations, and repair once if needed."""
        gen = self.settings.generation
        instruction = self.registry.mode_instruction(
            ANSWER_TASK, mode, max_words=gen.max_words, max_words_detailed=gen.max_words_detailed
        )
        try:
            raw, call_usage = self.llm.run_task(
                ANSWER_TASK, context=context, question=question, mode_instruction=instruction
            )
        except LLMBlockedError as exc:
            logger.warning("Answer blocked: %s", exc)
            return self._fixed("blocked"), usage
        usage = usage + call_usage
        assert isinstance(raw, AnswerResponse)
        check = citations.validate(raw, n_sources)
        if check.removed:
            logger.info("Removed invalid citations %s", check.removed)
        if not check.needs_repair:
            return check.response, usage
        if not gen.repair_on_invalid_citations:
            return check.response.model_copy(update={"confidence": "low"}), usage
        return self._repair(raw, check, question, context, n_sources, usage)

    def _repair(
        self,
        raw: AnswerResponse,
        check: citations.CitationCheck,
        question: str,
        context: str,
        n_sources: int,
        usage: Usage,
    ) -> tuple[AnswerResponse, Usage]:
        problem = (
            self.registry.format("problem_invalid_ids", ids=", ".join(map(str, check.removed)))
            if check.removed
            else self.registry.format("problem_no_citations")
        )
        try:
            repaired, call_usage = self.llm.run_task(
                "repair_answer",
                problem=problem,
                valid_ids=", ".join(str(i) for i in range(1, n_sources + 1)),
                previous=raw.answer,
                context=context,
                question=question,
            )
        except LLMBlockedError:
            return check.response.model_copy(update={"confidence": "low"}), usage
        assert isinstance(repaired, AnswerResponse)
        second = citations.validate(repaired, n_sources)
        response = second.response
        if second.needs_repair:  # still ungrounded after one repair: keep it, flag it
            response = response.model_copy(update={"confidence": "low"})
        return response, usage + call_usage

    # -------------------------------------------------------------- helpers

    def _fixed(self, message: str) -> AnswerResponse:
        return AnswerResponse(
            answer=self.registry.message(message),
            cited_sources=[],
            answerable=False,
            confidence="low",
        )

    @staticmethod
    def _result(
        question: str,
        standalone: str,
        response: AnswerResponse,
        sources: list[RetrievedChunk],
        timings: dict[str, float],
        usage: Usage,
        started: float,
    ) -> AskResult:
        timings["total_ms"] = (time.perf_counter() - started) * MS_PER_S
        return AskResult(
            question=question,
            standalone_question=standalone,
            response=response,
            sources=sources,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                "llm_calls": usage.api_calls,
                "cache_hits": usage.cache_hits,
            },
        )


def build_retriever(settings: Settings) -> HybridRetriever:
    """Open the index at ``app.index_dir`` and create the retriever (models load lazily).

    Raises:
        IndexMismatchError: If the index was built with a different embedding model.
    """
    paths = IndexPaths(settings.app.index_dir)
    embedder = build_embedder(settings.embeddings)
    manifest = Manifest.load(paths.manifest)
    if manifest.files and manifest.embedding_model != embedder.model_name:
        raise IndexMismatchError(
            f"The index was built with {manifest.embedding_model} but config uses "
            f"{embedder.model_name}. {REBUILD_HINT}"
        )
    store = VectorStore(paths.chroma)
    reranker = CrossEncoderReranker(settings.retrieval.reranker_model, settings.embeddings.device)
    return HybridRetriever(
        embedder,
        store,
        load_or_build_bm25(settings, store),
        reranker,
        settings.retrieval,
        settings.ingestion.add_context_header,
    )


def build_pipeline(settings: Settings) -> RAGPipeline:
    """The single factory for the question-answering stack (CLI and Streamlit share it).

    Raises:
        ConfigError: If the API key is missing.
        IndexMismatchError: If the index does not match the configured embedding model.
    """
    registry = PromptRegistry(Path(settings.prompts.path))
    llm = build_llm_client(settings, registry)
    return RAGPipeline(settings, build_retriever(settings), llm, registry)
