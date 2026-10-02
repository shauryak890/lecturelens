"""Shared retrieve-then-generate logic for the study tools (quiz, summary, flashcards).

Every tool works on a :class:`Scope`: either a topic (the best-matching chunks are retrieved)
or a whole document (chunks are sampled evenly from start to end, so a summary covers the
entire lecture rather than one slide). The chunks become a numbered ``[S#]`` context, one LLM
call produces the structured output, and ``source_ids`` that point outside the context are
removed, the same check the RAG pipeline applies to citations.
"""

import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from lecturelens.config import Settings
from lecturelens.errors import NoContentError
from lecturelens.llm.client import LLMClient
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.rag.context import build_context
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import RetrievedChunk, Usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Scope:
    """What a study tool should cover: a ``topic`` or one document (``doc_id``)."""

    topic: str | None = None
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if bool(self.topic and self.topic.strip()) == bool(self.doc_id):
            raise ValueError("Give exactly one of topic or doc_id")


@dataclass
class StudyResult(Generic[T]):
    """A tool's validated output plus the sources behind its ``source_ids``."""

    output: T
    sources: list[RetrievedChunk]  # source_ids are 1-based indexes into this list
    usage: Usage
    scope_label: str
    notes: list[str] = field(default_factory=list)  # post-check messages for the UI


def valid_ids(ids: list[int], n_sources: int) -> list[int]:
    """Keep the source ids that exist in a context of ``n_sources`` excerpts (sorted, unique)."""
    return sorted({i for i in ids if 1 <= i <= n_sources})


def evenly_spaced(items: list, n: int) -> list:
    """Pick ``n`` items spread evenly from first to last (all of them if there are fewer)."""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


class StudyTool:
    """Base class: gathers context for a scope and runs one prompt-file task over it."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMClient,
        registry: PromptRegistry,
    ) -> None:
        """Create the tool from shared services (see :mod:`lecturelens.services`)."""
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.registry = registry

    def _gather(self, scope: Scope, n_chunks: int) -> tuple[list[RetrievedChunk], str]:
        """Chunks for the scope, best first, and a label naming the scope in the prompt."""
        if scope.doc_id:
            chunks = self.retriever.store.doc_chunks(scope.doc_id)
            if not chunks:
                raise NoContentError(f"Document {scope.doc_id} has no indexed text")
            picked = evenly_spaced(chunks, n_chunks)
            name = chunks[0].file_name
            return [RetrievedChunk(chunk=c, score=0.0, source="dense") for c in picked], name
        assert scope.topic is not None
        topic = scope.topic.strip()
        found = self.retriever.retrieve(topic, final_k=n_chunks)
        if not found:
            raise NoContentError(f"Nothing in your course material matches {topic!r}")
        return found, topic

    def _retrieve_and_generate(
        self, task: str, scope: Scope, n_chunks: int, label_var: str, **variables: object
    ) -> tuple[BaseModel, list[RetrievedChunk], Usage, str]:
        """Gather context for ``scope`` and run prompt-file ``task`` over it once.

        Args:
            task: Prompt-file task name.
            scope: Topic or document.
            n_chunks: How many chunks to gather (the context token budget still applies).
            label_var: Template variable that receives the scope label (the topic, or the
                document's file name).
            **variables: Other template variables.

        Returns:
            The validated output, the sources behind ``[S1]..[Sn]``, usage and the label.
        """
        chunks, label = self._gather(scope, n_chunks)
        context, sources = build_context(
            chunks, self.settings.retrieval.max_context_tokens, self.registry
        )
        output, usage = self.llm.run_task(task, context=context, **{label_var: label}, **variables)
        logger.info("%s on %r: %d sources, %s", task, label, len(sources), usage)
        return output, sources, usage, label
