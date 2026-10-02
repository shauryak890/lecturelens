"""Pydantic data contracts shared across the package.

Two groups of models live here:

* Pipeline data (``Page``, ``Chunk``, ``RetrievedChunk``, ``AskResult``, ``Usage``,
  ``LLMCallRecord``) passed between modules.
* LLM output schemas (``CondensedQuestion``, ``AnswerResponse``, ``Quiz``, ...) that are sent to
  the API as JSON schema and used to validate the response. The prompt file refers to them by
  class name; :data:`LLM_SCHEMAS` maps those names to classes.
"""

from typing import Literal

from pydantic import BaseModel, Field

from lecturelens.errors import PromptError

# ---------------------------------------------------------------- pipeline data


class Page(BaseModel):
    """One page of a parsed source document (MD/TXT files are a single page)."""

    doc_id: str  # sha256 of file content (first 16 hex chars)
    file_name: str
    page: int  # 1-based page number shown to users
    text: str


class Chunk(BaseModel):
    """A retrievable piece of a single page, with its citation metadata."""

    chunk_id: str  # f"{doc_id}:{page}:{idx}"
    doc_id: str
    file_name: str
    page: int
    heading_path: str = ""
    chunk_index: int
    text: str
    n_tokens: int


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its score and per-retriever ranks."""

    chunk: Chunk
    score: float
    source: Literal["dense", "bm25", "hybrid", "rerank"]
    ranks: dict[str, int] = {}  # e.g. {"dense": 3, "bm25": 1} for explainability


# ---------------------------------------------------------------- LLM output schemas


class CondensedQuestion(BaseModel):
    """Output of the condense_question task."""

    standalone_question: str


class AnswerResponse(BaseModel):
    """Output of the answer and repair_answer tasks."""

    answer: str = Field(description="Markdown answer with inline [S#] citations")
    cited_sources: list[int] = Field(description="Source numbers actually used, e.g. [1, 3]")
    answerable: bool = Field(description="False if the excerpts do not contain the answer")
    confidence: Literal["high", "medium", "low"]


class QuizQuestion(BaseModel):
    """A single multiple-choice question."""

    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    source_ids: list[int]
    difficulty: Literal["easy", "medium", "hard"]
    bloom_level: Literal["remember", "understand", "apply", "analyze"]


class Quiz(BaseModel):
    """Output of the quiz task."""

    topic: str
    questions: list[QuizQuestion]


class KeyPoint(BaseModel):
    """A cited key point in a summary."""

    point: str
    source_ids: list[int]


class GlossaryItem(BaseModel):
    """A cited glossary term in a summary."""

    term: str
    definition: str
    source_ids: list[int]


class Summary(BaseModel):
    """Output of the summarize task."""

    title: str
    overview: str
    key_points: list[KeyPoint]
    glossary: list[GlossaryItem]


class Flashcard(BaseModel):
    """A single question/answer flashcard."""

    front: str
    back: str
    source_ids: list[int]


class FlashcardSet(BaseModel):
    """Output of the flashcards task."""

    cards: list[Flashcard]


class FaithfulnessJudgement(BaseModel):
    """Output of the judge_faithfulness task (claims and parallel support flags)."""

    claims: list[str]
    supported: list[bool]
    reasoning: str


class RelevancyJudgement(BaseModel):
    """Output of the judge_relevancy task."""

    score: int = Field(ge=1, le=5)
    reasoning: str


# ---------------------------------------------------------------- results and telemetry


class ChatTurn(BaseModel):
    """One earlier question/answer exchange, used to condense follow-up questions."""

    question: str
    answer: str


class AskResult(BaseModel):
    """What the RAG pipeline returns to the UI and CLI."""

    question: str
    standalone_question: str
    response: AnswerResponse
    sources: list[RetrievedChunk]  # index i corresponds to [S{i+1}]
    timings_ms: dict[str, float]
    usage: dict[str, int]  # prompt_tokens, output_tokens, llm_calls, cache_hits across calls
    removed_citations: list[int] = []  # invalid [S#] ids stripped from the first answer
    repaired: bool = False  # whether the one-time citation repair call was made


class Usage(BaseModel):
    """Token and request counts of one or more LLM calls (add them with ``+``)."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0  # HTTP requests actually sent (retries and repairs included)
    cache_hits: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            api_calls=self.api_calls + other.api_calls,
            cache_hits=self.cache_hits + other.cache_hits,
        )


class LLMCallRecord(BaseModel):
    """One line of logs/llm_calls.jsonl. Never holds the API key or document text."""

    ts: str
    task: str
    model: str
    prompt_version: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    attempts: int = 1
    fallback_used: bool = False
    status: str = "ok"


LLM_SCHEMAS: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in (
        CondensedQuestion,
        AnswerResponse,
        Quiz,
        Summary,
        FlashcardSet,
        FaithfulnessJudgement,
        RelevancyJudgement,
    )
}


def get_schema(name: str) -> type[BaseModel]:
    """Return the LLM output schema class registered under ``name``.

    Args:
        name: Class name as written in the prompt file's ``schema`` key.

    Returns:
        The Pydantic model class.

    Raises:
        PromptError: If no schema with that name exists.
    """
    try:
        return LLM_SCHEMAS[name]
    except KeyError:
        known = ", ".join(sorted(LLM_SCHEMAS))
        raise PromptError(f"Unknown output schema {name!r}. Known schemas: {known}") from None
