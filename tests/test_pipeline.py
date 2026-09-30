"""End-to-end RAGPipeline tests: real index of the sample notes, FakeLLMClient, no network."""

from pathlib import Path

import pytest

from lecturelens.errors import LLMBlockedError
from lecturelens.indexing.indexer import Indexer
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.rag.pipeline import RAGPipeline
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import AnswerResponse, AskResult, ChatTurn, CondensedQuestion

from .fakes import FakeLLMClient, FakeReranker

GROUNDED = AnswerResponse(
    answer="Kneser-Ney uses a continuation probability [S1].",
    cited_sources=[1],
    answerable=True,
    confidence="high",
)


def _pipeline(
    indexer: Indexer,
    registry: PromptRegistry,
    responses: dict,
    reranker: FakeReranker | None = None,
) -> tuple[RAGPipeline, FakeLLMClient]:
    s = indexer.settings
    retriever = HybridRetriever(
        indexer.embedder,
        indexer.store,
        indexer.load_bm25(),
        reranker,
        s.retrieval,
        s.ingestion.add_context_header,
    )
    llm = FakeLLMClient(s.llm, registry, responses)
    return RAGPipeline(s, retriever, llm, registry), llm


def test_ask_returns_grounded_result(tiny_index: Indexer, registry: PromptRegistry) -> None:
    pipeline, llm = _pipeline(tiny_index, registry, {"answer": [GROUNDED]})
    result = pipeline.ask("What is Kneser-Ney smoothing?")
    assert isinstance(result, AskResult)
    assert result.response == GROUNDED
    assert result.standalone_question == result.question  # first turn: no condense call
    assert llm.tasks_called() == ["answer"]
    assert 0 < len(result.sources) <= tiny_index.settings.retrieval.final_k
    assert set(result.timings_ms) == {"condense_ms", "retrieve_ms", "generate_ms", "total_ms"}
    assert result.usage["llm_calls"] == 1 and result.usage["prompt_tokens"] > 0

    prompt = llm.calls[0]
    assert prompt.system == registry.system("tutor")
    assert prompt.schema is AnswerResponse
    first = result.sources[0].chunk
    assert f"[S1] ({first.file_name}, p.{first.page}" in prompt.user  # sources match context
    assert "<excerpts>" in prompt.user and "at most 180 words" in prompt.user  # concise mode


def test_answer_mode_changes_instruction(tiny_index: Indexer, registry: PromptRegistry) -> None:
    pipeline, llm = _pipeline(tiny_index, registry, {"answer": [GROUNDED]})
    pipeline.ask("What is Kneser-Ney smoothing?", mode="detailed")
    assert "up to 450 words" in llm.calls[0].user


def test_follow_up_triggers_condense_and_uses_standalone_query(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    standalone = "What are the limitations of n-gram language models?"
    pipeline, llm = _pipeline(
        tiny_index,
        registry,
        {
            "condense_question": [CondensedQuestion(standalone_question=standalone)],
            "answer": [GROUNDED],
        },
    )
    history = [ChatTurn(question="What is an n-gram model?", answer="It predicts words.")]
    result = pipeline.ask("what are its limitations?", history=history)
    assert llm.tasks_called() == ["condense_question", "answer"]
    assert "Student: What is an n-gram model?" in llm.calls[0].user
    assert result.standalone_question == standalone
    assert result.question == "what are its limitations?"
    assert standalone in llm.calls[1].user  # the answer prompt uses the standalone question
    assert result.sources[0].chunk.file_name == "02_ngram_language_models.md"


def test_empty_index_abstains_without_llm_call(
    tiny_settings,
    registry: PromptRegistry,
    tmp_path: Path,
) -> None:
    from .fakes import HashEmbedder, WhitespaceCounter

    empty = Indexer(
        tiny_settings,
        HashEmbedder(dim=128),
        WhitespaceCounter(),
        store=VectorStore(tmp_path / "empty_chroma"),
    )
    pipeline, llm = _pipeline(empty, registry, {})
    result = pipeline.ask("What is BPE?")
    assert llm.calls == []
    assert not result.response.answerable and result.sources == []
    assert result.response.answer == registry.message("empty_index")


def test_nothing_relevant_abstains_without_llm_call(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    pipeline, llm = _pipeline(tiny_index, registry, {}, reranker=FakeReranker(offset=100.0))
    result = pipeline.ask("How does RLHF work?")
    assert llm.calls == [] and result.sources == []
    assert result.response.answer == registry.message("not_found")
    assert result.usage["llm_calls"] == 0


def test_invalid_citation_is_stripped(tiny_index: Indexer, registry: PromptRegistry) -> None:
    hallucinated = GROUNDED.model_copy(
        update={"answer": "Continuation probability [S1][S42].", "cited_sources": [1, 42]}
    )
    pipeline, llm = _pipeline(tiny_index, registry, {"answer": [hallucinated]})
    result = pipeline.ask("What is Kneser-Ney smoothing?")
    assert result.response.answer == "Continuation probability [S1]."
    assert result.response.cited_sources == [1]
    assert llm.tasks_called() == ["answer"]  # a valid citation remains: no repair needed


def test_uncited_answer_is_repaired_once(tiny_index: Indexer, registry: PromptRegistry) -> None:
    uncited = GROUNDED.model_copy(
        update={"answer": "Uses continuation counts.", "cited_sources": []}
    )
    pipeline, llm = _pipeline(
        tiny_index, registry, {"answer": [uncited], "repair_answer": [GROUNDED]}
    )
    result = pipeline.ask("What is Kneser-Ney smoothing?")
    assert llm.tasks_called() == ["answer", "repair_answer"]
    assert "cites no valid excerpt" in llm.calls[1].user
    assert result.response == GROUNDED and result.usage["llm_calls"] == 2


def test_still_uncited_after_repair_gets_low_confidence(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    uncited = GROUNDED.model_copy(update={"answer": "No citations.", "cited_sources": [9]})
    pipeline, llm = _pipeline(
        tiny_index, registry, {"answer": [uncited], "repair_answer": [uncited]}
    )
    result = pipeline.ask("What is Kneser-Ney smoothing?")
    assert "do not exist: 9" in llm.calls[1].user
    assert result.response.confidence == "low" and result.response.cited_sources == []


def test_blocked_answer_becomes_friendly_abstention(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    pipeline, _ = _pipeline(tiny_index, registry, {"answer": [LLMBlockedError("SAFETY")]})
    result = pipeline.ask("What is Kneser-Ney smoothing?")
    assert result.response.answer == registry.message("blocked")
    assert not result.response.answerable


@pytest.mark.parametrize("turns", [0, 5])
def test_history_window(tiny_index: Indexer, registry: PromptRegistry, turns: int) -> None:
    s = tiny_index.settings.with_overrides({"generation.history_turns": 3})
    tiny_index.settings = s
    pipeline, llm = _pipeline(
        tiny_index,
        registry,
        {
            "condense_question": [CondensedQuestion(standalone_question="What is BPE?")],
            "answer": [GROUNDED],
        },
    )
    history = [ChatTurn(question=f"q{i}", answer=f"a{i}") for i in range(turns)]
    pipeline.ask("and its merges?", history=history)
    if turns == 0:
        assert llm.tasks_called() == ["answer"]
    else:  # only the last history_turns turns are sent
        assert "q1" not in llm.calls[0].user and "q2" in llm.calls[0].user
