"""Tests for the study tools and the Services container (offline, sample notes + fakes)."""

import csv
import io
from pathlib import Path

import pytest

from lecturelens.errors import NoContentError
from lecturelens.indexing.indexer import Indexer, find_document
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import (
    Flashcard,
    FlashcardSet,
    GlossaryItem,
    KeyPoint,
    Quiz,
    QuizQuestion,
    Summary,
)
from lecturelens.services import Services
from lecturelens.tools.base import Scope, evenly_spaced
from lecturelens.tools.flashcards import FlashcardGenerator, to_csv
from lecturelens.tools.quiz import QuizGenerator, jaccard, post_check, shuffle_options
from lecturelens.tools.summarizer import Summarizer

from .fakes import FakeLLMClient, FakeReranker


def _q(
    text: str, options: list[str] | None = None, correct: int = 0, sources: list[int] | None = None
) -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=options or ["right answer", "wrong one", "wrong two", "wrong three"],
        correct_index=correct,
        explanation="Because.",
        source_ids=[1] if sources is None else sources,
        difficulty="easy",
        bloom_level="remember",
    )


def _tool(
    cls,
    indexer: Indexer,
    registry: PromptRegistry,
    responses: dict,
    reranker: FakeReranker | None = None,
):
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
    return cls(s, retriever, llm, registry), llm


def _doc_id(indexer: Indexer, file_name: str) -> str:
    return next(c.doc_id for c in indexer.store.all_chunks() if c.file_name == file_name)


# ---------------------------------------------------------------- helpers


def test_scope_needs_exactly_one_of_topic_or_doc() -> None:
    Scope(topic="BPE")
    Scope(doc_id="abc")
    with pytest.raises(ValueError):
        Scope()
    with pytest.raises(ValueError):
        Scope(topic="BPE", doc_id="abc")
    with pytest.raises(ValueError):
        Scope(topic="   ")


def test_evenly_spaced_covers_start_and_end() -> None:
    assert evenly_spaced(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced([1, 2], 5) == [1, 2]
    assert evenly_spaced(list(range(5)), 1) == [0]


# ---------------------------------------------------------------- quiz post-checks


def test_shuffle_keeps_the_correct_answer() -> None:
    import random

    q = _q("Q?", ["a", "b", "c", "d"], correct=2)
    shuffled = shuffle_options(q, random.Random(7))
    assert sorted(shuffled.options) == ["a", "b", "c", "d"]
    assert shuffled.options[shuffled.correct_index] == "c"


def test_post_check_drops_bad_questions_and_explains() -> None:
    quiz = Quiz(
        topic="t",
        questions=[
            _q("What does BPE merge first?"),
            _q("Duplicate options?", ["same", "Same", "x", "y"]),
            _q("Catch-all?", ["a", "b", "c", "All of the above"]),
            _q("Ungrounded question?", sources=[9]),
            _q("What does BPE merge first exactly?"),  # near-duplicate of question 1
            _q("How is WordPiece different from BPE?"),
        ],
    )
    cleaned, notes = post_check(quiz, n_sources=3, n=5, seed=42, dedup_similarity=0.6)
    assert [q.question for q in cleaned.questions] == [
        "What does BPE merge first?",
        "How is WordPiece different from BPE?",
    ]
    assert len(notes) == 4
    assert any("near-duplicate" in n for n in notes) and any("no valid source" in n for n in notes)


def test_post_check_shuffle_is_seeded_and_varies_position() -> None:
    quiz = Quiz(
        topic="t",
        questions=[
            _q(f"Distinct question number {w}?")
            for w in ("one", "two", "three", "four", "five", "six")
        ],
    )
    first, _ = post_check(quiz, 1, n=6, seed=42, dedup_similarity=0.95)
    again, _ = post_check(quiz, 1, n=6, seed=42, dedup_similarity=0.95)
    assert first == again  # reproducible
    assert len({q.correct_index for q in first.questions}) > 1  # answer is not always "A"
    assert all(q.options[q.correct_index] == "right answer" for q in first.questions)


def test_post_check_truncates_to_n() -> None:
    quiz = Quiz(
        topic="t", questions=[_q(f"Unrelated question {w}?") for w in ("alpha", "beta", "gamma")]
    )
    cleaned, _ = post_check(quiz, 1, n=2, seed=1, dedup_similarity=0.9)
    assert len(cleaned.questions) == 2


def test_jaccard() -> None:
    assert jaccard("What is BPE?", "what is bpe") == 1.0
    assert jaccard("a b", "c d") == 0.0


# ---------------------------------------------------------------- generators


def test_quiz_on_topic_uses_retrieved_context(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    quiz = Quiz(topic="x", questions=[_q("What does Kneser-Ney use?", sources=[1, 2, 99])])
    tool, llm = _tool(QuizGenerator, tiny_index, registry, {"quiz": [quiz]})
    result = tool.generate(Scope(topic="Kneser-Ney smoothing"), n=99, difficulty="hard")
    prompt = llm.calls[0].user
    assert "Create 15 multiple-choice questions about: Kneser-Ney smoothing" in prompt  # max_n
    assert "Difficulty: hard" in prompt and "[S1]" in prompt
    assert result.scope_label == "Kneser-Ney smoothing" == result.output.topic
    assert result.output.questions[0].source_ids == [1, 2]  # 99 does not exist
    assert result.sources and any("1 of 15" in n for n in result.notes)


def test_quiz_on_document_samples_across_it(tiny_index: Indexer, registry: PromptRegistry) -> None:
    doc_id = _doc_id(tiny_index, "03_word_embeddings.md")
    quiz = Quiz(topic="x", questions=[_q("What is skip-gram?")])
    tool, llm = _tool(QuizGenerator, tiny_index, registry, {"quiz": [quiz]})
    result = tool.generate(Scope(doc_id=doc_id))
    assert result.scope_label == "03_word_embeddings.md"
    assert "about: 03_word_embeddings.md" in llm.calls[0].user
    doc_chunks = tiny_index.store.doc_chunks(doc_id)
    used = [s.chunk.chunk_id for s in result.sources]
    assert used[0] == doc_chunks[0].chunk_id  # sampling starts at the beginning...
    assert {s.chunk.file_name for s in result.sources} == {"03_word_embeddings.md"}


def test_quiz_with_no_usable_question_raises(tiny_index: Indexer, registry: PromptRegistry) -> None:
    bad = Quiz(topic="x", questions=[_q("Ungrounded?", sources=[])])
    tool, _ = _tool(QuizGenerator, tiny_index, registry, {"quiz": [bad]})
    with pytest.raises(NoContentError, match="no usable questions"):
        tool.generate(Scope(topic="n-gram models"))


def test_nothing_relevant_raises_without_llm_call(
    tiny_index: Indexer, registry: PromptRegistry
) -> None:
    tool, llm = _tool(QuizGenerator, tiny_index, registry, {}, reranker=FakeReranker(offset=100.0))
    with pytest.raises(NoContentError, match="RLHF"):
        tool.generate(Scope(topic="RLHF"))
    assert llm.calls == []


def test_summary_strips_invalid_source_ids(tiny_index: Indexer, registry: PromptRegistry) -> None:
    summary = Summary(
        title="N-grams",
        overview="Overview.",
        key_points=[KeyPoint(point="Markov assumption", source_ids=[1, 42])],
        glossary=[
            GlossaryItem(term="Perplexity", definition="Inverse probability.", source_ids=[42])
        ],
    )
    tool, llm = _tool(Summarizer, tiny_index, registry, {"summarize": [summary]})
    result = tool.summarize(Scope(topic="n-gram language models"))
    assert result.output.key_points[0].source_ids == [1]
    assert result.output.glossary[0].source_ids == []
    assert "Summarise the material about: n-gram language models" in llm.calls[0].user
    assert "7 points" in llm.calls[0].user and "up to 8" in llm.calls[0].user


def test_flashcards_dedupe_and_csv(tiny_index: Indexer, registry: PromptRegistry) -> None:
    cards = FlashcardSet(
        cards=[
            Flashcard(front="What is BPE?", back="Byte Pair Encoding.", source_ids=[1]),
            Flashcard(front="what is bpe?", back="Duplicate front.", source_ids=[1]),
            Flashcard(front="  ", back="Empty front.", source_ids=[1]),
            Flashcard(
                front="WordPiece?", back='Used by BERT, "##" marks pieces.', source_ids=[1, 2]
            ),
        ]
    )
    tool, _ = _tool(FlashcardGenerator, tiny_index, registry, {"flashcards": [cards]})
    result = tool.generate(Scope(topic="subword tokenization"), n=10)
    assert [c.front for c in result.output.cards] == ["What is BPE?", "WordPiece?"]
    assert result.notes == ["Dropped 2 empty or duplicate card(s)."]

    rows = list(csv.reader(io.StringIO(to_csv(result.output, result.sources))))
    assert rows[0] == ["front", "back", "source"]
    assert rows[2][1] == 'Used by BERT, "##" marks pieces.'  # quoting round-trips
    first = result.sources[0].chunk
    assert rows[1][2] == f"{first.file_name} p.{first.page}"
    assert rows[2][2].count(";") == 1


# ---------------------------------------------------------------- services and lookup


def test_find_document_by_name(tiny_index: Indexer) -> None:
    s = tiny_index.settings
    assert find_document(s, "03_word_embeddings.md").file_name == "03_word_embeddings.md"
    assert find_document(s, "EMBEDDINGS").file_name == "03_word_embeddings.md"  # substring
    with pytest.raises(NoContentError, match="several"):
        find_document(s, ".md")
    with pytest.raises(NoContentError, match="no indexed document"):
        find_document(s, "missing.pdf")


def test_services_refresh_bm25_after_ingest_and_delete(tiny_index: Indexer, tmp_path: Path) -> None:
    services = Services(tiny_index.settings)
    services.__dict__.update(
        embedder=tiny_index.embedder, store=tiny_index.store, indexer=tiny_index
    )  # inject offline components
    query = "zygomorphic orchids pollination"
    assert not services.retriever.retrieve(query, mode="bm25", rerank=False)

    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "botany.md").write_bytes(b"# Botany\nZygomorphic orchids rely on pollination.\n")
    services.ingest(extra)
    hits = services.retriever.retrieve(query, mode="bm25", rerank=False)
    assert hits and hits[0].chunk.file_name == "botany.md"

    name = services.delete_document(hits[0].chunk.doc_id)
    assert name == "botany.md"
    assert not services.retriever.retrieve(query, mode="bm25", rerank=False)
    assert services.stats().n_docs == 3
