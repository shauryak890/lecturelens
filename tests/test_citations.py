"""Tests for rag.citations and rag.context."""

from lecturelens.llm.prompts import PromptRegistry
from lecturelens.rag.citations import extract, normalize_citations, validate
from lecturelens.rag.context import build_context
from lecturelens.schemas import AnswerResponse, Chunk, RetrievedChunk


def _answer(text: str, cited: list[int], answerable: bool = True) -> AnswerResponse:
    return AnswerResponse(
        answer=text, cited_sources=cited, answerable=answerable, confidence="high"
    )


def _rc(n: int, tokens: int, heading: str = "Unit 1") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"d:{n}:0",
        doc_id="d",
        file_name="notes.pdf",
        page=n,
        heading_path=heading,
        chunk_index=0,
        text=f"Text of page {n}.",
        n_tokens=tokens,
    )
    return RetrievedChunk(chunk=chunk, score=1.0, source="hybrid")


def test_extract_in_order_of_first_appearance() -> None:
    assert extract("A [S1][S3]. B [S1] and [S2].") == [1, 3, 2]
    assert extract("no citations here") == []


def test_grouped_citations_are_normalised() -> None:
    assert normalize_citations("Fact [S1, S3]. Other [S2,S4]; x [S5; 6].") == (
        "Fact [S1][S3]. Other [S2][S4]; x [S5][S6]."
    )
    assert extract("Fact [S1, S3].") == [1, 3]


def test_invalid_id_stripped_from_text_and_list() -> None:
    check = validate(_answer("TF-IDF weighs terms [S1][S9]. Rare terms matter [S9].", [1, 9]), 5)
    assert check.removed == [9]
    assert check.response.answer == "TF-IDF weighs terms [S1]. Rare terms matter."
    assert check.response.cited_sources == [1]
    assert not check.needs_repair


def test_cited_sources_synced_with_text() -> None:
    check = validate(_answer("A [S2]. B [S3].", [2]), 3)
    assert check.response.cited_sources == [2, 3]
    assert check.removed == []


def test_answerable_without_valid_citations_is_flagged() -> None:
    assert validate(_answer("An unsupported claim.", []), 5).needs_repair
    only_invalid = validate(_answer("Claim [S7].", [7]), 5)
    assert only_invalid.needs_repair and only_invalid.removed == [7]


def test_abstention_without_citations_is_fine() -> None:
    check = validate(_answer("I couldn't find this in your course material.", [], False), 5)
    assert not check.needs_repair and check.removed == []


def test_build_context_numbers_sources_and_respects_budget(registry: PromptRegistry) -> None:
    chunks = [_rc(1, 100), _rc(2, 100, heading=""), _rc(3, 100)]
    text, used = build_context(chunks, max_tokens=250, registry=registry)
    assert used == chunks[:2]
    assert text == (
        "[S1] (notes.pdf, p.1, Unit 1)\nText of page 1.\n\n[S2] (notes.pdf, p.2)\nText of page 2."
    )


def test_build_context_always_keeps_the_first_chunk(registry: PromptRegistry) -> None:
    text, used = build_context([_rc(1, 5000)], max_tokens=100, registry=registry)
    assert len(used) == 1 and text.startswith("[S1]")
