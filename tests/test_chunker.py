"""Tests for ingestion.chunker (SPEC 7.2), using a whitespace token counter (1 word = 1 token)."""

import pytest

from lecturelens.config import IngestionCfg, Settings
from lecturelens.ingestion.chunker import chunk_pages, indexed_text
from lecturelens.schemas import Chunk, Page

from .fakes import WhitespaceCounter

SIZE, OVERLAP, MIN = 40, 8, 6
COUNTER = WhitespaceCounter()


@pytest.fixture
def cfg(settings: Settings) -> IngestionCfg:
    return settings.with_overrides(
        {
            "ingestion.chunk_size_tokens": SIZE,
            "ingestion.chunk_overlap_tokens": OVERLAP,
            "ingestion.min_chunk_tokens": MIN,
        }
    ).ingestion


def _sentences(n: int, prefix: str = "w") -> str:
    """n sentences of 7 words each, every word unique so overlap is detectable."""
    return " ".join(
        " ".join(f"{prefix}{i}x{j}" for j in range(6)) + f" end{prefix}{i}." for i in range(n)
    )


def _page(text: str, page: int = 1, doc_id: str = "d1") -> Page:
    return Page(doc_id=doc_id, file_name=f"{doc_id}.pdf", page=page, text=text)


def _words(chunk: Chunk) -> list[str]:
    return chunk.text.split()


def test_no_chunk_exceeds_size(cfg: IngestionCfg) -> None:
    text = f"# Intro\n{_sentences(30)}\n\n## Details\n{_sentences(25, 'v')}"
    chunks = chunk_pages([_page(text)], COUNTER, cfg)
    assert len(chunks) > 3
    assert all(c.n_tokens <= SIZE for c in chunks)
    assert all(c.n_tokens == COUNTER.count(c.text) for c in chunks)


def test_long_unbroken_text_is_split_by_words_then_characters(cfg: IngestionCfg) -> None:
    no_punctuation = " ".join(f"tok{i}" for i in range(200))
    giant_word = "x" * 5  # counts as 1 token, never needs a hard split
    chunks = chunk_pages([_page(no_punctuation + " " + giant_word)], COUNTER, cfg)
    assert all(c.n_tokens <= SIZE for c in chunks)
    assert chunks[-1].text.endswith(giant_word)


def test_consecutive_chunks_overlap(cfg: IngestionCfg) -> None:
    chunks = chunk_pages([_page(_sentences(30))], COUNTER, cfg)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = set(_words(prev)[-OVERLAP:]) & set(_words(nxt)[:OVERLAP])
        assert shared, f"no overlap between {prev.chunk_id} and {nxt.chunk_id}"
        # overlap never exceeds the budget
        head = _words(nxt)
        n_overlap = next(i for i, w in enumerate(head) if w not in set(_words(prev)))
        assert n_overlap <= OVERLAP


def test_all_text_is_covered(cfg: IngestionCfg) -> None:
    text = _sentences(20)
    chunks = chunk_pages([_page(text)], COUNTER, cfg)
    covered = {w for c in chunks for w in _words(c)}
    assert covered == set(text.split())


def test_page_numbers_preserved_and_pages_never_mixed(cfg: IngestionCfg) -> None:
    pages = [_page(_sentences(8, f"p{n}_"), page=n) for n in (3, 4, 7)]
    chunks = chunk_pages(pages, COUNTER, cfg)
    assert {c.page for c in chunks} == {3, 4, 7}
    for c in chunks:
        assert all(w.startswith((f"p{c.page}_", f"endp{c.page}_")) for w in _words(c))


def test_heading_path_captured_and_nested(cfg: IngestionCfg) -> None:
    text = (
        "# Unit 2\n\n## Word **Embeddings**\n\n### Skip-gram\n"
        + _sentences(3, "s")
        + "\n\n## Language Models\n"
        + _sentences(3, "l")
    )
    chunks = chunk_pages([_page(text)], COUNTER, cfg)
    paths = {c.heading_path for c in chunks}
    assert "Unit 2 > Word Embeddings > Skip-gram" in paths  # emphasis markers stripped
    assert "Unit 2 > Language Models" in paths  # sibling heading pops the deeper levels


def test_heading_path_carries_across_pages_but_not_documents(cfg: IngestionCfg) -> None:
    pages = [
        _page("# Unit 1\n" + _sentences(2, "a"), page=1),
        _page(_sentences(2, "b"), page=2),  # continuation slide without a heading
        _page(_sentences(2, "c"), page=1, doc_id="d2"),
    ]
    chunks = chunk_pages(pages, COUNTER, cfg)
    by_page = {(c.doc_id, c.page): c.heading_path for c in chunks}
    assert by_page[("d1", 2)] == "Unit 1"
    assert by_page[("d2", 1)] == ""


def test_tiny_tail_merged_into_predecessor(cfg: IngestionCfg) -> None:
    # 35 words then a 3-word tail: a separate chunk would hold only 3 new tokens (< MIN)
    text = " ".join(f"a{i}" for i in range(35)) + ". Tail words here."
    chunks = chunk_pages([_page(text)], COUNTER, cfg)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("Tail words here.")
    assert chunks[0].n_tokens <= SIZE


def test_tail_not_merged_when_it_would_overflow(cfg: IngestionCfg) -> None:
    text = " ".join(f"a{i}" for i in range(39)) + ". Tail words here."
    chunks = chunk_pages([_page(text)], COUNTER, cfg)
    assert len(chunks) == 2
    assert all(c.n_tokens <= SIZE for c in chunks)


def test_ids_are_deterministic_and_unique(cfg: IngestionCfg) -> None:
    pages = [_page(_sentences(15), page=1), _page(_sentences(15, "q"), page=2)]
    first = chunk_pages(pages, COUNTER, cfg)
    second = chunk_pages(pages, COUNTER, cfg)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert ids[0] == "d1:1:0"
    assert all(c.chunk_id == f"{c.doc_id}:{c.page}:{c.chunk_index}" for c in first)


def test_short_page_is_one_chunk(cfg: IngestionCfg) -> None:
    chunks = chunk_pages([_page("# Title\nJust a short slide.")], COUNTER, cfg)
    assert len(chunks) == 1
    assert chunks[0].text == "# Title\nJust a short slide."
    assert chunks[0].heading_path == "Title"


def test_indexed_text_adds_context_header_only_when_enabled() -> None:
    chunk = Chunk(
        chunk_id="d:1:0",
        doc_id="d",
        file_name="lecture3.pdf",
        page=1,
        heading_path="Unit 2 > Word2Vec",
        chunk_index=0,
        text="Advantages: fast.",
        n_tokens=2,
    )
    assert indexed_text(chunk, True) == "lecture3.pdf | Unit 2 > Word2Vec\nAdvantages: fast."
    assert indexed_text(chunk, False) == "Advantages: fast."
    no_heading = chunk.model_copy(update={"heading_path": ""})
    assert indexed_text(no_heading, True) == "lecture3.pdf\nAdvantages: fast."
