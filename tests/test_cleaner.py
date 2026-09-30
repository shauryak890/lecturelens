"""Tests for ingestion.cleaner (SPEC 7.1)."""

import pytest

from lecturelens.config import IngestionCfg, Settings
from lecturelens.ingestion.cleaner import clean_pages, find_repeated_lines, normalize_text
from lecturelens.schemas import Page

TOPICS = ["tokenization", "smoothing", "perplexity", "embeddings", "attention", "parsing"]
BODY = "This page explains {topic} in enough words to survive cleaning."


def body(n: int) -> str:
    """Unique line per page. Varies by words: digits are masked by footer detection."""
    return BODY.format(topic=TOPICS[n])


@pytest.fixture
def cfg(settings: Settings) -> IngestionCfg:
    return settings.ingestion


def _pages(texts: list[str], doc_id: str = "doc1") -> list[Page]:
    return [
        Page(doc_id=doc_id, file_name=f"{doc_id}.pdf", page=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


def test_dehyphenates_line_breaks() -> None:
    assert normalize_text("tokeni-\nzation matters") == "tokenization matters"


def test_keeps_real_hyphens() -> None:
    assert normalize_text("state-of-the-art word-level") == "state-of-the-art word-level"


def test_nfkc_fixes_ligatures_and_fullwidth() -> None:
    assert normalize_text("ﬁnite ﬂow ＢＥＲＴ") == "finite flow BERT"


def test_collapses_whitespace_and_blank_lines() -> None:
    text = "a   b\t\tc  \r\n\n\n\n\nd"
    assert normalize_text(text) == "a b c\n\nd"


def test_repeated_footer_removed_with_page_numbers_masked(cfg: IngestionCfg) -> None:
    texts = [f"# {TOPICS[i].title()}\n{body(i)}\nDSE4150 | MUJ | {i + 10}" for i in range(5)]
    cleaned = clean_pages(_pages(texts), cfg)
    assert len(cleaned) == 5
    for i, page in enumerate(cleaned):
        assert "DSE4150" not in page.text
        assert page.text == f"# {TOPICS[i].title()}\n{body(i)}"  # unique lines kept


def test_unique_lines_kept(cfg: IngestionCfg) -> None:
    texts = [f"Topic {chr(65 + i)} has its own sentence here for page {i}." for i in range(4)]
    cleaned = clean_pages(_pages(texts), cfg)
    assert [p.text for p in cleaned] == texts


def test_line_on_half_the_pages_is_kept(cfg: IngestionCfg) -> None:
    # threshold is "more than 50%": a line on exactly 2 of 4 pages stays
    texts = [f"Shared remark\n{body(i)}" if i < 2 else body(i) for i in range(4)]
    cleaned = clean_pages(_pages(texts), cfg)
    assert sum("Shared remark" in p.text for p in cleaned) == 2


def test_short_documents_are_not_stripped(cfg: IngestionCfg) -> None:
    # every line of a 1-2 page document "repeats" on 100% of pages; it must survive
    pages = _pages(["Only page with important content about BM25 scoring."])
    assert find_repeated_lines(pages, cfg.header_footer_threshold, 3) == set()
    assert clean_pages(pages, cfg)[0].text == pages[0].text


def test_header_detection_is_per_document(cfg: IngestionCfg) -> None:
    footer = "Course footer line"
    doc_a = _pages([f"{body(i)}\n{footer}" for i in range(3)], "a")
    doc_b = _pages([f"{body(i)}\n{footer}" if i == 0 else body(i) for i in range(3)], "b")
    cleaned = clean_pages(doc_a + doc_b, cfg)
    assert not any(footer in p.text for p in cleaned if p.doc_id == "a")
    assert sum(footer in p.text for p in cleaned if p.doc_id == "b") == 1


def test_empty_and_near_empty_pages_dropped(cfg: IngestionCfg) -> None:
    pages = _pages(["", "   \n\n  ", "Title", body(1)])
    cleaned = clean_pages(pages, cfg)
    assert [p.page for p in cleaned] == [4]  # page numbers are preserved, not renumbered


def test_lines_differing_only_in_numbers_count_as_repeated(cfg: IngestionCfg) -> None:
    """Digit masking (SPEC 7.1) makes "Page 3 of 20" and "Page 4 of 20" the same footer."""
    texts = [body(i) + f"\nPage {i + 1} of 20" for i in range(4)]
    cleaned = clean_pages(_pages(texts), cfg)
    assert all("of 20" not in p.text for p in cleaned)
    assert [p.text for p in cleaned] == [body(i) for i in range(4)]
