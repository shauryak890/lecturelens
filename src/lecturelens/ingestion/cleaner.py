"""Text cleaning (SPEC 7.1): Unicode normalisation, de-hyphenation, whitespace,
repeated header/footer removal and near-empty page removal."""

import re
import unicodedata
from collections import Counter, defaultdict

from lecturelens.config import IngestionCfg
from lecturelens.schemas import Page

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "tokeni-\nzation" -> "tokenization"
_SPACE_RUN = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")  # at most 2 consecutive newlines
_DIGITS = re.compile(r"\d+")
_NON_SPACE = re.compile(r"\S")


def normalize_text(text: str) -> str:
    """Normalise one page of text.

    Applies NFKC (ligatures such as "ﬁ" become "fi", full-width characters become ASCII),
    joins words split by a line-end hyphen, collapses runs of spaces, strips each line and
    keeps at most one blank line in a row.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACE_RUN.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _line_key(line: str) -> str:
    """Key used to spot repeated lines: page numbers differ, so digits are masked."""
    return _DIGITS.sub("#", line.strip().lower())


def find_repeated_lines(pages: list[Page], threshold: float, min_pages: int) -> set[str]:
    """Return keys of lines that appear on more than ``threshold`` of a document's pages.

    Args:
        pages: Pages of a single document (already normalised).
        threshold: Fraction of pages a line must exceed to count as header/footer.
        min_pages: Documents with fewer pages are left alone; on a 1-2 page document every
            line would look "repeated".

    Returns:
        Masked line keys (see ``_line_key``) to drop.
    """
    if len(pages) < min_pages:
        return set()
    counts = Counter(
        key for page in pages for key in {_line_key(ln) for ln in page.text.split("\n")} if key
    )
    return {key for key, n in counts.items() if n / len(pages) > threshold}


def count_visible_chars(text: str) -> int:
    """Number of non-whitespace characters in ``text``."""
    return len(_NON_SPACE.findall(text))


def clean_pages(pages: list[Page], cfg: IngestionCfg) -> list[Page]:
    """Clean pages and drop those that end up (nearly) empty.

    Header/footer detection runs per document, so pages from several files may be passed
    together.

    Args:
        pages: Parsed pages.
        cfg: Ingestion settings (thresholds).

    Returns:
        Cleaned pages, in input order, without pages under ``cfg.min_page_chars`` visible
        characters.
    """
    normalised = [page.model_copy(update={"text": normalize_text(page.text)}) for page in pages]
    by_doc: dict[str, list[Page]] = defaultdict(list)
    for page in normalised:
        by_doc[page.doc_id].append(page)
    repeated = {
        doc_id: find_repeated_lines(
            doc_pages, cfg.header_footer_threshold, cfg.header_footer_min_pages
        )
        for doc_id, doc_pages in by_doc.items()
    }

    cleaned: list[Page] = []
    for page in normalised:
        drop = repeated[page.doc_id]
        text = page.text
        if drop:
            kept = [ln for ln in text.split("\n") if _line_key(ln) not in drop or not ln.strip()]
            text = _BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()
        if count_visible_chars(text) >= cfg.min_page_chars:
            cleaned.append(page.model_copy(update={"text": text}))
    return cleaned
