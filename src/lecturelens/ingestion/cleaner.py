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
_DIGIT_MASK = "\x00"  # stands in for digit runs in line keys


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
    """Key used to spot repeated lines: page numbers differ, so digits are masked.

    The mask is a NUL character, which never occurs in page text, so it cannot collide
    with a literal "#" from a Markdown heading when the key is turned into a regex.
    """
    return _DIGITS.sub(_DIGIT_MASK, line.strip().lower())


def _embedded_boilerplate(keys: set[str], min_chars: int) -> re.Pattern[str] | None:
    """Regex matching repeated lines of at least ``min_chars`` inside a longer line.

    PDF extraction sometimes glues a footer or watermark onto the end of a content line,
    so exact line matching misses it on that page. Short keys are excluded so a repeated
    "2" or "Q&A" never carves words out of real text. Masked digits match any number.
    """
    parts = [
        r"\d+".join(re.escape(part) for part in key.split(_DIGIT_MASK))
        for key in sorted(keys, key=len, reverse=True)  # longest first
        if count_visible_chars(key) >= min_chars
    ]
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def _strip_boilerplate(text: str, drop: set[str], embedded: re.Pattern[str] | None) -> str:
    """Remove repeated lines, and repeated text glued inside other lines."""
    kept: list[str] = []
    for line in text.split("\n"):
        if line.strip() and _line_key(line) in drop:
            continue
        if embedded is not None and line.strip():
            stripped = embedded.sub(" ", line)
            if stripped != line:
                line = _SPACE_RUN.sub(" ", stripped).strip()
                if not line:
                    continue
        kept.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


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

    embedded = {
        doc_id: _embedded_boilerplate(keys, cfg.header_footer_substring_min_chars)
        for doc_id, keys in repeated.items()
    }

    cleaned: list[Page] = []
    for page in normalised:
        text = page.text
        if repeated[page.doc_id]:
            text = _strip_boilerplate(text, repeated[page.doc_id], embedded[page.doc_id])
        if count_visible_chars(text) >= cfg.min_page_chars:
            cleaned.append(page.model_copy(update={"text": text}))
    return cleaned
