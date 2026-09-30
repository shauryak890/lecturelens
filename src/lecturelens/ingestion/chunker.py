"""Token-aware recursive chunking with overlap and heading-path metadata (SPEC 7.2).

Lengths are measured with the embedding model's own tokenizer, so ``chunk_size_tokens`` means
the same thing the embedder sees. Chunks never cross a page boundary, so every chunk has
exactly one page number to cite.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from lecturelens.config import IngestionCfg
from lecturelens.schemas import Chunk, Page

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_EMPHASIS = re.compile(r"[*_`]+")
# Recursive separators after headings: paragraph, line, sentence, word.
_SEPARATORS = (
    re.compile(r"\n\s*\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)
_WORD_SEPARATOR = _SEPARATORS[-1]
HEADING_JOINER = " > "
CONTEXT_HEADER_JOINER = " | "


class TokenCounter(Protocol):
    """Anything that counts tokens the way the embedding model does."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` (without special tokens)."""
        ...


class HFTokenCounter:
    """Token counter backed by a Hugging Face ``tokenizers`` tokenizer (e.g. bge-small's)."""

    def __init__(self, model_name: str) -> None:
        """Create the counter; the tokenizer is loaded on first use.

        Args:
            model_name: Hugging Face model id, e.g. ``"BAAI/bge-small-en-v1.5"``.
        """
        self.model_name = model_name
        self._tokenizer: Any = None

    def _load(self) -> Any:
        from tokenizers import Tokenizer  # light: avoids importing transformers

        tokenizer = Tokenizer.from_pretrained(self.model_name)  # cached after first download
        tokenizer.no_truncation()  # count every token, not just the first 512
        tokenizer.no_padding()
        return tokenizer

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` (without [CLS]/[SEP])."""
        if self._tokenizer is None:
            self._tokenizer = self._load()
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


@dataclass(frozen=True)
class _Piece:
    """An unsplittable-for-now span of text with its token count and section heading."""

    text: str
    n_tokens: int
    heading_path: str


@dataclass
class _Draft:
    """A chunk under construction: leading ``n_overlap`` pieces are copied from the previous."""

    pieces: list[_Piece]
    n_overlap: int = 0

    @property
    def n_tokens(self) -> int:
        return sum(p.n_tokens for p in self.pieces)

    @property
    def new_tokens(self) -> int:
        return sum(p.n_tokens for p in self.pieces[self.n_overlap :])


def indexed_text(chunk: Chunk, add_context_header: bool) -> str:
    """Text to embed and BM25-index for ``chunk``.

    With ``add_context_header`` the chunk is prefixed with ``"{file_name} | {heading_path}"``
    so a slide that only says "Advantages:" still carries its topic. Users see ``chunk.text``.
    """
    if not add_context_header:
        return chunk.text
    header = CONTEXT_HEADER_JOINER.join(p for p in (chunk.file_name, chunk.heading_path) if p)
    return f"{header}\n{chunk.text}"


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split ``text`` at ``pattern``, keeping each separator at the end of the preceding part."""
    parts, start = [], 0
    for match in pattern.finditer(text):
        if match.end() > start:
            parts.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [p for p in parts if p.strip()]


def _clean_heading(title: str) -> str:
    return _EMPHASIS.sub("", title).strip()


def _sections(text: str, stack: list[tuple[int, str]]) -> list[tuple[str, str]]:
    """Split page text at Markdown headings into ``(heading_path, section_text)``.

    ``stack`` holds ``(level, title)`` of open headings and is updated in place, so a page
    that continues a section inherits the heading path from earlier pages.
    """
    sections: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            level, title = len(match.group(1)), _clean_heading(match.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
        if match or not sections:
            sections.append((HEADING_JOINER.join(t for _, t in stack), []))
        sections[-1][1].append(line)
    return [(path, "\n".join(lines) + "\n") for path, lines in sections if "".join(lines).strip()]


def _hard_split(text: str, heading: str, counter: TokenCounter, max_tokens: int) -> list[_Piece]:
    """Last resort for a single "word" longer than the limit: halve it by characters."""
    n = counter.count(text)
    if n <= max_tokens or len(text) <= 1:
        return [_Piece(text, n, heading)]
    mid = len(text) // 2
    return _hard_split(text[:mid], heading, counter, max_tokens) + _hard_split(
        text[mid:], heading, counter, max_tokens
    )


def _split_recursive(
    text: str, heading: str, counter: TokenCounter, max_tokens: int, level: int = 0
) -> list[_Piece]:
    """Split ``text`` into pieces of at most ``max_tokens`` using ever finer separators."""
    n = counter.count(text)
    if n <= max_tokens:
        return [_Piece(text, n, heading)]
    if level >= len(_SEPARATORS):
        return _hard_split(text, heading, counter, max_tokens)
    parts = _split_keep(text, _SEPARATORS[level])
    if len(parts) <= 1:
        return _split_recursive(text, heading, counter, max_tokens, level + 1)
    return [
        piece
        for part in parts
        for piece in _split_recursive(part, heading, counter, max_tokens, level + 1)
    ]


def _overlap_tail(pieces: list[_Piece], budget: int, counter: TokenCounter) -> list[_Piece]:
    """Trailing text of ``pieces`` worth at most ``budget`` tokens, cut at word boundaries."""
    tail: list[_Piece] = []
    remaining = budget
    for piece in reversed(pieces):
        if piece.n_tokens <= remaining:
            tail.insert(0, piece)
            remaining -= piece.n_tokens
            continue
        words: list[str] = []
        for word in reversed(_split_keep(piece.text, _WORD_SEPARATOR)):
            n = counter.count(word)
            if n > remaining:
                break
            words.insert(0, word)
            remaining -= n
        if words:
            text = "".join(words)
            tail.insert(0, _Piece(text, counter.count(text), piece.heading_path))
        break
    return tail


def _pack(pieces: list[_Piece], cfg: IngestionCfg, counter: TokenCounter) -> list[_Draft]:
    """Greedily pack pieces into drafts with overlap, then merge a tiny trailing draft."""
    size, overlap = cfg.chunk_size_tokens, cfg.chunk_overlap_tokens
    drafts: list[_Draft] = []
    current = _Draft([])
    for piece in pieces:
        if current.new_tokens and current.n_tokens + piece.n_tokens > size:
            drafts.append(current)
            tail = _overlap_tail(current.pieces, overlap, counter)
            while tail and sum(p.n_tokens for p in tail) + piece.n_tokens > size:
                tail.pop(0)
            current = _Draft(list(tail), n_overlap=len(tail))
        current.pieces.append(piece)
    if current.new_tokens:
        drafts.append(current)

    if len(drafts) >= 2:
        last, prev = drafts[-1], drafts[-2]
        if last.new_tokens < cfg.min_chunk_tokens and prev.n_tokens + last.new_tokens <= size:
            prev.pieces.extend(last.pieces[last.n_overlap :])
            drafts.pop()
    return drafts


def _dominant_heading(draft: _Draft) -> str:
    """Heading path of the section contributing most tokens to the draft (first on ties)."""
    weights: Counter[str] = Counter()
    for piece in draft.pieces:
        weights[piece.heading_path] += piece.n_tokens
    return max(weights, key=lambda path: weights[path], default="")


def chunk_pages(pages: list[Page], counter: TokenCounter, cfg: IngestionCfg) -> list[Chunk]:
    """Split cleaned pages into chunks.

    Args:
        pages: Cleaned pages, in document order (several documents may be mixed; the heading
            path resets whenever ``doc_id`` changes).
        counter: Token counter matching the embedding model.
        cfg: Ingestion settings (chunk size, overlap, minimum chunk size).

    Returns:
        Chunks with ids ``"{doc_id}:{page}:{index}"``; ``chunk_index`` counts within the page.
    """
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    current_doc: str | None = None
    for page in pages:
        if page.doc_id != current_doc:
            stack, current_doc = [], page.doc_id
        pieces = [
            piece
            for heading, text in _sections(page.text, stack)
            for piece in _split_recursive(text, heading, counter, cfg.chunk_size_tokens)
        ]
        for index, draft in enumerate(_pack(pieces, cfg, counter)):
            text = "".join(p.text for p in draft.pieces).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}:{page.page}:{index}",
                    doc_id=page.doc_id,
                    file_name=page.file_name,
                    page=page.page,
                    heading_path=_dominant_heading(draft),
                    chunk_index=index,
                    text=text,
                    n_tokens=counter.count(text),
                )
            )
    logger.debug("Chunked %d pages into %d chunks", len(pages), len(chunks))
    return chunks
