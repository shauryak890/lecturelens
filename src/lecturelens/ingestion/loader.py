"""File discovery, content hashing and parsing of PDF/Markdown/text files into pages."""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lecturelens.errors import IngestionError
from lecturelens.schemas import Page

logger = logging.getLogger(__name__)

HASH_BLOCK_BYTES = 1024 * 1024  # stream files in 1 MB blocks
DOC_ID_LENGTH = 16  # hex chars of the SHA-256 used as doc_id
PDF_SUFFIX = ".pdf"
TEXT_ENCODINGS = ("utf-8-sig", "cp1252")  # tried in order for .md/.txt files


def discover_files(directory: Path, extensions: Iterable[str]) -> list[Path]:
    """Recursively find files with the given extensions, sorted, skipping hidden entries.

    Args:
        directory: Folder to search.
        extensions: Suffixes to keep, e.g. ``[".pdf", ".md"]`` (case-insensitive).

    Returns:
        Sorted file paths.

    Raises:
        IngestionError: If ``directory`` does not exist.
    """
    if not directory.is_dir():
        raise IngestionError(f"Folder not found: {directory}")
    wanted = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in wanted
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def doc_id_from_hash(sha256_hex: str) -> str:
    """Derive the short, stable document id from a file's SHA-256."""
    return sha256_hex[:DOC_ID_LENGTH]


def load_pages(path: Path, doc_id: str | None = None) -> list[Page]:
    """Parse a file into pages.

    PDFs are converted page by page to Markdown with pymupdf4llm (``page_chunks=True``), which
    keeps headings and tables. Markdown and text files become a single page 1.

    Args:
        path: File to parse.
        doc_id: Precomputed document id; computed from the file hash if omitted.

    Returns:
        Pages in document order (possibly with empty text; cleaning drops those).

    Raises:
        IngestionError: If the file cannot be read or parsed. The message names the file.
    """
    doc_id = doc_id or doc_id_from_hash(file_hash(path))
    try:
        if path.suffix.lower() == PDF_SUFFIX:
            return _load_pdf(path, doc_id)
        return [Page(doc_id=doc_id, file_name=path.name, page=1, text=_read_text(path))]
    except IngestionError:
        raise
    except Exception as exc:  # parser libraries raise many unrelated types
        raise IngestionError(f"Could not parse {path.name}: {exc}") from exc


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError(f"Could not decode {path.name} as {' or '.join(TEXT_ENCODINGS)}")


def _load_pdf(path: Path, doc_id: str) -> list[Page]:
    import pymupdf4llm  # heavy import, only needed for PDFs

    chunks: list[dict[str, Any]] = pymupdf4llm.to_markdown(
        str(path), page_chunks=True, show_progress=False
    )
    pages = [
        Page(
            doc_id=doc_id,
            file_name=path.name,
            page=_page_number(chunk, fallback=i + 1),
            text=chunk.get("text") or "",
        )
        for i, chunk in enumerate(chunks)
    ]
    logger.info("Parsed %s: %d pages", path.name, len(pages))
    return pages


def _page_number(chunk: dict[str, Any], fallback: int) -> int:
    """1-based page number; the key differs between pymupdf4llm's layout and legacy paths."""
    metadata = chunk.get("metadata") or {}
    for key in ("page_number", "page"):
        if isinstance(metadata.get(key), int):
            return metadata[key]
    return fallback
