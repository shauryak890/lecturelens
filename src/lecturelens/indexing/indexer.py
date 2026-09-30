"""Ingestion orchestration (SPEC 3.1): discover -> hash/skip -> parse -> clean -> chunk ->
embed -> upsert into Chroma -> rebuild BM25 -> update manifest.

Used by the CLI ``ingest``/``stats`` commands and (in P3) the Streamlit Library tab.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from lecturelens.config import Settings
from lecturelens.errors import IndexMismatchError, IngestionError
from lecturelens.indexing.bm25_index import BM25Index
from lecturelens.indexing.embedder import Embedder
from lecturelens.indexing.manifest import FileEntry, Manifest
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.ingestion.chunker import TokenCounter, chunk_pages, indexed_text
from lecturelens.ingestion.cleaner import clean_pages
from lecturelens.ingestion.loader import discover_files, doc_id_from_hash, file_hash, load_pages
from lecturelens.logging_utils import utc_timestamp
from lecturelens.schemas import Chunk

logger = logging.getLogger(__name__)

CHROMA_DIRNAME = "chroma"
BM25_FILENAME = "bm25.pkl"
MANIFEST_FILENAME = "manifest.json"


class IndexPaths:
    """Locations of the index artefacts under ``app.index_dir``."""

    def __init__(self, root: Path) -> None:
        """Create the path set for index directory ``root``."""
        self.root = root
        self.chroma = root / CHROMA_DIRNAME
        self.bm25 = root / BM25_FILENAME
        self.manifest = root / MANIFEST_FILENAME


class IngestReport(BaseModel):
    """What one ingest run did."""

    indexed: list[str] = []
    skipped: list[str] = []  # unchanged since last ingest
    duplicates: dict[str, str] = {}  # file -> already-indexed file with identical content
    failed: dict[str, str] = {}  # file -> error message
    new_chunks: int = 0
    total_chunks: int = 0
    seconds: float = 0.0


class IndexStats(BaseModel):
    """Summary of the current index for the ``stats`` command and the UI sidebar."""

    n_docs: int
    n_pages: int
    n_chunks: int
    avg_tokens_per_chunk: float
    embedding_model: str | None
    dim: int | None
    bm25_ready: bool
    index_bytes: int
    files: list[FileEntry]


def chunking_settings(settings: Settings) -> dict[str, Any]:
    """Settings that change chunk boundaries or indexed text; a change requires a rebuild."""
    return settings.ingestion.model_dump(
        include={
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "min_chunk_tokens",
            "min_page_chars",
            "header_footer_threshold",
            "header_footer_min_pages",
            "header_footer_substring_min_chars",
            "use_ocr",
            "add_context_header",
        }
    )


class Indexer:
    """Builds and incrementally updates the dense + BM25 index."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        counter: TokenCounter,
        store: VectorStore | None = None,
    ) -> None:
        """Create the indexer.

        Args:
            settings: Application settings.
            embedder: Embedder used for chunk vectors (must match the one used for queries).
            counter: Token counter matching the embedder's tokenizer.
            store: Vector store; opened at ``app.index_dir`` if omitted.
        """
        self.settings = settings
        self.embedder = embedder
        self.counter = counter
        self.paths = IndexPaths(settings.app.index_dir)
        self.store = store or VectorStore(self.paths.chroma)

    def ingest(
        self,
        directory: Path,
        *,
        rebuild: bool = False,
        on_file: Callable[[Path], None] | None = None,
    ) -> IngestReport:
        """Index every supported file under ``directory``; unchanged files are skipped.

        Args:
            directory: Folder to ingest (searched recursively).
            rebuild: Drop the existing index first (needed after changing the embedding model
                or chunking settings).
            on_file: Optional callback invoked before each file is processed (progress UI).

        Returns:
            A report of indexed, skipped, duplicate and failed files.

        Raises:
            IngestionError: If ``directory`` does not exist.
            IndexMismatchError: If the index was built with other settings and ``rebuild`` is
                false.
        """
        started = time.perf_counter()
        files = discover_files(directory, self.settings.ingestion.extensions)
        manifest = self._prepare_manifest(rebuild)
        report = IngestReport()
        for path in files:
            if on_file:
                on_file(path)
            self._ingest_file(path, manifest, report)

        if report.indexed or rebuild or not self.paths.bm25.is_file():
            self.rebuild_bm25()
        report.total_chunks = self.store.count()
        report.seconds = time.perf_counter() - started
        logger.info("Ingest of %s finished: %s", directory, report.model_dump_json())
        return report

    def _prepare_manifest(self, rebuild: bool) -> Manifest:
        if rebuild:
            self.store.reset()
            self.paths.bm25.unlink(missing_ok=True)
            manifest = Manifest()
        else:
            manifest = Manifest.load(self.paths.manifest)
        chunking = chunking_settings(self.settings)
        # model name only: checking the dimension would load the model even when every file
        # is unchanged; check_dim() runs just before the first embedding instead
        manifest.check_compatible(self.embedder.model_name, chunking)
        manifest.set_settings(self.embedder.model_name, chunking)
        manifest.save(self.paths.manifest)
        return manifest

    def _ingest_file(self, path: Path, manifest: Manifest, report: IngestReport) -> None:
        key = path.resolve().as_posix()
        sha = file_hash(path)
        doc_id = doc_id_from_hash(sha)
        previous = manifest.files.get(key)
        if previous and previous.hash == sha:
            report.skipped.append(path.name)
            return
        duplicate_of = manifest.find_doc(doc_id)
        if duplicate_of and duplicate_of != key:
            report.duplicates[path.name] = manifest.files[duplicate_of].file_name
            return
        try:
            n_pages, chunks = self._parse_and_chunk(path, doc_id)
        except IngestionError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            report.failed[path.name] = str(exc)
            return

        vectors = self._embed(chunks, manifest)
        if previous:  # file content changed: replace its old chunks
            self.store.delete_doc(previous.doc_id)
        if chunks:
            self.store.upsert(chunks, vectors)
        manifest.files[key] = FileEntry(
            hash=sha,
            doc_id=doc_id,
            file_name=path.name,
            n_pages=n_pages,
            n_chunks=len(chunks),
            n_tokens=sum(c.n_tokens for c in chunks),
            indexed_at=utc_timestamp(),
        )
        manifest.save(self.paths.manifest)  # after every file, so an interrupted run resumes
        report.indexed.append(path.name)
        report.new_chunks += len(chunks)

    def _parse_and_chunk(self, path: Path, doc_id: str) -> tuple[int, list[Chunk]]:
        pages = load_pages(path, doc_id, use_ocr=self.settings.ingestion.use_ocr)
        cleaned = clean_pages(pages, self.settings.ingestion)
        chunks = chunk_pages(cleaned, self.counter, self.settings.ingestion)
        logger.info(
            "%s: %d pages parsed, %d kept, %d chunks",
            path.name,
            len(pages),
            len(cleaned),
            len(chunks),
        )
        return len(pages), chunks

    def _embed(self, chunks: list[Chunk], manifest: Manifest) -> np.ndarray:
        """Embed chunks' indexed text, verifying the vector dimension against the index."""
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        manifest.check_dim(self.embedder.dim)  # first access loads a lazy embedder
        add_header = self.settings.ingestion.add_context_header
        return self.embedder.embed_documents([indexed_text(c, add_header) for c in chunks])

    def rebuild_bm25(self) -> BM25Index:
        """Rebuild the BM25 index over every stored chunk and save it."""
        return rebuild_bm25(self.settings, self.store)

    def load_bm25(self) -> BM25Index:
        """Load the saved BM25 index, rebuilding it if missing or built with other settings."""
        return load_or_build_bm25(self.settings, self.store)


def rebuild_bm25(settings: Settings, store: VectorStore) -> BM25Index:
    """Build the BM25 index over every chunk in ``store`` and save it under ``app.index_dir``."""
    chunks = store.all_chunks()
    add_header = settings.ingestion.add_context_header
    index = BM25Index(settings.retrieval.bm25)
    index.build([c.chunk_id for c in chunks], [indexed_text(c, add_header) for c in chunks])
    index.save(IndexPaths(settings.app.index_dir).bm25)
    return index


def load_or_build_bm25(settings: Settings, store: VectorStore) -> BM25Index:
    """Load the saved BM25 index; rebuild it (cheap, no embeddings) if missing or stale."""
    path = IndexPaths(settings.app.index_dir).bm25
    if path.is_file():
        try:
            return BM25Index.load(path, settings.retrieval.bm25)
        except IndexMismatchError as exc:
            logger.info("%s Rebuilding BM25 (cheap, no embeddings needed).", exc)
    return rebuild_bm25(settings, store)


def build_indexer(settings: Settings) -> Indexer:
    """Create an :class:`Indexer` with the configured embedder and its matching tokenizer.

    Loads the embedding model (downloaded on first use, then cached by Hugging Face).
    """
    from lecturelens.indexing.embedder import build_embedder
    from lecturelens.ingestion.chunker import HFTokenCounter

    embedder = build_embedder(settings.embeddings)
    return Indexer(settings, embedder, HFTokenCounter(settings.embeddings.model))


def index_stats(settings: Settings, store: VectorStore | None = None) -> IndexStats:
    """Summarise the index at ``app.index_dir`` without loading any model."""
    paths = IndexPaths(settings.app.index_dir)
    manifest = Manifest.load(paths.manifest)
    store = store or VectorStore(paths.chroma)
    n_chunks = store.count()
    index_bytes = (
        sum(f.stat().st_size for f in paths.root.rglob("*") if f.is_file())
        if paths.root.is_dir()
        else 0
    )
    return IndexStats(
        n_docs=len(manifest.files),
        n_pages=sum(e.n_pages for e in manifest.files.values()),
        n_chunks=n_chunks,
        avg_tokens_per_chunk=manifest.n_tokens / manifest.n_chunks if manifest.n_chunks else 0.0,
        embedding_model=manifest.embedding_model,
        dim=manifest.dim,
        bm25_ready=paths.bm25.is_file(),
        index_bytes=index_bytes,
        files=sorted(manifest.files.values(), key=lambda e: e.file_name.lower()),
    )
