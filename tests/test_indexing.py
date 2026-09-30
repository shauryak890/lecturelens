"""Offline tests for loader, BM25 index, manifest and the Indexer (real Chroma in tmp_path)."""

from pathlib import Path

import pytest

from lecturelens.config import Settings
from lecturelens.errors import IndexMismatchError, IngestionError
from lecturelens.indexing.bm25_index import BM25Index, tokenize
from lecturelens.indexing.embedder import LazyEmbedder
from lecturelens.indexing.indexer import Indexer, index_stats
from lecturelens.indexing.manifest import Manifest
from lecturelens.ingestion.loader import discover_files, file_hash, load_pages

from .fakes import HashEmbedder, WhitespaceCounter

NOTE_A = "# BM25\nBM25 ranks documents with term frequency saturation k1 and length norm b.\n"
NOTE_B = "# Word2Vec\nSkip-gram predicts context words from the centre word.\n"


@pytest.fixture
def cfg(settings: Settings, tmp_path: Path) -> Settings:
    return settings.with_overrides(
        {"app.index_dir": str(tmp_path / "index"), "ingestion.min_page_chars": 5}
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "bm25.md").write_bytes(NOTE_A.encode("utf-8"))  # bytes: no CRLF on Windows
    (folder / "w2v.txt").write_bytes(NOTE_B.encode("utf-8"))
    return folder


def _indexer(cfg: Settings, embedder: HashEmbedder | None = None) -> Indexer:
    return Indexer(cfg, embedder or HashEmbedder(dim=32), WhitespaceCounter())


# ---------------------------------------------------------------- loader


def test_discover_files_filters_sorts_and_skips_hidden(tmp_path: Path) -> None:
    for rel in ("b.md", "a.PDF", "sub/c.txt", ".hidden/d.md", ".e.md", "f.docx"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    found = [
        p.relative_to(tmp_path).as_posix()
        for p in discover_files(tmp_path, [".pdf", ".md", ".txt"])
    ]
    assert found == ["a.PDF", "b.md", "sub/c.txt"]


def test_discover_files_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        discover_files(tmp_path / "nope", [".md"])


def test_markdown_is_one_page_and_hash_is_stable(corpus: Path) -> None:
    path = corpus / "bm25.md"
    pages = load_pages(path)
    assert len(pages) == 1 and pages[0].page == 1 and pages[0].text == NOTE_A
    assert pages[0].doc_id == file_hash(path)[:16]
    assert file_hash(path) == file_hash(path)


def test_pdf_pages_are_parsed_with_page_numbers(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "slides.pdf"
    doc = pymupdf.open()
    for text in ("Tokenization splits text into tokens.", "Perplexity measures surprise."):
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    pages = load_pages(path)
    assert [p.page for p in pages] == [1, 2]
    assert "Tokenization" in pages[0].text and "Perplexity" in pages[1].text


def test_corrupt_pdf_raises_ingestion_error_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(IngestionError, match="broken.pdf"):
        load_pages(path)


# ---------------------------------------------------------------- BM25


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("The TF-IDF of a Term") == ["tf", "idf", "term"]


def test_bm25_ranks_exact_terms_and_round_trips(settings: Settings, tmp_path: Path) -> None:
    index = BM25Index(settings.retrieval.bm25)
    index.build(["a", "b", "c"], ["BLEU measures n-gram overlap", "Viterbi decoding", "BLEU BLEU"])
    hits = index.search("what does BLEU measure", k=5)
    assert [cid for cid, _ in hits] == ["c", "a"]  # "b" has no matching term -> excluded
    path = tmp_path / "bm25.pkl"
    index.save(path)
    assert BM25Index.load(path, settings.retrieval.bm25).search("BLEU", k=5) == index.search(
        "BLEU", k=5
    )
    changed = settings.with_overrides({"retrieval.bm25.k1": 1.2}).retrieval.bm25
    with pytest.raises(IndexMismatchError):
        BM25Index.load(path, changed)


def test_bm25_idf_stays_positive_for_common_terms(settings: Settings) -> None:
    """Regression: classic Okapi IDF is 0 for a term in half the docs, so it never matched."""
    index = BM25Index(settings.retrieval.bm25)
    index.build(["a", "b"], ["skip-gram model", "cbow model"])
    assert [cid for cid, _ in index.search("skip-gram", k=5)] == ["a"]  # in 1 of 2 docs
    assert len(index.search("model", k=5)) == 2  # in every doc: still a (small) positive score


def test_bm25_stemming_matches_inflections(settings: Settings) -> None:
    stem_cfg = settings.with_overrides({"retrieval.bm25.stemming": True}).retrieval.bm25
    index = BM25Index(stem_cfg)
    index.build(["a", "b"], ["tokenizers tokenizing text", "unrelated words"])
    assert [cid for cid, _ in index.search("tokenization", k=5)] == ["a"]


def test_bm25_empty_index_returns_nothing(settings: Settings) -> None:
    index = BM25Index(settings.retrieval.bm25)
    index.build([], [])
    assert index.search("anything", k=5) == []


# ---------------------------------------------------------------- Indexer


def test_ingest_indexes_then_skips_unchanged_files(cfg: Settings, corpus: Path) -> None:
    first = _indexer(cfg).ingest(corpus)
    assert sorted(first.indexed) == ["bm25.md", "w2v.txt"]
    assert first.new_chunks == first.total_chunks == 2

    second = _indexer(cfg).ingest(corpus)  # FR-1: re-running adds 0 new chunks
    assert second.indexed == [] and second.new_chunks == 0
    assert sorted(second.skipped) == ["bm25.md", "w2v.txt"]
    assert second.total_chunks == 2


def test_changed_file_replaces_its_old_chunks(cfg: Settings, corpus: Path) -> None:
    indexer = _indexer(cfg)
    indexer.ingest(corpus)
    (corpus / "bm25.md").write_text(NOTE_A + "\nAn extra sentence about IDF.", encoding="utf-8")
    report = indexer.ingest(corpus)
    assert report.indexed == ["bm25.md"]
    assert report.total_chunks == 2  # old version's chunk deleted, not duplicated
    texts = [c.text for c in indexer.store.all_chunks()]
    assert any("IDF" in t for t in texts)


def test_duplicate_content_is_indexed_once(cfg: Settings, corpus: Path) -> None:
    (corpus / "copy_of_bm25.md").write_bytes((corpus / "bm25.md").read_bytes())
    report = _indexer(cfg).ingest(corpus)
    assert report.duplicates == {"copy_of_bm25.md": "bm25.md"}
    assert report.total_chunks == 2


def test_unparseable_file_is_reported_and_others_still_indexed(cfg: Settings, corpus: Path) -> None:
    (corpus / "broken.pdf").write_bytes(b"not a pdf")
    report = _indexer(cfg).ingest(corpus)
    assert "broken.pdf" in report.failed
    assert sorted(report.indexed) == ["bm25.md", "w2v.txt"]


def test_embedding_model_change_requires_rebuild(cfg: Settings, corpus: Path) -> None:
    _indexer(cfg).ingest(corpus)
    other = HashEmbedder(dim=16, model_name="other-model")
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        _indexer(cfg, other).ingest(corpus)
    report = _indexer(cfg, other).ingest(corpus, rebuild=True)
    assert sorted(report.indexed) == ["bm25.md", "w2v.txt"] and report.total_chunks == 2
    assert Manifest.load(cfg.app.index_dir / "manifest.json").embedding_model == "other-model"


def test_chunking_change_requires_rebuild(cfg: Settings, corpus: Path) -> None:
    _indexer(cfg).ingest(corpus)
    changed = cfg.with_overrides({"ingestion.chunk_size_tokens": 200})
    with pytest.raises(IndexMismatchError, match="Chunking settings changed"):
        _indexer(changed).ingest(corpus)


def test_dense_and_bm25_indexes_find_the_right_chunk(cfg: Settings, corpus: Path) -> None:
    indexer = _indexer(cfg)
    indexer.ingest(corpus)
    bm25_hits = indexer.load_bm25().search("skip-gram centre word", k=2)
    assert len(bm25_hits) == 1  # only the Word2Vec note contains those terms
    top_bm25 = indexer.store.get([bm25_hits[0][0]])[0]
    assert top_bm25.file_name == "w2v.txt"
    dense = indexer.store.query(indexer.embedder.embed_query("BM25 term frequency saturation"), 2)
    top_dense = indexer.store.get([dense[0][0]])[0]
    assert top_dense.file_name == "bm25.md"
    assert top_dense.heading_path == "BM25" and top_dense.page == 1
    assert dense[0][1] > dense[1][1]


def test_stats_summarise_the_index(cfg: Settings, corpus: Path) -> None:
    indexer = _indexer(cfg)
    indexer.ingest(corpus)
    stats = index_stats(cfg, store=indexer.store)
    assert (stats.n_docs, stats.n_pages, stats.n_chunks) == (2, 2, 2)
    assert stats.embedding_model == "hash-embedder" and stats.dim == 32
    assert stats.bm25_ready and stats.index_bytes > 0
    assert stats.avg_tokens_per_chunk > 0


def test_unchanged_reingest_never_loads_the_embedding_model(cfg: Settings, corpus: Path) -> None:
    _indexer(cfg).ingest(corpus)

    def must_not_load() -> HashEmbedder:
        raise AssertionError("model loaded although every file is unchanged")

    lazy = LazyEmbedder("hash-embedder", must_not_load)
    report = _indexer(cfg, lazy).ingest(corpus)  # type: ignore[arg-type]
    assert report.new_chunks == 0 and not lazy.loaded


def test_dimension_change_under_same_model_name_is_caught(cfg: Settings, corpus: Path) -> None:
    _indexer(cfg).ingest(corpus)
    (corpus / "new.md").write_bytes(b"# New\nA new note that must be embedded.\n")
    with pytest.raises(IndexMismatchError, match="32-dimensional"):
        _indexer(cfg, HashEmbedder(dim=16)).ingest(corpus)


def test_ingest_format_change_requires_rebuild(
    cfg: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code changes to cleaning/chunking cannot show up in config, so a version does."""
    from lecturelens.indexing import indexer as indexer_mod

    _indexer(cfg).ingest(corpus)
    monkeypatch.setattr(indexer_mod, "INGEST_FORMAT_VERSION", indexer_mod.INGEST_FORMAT_VERSION + 1)
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        _indexer(cfg).ingest(corpus)
    assert _indexer(cfg).ingest(corpus, rebuild=True).total_chunks == 2
