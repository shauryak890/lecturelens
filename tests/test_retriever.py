"""Tests for retrieval.retriever.HybridRetriever on the indexed sample notes (offline)."""

import pytest

from lecturelens.config import Settings
from lecturelens.indexing.indexer import Indexer
from lecturelens.retrieval.retriever import HybridRetriever

from .fakes import FakeReranker

QUERY = "Kneser-Ney continuation probability"


def _retriever(indexer: Indexer, reranker: FakeReranker | None = None) -> HybridRetriever:
    s = indexer.settings
    return HybridRetriever(
        indexer.embedder,
        indexer.store,
        indexer.load_bm25(),
        reranker,
        s.retrieval,
        s.ingestion.add_context_header,
    )


def _files(results) -> list[str]:
    return [r.chunk.file_name for r in results]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_mode_returns_ranked_chunks(tiny_index: Indexer, mode: str) -> None:
    results = _retriever(tiny_index).retrieve(QUERY, mode=mode, rerank=False)
    assert 0 < len(results) <= tiny_index.settings.retrieval.final_k
    assert {r.source for r in results} == {mode}
    target = "02_ngram_language_models.md"
    if mode == "dense":  # the hash embedder is crude (bucket collisions): just require a hit
        assert target in _files(results)
    else:
        assert results[0].chunk.file_name == target
    if mode != "hybrid":
        assert [r.ranks[mode] for r in results] == list(range(1, len(results) + 1))


def test_hybrid_records_both_ranks_and_modes_differ(tiny_index: Indexer) -> None:
    retriever = _retriever(tiny_index)
    hybrid = retriever.retrieve(QUERY, mode="hybrid", rerank=False)
    assert any({"dense", "bm25"} <= set(r.ranks) for r in hybrid)
    by_mode = {
        m: [r.chunk.chunk_id for r in retriever.retrieve(QUERY, mode=m, rerank=False)]
        for m in ("dense", "bm25", "hybrid")
    }
    assert len({tuple(ids) for ids in by_mode.values()}) > 1  # FR-3: modes change results


def test_doc_filter_restricts_all_modes(tiny_index: Indexer) -> None:
    doc_id = next(
        c.doc_id for c in tiny_index.store.all_chunks() if c.file_name == "03_word_embeddings.md"
    )
    retriever = _retriever(tiny_index)
    for mode in ("dense", "bm25", "hybrid"):
        results = retriever.retrieve(QUERY, mode=mode, doc_filter=[doc_id], rerank=False)
        assert set(_files(results)) <= {"03_word_embeddings.md"}


def test_final_k_is_respected(tiny_index: Indexer) -> None:
    assert len(_retriever(tiny_index).retrieve("language", final_k=2, rerank=False)) == 2


def test_rerank_reorders_and_records_rank(tiny_index: Indexer) -> None:
    reranker = FakeReranker()
    results = _retriever(tiny_index, reranker).retrieve(QUERY, rerank=True)
    assert reranker.calls and {r.source for r in results} == {"rerank"}
    assert [r.ranks["rerank"] for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    # the reranker sees the contextual header ("file | heading") like the indexes do
    assert all(p.startswith(("01_", "02_", "03_")) for p in reranker.calls[0][1])


def test_all_rerank_scores_below_threshold_returns_nothing(tiny_index: Indexer) -> None:
    hopeless = FakeReranker(offset=100.0)  # every score far below min_rerank_score (-5)
    assert _retriever(tiny_index, hopeless).retrieve(QUERY, rerank=True) == []


def test_rerank_override_needs_a_reranker(tiny_index: Indexer, settings: Settings) -> None:
    results = _retriever(tiny_index, None).retrieve(QUERY, rerank=True)  # no model configured
    assert results and {r.source for r in results} == {settings.retrieval.mode}


def test_config_rerank_off_skips_the_reranker(tiny_index: Indexer, settings: Settings) -> None:
    assert settings.retrieval.rerank is False  # default since the evaluation (see README)
    reranker = FakeReranker()
    results = _retriever(tiny_index, reranker).retrieve(QUERY)
    assert results and reranker.calls == [] and {r.source for r in results} == {"hybrid"}
