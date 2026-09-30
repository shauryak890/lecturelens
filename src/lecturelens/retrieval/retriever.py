"""HybridRetriever: dense | bm25 | hybrid retrieval, optional cross-encoder rerank (SPEC 6.5).

1. dense  = top ``top_k_dense`` chunks by cosine similarity (Chroma)
2. sparse = top ``top_k_bm25`` chunks by BM25
3. hybrid = Reciprocal Rank Fusion of both lists
4. rerank = cross-encoder rescoring of the top ``rerank_candidates``; if even the best score is
   below ``min_rerank_score`` nothing is returned, so the pipeline abstains without an LLM call
5. keep ``final_k`` (the token budget is applied when the context is built)
"""

import logging
from typing import Literal

from lecturelens.config import RetrievalCfg, RetrievalMode
from lecturelens.indexing.bm25_index import BM25Index
from lecturelens.indexing.embedder import Embedder
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.ingestion.chunker import indexed_text
from lecturelens.retrieval.fusion import reciprocal_rank_fusion
from lecturelens.retrieval.reranker import Reranker
from lecturelens.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

Candidate = tuple[str, float, dict[str, int]]  # (chunk_id, score, {retriever: rank})


class HybridRetriever:
    """Retrieves chunks for a query with the configured (or per-call) mode."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker | None,
        cfg: RetrievalCfg,
        add_context_header: bool,
    ) -> None:
        """Create the retriever.

        Args:
            embedder: Query embedder (same model the index was built with).
            store: Dense vector store.
            bm25: Lexical index over the same chunks.
            reranker: Cross-encoder, or ``None`` to disable reranking entirely.
            cfg: Retrieval settings.
            add_context_header: Whether chunks were indexed with the "file | heading" header;
                the reranker sees the same text.
        """
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.cfg = cfg
        self.add_context_header = add_context_header

    def is_empty(self) -> bool:
        """Whether the index holds no chunks at all."""
        return self.store.count() == 0

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode | None = None,
        final_k: int | None = None,
        doc_filter: list[str] | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Return the best chunks for ``query``, most relevant first.

        Args:
            query: Standalone search query.
            mode: ``dense``, ``bm25`` or ``hybrid``; defaults to ``retrieval.mode``.
            final_k: Number of chunks to return; defaults to ``retrieval.final_k``.
            doc_filter: Restrict results to these ``doc_id`` values.
            rerank: Override ``retrieval.rerank`` (ignored when no reranker is configured).

        Returns:
            Retrieved chunks with scores and per-retriever ranks. Empty if nothing matched or
            every reranked candidate scored below ``min_rerank_score``.
        """
        mode = mode or self.cfg.mode
        final_k = final_k or self.cfg.final_k
        use_rerank = (self.cfg.rerank if rerank is None else rerank) and self.reranker is not None

        candidates = self._candidates(query, mode, doc_filter)
        limit = max(self.cfg.rerank_candidates, final_k) if use_rerank else final_k
        results = self._materialise(candidates[:limit], mode)
        if use_rerank and results:
            results = self._rerank(query, results)
        return results[:final_k]

    def _candidates(
        self, query: str, mode: RetrievalMode, doc_filter: list[str] | None
    ) -> list[Candidate]:
        lists: dict[str, list[tuple[str, float]]] = {}
        if mode in ("dense", "hybrid"):
            where = {"doc_id": {"$in": doc_filter}} if doc_filter else None
            vector = self.embedder.embed_query(query)
            lists["dense"] = self.store.query(vector, self.cfg.top_k_dense, where)
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = self._bm25_search(query, doc_filter)
        if mode == "hybrid":
            ranked = {name: [cid for cid, _ in hits] for name, hits in lists.items()}
            return reciprocal_rank_fusion(ranked, self.cfg.rrf_k, self.cfg.weights)
        return [(cid, score, {mode: rank}) for rank, (cid, score) in enumerate(lists[mode], 1)]

    def _bm25_search(self, query: str, doc_filter: list[str] | None) -> list[tuple[str, float]]:
        if not doc_filter:
            return self.bm25.search(query, self.cfg.top_k_bm25)
        allowed = set(doc_filter)
        hits = self.bm25.search(query, len(self.bm25))  # BM25 has no filter: rank all, then keep
        return [h for h in hits if h[0].split(":", 1)[0] in allowed][: self.cfg.top_k_bm25]

    def _materialise(
        self, candidates: list[Candidate], mode: RetrievalMode
    ) -> list[RetrievedChunk]:
        chunks = {c.chunk_id: c for c in self.store.get([cid for cid, _, _ in candidates])}
        source: Literal["dense", "bm25", "hybrid"] = mode
        return [
            RetrievedChunk(chunk=chunks[cid], score=score, source=source, ranks=ranks)
            for cid, score, ranks in candidates
            if cid in chunks
        ]

    def _rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        assert self.reranker is not None
        texts = [indexed_text(r.chunk, self.add_context_header) for r in results]
        scores = self.reranker.score(query, texts)
        if max(scores) < self.cfg.min_rerank_score:
            logger.info(
                "All %d rerank scores below %.2f: abstaining",
                len(scores),
                self.cfg.min_rerank_score,
            )
            return []
        order = sorted(range(len(results)), key=lambda i: -scores[i])  # stable for ties
        return [
            results[i].model_copy(
                update={
                    "score": scores[i],
                    "source": "rerank",
                    "ranks": {**results[i].ranks, "rerank": position},
                }
            )
            for position, i in enumerate(order, start=1)
        ]
