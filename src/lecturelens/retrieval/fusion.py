"""Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, SIGIR 2009; SPEC 7.5).

BM25 scores and cosine similarities live on different scales, so adding them is meaningless.
RRF uses only ranks: ``score(d) = sum_i w_i / (k + rank_i(d))`` with ranks starting at 1. A chunk
ranked well by both retrievers rises to the top; a chunk found by only one still gets credit.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    k: int,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists.

    Args:
        ranked_lists: Retriever name -> ids, best first (e.g. ``{"dense": [...], "bm25": [...]}``).
        k: RRF constant; larger values damp the dominance of the top ranks (60 in the paper).
        weights: Optional per-retriever weights (default 1.0 each).

    Returns:
        ``(id, fused_score, {retriever: rank})`` sorted by score descending; ties are broken by
        id so the output is deterministic.
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, ids in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            if name in ranks[doc_id]:  # duplicate id within one list: keep its best rank
                continue
            scores[doc_id] += weight / (k + rank)
            ranks[doc_id][name] = rank
    return sorted(
        ((doc_id, score, ranks[doc_id]) for doc_id, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
