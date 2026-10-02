"""Retrieval metrics with page-level binary relevance (SPEC 12.2).

A retrieved chunk is relevant when its ``(file, page)`` is one of the question's gold pages.
Rankings are lists of ``(file, page)`` in retrieval order (one entry per chunk, so the same
page can appear twice; it only counts the first time).
"""

import math
from collections.abc import Sequence

Page = tuple[str, int]


def _first_hits(ranking: Sequence[Page], gold: set[Page], k: int) -> list[int]:
    """1-based ranks within the top ``k`` where a *new* gold page appears."""
    seen: set[Page] = set()
    ranks = []
    for rank, page in enumerate(ranking[:k], start=1):
        if page in gold and page not in seen:
            seen.add(page)
            ranks.append(rank)
    return ranks


def hit_at_k(ranking: Sequence[Page], gold: set[Page], k: int) -> float:
    """1.0 if any gold page is in the top ``k``, else 0.0."""
    return 1.0 if _first_hits(ranking, gold, k) else 0.0


def recall_at_k(ranking: Sequence[Page], gold: set[Page], k: int) -> float:
    """Fraction of gold pages found in the top ``k``."""
    return len(_first_hits(ranking, gold, k)) / len(gold) if gold else 0.0


def reciprocal_rank(ranking: Sequence[Page], gold: set[Page]) -> float:
    """1 / rank of the first gold page (0 if none was retrieved)."""
    hits = _first_hits(ranking, gold, len(ranking))
    return 1.0 / hits[0] if hits else 0.0


def ndcg_at_k(ranking: Sequence[Page], gold: set[Page], k: int) -> float:
    """Normalised DCG with binary relevance; each gold page earns credit once."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank in _first_hits(ranking, gold, k))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def first_gold_rank(ranking: Sequence[Page], gold: set[Page]) -> int | None:
    """Rank of the first gold page, or ``None`` (useful in per-question reports)."""
    hits = _first_hits(ranking, gold, len(ranking))
    return hits[0] if hits else None


def aggregate(
    rankings: Sequence[Sequence[Page]], golds: Sequence[set[Page]], k_values: Sequence[int]
) -> dict[str, float]:
    """Mean Hit@k, Recall@k, nDCG@k (for each k) and MRR over a set of questions."""
    n = len(rankings)
    if n == 0:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = (
            sum(hit_at_k(r, g, k) for r, g in zip(rankings, golds, strict=True)) / n
        )
        metrics[f"recall@{k}"] = (
            sum(recall_at_k(r, g, k) for r, g in zip(rankings, golds, strict=True)) / n
        )
        metrics[f"ndcg@{k}"] = (
            sum(ndcg_at_k(r, g, k) for r, g in zip(rankings, golds, strict=True)) / n
        )
    metrics["mrr"] = sum(reciprocal_rank(r, g) for r, g in zip(rankings, golds, strict=True)) / n
    return metrics
