"""Tests for retrieval.fusion.reciprocal_rank_fusion (SPEC 7.5)."""

import pytest

from lecturelens.retrieval.fusion import reciprocal_rank_fusion


def test_hand_computed_example_from_spec() -> None:
    # X: rank 1 in BM25 and rank 4 in dense -> 1/61 + 1/64; Y: rank 2 in dense only -> 1/62
    fused = reciprocal_rank_fusion({"dense": ["a", "Y", "b", "X"], "bm25": ["X", "c"]}, k=60)
    scores = {doc: score for doc, score, _ in fused}
    assert scores["X"] == pytest.approx(1 / 61 + 1 / 64)
    assert round(scores["X"], 4) == 0.0320
    assert scores["Y"] == pytest.approx(1 / 62)
    assert fused[0][0] == "X"
    assert fused[0][2] == {"bm25": 1, "dense": 4}


def test_item_in_both_lists_outranks_item_in_one() -> None:
    fused = reciprocal_rank_fusion({"dense": ["solo", "both"], "bm25": ["both"]}, k=60)
    assert [doc for doc, _, _ in fused] == ["both", "solo"]


def test_weights_are_respected() -> None:
    lists = {"dense": ["d"], "bm25": ["b"]}
    assert reciprocal_rank_fusion(lists, k=60, weights={"dense": 2.0, "bm25": 1.0})[0][0] == "d"
    assert reciprocal_rank_fusion(lists, k=60, weights={"dense": 1.0, "bm25": 2.0})[0][0] == "b"


def test_ties_are_broken_by_id_deterministically() -> None:
    fused = reciprocal_rank_fusion({"dense": ["z", "m"], "bm25": ["m", "z"]}, k=60)
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [doc for doc, _, _ in fused] == ["m", "z"]


def test_larger_k_damps_top_rank_advantage() -> None:
    lists = {"dense": ["a", "b"]}
    small = reciprocal_rank_fusion(lists, k=1)
    large = reciprocal_rank_fusion(lists, k=1000)
    assert small[0][1] / small[1][1] > large[0][1] / large[1][1]


def test_empty_and_duplicate_inputs() -> None:
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}, k=60) == []
    fused = reciprocal_rank_fusion({"dense": ["a", "a", "b"]}, k=60)
    assert dict((d, r) for d, _, r in fused)["a"] == {"dense": 1}
