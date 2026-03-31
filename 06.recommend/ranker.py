"""Reciprocal Rank Fusion (RRF)。"""

from __future__ import annotations

from collections import defaultdict


def rrf(
    rank_lists: list[list[int]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    rank_lists: 多个排序列表，每个列表为 ID 从好到差。
    weights: 与 rank_lists 等长，权重和不必为 1。
    返回 [(id, score), ...] 按 score 降序。
    """
    if len(rank_lists) != len(weights):
        raise ValueError("rank_lists 与 weights 长度须一致")
    scores: dict[int, float] = defaultdict(float)
    for ranks, w in zip(rank_lists, weights):
        if w <= 0:
            continue
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
