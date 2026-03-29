"""
用户期望情感（5 维 L1 概率向量）与诗歌 prob_l1_* 的余弦相似度排序。

返回 (ranked_ids, has_signal):
  has_signal=False 表示用户未指定任何情感，调用方应跳过本维度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

L1_COLS = [
    "prob_l1_negative",
    "prob_l1_implicit_negative",
    "prob_l1_neutral",
    "prob_l1_implicit_positive",
    "prob_l1_positive",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_ids_by_sentiment(
    df: pd.DataFrame, user_vec: list[float] | None
) -> tuple[list, bool]:
    """
    返回 (ranked_ids, has_signal)。
    user_vec 为 None 时 has_signal=False。
    """
    if not user_vec or len(user_vec) != 5:
        return df["ID"].tolist(), False
    u = np.array(user_vec, dtype=float)
    scores: list[tuple[object, float]] = []
    for _, row in df.iterrows():
        v = np.array([row[c] for c in L1_COLS], dtype=float)
        scores.append((row["ID"], cosine(u, v)))
    scores.sort(key=lambda x: -x[1])
    return [i for i, _ in scores], True
