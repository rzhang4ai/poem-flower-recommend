"""
用户查询与 9 维 IDF 加权语义向量的余弦相似度排序。

使用 dim_vector_idf（IDF 加权后 L2 归一化），使稀少但有区分力的
dim_virtue / dim_social / dim_people 得到足够权重。

返回 (ranked_ids, has_signal):
  has_signal=False 表示用户查询未命中任何关键词，调用方应跳过本维度。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIM_ORDER = [
    "dim_nature",
    "dim_season",
    "dim_space",
    "dim_people",
    "dim_virtue",
    "dim_artifact",
    "dim_biota",
    "dim_social",
    "dim_flower",
]


def load_dim_keywords(path: Path) -> list[list[str]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data.get(k, []) for k in DIM_ORDER]
    return data


def user_dim_vector(text: str, dim_kw: list[list[str]]) -> tuple[np.ndarray, bool]:
    """
    根据 text 命中关键词构建 9 维用户意向向量。
    返回 (unit_vector, has_signal)。
    """
    t = str(text or "")
    vec = np.zeros(len(DIM_ORDER), dtype=float)
    for i, kws in enumerate(dim_kw):
        for kw in kws:
            if kw and kw in t:
                vec[i] += 1.0

    n = np.linalg.norm(vec)
    has_signal = n > 1e-12
    if not has_signal:
        return vec, False
    return vec / n, True


def poem_dim_array(row: pd.Series) -> np.ndarray:
    """优先使用 IDF 加权向量；没有时退回原始向量。"""
    for col in ("dim_vector_idf", "dim_vector"):
        s = row.get(col)
        if not pd.isna(s) and str(s).strip():
            try:
                arr = json.loads(str(s))
                v = np.array(arr, dtype=float)
                # 兼容旧 8 维数据：补零至 9 维
                if len(v) < len(DIM_ORDER):
                    v = np.concatenate([v, np.zeros(len(DIM_ORDER) - len(v))])
                return v
            except (json.JSONDecodeError, ValueError):
                pass
    return np.zeros(len(DIM_ORDER))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_ids_by_semantic_dim(
    df: pd.DataFrame, text: str, dim_kw: list[list[str]]
) -> tuple[list, bool]:
    """
    返回 (ranked_ids, has_signal)。
    has_signal=False 时排序无意义，RRF 应将本维度权重置 0。
    """
    u, has_signal = user_dim_vector(text, dim_kw)
    if not has_signal:
        return df["ID"].tolist(), False
    scores: list[tuple[object, float]] = []
    for _, row in df.iterrows():
        p = poem_dim_array(row)
        scores.append((row["ID"], cosine(u, p)))
    scores.sort(key=lambda x: -x[1])
    return [i for i, _ in scores], True
