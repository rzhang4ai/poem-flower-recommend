"""
方向 B（在线步骤）：加载预计算的诗歌嵌入向量（由 embed_shangxi_gemini.py 离线生成），
对用户 query 实时嵌入（task_type=RETRIEVAL_QUERY），通过余弦相似度排序。

嵌入模型：默认 gemini-embedding-2-preview（可由 GEMINI_EMBED_MODEL 覆盖）
  - 诗词文档侧：RETRIEVAL_DOCUMENT（离线，embed_shangxi_gemini.py 生成）
  - 用户查询侧：RETRIEVAL_QUERY（在线，本文件实时调用）

若嵌入文件不存在或 API 不可用，自动跳过（has_signal=False）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from paths import OUTPUT_DIR

logger = logging.getLogger(__name__)

_IDS_PATH  = OUTPUT_DIR / "poems_embed_ids.json"
_EMBS_PATH = OUTPUT_DIR / "poems_embeddings.npy"

# 模块级缓存
_embed_matrix: np.ndarray | None = None
_embed_ids:    list[str]  | None = None


def _load_poem_embeddings() -> tuple[np.ndarray | None, list[str] | None]:
    global _embed_matrix, _embed_ids
    if _embed_matrix is not None:
        return _embed_matrix, _embed_ids
    if not _IDS_PATH.exists() or not _EMBS_PATH.exists():
        logger.info("诗歌嵌入文件不存在，跳过 embedding 通道。运行 embed_poems.py 生成。")
        return None, None
    try:
        _embed_ids    = json.loads(_IDS_PATH.read_text(encoding="utf-8"))
        _embed_matrix = np.load(str(_EMBS_PATH)).astype(np.float32)
        # L2 归一化（方便之后只做点积）
        norms = np.linalg.norm(_embed_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        _embed_matrix = _embed_matrix / norms
        logger.info("诗歌嵌入加载成功：shape=%s", _embed_matrix.shape)
        return _embed_matrix, _embed_ids
    except Exception as e:
        logger.warning("加载嵌入文件失败: %s", e)
        return None, None


def embed_query(text: str) -> np.ndarray | None:
    """
    对用户查询文本实时嵌入（task_type=RETRIEVAL_QUERY）。
    优先使用 Gemini，失败返回 None。
    """
    try:
        import gemini_client
        if not gemini_client.is_available():
            return None
        vecs = gemini_client.embed([text], task_type="RETRIEVAL_QUERY")
        v = np.array(vecs[0], dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return None
        # 与离线文档嵌入同模型时，余弦相似度前统一 L2 归一化
        return v / norm
    except Exception as e:
        logger.warning("query 嵌入失败: %s", e)
        return None


def rank_ids_by_embedding(
    df: pd.DataFrame,
    query_text: str,
    classical_query: str = "",
) -> tuple[list, bool]:
    """
    用嵌入相似度对 df 中的诗排序。

    Parameters
    ----------
    query_text      : 用户原始查询
    classical_query : LLM 生成的古典风格短语（可选，与原始 query 融合嵌入）

    Returns
    -------
    (ranked_ids, has_signal)
    has_signal=False 时排序无效，RRF 应将权重置 0。
    """
    matrix, all_ids = _load_poem_embeddings()
    if matrix is None or all_ids is None:
        return df["ID"].tolist(), False

    # 融合原始 query 与 classical_query 的嵌入（平均）
    texts_to_embed = [query_text]
    if classical_query and classical_query != query_text:
        texts_to_embed.append(classical_query)

    q_vecs: list[np.ndarray] = []
    for t in texts_to_embed:
        v = embed_query(t)
        if v is not None:
            q_vecs.append(v)

    if not q_vecs:
        return df["ID"].tolist(), False

    # 平均融合
    q_vec = np.mean(np.stack(q_vecs), axis=0).astype(np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-12:
        return df["ID"].tolist(), False
    q_vec /= q_norm

    # 只对当前候选集（df）的诗排序
    df_ids = set(str(i) for i in df["ID"].tolist())
    id_to_score: dict[str, float] = {}
    for idx, pid in enumerate(all_ids):
        if pid in df_ids:
            id_to_score[pid] = float(np.dot(matrix[idx], q_vec))

    # 按相似度降序，保留原始 ID 类型
    sorted_items = sorted(id_to_score.items(), key=lambda x: -x[1])

    # 转回原始 ID 类型（str 或 int）
    orig_ids = {str(i): i for i in df["ID"].tolist()}
    ranked = [orig_ids[pid] for pid, _ in sorted_items if pid in orig_ids]

    # 补充未出现在嵌入文件中的 ID（放到末尾）
    ranked_set = set(str(i) for i in ranked)
    for i in df["ID"].tolist():
        if str(i) not in ranked_set:
            ranked.append(i)

    return ranked, True
