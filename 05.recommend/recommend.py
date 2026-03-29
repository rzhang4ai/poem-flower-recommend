"""
多维匹配 + RRF 排序 + 花种均衡。供脚本与 Streamlit 共用。

匹配通道（动态权重，无信号时自动置 0）：
  1. BM25 意象词匹配      —— 始终参与，权重 0.30
     ·  包含 LLM 提取的 imagery_hints + classical_query（若有）
  2. 情感向量余弦          —— LLM 推断 / 规则 fallback；有信号权重 0.30
  3. 9 维语义 dim 余弦     —— 场景预设 > 文本关键词；有信号权重 0.25
  4. 嵌入相似度            —— 文件存在且 API 可用时权重 0.15；否则 0

各通道权重在有效通道间归一化，确保总和 = 1.0。

月份过滤已去除（当代花卉四季可购）。
per-flower capping 保证花种多样性。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explainer import explain_row
from matcher_embedding import rank_ids_by_embedding
from matcher_imagery import build_bm25, query_tokens, rank_ids_by_bm25
from matcher_semantic_dim import DIM_ORDER, load_dim_keywords, rank_ids_by_semantic_dim
from matcher_sentiment import rank_ids_by_sentiment
from paths import DATA_CSV, DIM_KEYWORDS_JSON, LEXICON_JSON, SCENE_PRESETS_JSON
from ranker import rrf

def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s < 1e-12:
        return [0.0] * len(vec)
    return [x / s for x in vec]


# ── 情感词典 ──────────────────────────────────────────────────────────────

def load_emotion_lexicon(path: Path) -> dict[str, list[float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_emotion_vec(
    lex: dict[str, list[float]],
    lex_key: str | None,
    query: str,
) -> tuple[list[float] | None, str | None]:
    """
    优先使用 LLM 推断的 lex_key，否则在 query 中扫描词典关键词。
    返回 (5维向量, 命中词); 未命中返回 (None, None)。
    """
    if lex_key and lex_key in lex:
        return lex[lex_key], lex_key
    q = str(query or "")
    for kw, vec in lex.items():
        if kw not in ("default",) and len(kw) >= 2 and kw in q:
            return vec, kw
    return None, None


# ── 场景预设 ──────────────────────────────────────────────────────────────

def load_scene_presets(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def scene_to_dim_vector(scene_key: str, scene_presets: dict[str, Any]) -> np.ndarray | None:
    preset = scene_presets.get(scene_key)
    if preset is None:
        return None
    raw = list(preset.get("vector", []))
    while len(raw) < len(DIM_ORDER):
        raw.append(0.0)
    normed = _l2_normalize(raw)
    return np.array(normed, dtype=float)


def rank_ids_by_scene_dim(df: pd.DataFrame, scene_vec: np.ndarray) -> tuple[list, bool]:
    from matcher_semantic_dim import cosine, poem_dim_array
    scores = [(row["ID"], cosine(scene_vec, poem_dim_array(row))) for _, row in df.iterrows()]
    scores.sort(key=lambda x: -x[1])
    return [i for i, _ in scores], True


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── 花种均衡 ──────────────────────────────────────────────────────────────

def _cap_by_flower(
    ranked: list[tuple[Any, float]],
    id_to_row: dict[Any, Any],
    max_per_flower: int = 2,
) -> list[tuple[Any, float]]:
    from collections import defaultdict
    flower_counts: dict[str, int] = defaultdict(int)
    result: list[tuple[Any, float]] = []
    for doc_id, score in ranked:
        row = id_to_row.get(doc_id)
        if row is None:
            continue
        flower = str(row.get("花名", ""))
        if flower_counts[flower] < max_per_flower:
            flower_counts[flower] += 1
            result.append((doc_id, score))
    return result


# ── 核心推荐函数 ──────────────────────────────────────────────────────────

def recommend(
    df: pd.DataFrame,
    dim_kw: list[list[str]],
    emotion_lex: dict[str, list[float]],
    scene_presets: dict[str, Any],
    query: str,
    intent=None,                          # IntentResult | None，来自 llm_intent_parser
    scene_key: str | None = None,         # UI 精调覆盖
    emotion_vec_override: list[float] | None = None,  # UI 精调覆盖
    top_k: int = 3,
    k_rrf: int = 60,
    max_per_flower: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    主推荐函数。

    Parameters
    ----------
    intent              : LLM 解析结果（IntentResult），None 时使用规则 fallback
    scene_key           : UI 精调指定场景（覆盖 intent.scene_key）
    emotion_vec_override: UI 精调指定情感向量（覆盖所有自动推断）

    Returns
    -------
    (results, signals)  signals 供 UI 展示系统理解
    """
    if df.empty:
        return [], {}

    # ── 从 intent 提取信号 ────────────────────────────────────────────────
    emo_lex_key   = None
    imagery_extra: list[str] = []
    classical_q   = ""
    if intent is not None:
        emo_lex_key   = intent.emotion_lex_key
        imagery_extra = list(intent.imagery_hints or [])
        classical_q   = intent.classical_query or ""

    # UI 精调优先级最高
    effective_scene = scene_key or (intent.scene_key if intent else None)

    # ── 通道 1：BM25（意象词 + LLM 提示词）────────────────────────────────
    bm25_f, _ = build_bm25(df)
    combined_q = " ".join(filter(None, [query, " ".join(imagery_extra), classical_q]))
    qtok = query_tokens(combined_q)
    ids_bm25 = rank_ids_by_bm25(df, bm25_f, qtok)

    # ── 通道 2：情感 ──────────────────────────────────────────────────────
    if emotion_vec_override:
        user_vec, emo_hit = emotion_vec_override, "（精调指定）"
    else:
        user_vec, emo_hit = resolve_emotion_vec(emotion_lex, emo_lex_key, query)
    ids_sent, sent_active = rank_ids_by_sentiment(df, user_vec)

    # ── 通道 3：9 维语义 dim ──────────────────────────────────────────────
    scene_vec = scene_to_dim_vector(effective_scene, scene_presets) if effective_scene else None
    if scene_vec is not None:
        ids_dim, dim_active = rank_ids_by_scene_dim(df, scene_vec)
    else:
        ids_dim, dim_active = rank_ids_by_semantic_dim(df, query, dim_kw)

    # ── 通道 4：嵌入相似度 ────────────────────────────────────────────────
    ids_emb, emb_active = rank_ids_by_embedding(df, query, classical_q)

    # ── 动态权重归一化 ────────────────────────────────────────────────────
    channel_weights = {
        "bm25": 0.30,
        "sent": 0.30 if sent_active else 0.0,
        "dim":  0.25 if dim_active  else 0.0,
        "emb":  0.15 if emb_active  else 0.0,
    }
    total = sum(channel_weights.values()) or 1.0
    lists   = [ids_bm25, ids_sent, ids_dim, ids_emb]
    weights = [channel_weights[k] / total
               for k in ("bm25", "sent", "dim", "emb")]

    # ── RRF 融合 ─────────────────────────────────────────────────────────
    fused = rrf(lists, weights, k=k_rrf)
    id_to_row = {r["ID"]: r for _, r in df.iterrows()}

    # ── 花种均衡截断 ──────────────────────────────────────────────────────
    candidates = _cap_by_flower(fused[: top_k * 6], id_to_row, max_per_flower)

    out: list[dict[str, Any]] = []
    for doc_id, rrf_score in candidates[:top_k]:
        row = id_to_row.get(doc_id)
        if row is None:
            continue
        exp = explain_row(row, query, emo_hit, sent_active, dim_active)
        exp["rrf_score"] = round(rrf_score, 6)
        out.append(exp)

    signals: dict[str, Any] = {
        "emo_hit":        emo_hit,
        "sent_active":    sent_active,
        "detected_scene": effective_scene,
        "dim_active":     dim_active,
        "emb_active":     emb_active,
        "from_llm":       getattr(intent, "from_llm", False),
        "classical_query": classical_q,
        "qtok":           qtok[:8],
        "weights":        {k: round(v, 3) for k, v in zip(
                             ("bm25","sent","dim","emb"), weights)},
    }
    return out, signals


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_data_and_index(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, list[list[str]], dict[str, list[float]], dict[str, Any]]:
    path = csv_path or DATA_CSV
    if not path.exists():
        raise FileNotFoundError(f"请先运行 prepare_dimensions.py 生成: {path}")
    df = pd.read_csv(path)
    dim_kw = load_dim_keywords(DIM_KEYWORDS_JSON)
    emo = load_emotion_lexicon(LEXICON_JSON)
    scenes = load_scene_presets(SCENE_PRESETS_JSON)
    return df, dim_kw, emo, scenes


# ── CLI 测试 ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="命令行测试推荐")
    ap.add_argument("query", nargs="?", default="想送给即将退休的老师一首花诗")
    ap.add_argument("--scene",   default=None)
    ap.add_argument("--no-llm",  action="store_true", help="强制不调用 LLM，仅规则")
    ap.add_argument("--top",     type=int, default=3)
    ap.add_argument("--cap",     type=int, default=2)
    args = ap.parse_args()

    df, dim_kw, emo, scenes = load_data_and_index()

    intent = None
    if not args.no_llm:
        try:
            import llm_intent_parser
            intent = llm_intent_parser.parse(args.query)
            print(f"LLM 解析：scene={intent.scene_key}, emo={intent.emotion_lex_key}, "
                  f"classical_q={intent.classical_query}, from_llm={intent.from_llm}")
        except Exception as e:
            print(f"LLM 解析异常（{e}），使用规则 fallback")

    res, signals = recommend(
        df, dim_kw, emo, scenes,
        args.query,
        intent=intent,
        scene_key=args.scene,
        top_k=args.top,
        max_per_flower=args.cap,
    )
    print("通道权重：", signals.get("weights"))
    for i, r in enumerate(res, 1):
        print(f"\n--- {i}. 《{r.get('诗名')}》· {r.get('作者')} · {r.get('花名')}")
        print(r.get("说明"))
        print("rrf:", r.get("rrf_score"))


if __name__ == "__main__":
    main()
