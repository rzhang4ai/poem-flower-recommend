"""
用户查询与每首诗「意象文档」的 BM25 匹配。

意象文档 = sxhy_raw_words（| 分隔，原词，保留多字词粒度）
          + ccpoem_top10（空格分隔单字 token）
          + confirmed_imagery（双重确认词，重复计入以提升权重）

用户查询：jieba 切词（适合现代汉语），同时提取 1-2 字古汉语词
"""

from __future__ import annotations

import re

import jieba
import pandas as pd

from bm25 import BM25Okapi

# 停用词（对排序没有实质意义的词，主要是现代汉语连接词）
_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "什么",
    "吗", "呢", "啊", "哦", "吧", "嗯", "么", "个", "给", "将",
    "中", "对", "可", "但", "而", "或", "以", "及", "等", "被",
}


def _doc_tokens(row: pd.Series) -> list[str]:
    """
    古汉语诗歌意象文档 token 化：
    - sxhy_raw_words: 直接按 | 分割，保留原词（含多字词如 '映雪''长门'）
    - ccpoem_top10:  按空格分割（已是 BERT 子词单字，直接用）
    - confirmed_imagery: 双重确认词重复一次以提升 BM25 得分
    """
    raw = str(row.get("sxhy_raw_words", "") or "")
    top10 = str(row.get("ccpoem_top10", "") or "")
    ci = str(row.get("confirmed_imagery", "") or "")

    tokens: list[str] = []

    # sxhy_raw_words：管道分隔，直接取词
    for w in raw.split("|"):
        w = w.strip()
        if w:
            tokens.append(w)

    # ccpoem_top10：空格分隔单字
    for w in top10.split():
        w = w.strip()
        if w:
            tokens.append(w)

    # confirmed_imagery：双重确认词，重复以提权
    if ci:
        for w in ci.split("|"):
            w = w.strip()
            if w:
                tokens.append(w)
                tokens.append(w)  # 重复一次，提升权重

    return tokens if tokens else ["__empty__"]


def query_tokens(query: str) -> list[str]:
    """
    用户查询 token 化：
    - jieba 切词处理现代汉语词（老师、思念、送别...）
    - 同时提取 query 中的所有 1-2 字子串，覆盖古汉语词
    - 过滤停用词
    """
    q = str(query or "").strip()
    if not q:
        return []

    tokens: list[str] = []

    # jieba 切词
    _punc = re.compile(r"^[\s\W]+$")
    for w in jieba.cut(q):
        w = w.strip()
        if w and w not in _STOPWORDS and not _punc.match(w):
            tokens.append(w)

    # 额外：提取 2 字子串（覆盖古汉语两字词，如'高洁''清冷''离别'）
    _punc1 = re.compile(r"[\s\W]")
    for i in range(len(q) - 1):
        bigram = q[i:i+2]
        if not _punc1.match(bigram):
            tokens.append(bigram)

    return tokens


def build_bm25(df: pd.DataFrame) -> tuple[BM25Okapi, list[list[str]]]:
    corpus = [_doc_tokens(row) for _, row in df.iterrows()]
    return BM25Okapi(corpus), corpus


def rank_ids_by_bm25(df: pd.DataFrame, bm25: BM25Okapi, qtok: list[str]) -> list:
    if not qtok:
        return df["ID"].tolist()
    scores = bm25.get_scores(qtok)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ids = df["ID"].tolist()
    return [ids[i] for i in order]
