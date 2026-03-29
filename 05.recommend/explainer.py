"""为单条推荐生成简短可追溯说明（中文）。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

DIM_ZH = {
    "dim_nature":   "自然天象",
    "dim_season":   "时令节序",
    "dim_space":    "地理空间",
    "dim_people":   "人物关系",
    "dim_virtue":   "品格志向",
    "dim_artifact": "器物生活",
    "dim_biota":    "动植物",
    "dim_social":   "人文社交",
    "dim_flower":   "花卉意象",
}

DIM_ORDER = list(DIM_ZH.keys())


def explain_row(
    row: pd.Series,
    query: str,
    emotion_label: str | None,
    sent_active: bool = True,
    dim_active: bool = True,
) -> dict[str, Any]:
    parts: list[str] = []

    # 情感维度
    parts.append(
        f"情感：L1={row.get('l1_polarity_zh', '')}，L3={row.get('l3_c3_zh', '')}。"
        f" 概率：负{row.get('prob_l1_negative', 0):.2f} / 隐负{row.get('prob_l1_implicit_negative', 0):.2f} / "
        f"中{row.get('prob_l1_neutral', 0):.2f} / 隐正{row.get('prob_l1_implicit_positive', 0):.2f} / 正{row.get('prob_l1_positive', 0):.2f}。"
    )
    if not sent_active:
        parts.append("（您未指定情感偏好，情感维度未参与排序。）")
    elif emotion_label:
        parts.append(f"您的情感关键词匹配到：「{emotion_label}」。")

    # 双重确认意象
    ci = row.get("confirmed_imagery", "")
    if ci and str(ci).strip():
        parts.append(f"双重确认意象（BERT top10 ∩ 诗学含英）：{ci}。")
    else:
        parts.append("双重确认意象：无（以 BM25 意象全集匹配为主）。")

    # 语义维度：优先用 IDF 加权向量
    for col in ("dim_vector_idf", "dim_vector"):
        s = row.get(col)
        if not pd.isna(s) and str(s).strip():
            try:
                dv = json.loads(str(s))
                v = list(dv)
                # 补零至 9 维
                while len(v) < len(DIM_ORDER):
                    v.append(0.0)
                top_i = max(range(len(DIM_ORDER)), key=lambda i: v[i])
                dim_label = DIM_ZH.get(DIM_ORDER[top_i], DIM_ORDER[top_i])
                parts.append(f"语义维度：最强为「{dim_label}」（IDF 加权后）。")
                break
            except (json.JSONDecodeError, ValueError):
                pass

    if not dim_active:
        parts.append("（您的输入未触发语义维度关键词，该维度未参与排序。）")

    return {
        "诗名":   row.get("诗名", ""),
        "作者":   row.get("作者", ""),
        "朝代":   row.get("朝代", ""),
        "花名":   row.get("花名", ""),
        "全文":   str(row.get("text", "") or row.get("正文_preview", "")),
        "正文摘录": str(row.get("正文_preview", ""))[:120],
        "说明":   " ".join(parts),
    }
