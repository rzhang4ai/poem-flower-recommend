"""
07_step1_pseudo_labeling.py
===========================
第一步：FCCPSL 词典精确匹配 + 极性逻辑对撞 -> 黄金训练集

输入：
  output/lexicon/FSPC_V1.0.json          -- 5000首古诗 + FSPC 5极性标签
  output/lexicon/fccpsl_terms_only.csv   -- FCCPSL 词条表（含 C3 分类）

输出：
  output/pseudo_label/golden_dataset_step1.csv  -- 黄金训练集
  output/pseudo_label/step1_funnel_report.txt   -- 漏斗统计

核心设计决策（均可通过常量调整）：
  1. MIN_WORD_LEN = 2
       单字词噪声严重（"光"、"美"等会命中大量无关诗歌），默认只使用 ≥2 字的词条。
       通过 SINGLE_CHAR_WHITELIST 可保留少量高辨识单字（如"愁"、"悲"）。

  2. USE_IDF_WEIGHT = True
       FCCPSL 各类词数差距悬殊（praise 5582 vs miss 127），
       启用逆类别频率权重：每次命中得分 = 1 / log2(类词条数 + 1)，
       避免大词典类别垄断匹配结果。关闭后退化为原始命中频次。

  3. 逻辑对撞规则（LOGIC_RULES）直接对应用户提供的 FCCPSL-C3 ↔ FSPC 映射表。

  4. 并列处理：最高得分同时属于 ≥2 个类别 → 标记 Uncertain，丢弃。
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ===========================================================================
# 路径配置
# ===========================================================================
_SCRIPT_DIR  = Path(__file__).resolve().parent
_LEXICON_DIR = _SCRIPT_DIR / "output" / "lexicon"
_OUTPUT_DIR  = _SCRIPT_DIR / "output" / "pseudo_label"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FSPC_PATH    = _LEXICON_DIR / "FSPC_V1.0.json"
FCCPSL_PATH  = _LEXICON_DIR / "fccpsl_terms_only.csv"
OUT_CSV      = _OUTPUT_DIR  / "golden_dataset_step1.csv"
OUT_REPORT   = _OUTPUT_DIR  / "step1_funnel_report.txt"

# ===========================================================================
# 可调参数
# ===========================================================================
MIN_WORD_LEN  = 2           # 最小匹配词长（过滤单字噪声；单字白名单见下）
MAX_WORD_LEN  = 2           # 最大匹配词长
                            # 诊断：FCCPSL 3/4字词几乎不出现于古诗正文（命中率<1%），
                            # 实际有效词只有2字词。设为 None 表示不限制上限。
MIN_TOP_SCORE = 0.001       # 最高得分须超过此阈值：只需至少1次有效命中即可
                            # IDF加权下单次命中得分≈0.08~0.16，保持>0即可
USE_IDF_WEIGHT = True       # True=逆类别频率加权；False=原始命中频次

# 高辨识单字白名单：即使 MIN_WORD_LEN=2，这些字仍参与匹配
SINGLE_CHAR_WHITELIST = {
    "愁", "悲", "哀", "泪", "怨", "苦", "痛",   # 悲伤类
    "喜", "乐", "欢", "悦",                       # 喜悦类
    "思", "念", "忆",                             # 思念类
}

# 否定词集合：情感词前一字是否定词时，本次匹配作废
NEGATION_CHARS = {"不", "无", "莫", "非", "未", "休", "勿"}

# ===========================================================================
# FCCPSL-C3 ↔ FSPC 极性逻辑对撞规则（直接对应用户提供的映射表）
# ===========================================================================
LOGIC_RULES: Dict[str, frozenset] = {
    "joy":       frozenset({"Positive"}),
    "ease":      frozenset({"Positive", "Implicit Positive"}),
    "praise":    frozenset({"Positive", "Implicit Positive"}),
    "like":      frozenset({"Positive", "Implicit Positive"}),
    "faith":     frozenset({"Positive", "Implicit Positive", "Neutral"}),
    "wish":      frozenset({"Positive", "Implicit Positive"}),
    "peculiar":  frozenset({"Neutral"}),
    "fear":      frozenset({"Negative"}),
    "sorrow":    frozenset({"Negative", "Implicit Negative"}),
    "guilt":     frozenset({"Implicit Negative"}),
    "miss":      frozenset({"Implicit Negative", "Neutral"}),
    "criticize": frozenset({"Negative", "Implicit Negative"}),
    "anger":     frozenset({"Negative", "Implicit Negative"}),
    "vexed":     frozenset({"Negative", "Implicit Negative"}),
    "misgive":   frozenset({"Implicit Negative", "Neutral"}),
}

FSPC_POL_MAP = {
    "1": "Negative",
    "2": "Implicit Negative",
    "3": "Neutral",
    "4": "Implicit Positive",
    "5": "Positive",
}

# C3 英文 -> 中文（用于输出表头）
C3_ZH: Dict[str, str] = {}   # 从 CSV 自动读取


# ===========================================================================
# 工具函数
# ===========================================================================
def clean_poem(text: str) -> str:
    """去除分隔符，仅保留汉字，用于匹配。"""
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def count_matches(text: str, term: str) -> int:
    """
    统计 term 在 text 中有效出现次数。
    term 前一字为否定词时本次匹配作废。
    """
    cnt, start = 0, 0
    tlen = len(term)
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        if idx > 0 and text[idx - 1] in NEGATION_CHARS:
            pass
        else:
            cnt += 1
        start = idx + 1
    return cnt


# ===========================================================================
# 数据加载
# ===========================================================================
def load_fccpsl(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    按 C3 类别加载 FCCPSL 词条，应用词长过滤和单字白名单。

    返回：
      vocab_by_c3  : {c3: [词条, ...]}（已过滤）
      idf_weight   : {c3: 逆频率权重} = 1 / log2(|c3_vocab| + 1)
    """
    global C3_ZH
    df = pd.read_csv(path)
    C3_ZH = dict(zip(df["C3"], df["C3_zh"]))

    vocab_by_c3: Dict[str, List[str]] = {}
    for c3, grp in df.groupby("C3"):
        words = []
        for _, row in grp.iterrows():
            w = str(row["词"]).strip()
            wlen = len(w)
            in_whitelist = (wlen == 1 and w in SINGLE_CHAR_WHITELIST)
            in_range = (wlen >= MIN_WORD_LEN and
                        (MAX_WORD_LEN is None or wlen <= MAX_WORD_LEN))
            if in_range or in_whitelist:
                words.append(w)
        # 去重，按词长降序（优先匹配长词，避免短词提前消耗匹配位置）
        words = sorted(set(words), key=lambda x: -len(x))
        vocab_by_c3[c3] = words

    idf_weight = {
        c3: 1.0 / math.log2(len(words) + 1) if words else 0.0
        for c3, words in vocab_by_c3.items()
    }
    return vocab_by_c3, idf_weight


def load_fspc(path: Path) -> pd.DataFrame:
    """
    读取 FSPC_V1.0.json，输出 DataFrame。
    字段：poet, title, dynasty, poem_raw, text（汉字正文）, polarity
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            poem_raw = d.get("poem", "")
            text     = clean_poem(poem_raw)
            hol      = str(d.get("setiments", {}).get("holistic", "3"))
            rows.append({
                "poet":     d.get("poet", ""),
                "title":    d.get("title", ""),
                "dynasty":  d.get("dynasty", ""),
                "poem_raw": poem_raw,
                "text":     text,
                "polarity": FSPC_POL_MAP.get(hol, "Neutral"),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 核心：伪标签生成
# ===========================================================================
def pseudo_label_single(
    text: str,
    vocab_by_c3: Dict[str, List[str]],
    idf_weight: Dict[str, float],
) -> Tuple[str, float, Dict[str, float]]:
    """
    对单首诗计算各 C3 类别得分，返回：
      (pseudo_label, top_score, score_dict)

    若并列最高 或 最高分 < MIN_TOP_SCORE → pseudo_label = "Uncertain"
    """
    scores: Dict[str, float] = {}
    for c3, words in vocab_by_c3.items():
        raw = sum(count_matches(text, w) for w in words)
        weight = idf_weight[c3] if USE_IDF_WEIGHT else 1.0
        scores[c3] = raw * weight

    top_score = max(scores.values()) if scores else 0.0
    best = [c3 for c3, s in scores.items() if s == top_score]

    if top_score < MIN_TOP_SCORE or len(best) != 1:
        return "Uncertain", top_score, scores
    return best[0], top_score, scores


def generate_pseudo_labels(
    df: pd.DataFrame,
    vocab_by_c3: Dict[str, List[str]],
    idf_weight: Dict[str, float],
) -> pd.DataFrame:
    """对 FSPC 全量诗歌做伪标签生成，返回增补字段后的 DataFrame。"""
    c3_cols    = sorted(vocab_by_c3.keys())
    pseudos, top_scores = [], []
    score_rows: List[Dict[str, float]] = []

    print(f"开始匹配 {len(df)} 首诗...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 500 == 0:
            print(f"  已处理 {i}/{len(df)} ...")
        label, top, scores = pseudo_label_single(row["text"], vocab_by_c3, idf_weight)
        pseudos.append(label)
        top_scores.append(top)
        score_rows.append({f"score_{c3}": scores.get(c3, 0.0) for c3 in c3_cols})

    df = df.copy()
    df["pseudo_label"]    = pseudos
    df["pseudo_label_zh"] = [C3_ZH.get(p, p) for p in pseudos]
    df["top_score"]       = top_scores
    score_df = pd.DataFrame(score_rows, index=df.index)
    return pd.concat([df, score_df], axis=1)


# ===========================================================================
# 第二步：极性逻辑对撞
# ===========================================================================
def logic_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    将伪标签与 FSPC 粗粒度极性做逻辑对撞（按 LOGIC_RULES）。
    返回 (黄金集, 漏斗统计字典)
    """
    n_raw = len(df)

    # 去掉 Uncertain
    df1 = df[df["pseudo_label"] != "Uncertain"].copy()
    n1  = len(df1)

    # 极性对撞
    mask = [
        row["polarity"] in LOGIC_RULES.get(row["pseudo_label"], frozenset())
        for _, row in df1.iterrows()
    ]
    df_gold = df1[mask].copy().reset_index(drop=True)
    n_gold  = len(df_gold)

    # 被丢弃的冲突样本统计（用于分析）
    conflict = df1[~pd.Series(mask, index=df1.index)].copy()

    funnel = {
        "raw":              n_raw,
        "after_uncertain":  n1,
        "conflict_dropped": len(conflict),
        "golden":           n_gold,
        "retention_rate":   f"{n_gold/n_raw*100:.1f}%",
    }
    return df_gold, funnel, conflict


# ===========================================================================
# 报告输出
# ===========================================================================
def print_and_save_report(
    df_all: pd.DataFrame,
    df_gold: pd.DataFrame,
    funnel: Dict,
    conflict: pd.DataFrame,
    vocab_by_c3: Dict[str, List[str]],
    idf_weight: Dict[str, float],
) -> None:

    lines = []
    sep = "=" * 62

    lines.append(sep)
    lines.append("  Step 1 伪标签生成 + 逻辑对撞  漏斗报告")
    lines.append(sep)

    lines.append(f"\n【参数配置】")
    lines.append(f"  MIN_WORD_LEN      = {MIN_WORD_LEN}")
    lines.append(f"  MIN_TOP_SCORE     = {MIN_TOP_SCORE}")
    lines.append(f"  USE_IDF_WEIGHT    = {USE_IDF_WEIGHT}")
    lines.append(f"  SINGLE_CHAR_WHITE = {sorted(SINGLE_CHAR_WHITELIST)}")

    lines.append(f"\n【词典覆盖（过滤后）】")
    for c3 in sorted(vocab_by_c3):
        n = len(vocab_by_c3[c3])
        w = idf_weight[c3]
        zh = C3_ZH.get(c3, "")
        lines.append(f"  {c3:12s}({zh:6s})  {n:4d}词  IDF权重={w:.4f}")

    lines.append(f"\n【样本漏斗】")
    lines.append(f"  原始 FSPC 样本              : {funnel['raw']:>5}")
    lines.append(f"  去掉 Uncertain (无匹配/并列)  : {funnel['after_uncertain']:>5}  "
                 f"({funnel['raw']-funnel['after_uncertain']} 丢弃)")
    lines.append(f"  逻辑对撞冲突丢弃             : {funnel['after_uncertain']-funnel['golden']:>5}  "
                 f"({funnel['conflict_dropped']} 条)")
    lines.append(f"  最终黄金集                   : {funnel['golden']:>5}  "
                 f"({funnel['retention_rate']})")

    lines.append(f"\n【伪标签分布（全量匹配结果）】")
    dist_all = df_all["pseudo_label"].value_counts()
    for lbl, cnt in dist_all.items():
        zh = C3_ZH.get(lbl, "")
        lines.append(f"  {lbl:12s}({zh:6s}): {cnt:5d}")

    lines.append(f"\n【黄金集标签分布】")
    dist_gold = df_gold["pseudo_label"].value_counts()
    for lbl, cnt in dist_gold.items():
        zh = C3_ZH.get(lbl, "")
        pct = cnt / len(df_gold) * 100
        lines.append(f"  {lbl:12s}({zh:6s}): {cnt:5d}  ({pct:.1f}%)")

    lines.append(f"\n【黄金集极性分布（来自 FSPC 原始标注）】")
    for pol, cnt in df_gold["polarity"].value_counts().items():
        pct = cnt / len(df_gold) * 100
        lines.append(f"  {pol:22s}: {cnt:5d}  ({pct:.1f}%)")

    lines.append(f"\n【冲突样本 Top 案例（最大10条）】")
    if len(conflict) > 0:
        lines.append(f"  {'伪标签':12s}  {'FSPC极性':22s}  诗歌（前20字）")
        for _, row in conflict.head(10).iterrows():
            poem_short = row["poem_raw"].replace("|","")[:20]
            lines.append(f"  {row['pseudo_label']:12s}  {row['polarity']:22s}  {poem_short}")
    else:
        lines.append("  （无冲突样本）")

    lines.append(f"\n【输出文件】")
    lines.append(f"  黄金集 CSV : {OUT_CSV}")
    lines.append(f"  本报告     : {OUT_REPORT}")
    lines.append(sep)

    report_str = "\n".join(lines)
    print(report_str)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    print(f"加载 FCCPSL 词典: {FCCPSL_PATH}")
    vocab_by_c3, idf_weight = load_fccpsl(FCCPSL_PATH)
    print(f"  -> {len(vocab_by_c3)} 个 C3 类别（过滤后）")

    print(f"加载 FSPC 数据: {FSPC_PATH}")
    df_fspc = load_fspc(FSPC_PATH)
    print(f"  -> {len(df_fspc)} 首诗")

    # ── Step 1：伪标签生成 ─────────────────────────────────────────────
    df_all = generate_pseudo_labels(df_fspc, vocab_by_c3, idf_weight)

    # ── Step 2：逻辑对撞清洗 ───────────────────────────────────────────
    print("逻辑对撞清洗...")
    df_gold, funnel, conflict = logic_validation(df_all)

    # ── 输出报告 ───────────────────────────────────────────────────────
    print_and_save_report(df_all, df_gold, funnel, conflict, vocab_by_c3, idf_weight)

    # ── 保存黄金集 ─────────────────────────────────────────────────────
    # 选取输出列（保留原始信息 + 伪标签 + 各类得分）
    score_cols = [c for c in df_gold.columns if c.startswith("score_")]
    out_cols = (
        ["poet", "title", "dynasty", "poem_raw", "text", "polarity",
         "pseudo_label", "pseudo_label_zh", "top_score"]
        + score_cols
    )
    df_gold[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n黄金集已保存 -> {OUT_CSV}  ({len(df_gold)} 行)")


if __name__ == "__main__":
    main()
