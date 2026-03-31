"""
02_sentiment_analyze.py
=======================
用 FCCPSL 合并词典对 1075 首项目诗词做情感分析。

方法（替代 SnowNLP + 手写EMOTION_LEXICON）：
  - 去除 SnowNLP（不适合古诗词）
  - 使用 FCCPSL 14,368 词（古诗词专属）+ NTUSD 补充
  - 输出 FCCPSL 三层得分（C3×15维 / C2×5维 / C1×2维极性）
  - 正文（古汉语）逐字匹配，赏析（现代汉语）jieba 分词后匹配
  - 否定词窗口处理（窗口=2字/词）

输出（output/results/）：
  sentiment_per_poem.csv  — 1075首诗的情感向量（含三层得分 + 主导情感）
  flower_sentiment.csv    — 79 种植物的情感画像（C2层均值）
  summary_report.txt      — 汇总统计报告

可追溯性：
  - 词典来源记录在 output/lexicon/lexicon_stats.txt
  - 分析参数（权重比例、否定窗口）记录在 summary_report.txt
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ─── 路径 ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
LEXICON_CSV  = ROOT / "output" / "lexicon" / "combined_lexicon.csv"
POEMS_CSV    = Path(__file__).parent.parent.parent / "00.poems_dataset" / "poems_dataset_merged_done.csv"
RESULTS_DIR  = ROOT / "output" / "results"
FIGURES_DIR  = ROOT / "output" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── 参数（记录在 summary_report.txt 保证可追溯）────────────────────────────
POEM_WEIGHT     = 0.6   # 正文（古汉语）权重
ANALYSIS_WEIGHT = 0.4   # 赏析（现代汉语）权重
NEG_WINDOW      = 2     # 否定词影响后续N个 token
NEUTRAL_THRESH  = 0.05  # |pos_score - neg_score| < 阈值 → neutral

# 否定词（古汉语+现代汉语）
NEGATION_WORDS = {
    "不", "无", "非", "未", "没", "莫", "勿", "别", "休", "难",
    "岂", "未曾", "不曾", "并非", "并不",
}

# FCCPSL C3 层 15 类（用于初始化分数向量）
C3_CLASSES = [
    "ease", "joy", "praise", "like", "faith", "wish", "peculiar",
    "sorrow", "miss", "fear", "guilt", "criticize", "anger", "vexed", "misgive",
]
C3_LABEL_ZH = {
    "ease": "平和", "joy": "喜悦", "praise": "赞美", "like": "喜爱",
    "faith": "信念", "wish": "期盼", "peculiar": "惊奇",
    "sorrow": "悲伤", "miss": "思念", "fear": "忧惧",
    "guilt": "愧疚", "criticize": "批判", "anger": "愤怒",
    "vexed": "苦闷", "misgive": "疑虑",
}

C3_TO_C2 = {
    "ease": "pleasure", "joy": "pleasure",
    "praise": "favour", "like": "favour", "faith": "favour", "wish": "favour",
    "peculiar": "surprise",
    "sorrow": "sadness", "miss": "sadness", "fear": "sadness", "guilt": "sadness",
    "criticize": "disgust", "anger": "disgust", "vexed": "disgust", "misgive": "disgust",
}
C2_TO_C1 = {
    "pleasure": "positive", "favour": "positive", "surprise": "positive",
    "sadness": "negative", "disgust": "negative",
}
C2_CLASSES = ["pleasure", "favour", "surprise", "sadness", "disgust"]

# 标点正则（诗文清洗）
PUNCT_RE = re.compile(r"[^\u4e00-\u9fff]")


# ─── 加载词典 ──────────────────────────────────────────────────────────────────
def load_lexicon(csv_path: Path) -> dict[str, dict]:
    """
    加载合并词典，返回 {词: {c3, c2, c1}} 字典。
    先尝试最长词匹配（多字词优先）。
    """
    df = pd.read_csv(csv_path)
    lexicon = {}
    for _, row in df.iterrows():
        w = str(row["词"]).strip()
        if w and w != "nan":
            lexicon[w] = {
                "c3": str(row["C3"]),
                "c2": str(row["C2"]),
                "c1": str(row["C1"]),
            }
    return lexicon


# ─── 文本匹配 ──────────────────────────────────────────────────────────────────
def extract_sentiment_tokens(text: str, lexicon: dict,
                              multi_words: set, is_modern: bool = False) -> list[tuple]:
    """
    从文本中提取情感词，返回 [(词, c3, c2, c1, is_negated), ...]
    
    古汉语（正文）：逐字匹配，优先多字词（最长优先）
    现代汉语（赏析）：使用 jieba 分词后匹配
    """
    if is_modern:
        try:
            import jieba
            tokens = list(jieba.cut(text))
        except ImportError:
            tokens = list(text)
    else:
        tokens = list(PUNCT_RE.sub("", text))

    results = []
    neg_countdown = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # 检查否定词
        if tok in NEGATION_WORDS:
            neg_countdown = NEG_WINDOW
            i += 1
            continue

        is_negated = (neg_countdown > 0)
        if neg_countdown > 0:
            neg_countdown -= 1

        # 尝试最长匹配（从4字到1字）
        matched = False
        for length in range(min(4, len(tokens) - i), 0, -1):
            chunk = "".join(tokens[i: i + length])
            if chunk in lexicon:
                info = lexicon[chunk]
                results.append((chunk, info["c3"], info["c2"], info["c1"], is_negated))
                i += length
                matched = True
                break

        if not matched:
            i += 1

    return results


# ─── 计算情感向量 ──────────────────────────────────────────────────────────────
def compute_sentiment_vector(tokens_with_info: list[tuple],
                              text_len: int) -> dict:
    """
    从情感词列表计算三层情感向量。
    
    否定词处理：is_negated=True 时，将该词贡献的 score 取反
    （否定一个 sorrow 词 → 减少 sorrow 分；否定 joy 词 → 减少 joy 分）
    归一化：除以文本字数（消除长度偏差）
    """
    c3_raw  = defaultdict(float)
    c3_neg  = defaultdict(float)

    n = max(text_len, 1)

    for (word, c3, c2, c1, is_negated) in tokens_with_info:
        weight = len(word)  # 多字词权重更高（字数作为权重）
        if is_negated:
            c3_neg[c3] += weight
        else:
            c3_raw[c3] += weight

    # 最终 C3 分数（取反后可能为负，表示否定该情感）
    c3_scores = {}
    for c3 in C3_CLASSES:
        raw  = c3_raw.get(c3, 0.0)
        neg  = c3_neg.get(c3, 0.0)
        c3_scores[c3] = round((raw - neg) / n, 6)

    # C2 分数（C3 加和）
    c2_scores = {}
    for c2 in C2_CLASSES:
        children = [c3 for c3, parent in C3_TO_C2.items() if parent == c2]
        c2_scores[c2] = round(sum(c3_scores.get(c3, 0) for c3 in children), 6)

    # C1 极性（正向 vs 负向）
    pos_score = sum(c2_scores.get(c2, 0) for c2 in ["pleasure", "favour", "surprise"])
    neg_score = sum(c2_scores.get(c2, 0) for c2 in ["sadness", "disgust"])
    diff = pos_score - neg_score

    if abs(diff) < NEUTRAL_THRESH:
        polarity = "neutral"
        confidence = 0.0
    elif diff > 0:
        polarity = "positive"
        confidence = round(diff / max(pos_score + neg_score, 1e-8), 4)
    else:
        polarity = "negative"
        confidence = round(-diff / max(pos_score + neg_score, 1e-8), 4)

    # 主导 C3 / C2
    dominant_c3 = max(C3_CLASSES, key=lambda c: c3_scores.get(c, 0))
    dominant_c2 = max(C2_CLASSES, key=lambda c: c2_scores.get(c, 0))
    dominant_c3_score = c3_scores.get(dominant_c3, 0)

    return {
        **{f"c3_{c3}": c3_scores.get(c3, 0) for c3 in C3_CLASSES},
        **{f"c2_{c2}": c2_scores.get(c2, 0) for c2 in C2_CLASSES},
        "pos_score":       round(pos_score, 6),
        "neg_score":       round(neg_score, 6),
        "polarity":        polarity,
        "polarity_conf":   confidence,
        "dominant_c3":     dominant_c3,
        "dominant_c3_zh":  C3_LABEL_ZH.get(dominant_c3, dominant_c3),
        "dominant_c3_score": round(dominant_c3_score, 6),
        "dominant_c2":     dominant_c2,
        "matched_words":   len(tokens_with_info),
    }


# ─── 对单首诗分析 ─────────────────────────────────────────────────────────────
def analyze_poem(poem_text: str, analysis_text: str,
                 lexicon: dict, multi_words: set) -> dict:
    """
    综合分析一首诗（正文 + 赏析），返回加权情感向量。
    
    策略：
      poem_text（正文，古汉语）  → 逐字匹配，权重 0.6
      analysis_text（赏析，现代）→ 分词匹配，权重 0.4
      合并版 = 加权平均
    """
    # 清洗文本
    poem_clean     = PUNCT_RE.sub("", str(poem_text or ""))
    analysis_clean = str(analysis_text or "")

    # 提取情感词
    poem_toks     = extract_sentiment_tokens(poem_clean, lexicon, multi_words, is_modern=False)
    analysis_toks = extract_sentiment_tokens(analysis_clean, lexicon, multi_words, is_modern=True)

    # 计算各自向量
    poem_vec     = compute_sentiment_vector(poem_toks,     len(poem_clean))
    analysis_vec = compute_sentiment_vector(analysis_toks, len(analysis_clean))

    # 加权合并（C3/C2/C1 分数）
    combined = {}
    for key in poem_vec:
        if isinstance(poem_vec[key], (int, float)):
            combined[key] = round(
                POEM_WEIGHT * poem_vec[key] + ANALYSIS_WEIGHT * analysis_vec[key], 6
            )
        else:
            # 非数值字段（dominant_c3 等）取正文版本
            combined[key] = poem_vec[key]

    # 重新判断极性（基于合并分数）
    pos = combined.get("pos_score", 0)
    neg = combined.get("neg_score", 0)
    diff = pos - neg
    if abs(diff) < NEUTRAL_THRESH:
        combined["polarity"] = "neutral"
        combined["polarity_conf"] = 0.0
    elif diff > 0:
        combined["polarity"] = "positive"
        combined["polarity_conf"] = round(diff / max(pos + neg, 1e-8), 4)
    else:
        combined["polarity"] = "negative"
        combined["polarity_conf"] = round(-diff / max(pos + neg, 1e-8), 4)

    # 附加元信息
    combined["poem_matched_words"]     = poem_vec["matched_words"]
    combined["analysis_matched_words"] = analysis_vec["matched_words"]

    return combined


# ─── 可视化 ───────────────────────────────────────────────────────────────────
def plot_sentiment_distribution(df: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        print("  [WARN] matplotlib 未安装，跳过可视化")
        return

    # 字体
    for font in ["PingFang SC", "Heiti TC", "STHeiti", "SimHei"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("FCCPSL 情感分析结果（1075首诗词）", fontsize=14, y=1.02)

    # 1. C1 极性分布（饼图）
    ax = axes[0]
    polarity_counts = df["polarity"].value_counts()
    colors = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#9E9E9E"}
    ax.pie(
        polarity_counts.values,
        labels=[f"{k}\n({v}首)" for k, v in polarity_counts.items()],
        colors=[colors.get(k, "#999") for k in polarity_counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("C1 极性分布")

    # 2. C2 情感族分布（柱状图）
    ax = axes[1]
    c2_cols  = [f"c2_{c2}" for c2 in C2_CLASSES]
    c2_means = [df[c].mean() for c in c2_cols]
    c2_colors = ["#FFB300", "#66BB6A", "#AB47BC", "#42A5F5", "#EF5350"]
    bars = ax.bar(C2_CLASSES, c2_means, color=c2_colors, alpha=0.85)
    ax.set_xlabel("C2 情感族")
    ax.set_ylabel("平均得分")
    ax.set_title("C2 情感族平均强度")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. C3 层 15 类分布（横向柱状图）
    ax = axes[2]
    c3_zh    = [C3_LABEL_ZH.get(c, c) for c in C3_CLASSES]
    c3_means = [df[f"c3_{c}"].mean() for c in C3_CLASSES]
    bar_colors = ["#4CAF50"] * 7 + ["#F44336"] * 8  # 正向绿，负向红
    bars = ax.barh(c3_zh, c3_means, color=bar_colors, alpha=0.8)
    ax.set_xlabel("平均得分")
    ax.set_title("C3 情感类（15维）平均强度")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    out_path = FIGURES_DIR / "sentiment_distribution.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figures/sentiment_distribution.png")


def plot_flower_heatmap(df_flower: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import seaborn as sns
    except ImportError:
        print("  [WARN] matplotlib/seaborn 未安装，跳过热图")
        return

    for font in ["PingFang SC", "Heiti TC", "STHeiti", "SimHei"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break

    c2_cols = [f"c2_{c2}" for c2 in C2_CLASSES]
    # 取前 40 种植物（按总词频排序）
    top_flowers = df_flower.sort_values("poem_count", ascending=False).head(40)
    heat_data   = top_flowers[c2_cols].values
    row_labels  = top_flowers["flower"].tolist()

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(C2_CLASSES)))
    ax.set_xticklabels(C2_CLASSES, rotation=20)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("植物 × C2情感族 共现热图（Top-40植物）")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = FIGURES_DIR / "flower_sentiment_heatmap.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figures/flower_sentiment_heatmap.png")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  02_sentiment_analyze.py — FCCPSL 情感分析")
    print("=" * 60)

    # 检查必要文件
    if not LEXICON_CSV.exists():
        print(f"❌ 词典文件不存在：{LEXICON_CSV}")
        print("   请先运行 01_parse_lexicons.py")
        return

    if not POEMS_CSV.exists():
        print(f"❌ 诗词数据集不存在：{POEMS_CSV}")
        return

    # 加载词典
    print("\n加载词典...")
    lexicon = load_lexicon(LEXICON_CSV)
    multi_words = {w for w in lexicon if len(w) >= 2}
    print(f"  词典总词数：{len(lexicon):,}（多字词 {len(multi_words):,}）")

    # 加载诗词数据集
    print(f"\n加载诗词数据集：{POEMS_CSV}")
    df = pd.read_csv(POEMS_CSV)
    print(f"  共 {len(df)} 首")

    # 确定字段名（兼容不同版本）
    text_col     = next((c for c in ["text", "正文", "poem", "content"] if c in df.columns), None)
    analysis_col = next((c for c in ["analysis", "赏析", "comment", "赏"] if c in df.columns), None)
    flower_col   = next((c for c in ["flower", "花名", "plant"] if c in df.columns), None)

    if text_col is None:
        print(f"❌ 找不到正文列，可用列：{list(df.columns)}")
        return

    print(f"  正文列：{text_col}，赏析列：{analysis_col}，植物列：{flower_col}")

    # 逐首分析
    print("\n开始情感分析...")
    records = []
    for i, row in df.iterrows():
        poem_text     = str(row.get(text_col, ""))
        analysis_text = str(row.get(analysis_col, "")) if analysis_col else ""
        flower        = str(row.get(flower_col, "")) if flower_col else ""
        poem_id       = row.get("id", i)
        title         = row.get("title", row.get("诗名", ""))
        dynasty       = row.get("dynasty", row.get("朝代", ""))

        vec = analyze_poem(poem_text, analysis_text, lexicon, multi_words)

        record = {
            "id":       poem_id,
            "title":    title,
            "dynasty":  dynasty,
            "flower":   flower,
            **vec,
        }
        records.append(record)

        if (i + 1) % 100 == 0:
            print(f"  进度：{i+1}/{len(df)}")

    result_df = pd.DataFrame(records)
    result_df.to_csv(RESULTS_DIR / "sentiment_per_poem.csv", index=False, encoding="utf-8-sig")
    print(f"\n✓ sentiment_per_poem.csv（{len(result_df)} 首）")

    # 植物情感画像
    if flower_col:
        flower_groups = result_df.groupby("flower")
        flower_records = []
        c2_cols = [f"c2_{c2}" for c2 in C2_CLASSES]
        c3_cols = [f"c3_{c3}" for c3 in C3_CLASSES]
        for fl, grp in flower_groups:
            if not fl or fl == "nan":
                continue
            rec = {"flower": fl, "poem_count": len(grp)}
            for col in c2_cols + c3_cols:
                rec[col] = round(grp[col].mean(), 6)
            dominant = grp["dominant_c3"].value_counts().index[0] if len(grp) > 0 else ""
            rec["dominant_c3"]    = dominant
            rec["dominant_c3_zh"] = C3_LABEL_ZH.get(dominant, dominant)
            rec["pos_ratio"]      = round((grp["polarity"] == "positive").sum() / len(grp), 4)
            rec["neg_ratio"]      = round((grp["polarity"] == "negative").sum() / len(grp), 4)
            rec["neutral_ratio"]  = round((grp["polarity"] == "neutral").sum() / len(grp), 4)
            flower_records.append(rec)

        df_flower = pd.DataFrame(flower_records).sort_values("poem_count", ascending=False)
        df_flower.to_csv(RESULTS_DIR / "flower_sentiment.csv", index=False, encoding="utf-8-sig")
        print(f"✓ flower_sentiment.csv（{len(df_flower)} 种植物）")

    # 汇总报告
    polarity_dist = result_df["polarity"].value_counts()
    dominant_dist = result_df["dominant_c3"].value_counts()
    avg_matched   = result_df["poem_matched_words"].mean()
    zero_match    = (result_df["poem_matched_words"] == 0).sum()

    report_lines = [
        "=" * 60,
        "  FCCPSL 情感分析汇总报告",
        "=" * 60,
        f"分析参数：",
        f"  正文权重：{POEM_WEIGHT}，赏析权重：{ANALYSIS_WEIGHT}",
        f"  否定词窗口：{NEG_WINDOW}",
        f"  中性阈值：|pos-neg| < {NEUTRAL_THRESH}",
        f"",
        f"分析结果：",
        f"  总诗词数：{len(result_df)}",
        f"  正文平均情感词命中：{avg_matched:.1f} 词/首",
        f"  正文0词命中（无法判断）：{zero_match} 首 ({zero_match/len(result_df)*100:.1f}%)",
        f"",
        f"C1 极性分布：",
    ]
    for pol, cnt in polarity_dist.items():
        report_lines.append(f"  {pol:10s}: {cnt} 首 ({cnt/len(result_df)*100:.1f}%)")

    report_lines += ["", "C3 主导情感 Top-10："]
    for c3, cnt in dominant_dist.head(10).items():
        zh = C3_LABEL_ZH.get(c3, c3)
        report_lines.append(f"  {c3:12s}（{zh}）: {cnt} 首")

    report = "\n".join(report_lines)
    (RESULTS_DIR / "summary_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # 可视化
    print("\n生成可视化...")
    plot_sentiment_distribution(result_df)
    if flower_col and "df_flower" in dir():
        plot_flower_heatmap(df_flower)

    print("\n✓ 情感分析完成")


if __name__ == "__main__":
    main()
