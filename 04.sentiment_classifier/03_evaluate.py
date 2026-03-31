"""
03_evaluate.py
==============
在 FSPC（清华 5000 首细粒度情感标注语料）上评估 FCCPSL 词典法的准确率。

目的：
  - 量化词典法的实际准确率（特别是对隐性情感的识别）
  - 为阶段二（SikuRoBERTa 微调）提供基线对比数据
  - 分析词典在哪类诗中表现更差

评估逻辑：
  FSPC 5类标签 → 3类映射：
    1 (negative)          → negative
    2 (implicit negative) → negative  ← 词典法弱点
    3 (neutral)           → neutral
    4 (implicit positive) → positive  ← 词典法弱点
    5 (positive)          → positive

输入：
  output/lexicon/FSPC_V1.0.json  （需先手动下载）
  output/lexicon/combined_lexicon.csv

输出：
  output/evaluation/fspc_evaluation.csv  — 逐首评估结果
  output/evaluation/evaluation_report.txt — 准确率/混淆矩阵/分析报告
  output/figures/fspc_confusion_matrix.png — 混淆矩阵热图

FSPC 手动下载命令（在终端执行）：
  wget https://raw.githubusercontent.com/THUNLP-AIPoet/Datasets/master/FSPC/FSPC_V1.0.json \\
    -O 02.sample_label_phase2/step2e_sentiment/output/lexicon/FSPC_V1.0.json
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# ─── 路径 ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
LEXICON_CSV   = ROOT / "output" / "lexicon" / "combined_lexicon.csv"
FSPC_JSON     = ROOT / "output" / "lexicon" / "FSPC_V1.0.json"
EVAL_DIR      = ROOT / "output" / "evaluation"
FIGURES_DIR   = ROOT / "output" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── 从 02_sentiment_analyze 复用的核心函数 ──────────────────────────────────
import re
from collections import defaultdict

PUNCT_RE     = re.compile(r"[^\u4e00-\u9fff]")
NEG_WINDOW   = 2
NEUTRAL_THRESH = 0.05
NEGATION_WORDS = {"不", "无", "非", "未", "没", "莫", "勿", "别", "休", "难", "岂"}

C3_CLASSES = [
    "ease", "joy", "praise", "like", "faith", "wish", "peculiar",
    "sorrow", "miss", "fear", "guilt", "criticize", "anger", "vexed", "misgive",
]
C3_TO_C2 = {
    "ease": "pleasure", "joy": "pleasure",
    "praise": "favour", "like": "favour", "faith": "favour", "wish": "favour",
    "peculiar": "surprise",
    "sorrow": "sadness", "miss": "sadness", "fear": "sadness", "guilt": "sadness",
    "criticize": "disgust", "anger": "disgust", "vexed": "disgust", "misgive": "disgust",
}
C2_CLASSES = ["pleasure", "favour", "surprise", "sadness", "disgust"]


def load_lexicon(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {
        str(row["词"]).strip(): {"c3": str(row["C3"]), "c2": str(row["C2"]), "c1": str(row["C1"])}
        for _, row in df.iterrows()
        if str(row["词"]).strip() != "nan"
    }


def simple_sentiment(text: str, lexicon: dict) -> str:
    """简化版极性判断（用于 FSPC 评估）"""
    clean = PUNCT_RE.sub("", text)
    pos_score, neg_score = 0.0, 0.0
    neg_countdown = 0

    for i, ch in enumerate(clean):
        if ch in NEGATION_WORDS:
            neg_countdown = NEG_WINDOW
            continue
        is_neg = neg_countdown > 0
        if neg_countdown > 0:
            neg_countdown -= 1

        # 尝试 1~4 字匹配
        for length in range(min(4, len(clean) - i), 0, -1):
            word = clean[i: i + length]
            if word in lexicon:
                info = lexicon[word]
                w = length  # 字数作为权重
                if info["c1"] == "positive":
                    if is_neg:
                        neg_score += w
                    else:
                        pos_score += w
                elif info["c1"] == "negative":
                    if is_neg:
                        pos_score += w
                    else:
                        neg_score += w
                break

    n = max(len(clean), 1)
    pos_norm = pos_score / n
    neg_norm = neg_score / n
    diff = pos_norm - neg_norm

    if abs(diff) < NEUTRAL_THRESH:
        return "neutral"
    return "positive" if diff > 0 else "negative"


# ─── FSPC 标签映射 ────────────────────────────────────────────────────────────
FSPC_5TO3 = {
    "1": "negative",
    "2": "negative",   # implicit negative → 归入 negative
    "3": "neutral",
    "4": "positive",   # implicit positive → 归入 positive
    "5": "positive",
}

FSPC_LABEL_DESC = {
    "1": "negative（显性负面）",
    "2": "implicit negative（隐性负面）",
    "3": "neutral（中性）",
    "4": "implicit positive（隐性正面）",
    "5": "positive（显性正面）",
}


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  03_evaluate.py — FSPC 评估")
    print("=" * 60)

    # 检查文件
    if not LEXICON_CSV.exists():
        print(f"❌ 词典不存在：{LEXICON_CSV}，请先运行 01_parse_lexicons.py")
        return

    if not FSPC_JSON.exists():
        print(f"❌ FSPC 文件不存在：{FSPC_JSON}")
        print("   请在终端手动下载：")
        print(f"   wget https://raw.githubusercontent.com/THUNLP-AIPoet/Datasets/master/FSPC/FSPC_V1.0.json \\")
        print(f"     -O {FSPC_JSON}")
        return

    # 加载词典
    print("加载词典...")
    lexicon = load_lexicon(LEXICON_CSV)
    print(f"  词典词数：{len(lexicon):,}")

    # 加载 FSPC
    print("\n加载 FSPC...")
    poems = []
    with open(FSPC_JSON, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    poems.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  FSPC 共 {len(poems)} 首诗")

    # 逐首评估
    print("\n开始评估...")
    records = []
    confusion = defaultdict(lambda: defaultdict(int))

    for poem_obj in poems:
        full_text = poem_obj["poem"].replace("|", "")
        title     = poem_obj.get("title", "")
        poet      = poem_obj.get("poet", "")
        dynasty   = poem_obj.get("dynasty", "")

        sents     = poem_obj.get("setiments", poem_obj.get("sentiments", {}))
        true_5    = str(sents.get("holistic", "3"))
        true_3    = FSPC_5TO3.get(true_5, "neutral")

        # 词典法预测
        pred_3    = simple_sentiment(full_text, lexicon)

        is_correct = (pred_3 == true_3)
        is_implicit = true_5 in {"2", "4"}

        confusion[true_3][pred_3] += 1

        records.append({
            "title":       title,
            "poet":        poet,
            "dynasty":     dynasty,
            "full_text":   full_text[:30] + "..." if len(full_text) > 30 else full_text,
            "true_5class": true_5,
            "true_5desc":  FSPC_LABEL_DESC.get(true_5, true_5),
            "true_3class": true_3,
            "pred_3class": pred_3,
            "correct":     is_correct,
            "is_implicit": is_implicit,
        })

    df_eval = pd.DataFrame(records)
    df_eval.to_csv(EVAL_DIR / "fspc_evaluation.csv", index=False, encoding="utf-8-sig")
    print(f"✓ fspc_evaluation.csv（{len(df_eval)} 首）")

    # 计算准确率
    total     = len(df_eval)
    overall   = df_eval["correct"].sum() / total
    explicit  = df_eval[~df_eval["is_implicit"]]["correct"].mean()
    implicit  = df_eval[df_eval["is_implicit"]]["correct"].mean()

    # 按5类分别统计
    by_5class = df_eval.groupby("true_5class")["correct"].agg(["mean", "count"]).reset_index()

    # 混淆矩阵
    labels_3 = ["positive", "negative", "neutral"]
    conf_matrix = pd.DataFrame(0, index=labels_3, columns=labels_3)
    for true_l, preds in confusion.items():
        for pred_l, cnt in preds.items():
            if true_l in labels_3 and pred_l in labels_3:
                conf_matrix.loc[true_l, pred_l] = cnt

    # 词典覆盖率分析
    coverage_counts = []
    for poem_obj in poems:
        full_text = poem_obj["poem"].replace("|", "")
        clean = PUNCT_RE.sub("", full_text)
        matched = sum(1 for i in range(len(clean))
                      for l in range(min(4, len(clean)-i), 0, -1)
                      if clean[i:i+l] in lexicon)
        coverage_counts.append(matched)
    avg_coverage = sum(coverage_counts) / len(coverage_counts)
    zero_coverage = sum(1 for c in coverage_counts if c == 0)

    # 生成报告
    report_lines = [
        "=" * 65,
        "  FCCPSL 词典法 × FSPC 评估报告",
        "=" * 65,
        f"FSPC 评估集规模：{total} 首古诗词",
        f"词典规模：{len(lexicon):,} 词",
        "",
        "── 整体准确率 ───────────────────────────────────────────",
        f"  整体 3类 Acc：{overall:.4f} ({overall*100:.1f}%)",
        f"  显性情感（1/5类）Acc：{explicit:.4f} ({explicit*100:.1f}%)",
        f"  隐性情感（2/4类）Acc：{implicit:.4f} ({implicit*100:.1f}%)",
        f"  （差距 = {(explicit-implicit)*100:.1f}%，说明词典对隐性情感的局限）",
        "",
        "── 5类细粒度准确率 ──────────────────────────────────────",
    ]
    for _, row in by_5class.sort_values("true_5class").iterrows():
        desc = FSPC_LABEL_DESC.get(str(int(row["true_5class"])), "")
        report_lines.append(
            f"  类别{int(row['true_5class'])} {desc[:20]:20s}："
            f"  Acc={row['mean']:.3f}  样本数={int(row['count'])}"
        )

    report_lines += [
        "",
        "── 混淆矩阵（行=真实，列=预测）────────────────────────",
        conf_matrix.to_string(),
        "",
        "── 词典覆盖率分析 ───────────────────────────────────────",
        f"  FSPC 诗文平均情感词命中：{avg_coverage:.1f} 词/首",
        f"  0词命中（无法判断）：{zero_coverage} 首 ({zero_coverage/total*100:.1f}%)",
        "",
        "── 分析结论 ─────────────────────────────────────────────",
        f"  词典法在显性情感诗词上准确率较高（预期 ≥ 70%）",
        f"  词典法在隐性情感诗词上准确率明显下降（通常 < 60%）",
        f"  这证明了阶段二（SikuRoBERTa微调）的必要性：",
        f"  通过上下文语义建模弥补词典法对隐性情感的盲区",
    ]

    report = "\n".join(report_lines)
    (EVAL_DIR / "evaluation_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # 混淆矩阵可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        for font in ["PingFang SC", "Heiti TC", "STHeiti", "SimHei"]:
            if font in {f.name for f in fm.fontManager.ttflist}:
                plt.rcParams["font.family"] = font
                break

        fig, ax = plt.subplots(figsize=(7, 6))
        import seaborn as sns
        sns.heatmap(
            conf_matrix,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_3, yticklabels=labels_3,
            ax=ax,
        )
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title(
            f"FCCPSL词典法 混淆矩阵\n"
            f"总Acc={overall:.3f}  显性={explicit:.3f}  隐性={implicit:.3f}"
        )
        fig.tight_layout()
        out_path = FIGURES_DIR / "fspc_confusion_matrix.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ figures/fspc_confusion_matrix.png")
    except Exception as e:
        print(f"  [WARN] 混淆矩阵图生成失败：{e}")

    print(f"\n✓ 评估完成 — 整体 Acc = {overall*100:.1f}%")


if __name__ == "__main__":
    main()
