"""
step2e_sentiment / 06_compare_siku_ccpoem.py
=============================================
将 SikuRoBERTa 与 BERT-CCPoem 在 1075 首项目诗词上的推断结果并排对比，
并输出包含“诗歌正文”的人工审查表。

输入
----
  output/results/sentiment_per_poem.csv
  00.poems_dataset/poems_dataset_merged_done.csv

输出
----
  output/results/siku_vs_ccpoem_sentiment_compare.csv
  output/results/siku_vs_ccpoem_summary.txt
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent.parent
RESULT_CSV = BASE_DIR / "output" / "results" / "sentiment_per_poem.csv"
POEMS_CSV = ROOT_DIR / "00.poems_dataset" / "poems_dataset_merged_done.csv"
COMPARE_CSV = BASE_DIR / "output" / "results" / "siku_vs_ccpoem_sentiment_compare.csv"
SUMMARY_TXT = BASE_DIR / "output" / "results" / "siku_vs_ccpoem_summary.txt"


def safe_col(df: pd.DataFrame, col: str, default=None):
    if col in df.columns:
        return df[col]
    return default


def main():
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"缺少结果文件：{RESULT_CSV}")
    if not POEMS_CSV.exists():
        raise FileNotFoundError(f"缺少原始诗词：{POEMS_CSV}")

    df_res = pd.read_csv(RESULT_CSV)
    df_poem = pd.read_csv(POEMS_CSV)

    required_cols = [
        "siku_label_zh",
        "siku_conf",
        "siku_polarity",
        "ccpoem_label_zh",
        "ccpoem_conf",
        "ccpoem_polarity",
    ]
    missing = [c for c in required_cols if c not in df_res.columns]
    if missing:
        raise ValueError(
            "sentiment_per_poem.csv 缺少以下列，请先运行对应推断脚本：\n"
            + ", ".join(missing)
        )

    # 按行序构造并排表（当前项目各阶段都按同一数据顺序处理）
    n = min(len(df_res), len(df_poem))
    df = pd.DataFrame({
        "id": safe_col(df_res, "id", pd.Series(range(n))).iloc[:n].values,
        "诗名": safe_col(df_poem, "诗名", pd.Series([""] * n)).iloc[:n].values,
        "花名": safe_col(df_poem, "花名", pd.Series([""] * n)).iloc[:n].values,
        "朝代": safe_col(df_poem, "朝代", pd.Series([""] * n)).iloc[:n].values,
        "作者": safe_col(df_poem, "作者", pd.Series([""] * n)).iloc[:n].values,
        "正文": safe_col(df_poem, "正文", pd.Series([""] * n)).iloc[:n].values,
        "赏析": safe_col(df_poem, "赏析", pd.Series([""] * n)).iloc[:n].values,
        "siku_label_zh": df_res["siku_label_zh"].iloc[:n].values,
        "siku_conf": df_res["siku_conf"].iloc[:n].values,
        "siku_polarity": df_res["siku_polarity"].iloc[:n].values,
        "ccpoem_label_zh": df_res["ccpoem_label_zh"].iloc[:n].values,
        "ccpoem_conf": df_res["ccpoem_conf"].iloc[:n].values,
        "ccpoem_polarity": df_res["ccpoem_polarity"].iloc[:n].values,
    })

    # 一致性字段
    df["label是否一致"] = (df["siku_label_zh"] == df["ccpoem_label_zh"]).astype(int)
    df["polarity是否一致"] = (df["siku_polarity"] == df["ccpoem_polarity"]).astype(int)
    df["conf差值(ccpoem-siku)"] = (df["ccpoem_conf"] - df["siku_conf"]).round(4)

    # 可读性字段：保留正文片段，便于快速浏览
    df["正文_preview"] = df["正文"].astype(str).str.replace("\n", " ", regex=False).str.slice(0, 80)

    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARE_CSV, index=False, encoding="utf-8-sig")

    label_agree = float(df["label是否一致"].mean())
    pol_agree = float(df["polarity是否一致"].mean())
    avg_delta = float(df["conf差值(ccpoem-siku)"].mean())

    summary = "\n".join([
        "=" * 70,
        "SikuRoBERTa vs BERT-CCPoem 情感推断并排对比汇总",
        "=" * 70,
        f"样本数: {len(df)}",
        f"5类标签一致率: {label_agree:.4f}",
        f"3类极性一致率: {pol_agree:.4f}",
        f"平均置信度差值(ccpoem-siku): {avg_delta:.4f}",
        "",
        f"并排对比文件: {COMPARE_CSV}",
        "说明: 对比表已包含“正文”与“赏析”列，便于人工直观判断。",
    ])
    SUMMARY_TXT.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
