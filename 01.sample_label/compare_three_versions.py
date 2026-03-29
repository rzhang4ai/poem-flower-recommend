"""
三版本分析对比：正文 / 赏析 / 正文+赏析
========================================
读取 Step2–Step4 已有输出，汇总三种输入下的差异并生成可读报告。

用法（在项目根目录）：
    source flower_env/bin/activate
    python 01.sample_label/compare_three_versions.py

输出：
    01.sample_label/output/three_versions_comparison_report.txt
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """加载情感与 TF-IDF 三版本数据"""
    sent_path = os.path.join(OUTPUT_DIR, "sentiment_scores.csv")
    if not os.path.exists(sent_path):
        print(f"❌ 未找到 {sent_path}，请先运行 Step 4")
        return None, None, None, None

    df_sent = pd.read_csv(sent_path, encoding="utf-8-sig")

    tfidf = {}
    for name, f in [
        ("poem", "tfidf_poem.csv"),
        ("analysis", "tfidf_analysis.csv"),
        ("combined", "tfidf_combined.csv"),
    ]:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            tfidf[name] = pd.read_csv(p, encoding="utf-8-sig")
        else:
            tfidf[name] = None

    return df_sent, tfidf.get("poem"), tfidf.get("analysis"), tfidf.get("combined")


def compare_sentiment(df: pd.DataFrame) -> list:
    """情感三版本对比"""
    lines = []
    lines.append("=" * 60)
    lines.append("一、情感分析：正文 vs 赏析 vs 正文+赏析")
    lines.append("=" * 60)

    for col, label in [
        ("snow_polarity_poem", "只看正文"),
        ("snow_polarity_analysis", "只看赏析"),
        ("snow_polarity_combined", "正文+赏析"),
    ]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        lines.append(f"\n【SnowNLP 极性】{label}")
        lines.append(f"  均值: {s.mean():.4f}  标准差: {s.std():.4f}  最小: {s.min():.4f}  最大: {s.max():.4f}")
        pos = (s > 0.6).sum()
        neg = (s < 0.4).sum()
        neu = ((s >= 0.4) & (s <= 0.6)).sum()
        lines.append(f"  正面(>0.6): {pos}  负面(<0.4): {neg}  中性: {neu}")

    emotion_cols = [
        ("dominant_emotion_poem", "只看正文"),
        ("dominant_emotion_analysis", "只看赏析"),
        ("dominant_emotion_combined", "正文+赏析"),
    ]
    for col, label in emotion_cols:
        if col not in df.columns:
            continue
        lines.append(f"\n【主导情感】{label}")
        vc = df[col].value_counts()
        for emo, cnt in vc.items():
            lines.append(f"  {emo:12s}: {cnt:3d} ({cnt/len(df):.1%})")

    lines.append("\n【三版本主导情感一致率】")
    if all(c in df.columns for c in ["dominant_emotion_poem", "dominant_emotion_analysis", "dominant_emotion_combined"]):
        agree_poem_ana = (df["dominant_emotion_poem"] == df["dominant_emotion_analysis"]).sum()
        agree_ana_comb = (df["dominant_emotion_analysis"] == df["dominant_emotion_combined"]).sum()
        agree_poem_comb = (df["dominant_emotion_poem"] == df["dominant_emotion_combined"]).sum()
        three_same = (
            (df["dominant_emotion_poem"] == df["dominant_emotion_analysis"])
            & (df["dominant_emotion_analysis"] == df["dominant_emotion_combined"])
        ).sum()
        n = len(df)
        lines.append(f"  正文 vs 赏析:     {agree_poem_ana}/{n} = {agree_poem_ana/n:.1%}")
        lines.append(f"  赏析 vs 合并:     {agree_ana_comb}/{n} = {agree_ana_comb/n:.1%}")
        lines.append(f"  正文 vs 合并:     {agree_poem_comb}/{n} = {agree_poem_comb/n:.1%}")
        lines.append(f"  三版本完全一致:   {three_same}/{n} = {three_same/n:.1%}")

    return lines


def compare_tfidf(df_poem, df_ana, df_comb) -> list:
    """TF-IDF 关键词三版本对比"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("二、TF-IDF 关键词：正文 vs 赏析 vs 正文+赏析")
    lines.append("=" * 60)

    if df_poem is None or df_ana is None or df_comb is None:
        lines.append("  (部分 tfidf_*.csv 缺失，跳过)")
        return lines

    col_preview = "top5_preview"
    merge_df = df_poem[["ID", "诗名", "花名"]].copy()
    merge_df = merge_df.merge(
        df_poem[["ID", col_preview]].rename(columns={col_preview: "top5_正文"}),
        on="ID", how="left"
    )
    merge_df = merge_df.merge(
        df_ana[["ID", col_preview]].rename(columns={col_preview: "top5_赏析"}),
        on="ID", how="left"
    )
    merge_df = merge_df.merge(
        df_comb[["ID", col_preview]].rename(columns={col_preview: "top5_合并"}),
        on="ID", how="left"
    )

    lines.append("\n【前 5 条诗的三版本 Top5 关键词对比】")
    for _, row in merge_df.head(5).iterrows():
        lines.append(f"\n  [{row['花名']}] {row['诗名'][:20]}")
        lines.append(f"    正文:   {row.get('top5_正文', '')}")
        lines.append(f"    赏析:   {row.get('top5_赏析', '')}")
        lines.append(f"    合并:   {row.get('top5_合并', '')}")

    def token_set(s):
        if pd.isna(s) or not s:
            return set()
        return set(str(s).replace("、", " ").split())

    same_poem_ana = sum(1 for _, row in merge_df.iterrows() if token_set(row.get("top5_正文")) & token_set(row.get("top5_赏析")))
    same_ana_comb = sum(1 for _, row in merge_df.iterrows() if token_set(row.get("top5_赏析")) & token_set(row.get("top5_合并")))
    n = len(merge_df)
    lines.append("\n【Top5 关键词重叠率】")
    lines.append(f"  正文与赏析至少 1 词重叠: {same_poem_ana}/{n} = {same_poem_ana/n:.1%}")
    lines.append(f"  赏析与合并至少 1 词重叠: {same_ana_comb}/{n} = {same_ana_comb/n:.1%}")

    return lines


def compare_lda() -> list:
    """LDA 仅对 赏析 / 合并 有输出"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("三、LDA 主题（赏析 vs 正文+赏析）")
    lines.append("=" * 60)
    lda_ana = os.path.join(OUTPUT_DIR, "lda_topics_analysis.csv")
    lda_comb = os.path.join(OUTPUT_DIR, "lda_topics_combined.csv")
    if os.path.exists(lda_ana) and os.path.exists(lda_comb):
        da = pd.read_csv(lda_ana, encoding="utf-8-sig")
        dc = pd.read_csv(lda_comb, encoding="utf-8-sig")
        if "dominant_topic" in da.columns and "dominant_topic" in dc.columns:
            agree = (da["dominant_topic"] == dc["dominant_topic"]).sum()
            n = len(da)
            lines.append("\n  流水线只对「赏析」和「正文+赏析」做了 LDA（正文过短未做）。")
            lines.append(f"  赏析 vs 合并 主导主题一致: {agree}/{n} = {agree/n:.1%}")
    else:
        lines.append("\n  (未找到 LDA 输出文件)")
    return lines


def main():
    print("三版本对比：正文 / 赏析 / 正文+赏析")
    print("读取 sentiment_scores.csv 与 tfidf_*.csv ...")
    df_sent, df_poem, df_ana, df_comb = load_data()
    if df_sent is None:
        return

    report_lines = [
        "诗花雅送 · 传统AI 三版本分析对比报告",
        "（只看正文 / 只看赏析 / 正文+赏析）",
        "",
    ]
    report_lines.extend(compare_sentiment(df_sent))
    report_lines.extend(compare_tfidf(df_poem, df_ana, df_comb))
    report_lines.extend(compare_lda())

    report_lines.append("\n" + "=" * 60)
    report_lines.append("如何自己在现有结果里看三版本差别")
    report_lines.append("=" * 60)
    report_lines.append("""
1) 情感（每首诗三条结果）
   - 文件: 01.sample_label/output/sentiment_scores.csv
   - 列: snow_polarity_poem / snow_polarity_analysis / snow_polarity_combined
         dominant_emotion_poem / dominant_emotion_analysis / dominant_emotion_combined
   - 用 Excel 按 ID 对齐即可逐条对比。

2) TF-IDF 关键词
   - 文件: tfidf_poem.csv, tfidf_analysis.csv, tfidf_combined.csv
   - 列: top5_preview（以及 top20_words）
   - 用 ID 把三张表 merge 即可并排看同一首诗在三种输入下的关键词。

3) LDA 主题
   - 文件: lda_topics_analysis.csv, lda_topics_combined.csv
   - 正文版未做 LDA，只有「赏析」和「正文+赏析」可对比。
""")

    report_text = "\n".join(report_lines)
    txt_path = os.path.join(OUTPUT_DIR, "three_versions_comparison_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n💾 已写入: {txt_path}")
    print("\n--- 报告摘要 ---")
    print(report_text[: 3200])
    if len(report_text) > 3200:
        print("... (详见上述文件)")


if __name__ == "__main__":
    main()
