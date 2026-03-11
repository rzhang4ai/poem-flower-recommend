"""
聚类簇含义说明：k-Means / 层次聚类的数字对应什么？
====================================================
聚类编号（0, 1, 2, ...）本身没有语义，本脚本通过「每个簇里有哪些诗、主导情感/场合/LDA主题分布」
来归纳每个簇的「含义」，便于解读。

用法（项目根目录）：
    source flower_env/bin/activate
    python 01.sample_label/interpret_clusters.py

输出：
    01.sample_label/output/cluster_interpretation.txt
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tables():
    """加载聚类、LDA、情感、规则标注等表"""
    base = OUTPUT_DIR
    out = {}
    # 聚类
    for name, f in [
        ("kmeans", "kmeans_labels_analysis.csv"),
        ("hier", "hierarchical_labels_analysis.csv"),
    ]:
        p = os.path.join(base, f)
        if os.path.exists(p):
            out[name] = pd.read_csv(p, encoding="utf-8-sig")
        else:
            out[name] = None

    # LDA 主题关键词（用于解释「主题编号」对应什么）
    lda_kw_path = os.path.join(base, "lda_topic_keywords_analysis.csv")
    if os.path.exists(lda_kw_path):
        out["lda_keywords"] = pd.read_csv(lda_kw_path, encoding="utf-8-sig")
    else:
        out["lda_keywords"] = None

    # 情感、规则（用于按簇汇总「主要是什么情感/场合」）
    for name, f in [
        ("sentiment", "sentiment_scores.csv"),
        ("rules", "rule_labels.csv"),
    ]:
        p = os.path.join(base, f)
        if os.path.exists(p):
            out[name] = pd.read_csv(p, encoding="utf-8-sig")
        else:
            out[name] = None

    # LDA 每首诗的主题（dominant_topic）
    lda_path = os.path.join(base, "lda_topics_analysis.csv")
    if os.path.exists(lda_path):
        out["lda_topics"] = pd.read_csv(lda_path, encoding="utf-8-sig")[["ID", "dominant_topic"]]
    else:
        out["lda_topics"] = None

    return out


def summarize_cluster(df_cluster: pd.DataFrame, id_col: str, label: str) -> list:
    """对一个簇的 DataFrame 做统计摘要（花名、场合、情感、LDA 主题）"""
    lines = []
    n = len(df_cluster)
    if n == 0:
        return lines

    if "花名" in df_cluster.columns:
        vc = df_cluster["花名"].value_counts().head(5)
        lines.append(f"    花名: {', '.join(f'{k}({v})' for k, v in vc.items())}")
    if "occasion_cn" in df_cluster.columns:
        occ = df_cluster["occasion_cn"].replace("", pd.NA).dropna()
        if len(occ) > 0:
            vc = occ.value_counts().head(3)
            lines.append(f"    场合: {', '.join(f'{k}({v})' for k, v in vc.items())}")
    if "relation_cn" in df_cluster.columns:
        rel = df_cluster["relation_cn"].replace("", pd.NA).dropna()
        if len(rel) > 0:
            vc = rel.value_counts().head(3)
            lines.append(f"    关系: {', '.join(f'{k}({v})' for k, v in vc.items())}")
    if "dominant_emotion_analysis" in df_cluster.columns:
        vc = df_cluster["dominant_emotion_analysis"].value_counts().head(3)
        lines.append(f"    情感: {', '.join(f'{k}({v})' for k, v in vc.items())}")
    if "dominant_topic" in df_cluster.columns:
        vc = df_cluster["dominant_topic"].value_counts().head(3)
        lines.append(f"    LDA主题: {', '.join(f'Topic{k}({v})' for k, v in vc.items())}")

    return lines


def main():
    tables = load_tables()
    lines = []

    lines.append("=" * 70)
    lines.append("聚类簇含义说明：数字对应的「内容」由簇内诗的统计归纳得出")
    lines.append("=" * 70)

    # ── LDA 主题编号含义（先给出「主题 ID → 关键词」对照表）────────────────────
    if tables.get("lda_keywords") is not None and not tables["lda_keywords"].empty:
        lines.append("\n【LDA 主题 ID 与关键词对照】（用于理解「主导主题」列）")
        for _, row in tables["lda_keywords"].iterrows():
            tid = row.get("topic_id", "")
            preview = row.get("top10_preview", "")
            cnt = row.get("doc_count", "")
            lines.append(f"  Topic {tid}: {preview}  （{cnt} 篇）")

    # 合并：需要 rule_labels 和 sentiment 带 occasion_cn, dominant_emotion_analysis 等
    merge_df = None
    if tables.get("kmeans") is not None:
        merge_df = tables["kmeans"][["ID", "kmeans_cluster", "花名", "诗名"]].copy()
    if tables.get("hier") is not None and merge_df is not None:
        merge_df = merge_df.merge(
            tables["hier"][["ID", "hier_cluster"]], on="ID", how="left"
        )
    elif tables.get("hier") is not None:
        merge_df = tables["hier"][["ID", "hier_cluster", "花名", "诗名"]].copy()

    if tables.get("lda_topics") is not None and merge_df is not None:
        merge_df = merge_df.merge(tables["lda_topics"], on="ID", how="left")
    if tables.get("rules") is not None and merge_df is not None:
        merge_df = merge_df.merge(
            tables["rules"][["ID", "occasion_cn", "relation_cn"]],
            on="ID", how="left",
        )
    if tables.get("sentiment") is not None and merge_df is not None:
        merge_df = merge_df.merge(
            tables["sentiment"][["ID", "dominant_emotion_analysis"]],
            on="ID", how="left",
        )

    if merge_df is None:
        lines.append("\n缺少聚类或 LDA/规则/情感表，无法生成簇含义。")
        report = "\n".join(lines)
        path = os.path.join(OUTPUT_DIR, "cluster_interpretation.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

    # ── k-Means 每个簇的含义 ─────────────────────────────────────────────────
    if "kmeans_cluster" in merge_df.columns:
        lines.append("\n" + "=" * 70)
        lines.append("一、k-Means 聚类（kmeans_cluster 0, 1, 2, ...）")
        lines.append("=" * 70)
        k_col = "kmeans_cluster"
        for cid in sorted(merge_df[k_col].dropna().unique()):
            sub = merge_df[merge_df[k_col] == cid]
            lines.append(f"\n  Cluster {cid}（共 {len(sub)} 条）")
            for ln in summarize_cluster(sub, "ID", str(cid)):
                lines.append(ln)
            # 示例诗名
            samples = sub["诗名"].head(3).tolist()
            lines.append(f"    示例: {samples[0][:20]}{'...' if len(str(samples[0]))>20 else ''}")

    # ── 层次聚类每个簇的含义 ───────────────────────────────────────────────────
    if "hier_cluster" in merge_df.columns:
        lines.append("\n" + "=" * 70)
        lines.append("二、层次聚类（hier_cluster 0, 1, 2, ...）")
        lines.append("=" * 70)
        h_col = "hier_cluster"
        for cid in sorted(merge_df[h_col].dropna().unique()):
            sub = merge_df[merge_df[h_col] == cid]
            lines.append(f"\n  Cluster {cid}（共 {len(sub)} 条）")
            for ln in summarize_cluster(sub, "ID", str(cid)):
                lines.append(ln)
            samples = sub["诗名"].head(3).tolist()
            lines.append(f"    示例: {samples[0][:20]}{'...' if len(str(samples[0]))>20 else ''}")

    lines.append("\n" + "=" * 70)
    lines.append("说明：聚类编号（0,1,2,...）由算法自动分配，无固定语义。")
    lines.append("      上表通过「簇内花名/场合/情感/LDA主题分布」归纳该簇的大致含义。")
    lines.append("=" * 70)

    report = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "cluster_interpretation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 已写入: {path}")
    print(report[: 2500])
    if len(report) > 2500:
        print("... (详见上述文件)")


if __name__ == "__main__":
    main()
