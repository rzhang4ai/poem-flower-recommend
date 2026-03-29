"""
step3_unsupervised/cluster_lda.py
====================================
无监督探索：LDA主题建模 + k-Means + 层次聚类

输出：
  lda_topics.csv              → 每首诗的主题分布向量
  lda_topic_keywords.csv      → 每个主题的top关键词
  kmeans_labels.csv           → k-Means聚类结果
  hierarchical_labels.csv     → 层次聚类结果
  figures/lda_coherence.png   → LDA主题数选择曲线
  figures/kmeans_elbow.png    → k-Means肘部法则曲线
  figures/dendrogram.png      → 层次聚类树状图

用法：
    python3 cluster_lda.py
    python3 cluster_lda.py --n_topics 7 --n_clusters 8
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 中文字体（matplotlib） ────────────────────────────────────────────────────
def setup_chinese_font():
    """尝试设置中文字体，找不到则用英文标签"""
    candidates = [
        'PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            return font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return None

# ── LDA 参数 ──────────────────────────────────────────────────────────────────
LDA_TOPIC_RANGE  = range(4, 13)   # 搜索4-12个主题
LDA_TOP_WORDS    = 15             # 每个主题展示的top词数
LDA_MAX_ITER     = 30
LDA_RANDOM_STATE = 42

# ── 聚类参数 ──────────────────────────────────────────────────────────────────
KMEANS_RANGE     = range(3, 13)   # 搜索3-12个簇
KMEANS_RANDOM    = 42
HIER_N_CLUSTERS  = 8              # 层次聚类默认簇数（可被命令行覆盖）


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def load_tokens(preprocess_dir: str, version: str) -> pd.DataFrame:
    path = os.path.join(preprocess_dir, f"tokens_{version}.csv")
    if not os.path.exists(path):
        print(f"  ⚠️  找不到 tokens_{version}.csv")
        return pd.DataFrame()
    return pd.read_csv(path, encoding='utf-8-sig')

def tokens_to_corpus(df: pd.DataFrame) -> list[str]:
    """将tokens列转为空格分隔字符串"""
    corpus = []
    for tokens_json in df['tokens']:
        try:
            tokens = json.loads(tokens_json) if tokens_json else []
        except Exception:
            tokens = []
        corpus.append(' '.join(tokens))
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# LDA 主题建模
# ══════════════════════════════════════════════════════════════════════════════

def compute_lda(corpus: list[str], df: pd.DataFrame,
                n_topics_range=LDA_TOPIC_RANGE) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    搜索最优主题数（基于perplexity + 手动检查coherence趋势）
    返回：(topic_dist_df, topic_keywords_df, best_n_topics)
    """
    # 构建词频矩阵
    vectorizer = CountVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.9,
        token_pattern=r'(?u)\S+'
    )
    dtm = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()

    # ── 搜索或使用指定的主题数 ─────────────────────────────────────────────────
    n_topics_range = list(n_topics_range)
    if len(n_topics_range) == 1:
        # 如果只给了一个主题数（例如通过 --n_topics 指定），直接使用该值，跳过perplexity搜索
        best_n = n_topics_range[0]
        print(f"  使用指定主题数: {best_n}")
    else:
        perplexities = []
        print("  搜索最优主题数:")
        for n in n_topics_range:
            lda = LatentDirichletAllocation(
                n_components=n,
                max_iter=LDA_MAX_ITER,
                random_state=LDA_RANDOM_STATE,
                learning_method='batch'
            )
            lda.fit(dtm)
            perp = lda.perplexity(dtm)
            perplexities.append(perp)
            print(f"    n={n:2d}  perplexity={perp:.1f}")

        # 选择perplexity下降趋缓的拐点（肘部法则）
        diffs = [perplexities[i] - perplexities[i+1] for i in range(len(perplexities)-1)]
        best_idx = int(np.argmax(diffs)) + 1   # 下降最大处后一个
        best_n = n_topics_range[best_idx]
        print(f"  → 推荐主题数: {best_n}")

        # ── 绘制perplexity曲线 ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(n_topics_range, perplexities, 'o-', color='#2D4A6A')
        ax.axvline(best_n, color='#8B3A2A', linestyle='--', label=f'推荐 n={best_n}')
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('Perplexity')
        ax.set_title('LDA Topic Count Selection (Perplexity)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "lda_coherence.png"), dpi=150)
        plt.close(fig)

    # ── 用最优主题数训练最终模型 ──────────────────────────────────────────────
    final_lda = LatentDirichletAllocation(
        n_components=best_n,
        max_iter=50,
        random_state=LDA_RANDOM_STATE,
        learning_method='batch'
    )
    topic_matrix = final_lda.fit_transform(dtm)   # shape: (n_docs, n_topics)

    # ── 主题分布（每首诗） ────────────────────────────────────────────────────
    topic_cols = [f'topic_{i}' for i in range(best_n)]
    topic_dist_df = df[['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名']].copy()
    for i, col in enumerate(topic_cols):
        topic_dist_df[col] = topic_matrix[:, i].round(4)
    topic_dist_df['dominant_topic'] = topic_matrix.argmax(axis=1)
    topic_dist_df['dominant_topic_score'] = topic_matrix.max(axis=1).round(4)

    # ── 主题关键词 ────────────────────────────────────────────────────────────
    topic_kw_rows = []
    for i, component in enumerate(final_lda.components_):
        top_idx = component.argsort()[-LDA_TOP_WORDS:][::-1]
        top_words = [(vocab[j], round(float(component[j]), 4)) for j in top_idx]
        topic_kw_rows.append({
            'topic_id':    i,
            'top_words':   json.dumps(top_words, ensure_ascii=False),
            'top10_preview': '、'.join([w for w, _ in top_words[:10]]),
            'doc_count':   int((topic_matrix.argmax(axis=1) == i).sum()),
        })
    topic_kw_df = pd.DataFrame(topic_kw_rows)

    return topic_dist_df, topic_kw_df, best_n


# ══════════════════════════════════════════════════════════════════════════════
# k-Means 聚类
# ══════════════════════════════════════════════════════════════════════════════

def compute_kmeans(tfidf_matrix: np.ndarray, df: pd.DataFrame,
                   k_range=KMEANS_RANGE) -> pd.DataFrame:
    """
    k-Means 聚类，基于肘部法则 + 轮廓系数选择最优K
    输入：TF-IDF 特征矩阵（已normalize）
    """
    X = normalize(tfidf_matrix)

    inertias   = []
    silhouettes = []
    print("\n  搜索最优聚类数 (k-Means):")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:.1f}  silhouette={sil:.3f}")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"  → 推荐 k={best_k} (最高轮廓系数)")

    # ── 绘制肘部曲线 ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, 'o-', color='#2D4A6A')
    ax1.axvline(best_k, color='#8B3A2A', linestyle='--', label=f'best k={best_k}')
    ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
    ax1.set_title('k-Means Elbow Curve'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(list(k_range), silhouettes, 's-', color='#2D6A4F')
    ax2.axvline(best_k, color='#8B3A2A', linestyle='--', label=f'best k={best_k}')
    ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score by k'); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "kmeans_elbow.png"), dpi=150)
    plt.close(fig)

    # ── 最优K的最终聚类 ───────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, random_state=KMEANS_RANDOM, n_init=10)
    labels_final = km_final.fit_predict(X)

    result = df[['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名']].copy()
    result['kmeans_cluster']      = labels_final
    result['kmeans_k']            = best_k
    result['kmeans_silhouette']   = round(silhouettes[list(k_range).index(best_k)], 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 层次聚类
# ══════════════════════════════════════════════════════════════════════════════

def compute_hierarchical(tfidf_matrix: np.ndarray, df: pd.DataFrame,
                         n_clusters: int = HIER_N_CLUSTERS) -> pd.DataFrame:
    """
    层次聚类 (Ward linkage)
    优势：不需要预设K，树状图直观展示文档间的层次关系
    """
    X = normalize(tfidf_matrix)

    # 树状图（取前50条文档，避免过密）
    sample_n = min(50, len(X))
    X_sample = X[:sample_n]
    labels_sample = df['诗名'].str[:8].values[:sample_n]

    Z = linkage(X_sample, method='ward')
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(
        Z,
        labels=labels_sample,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0.7 * max(Z[:, 2]),
        ax=ax
    )
    ax.set_title(f'Hierarchical Clustering Dendrogram (n={sample_n} samples)')
    ax.set_ylabel('Distance')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dendrogram.png"), dpi=150)
    plt.close(fig)

    # 完整数据集聚类
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_all = hc.fit_predict(X)

    result = df[['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名']].copy()
    result['hier_cluster']    = labels_all
    result['hier_n_clusters'] = n_clusters
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_dir', default='../step1_preprocess/output')
    parser.add_argument('--n_topics',   type=int, default=None,
                        help='强制指定LDA主题数（默认自动搜索）')
    parser.add_argument('--n_clusters', type=int, default=HIER_N_CLUSTERS,
                        help='层次聚类簇数（默认8）')
    args = parser.parse_args()

    font = setup_chinese_font()
    print(f"中文字体: {font or '未找到，图表使用英文'}")

    preprocess_dir = os.path.join(os.path.dirname(__file__), args.preprocess_dir)

    print("\n" + "=" * 55)
    print("Step 3: 无监督聚类探索")
    print("=" * 55)

    # 主要使用赏析版本（NLP效果最好）
    # 同时对combined版本运行，便于对比
    for version in ['analysis', 'combined']:
        df = load_tokens(preprocess_dir, version)
        if df.empty:
            continue
        corpus = tokens_to_corpus(df)
        non_empty = [(i, c) for i, c in enumerate(corpus) if c.strip()]
        print(f"\n── {version} 版本 ({len(non_empty)}/{len(corpus)} 条有效) ──")

        if len(non_empty) < 10:
            print("  有效文档不足10条，跳过")
            continue

        # TF-IDF矩阵（用于聚类的输入）
        tfidf_vec = TfidfVectorizer(
            max_features=300, min_df=2, max_df=0.9,
            token_pattern=r'(?u)\S+'
        )
        tfidf_matrix = tfidf_vec.fit_transform(corpus).toarray()

        # ── LDA ──────────────────────────────────────────────────────────────
        print(f"\n  [LDA]")
        topic_range = range(args.n_topics, args.n_topics + 1) if args.n_topics else LDA_TOPIC_RANGE
        topic_dist_df, topic_kw_df, best_n = compute_lda(corpus, df, topic_range)

        suffix = f"_{version}"
        topic_dist_df.to_csv(
            os.path.join(OUTPUT_DIR, f"lda_topics{suffix}.csv"),
            index=False, encoding='utf-8-sig')
        topic_kw_df.to_csv(
            os.path.join(OUTPUT_DIR, f"lda_topic_keywords{suffix}.csv"),
            index=False, encoding='utf-8-sig')
        print(f"  💾 lda_topics{suffix}.csv")
        print(f"  💾 lda_topic_keywords{suffix}.csv")
        print(f"\n  ── 主题关键词预览 (n={best_n}) ──")
        for _, row in topic_kw_df.iterrows():
            bar = '█' * row['doc_count']
            print(f"    Topic {row['topic_id']}: {row['top10_preview']}")
            print(f"           [{row['doc_count']:2d}篇] {bar}")

        # ── k-Means ───────────────────────────────────────────────────────────
        print(f"\n  [k-Means]")
        km_df = compute_kmeans(tfidf_matrix, df)
        km_df.to_csv(
            os.path.join(OUTPUT_DIR, f"kmeans_labels{suffix}.csv"),
            index=False, encoding='utf-8-sig')
        print(f"  💾 kmeans_labels{suffix}.csv")

        # ── 层次聚类 ──────────────────────────────────────────────────────────
        print(f"\n  [层次聚类]")
        hier_df = compute_hierarchical(tfidf_matrix, df, n_clusters=args.n_clusters)
        hier_df.to_csv(
            os.path.join(OUTPUT_DIR, f"hierarchical_labels{suffix}.csv"),
            index=False, encoding='utf-8-sig')
        print(f"  💾 hierarchical_labels{suffix}.csv")
        print(f"  💾 figures/dendrogram.png  (前50条树状图)")

    print("\n✅ Step 3 完成")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   图表目录: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
