"""
step2b_bert/visualize_embeddings.py
─────────────────────────────────────────────────────────────────────────────
回答三个核心问题：

Q1  如何可视化 embedding，直观判断正确性？
    → 用 PCA / t-SNE / UMAP 降至 2D，按花名/朝代着色
    → 若同花名的诗聚集在一起，说明 embedding 捕获到了语义主题

Q2  语义相似度是由什么特征驱动的？
    → 显示相似诗对的原文 + 共同高注意力 token
    → PCA 前3主成分载荷，看哪个"语义方向"最有区分力

Q3  top-10 token 是意象吗？
    → 对比同一首诗的 top-10 attention token 与意象词典的交集
    → 统计 top-10 中属于自然意象/人文意象的比例

输出（output/figures/）：
  fig1a_pca_poem_by_flower.png      正文 PCA 按花名着色
  fig1b_tsne_poem_by_flower.png     正文 t-SNE 按花名着色
  fig1c_umap_poem_by_flower.png     正文 UMAP 按花名着色（最推荐）
  fig1d_umap_ana_by_flower.png      赏析 UMAP 按花名着色
  fig1e_umap_both_side_by_side.png  正文+赏析对比
  fig2a_similar_pairs_text.txt      Top-5相似诗对原文对比（文本）
  fig2b_pca_components.png          PCA主成分载荷分析
  fig3a_token_imagery_overlap.png   top-10 token 与意象词典重叠分析
  fig3b_token_examples.png          典型诗句 token 重要性条形图

运行方式：
    cd /Users/rzhang/Documents/poem-flower-recommend
    source flower_env/bin/activate
    python 02.sample_label_phase2/step2b_bert/visualize_embeddings.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无窗口后端，适合脚本运行
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ─── 路径 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR   = SCRIPT_DIR / "output"
FIG_DIR      = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "models" / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# ─── 中文字体（macOS 系统字体） ───────────────────────────────────────────────
def setup_chinese_font():
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return prop.get_name()
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT_NAME = setup_chinese_font()


# ─── 意象词典（来自 Phase2 升级方案 §3.2） ────────────────────────────────────
NATURE_IMAGERY = {
    "月","日","风","雪","霜","云","水","山","雨","露",
    "松","竹","梅","兰","菊","莲","柳","桃","杏","桂",
    "鸿","雁","燕","鹤","蝉","萤","蝶","鱼","花","草",
    "叶","枝","根","香","寒","春","秋","冬","夏","天",
    "江","湖","河","海","溪","涧","石","峰","岭",
}
CULTURAL_IMAGERY = {
    "剑","琴","酒","烛","灯","镜","楼","舟","帆",
    "东篱","南山","折柳","芳草","茅屋","寒窗",
    "笛","笔","墨","砚","诗","书","长亭","短亭",
}
ACTION_IMAGERY = {
    "飞","落","凋","零","断","寄","归","别","残","空",
    "散","消","逝","沉","望","思","忆","吟","醉","哭",
}
IMAGERY_VOCAB = NATURE_IMAGERY | CULTURAL_IMAGERY | ACTION_IMAGERY


# ─── 调色板：为每种花分配颜色 ────────────────────────────────────────────────
def make_color_map(labels):
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    n = len(unique)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return {label: colors[i] for i, label in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════
# Q1  可视化 embedding（PCA / t-SNE / UMAP）
# ═══════════════════════════════════════════════════════════════════════════

def plot_2d_scatter(coords_2d, labels, title, filepath,
                    color_map=None, top_n_flowers=15,
                    annotation_col=None, df=None):
    """通用 2D 散点图，按标签着色，只显示出现最多的 top_n 类"""
    counter = Counter(labels)
    top_labels = {lab for lab, _ in counter.most_common(top_n_flowers)}

    if color_map is None:
        color_map = make_color_map(list(top_labels))

    fig, ax = plt.subplots(figsize=(12, 9))

    for label in sorted(top_labels):
        mask = [l == label for l in labels]
        xs = coords_2d[mask, 0]
        ys = coords_2d[mask, 1]
        ax.scatter(xs, ys,
                   c=[color_map.get(label, "#aaaaaa")],
                   label=f"{label}({sum(mask)})",
                   s=60, alpha=0.75, edgecolors="white", linewidths=0.3)

    # 不在 top_n 的用灰色点，不加图例
    other_mask = [l not in top_labels for l in labels]
    if any(other_mask):
        ax.scatter(coords_2d[other_mask, 0], coords_2d[other_mask, 1],
                   c="#cccccc", s=30, alpha=0.4, label="_nolegend_")

    # 为每个点标注诗名（可选，太密时关掉）
    if annotation_col is not None and df is not None:
        for i, (x, y) in enumerate(coords_2d):
            if labels[i] in top_labels:
                txt = str(df.iloc[i].get(annotation_col, ""))[:6]
                ax.annotate(txt, (x, y), fontsize=5, alpha=0.5,
                            xytext=(2, 2), textcoords="offset points")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("维度 1")
    ax.set_ylabel("维度 2")
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              framealpha=0.7, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150)
    plt.close()
    print(f"  保存：{filepath.name}")


def q1_visualize_embeddings(poem_emb, ana_emb, df):
    print("\n[Q1] 生成 embedding 可视化图 ...")
    flower_labels = list(df["花名"].fillna("未知"))
    dynasty_labels = list(df["朝代"].fillna("未知"))
    color_map_flower = make_color_map(
        [l for l, c in Counter(flower_labels).most_common(15)]
    )

    # —— 1a. 正文 PCA ─────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    poem_pca = pca.fit_transform(normalize(poem_emb))
    var_ratio = pca.explained_variance_ratio_
    plot_2d_scatter(
        poem_pca, flower_labels,
        f"正文 Embedding PCA（BERT-CCPoem）\n"
        f"PC1={var_ratio[0]:.1%}  PC2={var_ratio[1]:.1%}  共解释方差={sum(var_ratio):.1%}",
        FIG_DIR / "fig1a_pca_poem_by_flower.png",
        color_map=color_map_flower, annotation_col="诗名", df=df,
    )

    # —— 1b. 正文 t-SNE ───────────────────────────────────────────────────────
    print("    正在运行 t-SNE（约10秒）...")
    tsne = TSNE(n_components=2, perplexity=20, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    poem_tsne = tsne.fit_transform(normalize(poem_emb))
    plot_2d_scatter(
        poem_tsne, flower_labels,
        "正文 Embedding t-SNE（BERT-CCPoem）\n"
        "非线性降维，同类诗歌应聚集",
        FIG_DIR / "fig1b_tsne_poem_by_flower.png",
        color_map=color_map_flower, annotation_col="诗名", df=df,
    )

    # —— 1c. 正文 UMAP ────────────────────────────────────────────────────────
    try:
        import umap
        print("    正在运行 UMAP（约5秒）...")
        reducer = umap.UMAP(n_components=2, n_neighbors=12,
                            min_dist=0.1, random_state=42, metric="cosine")
        poem_umap = reducer.fit_transform(poem_emb)
        plot_2d_scatter(
            poem_umap, flower_labels,
            "正文 Embedding UMAP（BERT-CCPoem）\n"
            "余弦相似度空间，最接近真实语义拓扑",
            FIG_DIR / "fig1c_umap_poem_by_flower.png",
            color_map=color_map_flower, annotation_col="诗名", df=df,
        )
    except ImportError:
        print("    [跳过] UMAP 未安装")
        poem_umap = poem_tsne  # 备用

    # —— 1d. 赏析 UMAP ────────────────────────────────────────────────────────
    try:
        import umap
        reducer2 = umap.UMAP(n_components=2, n_neighbors=12,
                             min_dist=0.1, random_state=42, metric="cosine")
        ana_umap = reducer2.fit_transform(ana_emb)
        plot_2d_scatter(
            ana_umap, flower_labels,
            "赏析 Embedding UMAP（SikuRoBERTa）\n"
            "赏析语义空间：花名聚类 vs 情感/主题聚类",
            FIG_DIR / "fig1d_umap_ana_by_flower.png",
            color_map=color_map_flower, annotation_col="诗名", df=df,
        )

        # —— 1e. 并排对比 ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        for ax, coords, title in [
            (axes[0], poem_umap, "正文 UMAP\n（BERT-CCPoem，古诗词专用）"),
            (axes[1], ana_umap,  "赏析 UMAP\n（SikuRoBERTa，四库全书预训练）"),
        ]:
            top15 = {l for l, _ in Counter(flower_labels).most_common(15)}
            for label in sorted(top15):
                mask = np.array([l == label for l in flower_labels])
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[color_map_flower.get(label, "#aaa")],
                           label=f"{label}({mask.sum()})",
                           s=55, alpha=0.75, edgecolors="white", linewidths=0.3)
            other = np.array([l not in top15 for l in flower_labels])
            ax.scatter(coords[other, 0], coords[other, 1],
                       c="#cccccc", s=25, alpha=0.35)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.7)

        fig.suptitle("正文 vs 赏析 Embedding 空间对比（按花名着色）", fontsize=14)
        plt.tight_layout()
        plt.savefig(str(FIG_DIR / "fig1e_umap_both_side_by_side.png"), dpi=150)
        plt.close()
        print(f"  保存：fig1e_umap_both_side_by_side.png")

    except ImportError:
        pass

    return poem_pca, poem_tsne


# ═══════════════════════════════════════════════════════════════════════════
# Q2  语义相似度驱动因素：相似诗对原文对比 + PCA 主成分分析
# ═══════════════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    a, b = a / (np.linalg.norm(a) + 1e-8), b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def q2_similarity_analysis(poem_emb, ana_emb, df, token_df):
    print("\n[Q2] 语义相似度驱动因素分析 ...")

    # —— 2a. 找最高相似度诗对（正文） ──────────────────────────────────────────
    n = len(df)
    normed = normalize(poem_emb)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, -1)

    pairs = []
    for i in range(n):
        j = int(np.argmax(sim_matrix[i]))
        if i < j:  # 避免重复
            pairs.append((i, j, sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:20]

    # —— 文本报告 ──────────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "fig2a_similar_pairs_text.txt"
    tok_map = {str(r["ID"]): r for _, r in token_df.iterrows()}

    lines = [
        "=" * 70,
        "Q2：语义相似诗对分析（BERT-CCPoem 正文 embedding 余弦相似度）",
        "=" * 70,
        "",
        "★ 说明：BERT embedding 的相似度捕获了诗句在 512 维语义空间的整体相似性",
        "  不可解释为某单一维度，但从相似诗对的内容可以归纳出共同的：",
        "  ① 主题/题材（同写离别、思乡、咏物...）",
        "  ② 意象词汇（共用「月」「寒」「香」等高频意象）",
        "  ③ 情感基调（同属悲/喜/豪放/婉约）",
        "  ④ 朝代/文风（宋词 vs 唐诗 的用字风格有显著差异）",
        "",
    ]

    for rank, (i, j, score) in enumerate(top_pairs[:10], 1):
        row_i, row_j = df.iloc[i], df.iloc[j]

        # 获取各自的 top-5 token
        def get_top5(row):
            r = token_df[token_df["ID"] == row.get("ID", -1)]
            if r.empty:
                return []
            return [t for t, _ in json.loads(r.iloc[0]["top10_tokens"])[:5]]

        tok_i = get_top5(row_i)
        tok_j = get_top5(row_j)
        shared_tok = set(tok_i) & set(tok_j)
        shared_img = shared_tok & IMAGERY_VOCAB

        # 同花名？同朝代？
        same_flower  = "✅ 同花" if row_i["花名"] == row_j["花名"] else "❌ 不同花"
        same_dynasty = "✅ 同朝" if row_i.get("朝代","") == row_j.get("朝代","") else "❌ 不同朝"

        poem_i_short = str(row_i.get("正文",""))[:80].replace("\n"," ")
        poem_j_short = str(row_j.get("正文",""))[:80].replace("\n"," ")

        lines += [
            f"─── Top-{rank} 相似对  余弦相似度 = {score:.4f} ───────────────────────",
            f"  A: 【{row_i['花名']}】{row_i['诗名']} ({row_i.get('朝代','?')}·{row_i.get('作者','?')})",
            f"     正文：{poem_i_short}",
            f"     top-5 token：{'、'.join(tok_i)}",
            f"  B: 【{row_j['花名']}】{row_j['诗名']} ({row_j.get('朝代','?')}·{row_j.get('作者','?')})",
            f"     正文：{poem_j_short}",
            f"     top-5 token：{'、'.join(tok_j)}",
            f"  {same_flower}  {same_dynasty}",
            f"  共同高注意力 token：{'、'.join(shared_tok) if shared_tok else '（无）'}",
            f"  其中意象词：{'、'.join(shared_img) if shared_img else '（无）'}",
            "",
        ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  保存：{report_path.name}")

    # —— 2b. PCA 主成分载荷图（解释"哪个方向区分力最强"） ─────────────────────
    pca_full = PCA(n_components=10, random_state=42)
    pca_full.fit(normalize(poem_emb))
    variance = pca_full.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 方差解释比例
    axes[0].bar(range(1, 11), variance * 100, color="#4472C4", alpha=0.8)
    axes[0].set_xlabel("主成分编号")
    axes[0].set_ylabel("解释方差 (%)")
    axes[0].set_title("正文 Embedding PCA 方差解释比例\n（BERT-CCPoem，512维→前10主成分）")
    axes[0].set_xticks(range(1, 11))
    for i, v in enumerate(variance):
        axes[0].text(i + 1, v * 100 + 0.1, f"{v:.1%}", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(variance) * 100 * 1.3)

    # 花名在 PC1-PC2 上的平均位置
    coords = pca_full.transform(normalize(poem_emb))
    flower_series = df["花名"].fillna("未知")
    top10_flowers = [f for f, _ in Counter(flower_series).most_common(10)]
    axes[1].set_title("各花名在 PC1-PC2 上的中心位置\n（反映不同花的语义方向差异）")
    for flower in top10_flowers:
        mask = (flower_series == flower).values
        if mask.sum() >= 2:
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            axes[1].scatter(cx, cy, s=80, label=flower, zorder=5)
            axes[1].annotate(flower, (cx, cy), fontsize=9,
                             xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig2b_pca_components.png"), dpi=150)
    plt.close()
    print(f"  保存：fig2b_pca_components.png")


# ═══════════════════════════════════════════════════════════════════════════
# Q3  top-10 token 与意象词典的关系
# ═══════════════════════════════════════════════════════════════════════════

def q3_token_imagery_analysis(token_df, df):
    print("\n[Q3] top-10 token 与意象词汇分析 ...")

    # 对每首诗统计 top-10 token 中意象词的比例
    rows = []
    all_top_tokens = []

    for _, r in token_df.iterrows():
        tokens = json.loads(r["top10_tokens"])  # [(token, score), ...]
        tok_list = [t for t, _ in tokens]
        all_top_tokens.extend(tok_list)

        nature_hits  = [t for t in tok_list if t in NATURE_IMAGERY]
        culture_hits = [t for t in tok_list if t in CULTURAL_IMAGERY]
        action_hits  = [t for t in tok_list if t in ACTION_IMAGERY]
        imagery_hits = nature_hits + culture_hits + action_hits

        rows.append({
            "ID":             r.get("ID", ""),
            "诗名":           r.get("诗名", ""),
            "花名":           r.get("花名", ""),
            "top10_count":    len(tok_list),
            "imagery_count":  len(imagery_hits),
            "imagery_ratio":  len(imagery_hits) / max(len(tok_list), 1),
            "nature_count":   len(nature_hits),
            "culture_count":  len(culture_hits),
            "action_count":   len(action_hits),
            "imagery_tokens": "、".join(imagery_hits),
            "non_imagery_tokens": "、".join([t for t in tok_list if t not in IMAGERY_VOCAB]),
        })

    stat_df = pd.DataFrame(rows)

    # —— 3a. 意象覆盖率分布图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 意象比例直方图
    axes[0].hist(stat_df["imagery_ratio"] * 100, bins=15,
                 color="#70AD47", edgecolor="white", alpha=0.85)
    mean_ratio = stat_df["imagery_ratio"].mean()
    axes[0].axvline(mean_ratio * 100, color="red", lw=2,
                    label=f"均值 {mean_ratio:.1%}")
    axes[0].set_xlabel("top-10 token 中意象词比例 (%)")
    axes[0].set_ylabel("诗首数")
    axes[0].set_title("top-10 attention token 中意象词占比分布\n（Q3核心：attention ≈ 意象？）")
    axes[0].legend()

    # 意象类型饼图
    totals = {
        "自然意象\n(月/风/松/雁…)": stat_df["nature_count"].sum(),
        "人文意象\n(剑/琴/舟/长亭…)": stat_df["culture_count"].sum(),
        "动态意象\n(飞/落/归/别…)": stat_df["action_count"].sum(),
        "非意象字符\n(功能词/其他)":
            (stat_df["top10_count"] - stat_df["imagery_count"]).sum(),
    }
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#BBBBBB"]
    wedges, texts, autotexts = axes[1].pie(
        totals.values(), labels=totals.keys(),
        colors=colors, autopct="%1.1f%%",
        startangle=140, textprops={"fontsize": 8},
    )
    axes[1].set_title("top-10 token 构成分类\n（200首诗合计统计）")

    # 各花的意象覆盖率均值（横条图）
    flower_img = (
        stat_df.groupby("花名")["imagery_ratio"]
        .mean()
        .sort_values(ascending=True)
        .tail(15)
    )
    axes[2].barh(range(len(flower_img)), flower_img.values * 100,
                 color="#4472C4", alpha=0.8)
    axes[2].set_yticks(range(len(flower_img)))
    axes[2].set_yticklabels(flower_img.index, fontsize=8)
    axes[2].set_xlabel("意象词比例均值 (%)")
    axes[2].set_title("各花名 top-10 token 中\n意象词比例均值（前15花）")
    axes[2].axvline(mean_ratio * 100, color="red", lw=1.5, linestyle="--",
                    label=f"全局均值{mean_ratio:.0%}")
    axes[2].legend(fontsize=8)

    plt.suptitle("Q3：BERT top-10 attention token 的[意象性]分析", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig3a_token_imagery_overlap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存：fig3a_token_imagery_overlap.png")

    # —— 3b. 典型诗句 token 重要性条形图（6首示例） ────────────────────────────
    examples = token_df.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (_, r) in zip(axes, examples.iterrows()):
        tokens_scored = json.loads(r["top10_tokens"])
        if not tokens_scored:
            continue
        toks  = [t for t, _ in tokens_scored]
        scores = [s for _, s in tokens_scored]
        # 着色：意象词=蓝色，非意象词=灰色
        bar_colors = ["#4472C4" if t in IMAGERY_VOCAB else "#BBBBBB"
                      for t in toks]
        bars = ax.barh(range(len(toks))[::-1], scores,
                       color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(toks))[::-1])
        ax.set_yticklabels(toks, fontsize=10)
        ax.set_title(
            f"【{r.get('花名','')}】{str(r.get('诗名',''))[:10]}\n"
            f"朝代：{r.get('朝代','')}",
            fontsize=9
        )
        ax.set_xlabel("attention 分数", fontsize=8)
        # 右侧注释：是否意象词
        for i, (t, s) in enumerate(tokens_scored):
            tag = "★意象" if t in IMAGERY_VOCAB else ""
            ax.text(s + max(scores) * 0.01, len(toks) - 1 - i,
                    tag, va="center", fontsize=7, color="#4472C4")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label="意象词"),
        Patch(facecolor="#BBBBBB", label="非意象词（功能字/其他）"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        "Q3：6首示例诗的 BERT top-10 attention token\n"
        "蓝色=在意象词典中，灰色=功能字/其他",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig3b_token_examples.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存：fig3b_token_examples.png")

    # —— 3c. 全局 top-30 高频 token + 意象分析 ────────────────────────────────
    counter = Counter(all_top_tokens)
    top30 = counter.most_common(30)
    toks30  = [t for t, _ in top30]
    cnts30  = [c for _, c in top30]
    colors30 = []
    for t in toks30:
        if t in NATURE_IMAGERY:   colors30.append("#4472C4")
        elif t in CULTURAL_IMAGERY: colors30.append("#ED7D31")
        elif t in ACTION_IMAGERY:  colors30.append("#A9D18E")
        else:                      colors30.append("#BBBBBB")

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(top30)), cnts30, color=colors30, alpha=0.85)
    ax.set_xticks(range(len(top30)))
    ax.set_xticklabels(toks30, fontsize=11)
    ax.set_ylabel("出现在 top-10 中的次数（200首合计）")
    ax.set_title(
        "Q3：200首诗 BERT top-10 attention token 高频统计（前30字）\n"
        "蓝=自然意象  橙=人文意象  绿=动态意象  灰=非意象"
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label="自然意象（月/风/雪…）"),
        Patch(facecolor="#ED7D31", label="人文意象（剑/琴/酒…）"),
        Patch(facecolor="#A9D18E", label="动态意象（飞/落/归…）"),
        Patch(facecolor="#BBBBBB", label="非意象词"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "fig3c_top30_tokens_global.png"), dpi=150)
    plt.close()
    print(f"  保存：fig3c_top30_tokens_global.png")

    # 保存统计 CSV
    stat_df.to_csv(str(OUTPUT_DIR / "bert_token_imagery_stats.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"  保存：bert_token_imagery_stats.csv")

    # 终端打印摘要
    print(f"\n  ── Q3 统计摘要 ──────────────────────────────────────")
    print(f"  top-10 token 中意象词平均比例：{mean_ratio:.1%}")
    print(f"  ≥50% token 是意象词的诗：{(stat_df['imagery_ratio']>=0.5).sum()} 首")
    print(f"  0 个意象词的诗：{(stat_df['imagery_count']==0).sum()} 首")
    print(f"  全局最高频 token（top5）：{'  '.join([t for t,_ in top30[:5]])}")
    print(f"  全局最高频意象 token：{[t for t,_ in top30 if t in IMAGERY_VOCAB][:8]}")


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  step2b_bert 可视化分析")
    print("=" * 65)

    # 读取数据
    poem_emb = np.load(str(OUTPUT_DIR / "bert_ccpoem_embeddings.npy"))
    ana_emb  = np.load(str(OUTPUT_DIR / "bert_analysis_embeddings.npy"))
    df       = pd.read_csv(str(PROJECT_ROOT / "01.sample_label" / "output" / "sample_200.csv"))
    token_df = pd.read_csv(str(OUTPUT_DIR / "bert_token_importance.csv"))

    print(f"\n数据：{len(df)} 首诗")
    print(f"正文 embedding：{poem_emb.shape}  赏析 embedding：{ana_emb.shape}")

    q1_visualize_embeddings(poem_emb, ana_emb, df)
    q2_similarity_analysis(poem_emb, ana_emb, df, token_df)
    q3_token_imagery_analysis(token_df, df)

    print("\n" + "=" * 65)
    print("✅ 可视化完成！所有图表保存在：")
    print(f"   {FIG_DIR}")
    figs = sorted(FIG_DIR.glob("fig*.png"))
    for f in figs:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name}  ({size_kb:.0f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
