"""
可视化3：Top 10 诗人分析——每位诗人写过哪些花、各花的情感极性与核心意象。

每位诗人输出一张图，包含：
  - 标题：诗人姓名 + 总诗数
  - 每种花一行（按该花诗数降序），包含：
      左：花名 + 诗数
      中：情感分布横柱（10色 L3）
      右：Top5 核心意象词（sxhy_raw_words，双重确认词红色加粗）

另输出一张汇总图（poet_summary_grid.png）：
  10位诗人 × 所写花，色块=主导情感。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "03.final_labels/poems_structured_shangxi_wip.csv"
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)
FONT_PATH = "/System/Library/Fonts/STHeiti Light.ttc"

# ── 情感配色 ──────────────────────────────────────────────────────────────────
L3_ORDER = ["praise", "like", "joy", "ease", "faith",
            "sorrow", "miss", "misgive", "criticize", "vexed"]
L3_ZH = {
    "praise": "赞美", "like": "喜爱", "joy": "喜悦",
    "ease": "闲适", "faith": "信念",
    "sorrow": "悲伤", "miss": "思念", "misgive": "忧虑",
    "criticize": "讽刺", "vexed": "烦闷",
}
L3_COLORS = {
    "praise": "#E67E22", "like": "#F39C12", "joy": "#F1C40F",
    "ease": "#2ECC71",   "faith": "#27AE60",
    "sorrow": "#3498DB", "miss": "#2980B9", "misgive": "#1ABC9C",
    "criticize": "#E74C3C", "vexed": "#C0392B",
}
L1_COLORS = {
    "Positive": "#27AE60",
    "Implicit Positive": "#82E0AA",
    "Neutral": "#BDC3C7",
    "Implicit Negative": "#85C1E9",
    "Negative": "#E74C3C",
}
POSITIVE_SET = {"Positive", "Implicit Positive"}


def _fp(size=11):
    try:
        return fm.FontProperties(fname=FONT_PATH, size=size)
    except Exception:
        return None


def get_top_imagery(sub: pd.DataFrame, topn: int = 5) -> list[str]:
    counter: Counter = Counter()
    noise = {"花", "风", "月", "春", "云", "天", "日", "水", "山", "红"}
    for raw in sub["sxhy_raw_words"].dropna():
        for w in str(raw).split("|"):
            w = w.strip()
            if w and w not in noise:
                counter[w] += 1
    return [w for w, _ in counter.most_common(topn)]


def get_confirmed(sub: pd.DataFrame) -> set[str]:
    confirmed = set()
    for c in sub["confirmed_imagery"].dropna():
        confirmed.update(w.strip() for w in str(c).split("|") if w.strip())
    return confirmed


def plot_poet(poet: str, df_poet: pd.DataFrame, save_path: Path):
    flowers = df_poet["花名"].value_counts()
    n_flowers = len(flowers)

    fig_h = max(5, n_flowers * 0.9 + 2.5)
    fig, axes = plt.subplots(
        n_flowers, 3,
        figsize=(16, fig_h),
        gridspec_kw={"width_ratios": [1.2, 3, 2.5]}
    )
    if n_flowers == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FAFAFA")

    for i, (flower, count) in enumerate(flowers.items()):
        sub = df_poet[df_poet["花名"] == flower]
        ax_info, ax_bar, ax_words = axes[i]

        # ── 左格：花名 + 情感极性饼/标注 ───────────────────────────────────
        ax_info.set_facecolor("#F0F3F4")
        ax_info.axis("off")
        pos_n = sub["l1_polarity"].isin(POSITIVE_SET).sum()
        neg_n = len(sub) - pos_n
        pos_r = pos_n / len(sub) if len(sub) > 0 else 0

        ax_info.text(0.5, 0.80, flower, ha="center", va="center",
                     fontproperties=_fp(14), color="#2C3E50", fontweight="bold")
        ax_info.text(0.5, 0.52, f"{count} 首", ha="center", va="center",
                     fontproperties=_fp(11), color="#7F8C8D")

        # 小饼图：正/负比例
        pie_colors = ["#E67E22" if pos_r >= 0.5 else "#3498DB", "#EEEEEE"]
        ax_info.pie([pos_r, 1 - pos_r], colors=pie_colors,
                    startangle=90, radius=0.28,
                    center=(0.5, 0.22),
                    wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        pol_label = "正向为主" if pos_r >= 0.5 else "负向为主"
        ax_info.text(0.5, 0.05, f"{pol_label} {pos_r:.0%}", ha="center",
                     va="center", fontproperties=_fp(9),
                     color="#E67E22" if pos_r >= 0.5 else "#3498DB")

        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        # ── 中格：L3情感堆叠横柱 ─────────────────────────────────────────────
        ax_bar.set_facecolor("#F8F9FA")
        em_counts = sub["l3_c3"].value_counts()
        total = em_counts.sum()
        left = 0.0
        for em in L3_ORDER:
            val = em_counts.get(em, 0) / total if total > 0 else 0
            if val > 0:
                ax_bar.barh(0, val, left=left, height=0.55,
                             color=L3_COLORS[em], label=L3_ZH[em])
                if val >= 0.1:
                    ax_bar.text(left + val / 2, 0, L3_ZH[em],
                                ha="center", va="center",
                                color="white", fontproperties=_fp(9),
                                fontweight="bold")
                left += val

        # L1极性分布（小柱子，仅颜色参考）
        l1_counts = sub["l1_polarity"].value_counts()
        l1_total = l1_counts.sum()
        left1 = 0.0
        for l1, color in L1_COLORS.items():
            val = l1_counts.get(l1, 0) / l1_total if l1_total > 0 else 0
            ax_bar.barh(-0.45, val, left=left1, height=0.25,
                         color=color, alpha=0.7)
            left1 += val

        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(-0.75, 0.4)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                                fontproperties=_fp(8))
        ax_bar.text(0.5, 0.32, "L3细粒度情感", ha="center",
                    fontproperties=_fp(8), color="#888888", transform=ax_bar.transAxes)
        ax_bar.text(0.5, 0.10, "L1极性", ha="center",
                    fontproperties=_fp(7), color="#AAAAAA", transform=ax_bar.transAxes)
        ax_bar.axvline(0.5, color="#CCCCCC", lw=0.8, ls="--")
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        # ── 右格：意象词 ──────────────────────────────────────────────────────
        ax_words.set_facecolor("#FAFAFA")
        ax_words.axis("off")
        ax_words.set_xlim(0, 1)
        ax_words.set_ylim(0, 1)

        imagery = get_top_imagery(sub, topn=6)
        confirmed = get_confirmed(sub)

        # 所有出现词频统计
        all_counter: Counter = Counter()
        for raw in sub["sxhy_raw_words"].dropna():
            for w in str(raw).split("|"):
                w = w.strip()
                if w:
                    all_counter[w] += 1

        x, y = 0.08, 0.82
        for w in imagery:
            freq = all_counter.get(w, 0)
            is_confirmed = w in confirmed
            color = "#C0392B" if is_confirmed else "#2C3E50"
            weight = "bold" if is_confirmed else "normal"
            marker = "★" if is_confirmed else "·"
            ax_words.text(x, y, f"{marker} {w}（{freq}）",
                          ha="left", va="center",
                          color=color, fontweight=weight,
                          fontproperties=_fp(10))
            y -= 0.17
            if y < 0.05:
                break

        if not imagery:
            ax_words.text(0.5, 0.5, "（无意象数据）", ha="center", va="center",
                          fontproperties=_fp(9), color="#AAAAAA")

        # 分割线
        if i < n_flowers - 1:
            fig.add_artist(plt.Line2D(
                [0.02, 0.98], [(fig_h - (i + 1) * 0.9 - 2.5 + 0.05) / fig_h] * 2,
                color="#DDDDDD", lw=0.5, transform=fig.transFigure, zorder=100))

    # 图例
    handles = [mpatches.Patch(color=L3_COLORS[em], label=L3_ZH[em])
               for em in L3_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01),
               prop=_fp(9), framealpha=0.9)

    note = ("红色★ = BERT注意力 × 诗学含英双重确认意象  |  "
            "中柱=L3细粒度情感分布  |  下条=L1极性分布")
    fig.text(0.5, -0.025, note, ha="center", fontproperties=_fp(8),
             color="#888888")

    total_poems = len(df_poet)
    unique_flowers = len(flowers)
    fig.suptitle(f"{poet}  ·  共 {total_poems} 首  ·  涉及 {unique_flowers} 种花",
                 fontproperties=_fp(16), y=1.01, fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── 汇总网格图 ─────────────────────────────────────────────────────────────────

def plot_summary_grid(df: pd.DataFrame, top10_poets: list[str], save_path: Path):
    """10位诗人 × 各自写过的花，色块=主导L1情感极性。"""
    # 收集所有出现的花种
    all_flowers_ordered = (
        df[df["作者"].isin(top10_poets)]["花名"]
        .value_counts().index.tolist()
    )

    n_poets = len(top10_poets)
    n_flowers = len(all_flowers_ordered)
    cell_w, cell_h = 1.2, 0.9
    fig, ax = plt.subplots(figsize=(n_flowers * cell_w + 2, n_poets * cell_h + 1.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F0F0F0")

    poet_totals = df["作者"].value_counts()

    for i, poet in enumerate(top10_poets):
        for j, flower in enumerate(all_flowers_ordered):
            sub = df[(df["作者"] == poet) & (df["花名"] == flower)]
            if len(sub) == 0:
                ax.add_patch(plt.Rectangle((j, n_poets - 1 - i), 1, 1,
                                            color="#EEEEEE", lw=0))
                continue

            # 确定主导L1情感色
            top_l1 = sub["l1_polarity"].value_counts().idxmax()
            color = L1_COLORS.get(top_l1, "#BDC3C7")
            ax.add_patch(plt.Rectangle((j, n_poets - 1 - i), 1, 1,
                                        color=color, lw=0.5, edgecolor="white"))

            # 诗数标注
            ax.text(j + 0.5, n_poets - 1 - i + 0.55, str(len(sub)),
                    ha="center", va="center",
                    fontproperties=_fp(10), color="white", fontweight="bold")
            # 主导L3情感
            l3_vc = sub["l3_c3"].value_counts()
            top_l3 = l3_vc.idxmax() if len(l3_vc) > 0 else ""
            ax.text(j + 0.5, n_poets - 1 - i + 0.2, L3_ZH.get(top_l3, ""),
                    ha="center", va="center",
                    fontproperties=_fp(7.5), color="white", alpha=0.9)

    ax.set_xlim(0, n_flowers)
    ax.set_ylim(0, n_poets)

    ax.set_xticks([j + 0.5 for j in range(n_flowers)])
    ax.set_xticklabels(all_flowers_ordered,
                        fontproperties=_fp(9), rotation=45, ha="right")
    ax.set_yticks([n_poets - 1 - i + 0.5 for i in range(n_poets)])
    ax.set_yticklabels(
        [f"{p}（{poet_totals[p]}首）" for p in top10_poets],
        fontproperties=_fp(10))

    for spine in ax.spines.values():
        spine.set_visible(False)

    # 图例：L1情感色
    l1_order = ["Positive", "Implicit Positive", "Neutral",
                 "Implicit Negative", "Negative"]
    l1_zh = {
        "Positive": "明确正向", "Implicit Positive": "隐含正向",
        "Neutral": "中性", "Implicit Negative": "隐含负向",
        "Negative": "明确负向",
    }
    handles = [mpatches.Patch(color=L1_COLORS[l], label=l1_zh[l])
               for l in l1_order]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04),
               prop=_fp(9), framealpha=0.9)

    fig.suptitle("Top 10 诗人 × 花种  ·  数字=诗数  ·  颜色=L1情感极性  ·  小字=主导L3情感",
                 fontproperties=_fp(12), y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 汇总网格图已保存: {save_path}")


def main():
    df = pd.read_csv(DATA)
    top10_poets = df["作者"].value_counts().head(10).index.tolist()

    print("Top 10 诗人:", top10_poets)

    # 每位诗人详细图
    for poet in top10_poets:
        sub = df[df["作者"] == poet].copy()
        save_path = OUTPUT / f"poet_{poet}.png"
        print(f"  生成 {poet}（{len(sub)}首）…")
        plot_poet(poet, sub, save_path)
        print(f"    ✓ {save_path.name}")

    # 汇总网格图
    plot_summary_grid(df, top10_poets, OUTPUT / "poet_summary_grid.png")


if __name__ == "__main__":
    main()
