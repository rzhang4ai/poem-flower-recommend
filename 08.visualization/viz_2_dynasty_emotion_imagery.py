"""
可视化2：Top 20 花 × 朝代的情感与意象词变化。

布局：每种花一行，分两个面板：
  左：各朝代情感分布堆叠柱（L3细粒度，10色）
  右：各朝代 Top5 高频意象词（文字标注）

朝代合并策略：魏晋南北朝 / 隋唐 / 宋金 / 元 / 明 / 清
最少3首才显示数据（<3首用"数据不足"灰色标注）

输出：
  output/dynasty_emotion_imagery_<花名>.png  每种花单张
  output/dynasty_overview.png               概览热力图（格内色=主导 L1 五极性；若该花在所有有效朝代 L3 均为「赞美」，则格内文字同时显示该朝代 L3 第2、第3 名）
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "03.final_labels/poems_structured_shangxi_wip.csv"
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)
FONT_PATH = "/System/Library/Fonts/STHeiti Light.ttc"

# ── 常量 ──────────────────────────────────────────────────────────────────────
DYNASTY_ORDER = ["魏晋南北朝", "隋唐", "宋金", "元", "明", "清"]
DYNASTY_MAP = {
    "魏": "魏晋南北朝", "晋": "魏晋南北朝",
    "南朝宋": "魏晋南北朝", "南朝齐": "魏晋南北朝",
    "南朝梁": "魏晋南北朝", "南朝陈": "魏晋南北朝", "北朝周": "魏晋南北朝",
    "隋": "隋唐", "唐": "隋唐",
    "五代": "宋金", "宋": "宋金", "金": "宋金",
    "元": "元", "明": "明", "清": "清",
}

L3_LABELS_ZH = {
    "praise": "赞美", "like": "喜爱", "joy": "喜悦",
    "ease": "闲适", "faith": "信念",
    "sorrow": "悲伤", "miss": "思念", "misgive": "忧虑",
    "criticize": "讽刺", "vexed": "烦闷",
}
L3_ORDER = ["praise", "like", "joy", "ease", "faith",
            "sorrow", "miss", "misgive", "criticize", "vexed"]
L3_COLORS = {
    "praise": "#E67E22", "like": "#F39C12", "joy": "#F1C40F",
    "ease": "#2ECC71",   "faith": "#27AE60",
    "sorrow": "#3498DB", "miss": "#2980B9", "misgive": "#1ABC9C",
    "criticize": "#E74C3C", "vexed": "#C0392B",
}
MIN_POEMS = 3  # 不足此数则不显示

# L1 五极性配色（与产品情感图一致：积极黄 / 隐性积极浅绿 / 中性灰 / 隐性消极紫 / 消极深褐）
# 热力图数值编码 0..4 与下列顺序一致
L1_HEATMAP_ORDER: list[tuple[str, str, str]] = [
    ("Negative", "消极", "#3E2723"),              # 深褐/近黑
    ("Implicit Negative", "隐性消极", "#7E57C2"),  # 紫
    ("Neutral", "中性", "#9E9E9E"),               # 灰
    ("Implicit Positive", "隐性积极", "#A8E6CF"),  # 浅绿（图示含粉绿意象）
    ("Positive", "积极", "#FFEB3B"),              # 亮黄
]
L1_POLARITY_TO_CODE = {name: i for i, (name, _, _) in enumerate(L1_HEATMAP_ORDER)}
L1_OVERVIEW_CMAP = ListedColormap([c for _, _, c in L1_HEATMAP_ORDER])
L1_OVERVIEW_NORM = BoundaryNorm(
    np.linspace(-0.5, 4.5, 6), ncolors=len(L1_HEATMAP_ORDER)
)


def get_top_imagery(sub: pd.DataFrame, topn: int = 6) -> list[str]:
    """统计该组诗的高频意象词（sxhy_raw_words）。"""
    counter: Counter = Counter()
    noise = {"花", "风", "月", "春", "云", "天", "日", "水", "山", "红"}
    for raw in sub["sxhy_raw_words"].dropna():
        for w in str(raw).split("|"):
            w = w.strip()
            if w and w not in noise:
                counter[w] += 1
    return [w for w, _ in counter.most_common(topn)]


def positive_ratio(sub: pd.DataFrame) -> float:
    """计算正向情感（L1 Positive + Implicit Positive）比例。"""
    pos = sub["l1_polarity"].isin(["Positive", "Implicit Positive"]).sum()
    return pos / len(sub) if len(sub) > 0 else np.nan


# ── 每种花的详细图 ─────────────────────────────────────────────────────────────

def plot_flower_detail(flower: str, df_flower: pd.DataFrame, save_path: Path):
    try:
        fp = fm.FontProperties(fname=FONT_PATH)
    except Exception:
        fp = None

    def _prop(size=11):
        if fp:
            fp2 = fm.FontProperties(fname=FONT_PATH, size=size)
            return {"fontproperties": fp2}
        return {"fontsize": size}

    dynasties_present = [d for d in DYNASTY_ORDER
                         if (df_flower["朝代段"] == d).sum() >= MIN_POEMS]
    if not dynasties_present:
        return

    n_d = len(dynasties_present)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, n_d * 1.0 + 1.5)),
                              gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#FAFAFA")

    # ── 左：情感堆叠横柱 ─────────────────────────────────────────────────────
    ax_bar = axes[0]
    bar_data = {}
    for d in dynasties_present:
        sub = df_flower[df_flower["朝代段"] == d]
        counts = sub["l3_c3"].value_counts()
        total = counts.sum()
        bar_data[d] = {em: counts.get(em, 0) / total for em in L3_ORDER}

    lefts = np.zeros(len(dynasties_present))
    yticks = range(len(dynasties_present))
    for em in L3_ORDER:
        vals = [bar_data[d][em] for d in dynasties_present]
        bars = ax_bar.barh(list(yticks), vals, left=lefts,
                            color=L3_COLORS[em], label=L3_LABELS_ZH[em],
                            height=0.6)
        lefts += np.array(vals)

    ax_bar.set_yticks(list(yticks))
    ax_bar.set_yticklabels(
        [f"{d}\n({(df_flower['朝代段']==d).sum()}首)" for d in dynasties_present],
        **_prop(10))
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("情感占比", **_prop(10))
    ax_bar.set_title(f"{flower} · 各朝代情感分布", **_prop(12))
    ax_bar.axvline(0.5, color="#CCCCCC", lw=0.8, ls="--")
    ax_bar.set_facecolor("#F8F9FA")

    # 在柱子内标注最大情感类别
    for i, d in enumerate(dynasties_present):
        sub = df_flower[df_flower["朝代段"] == d]
        top_em = sub["l3_c3"].value_counts().idxmax() if len(sub) > 0 else ""
        ax_bar.text(0.01, i, L3_LABELS_ZH.get(top_em, ""), va="center",
                    color="white", fontweight="bold", **_prop(9))

    # ── 右：意象词文字列表 ────────────────────────────────────────────────────
    ax_txt = axes[1]
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(-0.5, len(dynasties_present) - 0.5)
    ax_txt.axis("off")
    ax_txt.set_title(f"{flower} · 各朝代 Top 意象词", **_prop(12))
    ax_txt.set_facecolor("#F8F9FA")

    for i, d in enumerate(dynasties_present):
        sub = df_flower[df_flower["朝代段"] == d]
        words = get_top_imagery(sub, topn=8)

        # 构建情感强调词（confirmed_imagery）
        confirmed = set()
        for c in sub["confirmed_imagery"].dropna():
            confirmed.update(w.strip() for w in str(c).split("|"))

        y_pos = len(dynasties_present) - 1 - i
        ax_txt.axhline(y_pos + 0.45, color="#DDDDDD", lw=0.5)

        # 正向情感占比标注
        pos_r = positive_ratio(sub)
        bar_color = "#E67E22" if pos_r >= 0.5 else "#3498DB"
        ax_txt.barh(y_pos, pos_r, height=0.25, left=0,
                     color=bar_color, alpha=0.25, zorder=0)

        # 朝代名称
        ax_txt.text(-0.02, y_pos, d, va="center", ha="right",
                    color="#555555", **_prop(9))

        # 意象词（双重确认词加粗+橙色）
        x_cur = 0.04
        for w in words:
            is_confirmed = w in confirmed
            color = "#C0392B" if is_confirmed else "#2C3E50"
            weight = "bold" if is_confirmed else "normal"
            txt_obj = ax_txt.text(x_cur, y_pos, w, va="center",
                                   color=color, fontweight=weight, **_prop(10))
            x_cur += len(w) * 0.065 + 0.04
            if x_cur > 0.95:
                break

    # ── 图例 ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=L3_COLORS[em], label=L3_LABELS_ZH[em])
        for em in L3_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04),
               prop=fm.FontProperties(fname=FONT_PATH, size=9) if fp else None)

    note = ax_txt.text(
        0.5, -0.42,
        "红色粗体 = BERT×诗学含英双重确认意象",
        ha="center", va="center", color="#C0392B", **_prop(8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── 概览热力图 ─────────────────────────────────────────────────────────────────

def _dominant_l1_code(sub: pd.DataFrame) -> float:
    s = sub["l1_polarity"].dropna()
    if len(s) == 0:
        return np.nan
    top = s.value_counts().idxmax()
    return float(L1_POLARITY_TO_CODE.get(top, np.nan))


def _l3_top_labels(sub: pd.DataFrame, k: int = 3) -> list[str]:
    vc = sub["l3_c3"].dropna().value_counts()
    out: list[str] = []
    for em in list(vc.index)[:k]:
        out.append(L3_LABELS_ZH.get(str(em), str(em)))
    return out


def _annot_text_color(l1_code: float) -> str:
    if np.isnan(l1_code):
        return "#333333"
    # 深褐、紫底用白字；灰/浅绿/黄底用深字
    if l1_code <= 1.0:
        return "#FFFFFF"
    return "#212121"


def plot_overview_heatmap(df: pd.DataFrame, top20: list[str], save_path: Path):
    try:
        fp = fm.FontProperties(fname=FONT_PATH, size=11)
    except Exception:
        fp = None

    n_rows, n_cols = len(top20), len(DYNASTY_ORDER)
    matrix = np.full((n_rows, n_cols), np.nan)
    annot = np.empty((n_rows, n_cols), dtype=object)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    # 每个有效格的主导 L3（用于判断是否「全朝代赞美」）
    dom_l3 = np.empty((n_rows, n_cols), dtype=object)

    for i, flower in enumerate(top20):
        for j, dynasty in enumerate(DYNASTY_ORDER):
            sub = df[(df["花名"] == flower) & (df["朝代段"] == dynasty)]
            n = len(sub)
            counts[i, j] = n
            if n >= MIN_POEMS:
                matrix[i, j] = _dominant_l1_code(sub)
                top_em = sub["l3_c3"].dropna().value_counts()
                dom_l3[i, j] = top_em.index[0] if len(top_em) > 0 else None
                imagery = get_top_imagery(sub, topn=3)
                em_line = L3_LABELS_ZH.get(dom_l3[i, j], dom_l3[i, j] or "")
                annot[i, j] = f"{em_line}\n{' '.join(imagery)}"
            else:
                dom_l3[i, j] = None
                annot[i, j] = f"n={n}" if n > 0 else ""

    # 若某花在所有「有效朝代」格内主导 L3 均为 praise，则格内显示 L3 前三名（削弱仅写「赞美」的干扰）
    for i, flower in enumerate(top20):
        valid_js = [j for j in range(n_cols) if counts[i, j] >= MIN_POEMS]
        if not valid_js:
            continue
        if not all(dom_l3[i, j] == "praise" for j in valid_js):
            continue
        for j in valid_js:
            sub = df[(df["花名"] == flower) & (df["朝代段"] == DYNASTY_ORDER[j])]
            labels = _l3_top_labels(sub, k=3)
            imagery = get_top_imagery(sub, topn=3)
            em_line = " · ".join(labels) if labels else ""
            annot[i, j] = f"{em_line}\n{' '.join(imagery)}"

    fig, ax = plt.subplots(figsize=(13, 12))
    fig.patch.set_facecolor("#FAFAFA")

    cmap = L1_OVERVIEW_CMAP.copy()
    cmap.set_bad("#EEEEEE")

    im = ax.imshow(matrix, cmap=cmap, norm=L1_OVERVIEW_NORM, aspect="auto")

    ax.set_xticks(range(len(DYNASTY_ORDER)))
    ax.set_yticks(range(len(top20)))
    ax.set_xticklabels(DYNASTY_ORDER,
                        fontproperties=fp if fp else None, fontsize=11)
    ax.set_yticklabels(
        [f"{f}（{(df['花名']==f).sum()}首）" for f in top20],
        fontproperties=fp if fp else None, fontsize=10)

    for i in range(len(top20)):
        for j in range(len(DYNASTY_ORDER)):
            text = annot[i, j]
            if text:
                val = matrix[i, j]
                txt_color = _annot_text_color(val)
                ax.text(j, i, text, ha="center", va="center",
                        color=txt_color, fontsize=7.5,
                        fontproperties=fm.FontProperties(fname=FONT_PATH, size=7.5) if fp else None)

    cbar = fig.colorbar(
        im, ax=ax, fraction=0.03, pad=0.02,
        ticks=[0, 1, 2, 3, 4],
    )
    cbar.ax.set_yticklabels(
        [zh for _, zh, _ in L1_HEATMAP_ORDER],
        fontproperties=fp,
    )
    cbar.set_label("格内底色 = 该格主导 L1 五极性", fontproperties=fp)

    ax.set_title(
        "Top 20 花卉 × 朝代 · L1 五极性（底色）+ 主导 L3 + Top3 意象词\n"
        "若该花在所有有效朝代主导 L3 均为「赞美」，则显示该朝代 L3 前三名；灰格=诗数不足3首",
        fontproperties=fm.FontProperties(fname=FONT_PATH, size=12) if fp else None,
        pad=12,
    )

    # 朝代分割线
    for j in range(1, len(DYNASTY_ORDER)):
        ax.axvline(j - 0.5, color="#888888", lw=0.5)
    for i in range(1, len(top20)):
        ax.axhline(i - 0.5, color="#CCCCCC", lw=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 概览热力图已保存: {save_path}")


def main():
    df = pd.read_csv(DATA)
    df["朝代段"] = df["朝代"].map(DYNASTY_MAP)
    top20 = df["花名"].value_counts().head(20).index.tolist()

    # 每种花详细图
    for flower in top20:
        sub = df[df["花名"] == flower].copy()
        dynasties_present = [d for d in DYNASTY_ORDER
                             if (sub["朝代段"] == d).sum() >= MIN_POEMS]
        if len(dynasties_present) < 2:
            print(f"  {flower}: 有效朝代段不足2个，跳过详细图")
            continue
        save_path = OUTPUT / f"dynasty_{flower}.png"
        print(f"  生成 {flower}…")
        plot_flower_detail(flower, sub, save_path)
        print(f"    ✓ {save_path.name}")

    # 概览热力图
    plot_overview_heatmap(df, top20, OUTPUT / "dynasty_overview.png")


if __name__ == "__main__":
    main()
