"""
课题 F：按「月份数字」与花名的极坐标散点（角度=月份，半径=log(1+诗数)），
颜色表示该组合下 L1 主导极性（取该组内众数）。

用法：
  python 04.visualization/viz_f_month_polar.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA_WITH_DIMS, OUTPUT_DIR

L1_ORDER = ["Negative", "Implicit Negative", "Neutral", "Implicit Positive", "Positive"]
L1_ZH = ["负面", "隐性负面", "中性", "隐性积极", "积极"]


def _setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cell-count", type=int, default=3, help="花×月格子最少诗数才打点")
    args = ap.parse_args()

    _setup_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_WITH_DIMS)
    if "月份数字" not in df.columns or df["月份数字"].isna().all():
        raise SystemExit("缺少有效的 月份数字 列，请先运行 03.final_labels/prepare_dimensions.py")

    df = df.dropna(subset=["月份数字"])
    df["月份数字"] = df["月份数字"].astype(int)

    g = df.groupby(["花名", "月份数字"])
    rows = []
    for (flower, mth), sub in g:
        c = len(sub)
        if c < args.min_cell_count:
            continue
        mode = sub["l1_polarity"].mode()
        l1 = mode.iloc[0] if len(mode) else "Neutral"
        rows.append(
            {
                "花名": flower,
                "月份数字": mth,
                "count": c,
                "l1_polarity": l1,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise SystemExit("没有满足 min_cell_count 的花×月组合，请降低阈值")

    # 角度：(月份-1)/12 * 2pi，正月=1 -> 0
    plot_df["theta"] = (plot_df["月份数字"] - 1) / 12.0 * 2 * np.pi
    plot_df["r"] = np.log1p(plot_df["count"])

    color_map = {p: f"C{i}" for i, p in enumerate(L1_ORDER)}
    colors = plot_df["l1_polarity"].map(lambda x: color_map.get(x, "0.5"))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    ax.scatter(
        plot_df["theta"],
        plot_df["r"],
        c=[color_map.get(p, "gray") for p in plot_df["l1_polarity"]],
        s=plot_df["count"] * 3,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )

    month_labels = [f"{i}月" for i in range(1, 13)]
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(month_labels)
    ax.set_title("花卉×月份 分布（半径=log(1+诗数)，颜色=L1 情感众数）", pad=20)

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=color_map[p], edgecolor="k", label=f"{p} / {L1_ZH[i]}") for i, p in enumerate(L1_ORDER)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.15, 1.05), fontsize=8)

    # 旁注：点太多时标注花名会糊，仅保存散点；需要时可导出 CSV
    out_png = OUTPUT_DIR / "viz_f_month_polar_scatter.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_png}")

    csv_out = OUTPUT_DIR / "viz_f_month_polar_points.csv"
    plot_df.drop(columns=["theta", "r"], errors="ignore").to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"已保存: {csv_out}")


if __name__ == "__main__":
    main()
