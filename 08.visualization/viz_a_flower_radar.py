"""
课题 A：花卉 8 维语义画像雷达图。
诗数 >= min_count 的花各一张子图；另输出一张 Top-N 花合并对比图。

用法（在项目根目录，激活 flower_env）：
  python 04.visualization/viz_a_flower_radar.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA_WITH_DIMS, OUTPUT_DIR

DIM_LABELS_ZH = ["自然天象", "时令节序", "地理空间", "人物关系", "品格志向", "器物生活", "动植物", "人文社交"]
DIM_COLS = [
    "dim_nature",
    "dim_season",
    "dim_space",
    "dim_people",
    "dim_virtue",
    "dim_artifact",
    "dim_biota",
    "dim_social",
]


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


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """按行 L1 归一化，避免量纲差异；全零行保持为零。"""
    s = mat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return mat / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=20, help="最少诗数才入图")
    ap.add_argument("--top", type=int, default=12, help="合并对比图取诗数最多的前 N 种花")
    args = ap.parse_args()

    _setup_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_WITH_DIMS)
    counts = df.groupby("花名").size()
    flowers = counts[counts >= args.min_count].index.tolist()
    flowers.sort(key=lambda f: (-counts[f], f))

    if not flowers:
        raise SystemExit(f"没有诗数 >= {args.min_count} 的花，请降低 --min-count")

    # 每种花：均值向量 -> L1 归一化
    agg = df[df["花名"].isin(flowers)].groupby("花名")[DIM_COLS].mean()
    mat = agg.values.astype(float)
    mat_n = _normalize_rows(mat)

    n = len(flowers)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), subplot_kw=dict(projection="polar"))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(DIM_LABELS_ZH), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for ax, flower, row in zip(axes, flowers, mat_n):
        vals = np.concatenate([row, [row[0]]])
        ax.plot(angles, vals, "o-", linewidth=1.5)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_LABELS_ZH, size=8)
        ax.set_title(f"{flower}（n={int(counts[flower])}）", size=10)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out1 = OUTPUT_DIR / "viz_a_flower_radar_by_flower.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out1}")

    # Top-N 合并对比（同一坐标系多条线）
    topn = min(args.top, len(flowers))
    top_flowers = flowers[:topn]
    idx = [flowers.index(f) for f in top_flowers]
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection="polar")
    for i in idx:
        f = flowers[i]
        row = mat_n[i]
        vals = np.concatenate([row, [row[0]]])
        ax2.plot(angles, vals, "o-", linewidth=1.2, label=f"{f}({int(counts[f])})")
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(DIM_LABELS_ZH, size=9)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax2.set_title(f"花卉语义画像对比（诗数 Top {topn}，min_count={args.min_count}）", size=12)
    out2 = OUTPUT_DIR / "viz_a_flower_radar_topN.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"已保存: {out2}")


if __name__ == "__main__":
    main()
