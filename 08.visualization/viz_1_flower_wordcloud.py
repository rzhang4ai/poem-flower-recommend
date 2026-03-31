"""
可视化1：每种花的意象词云（常规矩形云形，无自定义轮廓 mask）。

规则：
  - 排除该花自身名称及常见简称（如「梅花」排除 梅花、梅；「水仙花」排除 水仙花、水仙）。
  - 词频 = 诗学含英原词出现次数 × 权重 + BERT Top10 注意力分数累计（体现频率与模型重要性）。
  - 仍过滤极泛化单字（雪、月等）以免淹没意象；可按需调整 GENERIC_STOP。

输出：
  - output/wordcloud_<花名>.png   每种花各一张（数据集中全部花名，按诗量降序）
  - output/wordcloud_grid.png    全部花合并一览（多列网格）
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

# 合并总览图列数（约 sqrt(N) 量级，79 花≈8 列×10 行）
GRID_NCOLS = 8

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "03.final_labels/poems_structured_shangxi_wip.csv"
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

FONT_PATH = "/System/Library/Fonts/STHeiti Light.ttc"

# 泛化程度极高的字，保留少量可突出更具体的意象词（可按研究需要删减）
GENERIC_STOP = {
    "花", "风", "月", "春", "云", "天", "日", "水", "山",
    "红", "绿", "白", "香", "雨", "露", "叶", "枝",
}

# 诗学含英原词权重（相对 BERT 分数量级，与 SCORE_SCALE 配合）
W_SXHY_COUNT = 8.0
# BERT 注意力分数乘数，使与 sxhy 同量级可比
SCORE_SCALE = 120.0

# 除完整花名外，额外排除的别称（键=数据集花名列值）
EXTRA_SELF_TOKENS: dict[str, set[str]] = {
    "木芙蓉": {"芙蓉"},
    "红梅": {"梅"},
    "荷花": {"荷"},
    "莲花": {"莲"},
    "酴醿": {"酴", "醿"},
    "虞美人": set(),  # 仅排除四字全名，避免误伤「美人」等
}

# ── 花卉配色 ───────────────────────────────────────────────────────────────────
FLOWER_COLORS: dict[str, list[str]] = {
    "梅花":   ["#C0392B", "#922B21", "#E74C3C", "#F1948A", "#7B241C"],
    "菊花":   ["#D4AC0D", "#B7950B", "#F4D03F", "#A9770E", "#F0C040"],
    "牡丹":   ["#CB4335", "#A93226", "#E91E8C", "#C0392B", "#F8BBD9"],
    "海棠":   ["#E91E8C", "#AD1457", "#F48FB1", "#C2185B", "#FCE4EC"],
    "木芙蓉": ["#E67E22", "#CA6F1E", "#F39C12", "#B9770E", "#FAD7A0"],
    "桃花":   ["#E91E8C", "#E74C3C", "#F8BBD9", "#FF80AB", "#FFCDD2"],
    "杨花":   ["#7F8C8D", "#566573", "#ABB2B9", "#BDC3C7", "#5D6D7E"],
    "杏花":   ["#F5CBA7", "#DC7633", "#F0B27A", "#E59866", "#FDEBD0"],
    "荷花":   ["#C0392B", "#922B21", "#F9EBEA", "#F1948A", "#D98880"],
    "酴醿":   ["#85929E", "#5D6D7E", "#BDC3C7", "#D5D8DC", "#566573"],
    "桂花":   ["#E67E22", "#D35400", "#F39C12", "#FAD7A0", "#A04000"],
    "水仙花": ["#5DADE2", "#3498DB", "#85C1E9", "#AED6F1", "#2E86AB"],
    "虞美人": ["#CB4335", "#B03A2E", "#E74C3C", "#C0392B", "#FDEDEC"],
    "红梅":   ["#C0392B", "#922B21", "#E74C3C", "#FF1744", "#B71C1C"],
    "梨花":   ["#5D6D7E", "#85929E", "#BDC3C7", "#AAB7B8", "#D5D8DC"],
    "蔷薇":   ["#E91E8C", "#AD1457", "#F48FB1", "#C2185B", "#E040FB"],
    "莲花":   ["#C0392B", "#E91E8C", "#F48FB1", "#FCE4EC", "#880E4F"],
    "樱桃花": ["#E91E8C", "#F48FB1", "#FFCDD2", "#E57373", "#EF9A9A"],
    "兰花":   ["#7D3C98", "#6C3483", "#A569BD", "#D7BDE2", "#9B59B6"],
    "芍药":   ["#E91E8C", "#C0392B", "#F48FB1", "#EF9A9A", "#FFDDE1"],
}
DEFAULT_COLORS = ["#2E4057", "#048A81", "#54C6EB", "#1A535C", "#8EE3EF"]


def self_tokens_for_flower(flower: str) -> set[str]:
    """该花主题下需从词云中剔除的「花自身」称谓（精确匹配 token）。"""
    banned: set[str] = {flower}
    banned |= EXTRA_SELF_TOKENS.get(flower, set())
    if len(flower) >= 2 and flower.endswith("花"):
        stem = flower[:-1]
        if stem:
            banned.add(stem)
    return banned


def _parse_score_json(cell) -> list[tuple[str, float]]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(data, list):
        return []
    out: list[tuple[str, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            tok, sc = item[0], item[1]
            if isinstance(tok, str) and tok.strip():
                try:
                    out.append((tok.strip(), float(sc)))
                except (TypeError, ValueError):
                    pass
    return out


def build_word_freq(flower: str, sub: pd.DataFrame) -> dict[str, float]:
    """
    合并：
      - sxhy_raw_words：每出现一次 +W_SXHY_COUNT（意象词典命中 = 频率）
      - ccpoem_top10_with_score：按注意力分数累加 × SCORE_SCALE（重要性）
    """
    banned = self_tokens_for_flower(flower)
    counter: Counter[str] = Counter()

    for raw in sub["sxhy_raw_words"].dropna():
        for w in str(raw).split("|"):
            w = w.strip()
            if not w or w in banned or w in GENERIC_STOP:
                continue
            counter[w] += W_SXHY_COUNT

    for _, row in sub.iterrows():
        scored = _parse_score_json(row.get("ccpoem_top10_with_score"))
        if scored:
            for tok, score in scored:
                if not tok or tok in banned or tok in GENERIC_STOP:
                    continue
                counter[tok] += score * SCORE_SCALE
            continue
        cell10 = row.get("ccpoem_top10")
        if pd.notna(cell10):
            for tok in str(cell10).split():
                tok = tok.strip()
                if not tok or tok in banned or tok in GENERIC_STOP:
                    continue
                counter[tok] += SCORE_SCALE * 0.08

    return dict(counter)


def color_func_for(flower: str):
    colors = FLOWER_COLORS.get(flower, DEFAULT_COLORS)

    def _cf(word, font_size, position, orientation, random_state=None, **kwargs):
        return colors[random_state.randint(0, len(colors) - 1)]

    return _cf


def _placeholder_image(flower: str, n_poems: int) -> Image.Image:
    """无可用词时生成与词云同尺寸的占位图。"""
    w, h = 900, 600
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 36)
        font_small = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = font_small = ImageFont.load_default()
    msg = f"{flower}"
    subline = f"（{n_poems}首 · 无可用意象词）"
    b1 = draw.textbbox((0, 0), msg, font=font)
    tw1, th1 = b1[2] - b1[0], b1[3] - b1[1]
    b2 = draw.textbbox((0, 0), subline, font=font_small)
    tw2, th2 = b2[2] - b2[0], b2[3] - b2[1]
    gap = 12
    total_h = th1 + gap + th2
    y0 = (h - total_h) // 2
    draw.text(((w - tw1) // 2, y0), msg, fill="#7F8C8D", font=font)
    draw.text(((w - tw2) // 2, y0 + th1 + gap), subline, fill="#BDC3C7", font=font_small)
    return img


def make_flower_wordcloud(flower: str, sub: pd.DataFrame, save_path: Path):
    freq = build_word_freq(flower, sub)
    n_poems = len(sub)
    if not freq:
        img = _placeholder_image(flower, n_poems)
        img.save(str(save_path))
        return img

    wc = WordCloud(
        font_path=FONT_PATH,
        width=900,
        height=600,
        background_color="white",
        max_words=180,
        relative_scaling=0.45,
        min_font_size=10,
        prefer_horizontal=0.88,
        colormap=None,
        color_func=color_func_for(flower),
        margin=4,
    )
    wc.generate_from_frequencies(freq)
    wc.to_file(str(save_path))
    return wc.to_image()


def main():
    df = pd.read_csv(DATA)
    # 全部花名，按诗量降序（与数据集中出现频次一致）
    all_flowers = df["花名"].value_counts().index.tolist()
    n_species = len(all_flowers)

    images: list[tuple[str, object]] = []
    for flower in all_flowers:
        sub = df[df["花名"] == flower]
        save_path = OUTPUT / f"wordcloud_{flower}.png"
        print(f"  生成 {flower} ({len(sub)}首)…")
        img = make_flower_wordcloud(flower, sub, save_path)
        images.append((flower, img))

    ncols = GRID_NCOLS
    nrows = int(np.ceil(n_species / ncols))
    fig_w = ncols * 3.6
    fig_h = nrows * 2.85
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FAFAFA")

    ax_flat = np.atleast_1d(axes).ravel()

    try:
        title_font = fm.FontProperties(fname=FONT_PATH, size=9)
    except Exception:
        title_font = None

    for idx, (flower, img) in enumerate(images):
        ax = ax_flat[idx]
        ax.imshow(img)
        ax.axis("off")
        count = len(df[df["花名"] == flower])
        title_kwargs = {"fontproperties": title_font} if title_font else {}
        ax.set_title(f"{flower}（{count}）", fontsize=9, pad=3, **title_kwargs)

    for idx in range(len(images), len(ax_flat)):
        ax_flat[idx].axis("off")

    try:
        suptitle_fp = fm.FontProperties(fname=FONT_PATH, size=14)
    except Exception:
        suptitle_fp = None
    plt.suptitle(
        f"全部 {n_species} 种花 · 意象词云（已排除花名；词重=诗学含英频次+BERT注意力）",
        fontsize=14,
        y=1.002,
        fontproperties=suptitle_fp,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    grid_path = OUTPUT / "wordcloud_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✓ 共 {n_species} 张词云 + 合并图: {grid_path}")


if __name__ == "__main__":
    main()
