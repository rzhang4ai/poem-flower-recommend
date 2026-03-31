"""
extract_sxhy_catalog_categories.py
=================================
将《诗学含英》目录文本解析为结构化类别表，输出到 step2c_imagery/output。

默认输入：
  /Volumes/aiworkbench/datasets/04_structured/book_105/doubao_pages/merged_simp目录.txt

输出：
  output/sxhy_catalog_category_table.csv
  output/sxhy_catalog_category_summary.txt

说明：
  - category_level: 卷级大类 / 类别
  - emotion_hint: 基于关键词的情感倾向初判（positive / negative / neutral / mixed）
    该列用于人工校对前的粗筛，不替代人工标注。
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TXT = Path("/Volumes/aiworkbench/datasets/04_structured/book_105/doubao_pages/merged_simp目录.txt")
OUT_CSV = OUT_DIR / "sxhy_catalog_category_table.csv"
OUT_SUMMARY = OUT_DIR / "sxhy_catalog_category_summary.txt"

VOL_RE = re.compile(r"卷[一二三四五六七八九十百]+")
TOKEN_RE = re.compile(r"^[\u4e00-\u9fff0-9]+$")

# 规则法：情感倾向关键词（可后续扩展）
POS_KEYS = {"喜", "乐", "庆", "贺", "寿", "祥", "瑞", "福", "吉", "春", "晴"}
NEG_KEYS = {"悲", "哀", "怨", "愁", "病", "丧", "吊", "伤", "苦", "寒", "凶", "悼", "亡"}


def clean_header(raw: str) -> str:
    s = raw.strip()
    s = s.lstrip("#").strip()
    return s


def infer_emotion_hint(text: str) -> str:
    pos_hit = any(k in text for k in POS_KEYS)
    neg_hit = any(k in text for k in NEG_KEYS)
    if pos_hit and neg_hit:
        return "mixed"
    if pos_hit:
        return "positive"
    if neg_hit:
        return "negative"
    return "neutral"


def parse_catalog(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"输入目录文件不存在: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    current_volume = ""
    current_top = ""
    current_category = ""
    rows = []

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        if "诗学含英目录" in line:
            continue

        if line.startswith("#"):
            header = clean_header(line)
            if not header:
                continue

            vol_m = VOL_RE.search(header)
            if vol_m:
                current_volume = vol_m.group(0)

            # 去掉卷标记后保留剩余标题
            header_wo_vol = VOL_RE.sub("", header).strip()
            if header_wo_vol:
                current_top = header_wo_vol
                # 若本身就是“xx类”，同时视为当前类别
                if header_wo_vol.endswith("类"):
                    current_category = header_wo_vol
                else:
                    # 例如 "# 卷三" 下一行再给 "游眺类"
                    current_category = ""
            continue

        # 非标题行：优先识别“xxx类”单行作为当前类别
        if line.endswith("类") and " " not in line and len(line) <= 8:
            current_category = line
            continue

        # 目录条目行：按空白切分
        tokens = [t.strip() for t in line.split() if t.strip()]
        tokens = [t for t in tokens if TOKEN_RE.match(t)]
        if not tokens:
            continue

        for token in tokens:
            rows.append({
                "volume": current_volume,
                "top_header": current_top,
                "category": current_category if current_category else current_top,
                "entry": token,
                "entry_len": len(token),
                "emotion_hint": infer_emotion_hint(token),
                "source_line": i,
            })

    return pd.DataFrame(rows)


def main():
    df = parse_catalog(INPUT_TXT)
    if df.empty:
        raise RuntimeError("解析结果为空，请检查输入文本格式。")

    df = df.drop_duplicates(subset=["volume", "category", "entry"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    by_cat = df.groupby(["volume", "category"])["entry"].count().reset_index(name="entry_count")
    emo_cnt = Counter(df["emotion_hint"])

    lines = [
        "=" * 68,
        "诗学含英目录类别表提取报告",
        "=" * 68,
        f"输入文件: {INPUT_TXT}",
        f"总条目数: {len(df)}",
        f"卷数: {df['volume'].nunique()}",
        f"类别数: {df['category'].nunique()}",
        "",
        "情感倾向初判统计:",
        f"  positive: {emo_cnt.get('positive', 0)}",
        f"  negative: {emo_cnt.get('negative', 0)}",
        f"  mixed:    {emo_cnt.get('mixed', 0)}",
        f"  neutral:  {emo_cnt.get('neutral', 0)}",
        "",
        "各类别条目数 Top-20:",
    ]

    top20 = by_cat.sort_values("entry_count", ascending=False).head(20)
    for _, r in top20.iterrows():
        lines.append(f"  {r['volume']:<4} {r['category']:<12} {int(r['entry_count']):>4}")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ 已输出: {OUT_CSV}")
    print(f"✅ 已输出: {OUT_SUMMARY}")
    print(f"  总条目: {len(df)}  类别: {df['category'].nunique()}")


if __name__ == "__main__":
    main()
