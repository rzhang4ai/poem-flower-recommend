"""
digi_sxhy_pipeline.py
=====================
《诗学含英》数字化本（digi-sxhy）处理流水线：

1. 从 merged_simp目录.txt 解析核心意象分类结构 → CSV + 摘要
2. 合并 卷一 / 卷二三 / 卷四 clean 为 merged_simp_full.txt（统一词典正文）
3. 解析合并词典，对 poems_dataset_merged_done.csv 做最长优先匹配，输出每首诗命中意象

依赖路径（相对项目根）：
  literature review/digi-sxhy/merged_simp目录.txt
  literature review/digi-sxhy/merged_simp_卷一clean.txt
  literature review/digi-sxhy/merged_simp_卷二三clean.txt
  literature review/digi-sxhy/merged_simp_卷四clean.txt
  00.poems_dataset/poems_dataset_merged_done.csv

输出（step2c_imagery/output/）：
  digi_sxhy_catalog_table.csv
  digi_sxhy_catalog_summary.txt
  digi_sxhy_vocab.csv
  digi_sxhy_imagery_per_poem.csv
  digi_sxhy_imagery_frequency.csv
  digi_sxhy_build_report.txt

词典正文输出：
  literature review/digi-sxhy/merged_simp_full.txt
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ── 根路径 ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent.parent
DIGI_DIR = ROOT / "literature review" / "digi-sxhy"

CATALOG_TXT = DIGI_DIR / "merged_simp目录.txt"
CLEAN_PARTS = [
    DIGI_DIR / "merged_simp_卷一clean.txt",
    DIGI_DIR / "merged_simp_卷二三clean.txt",
    DIGI_DIR / "merged_simp_卷四clean.txt",
]
MERGED_LEXICON = DIGI_DIR / "merged_simp_full.txt"
POEMS_CSV = ROOT / "00.poems_dataset" / "poems_dataset_merged_done.csv"

OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOL_RE = re.compile(r"卷[一二三四五六七八九十百]+")
TOKEN_RE = re.compile(r"^[\u4e00-\u9fff0-9]+$")
_PURE_HAN = re.compile(r"^[\u4e00-\u9fff]+$")

POS_KEYS = {"喜", "乐", "庆", "贺", "寿", "祥", "瑞", "福", "吉", "春", "晴"}
NEG_KEYS = {"悲", "哀", "怨", "愁", "病", "丧", "吊", "伤", "苦", "寒", "凶", "悼", "亡"}


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


def clean_header(raw: str) -> str:
    s = raw.strip().lstrip("#").strip()
    return s


def parse_catalog(path: Path) -> pd.DataFrame:
    """
    解析 merged_simp目录.txt：卷 / 大类 / 目录行词条 → 结构化表。
    逻辑与 extract_sxhy_catalog_categories.py 对齐，适配 #卷二 等无空格写法。
    """
    if not path.exists():
        raise FileNotFoundError(f"目录文件不存在: {path}")

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
            header_wo_vol = VOL_RE.sub("", header).strip()
            if header_wo_vol:
                current_top = header_wo_vol
                if header_wo_vol.endswith("类"):
                    current_category = header_wo_vol
                else:
                    current_category = ""
            continue

        if line.endswith("类") and " " not in line and len(line) <= 12:
            current_category = line
            continue

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

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["volume", "category", "entry"]).reset_index(drop=True)


def merge_clean_files(paths: list[Path], out_path: Path) -> None:
    chunks = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"缺失 clean 文件: {p}")
        text = p.read_text(encoding="utf-8")
        chunks.append(f"\n\n# ====== 来源: {p.name} ======\n\n")
        chunks.append(text)
    out_path.write_text("".join(chunks), encoding="utf-8")


def is_skip_line(term: str) -> bool:
    return "诗学含英" in term or "诗 学 含 英" in term


def is_category_header(term: str) -> bool:
    return term.endswith("类")


def parse_sxhy_lexicon(path: Path) -> tuple[list[dict], dict[str, str]]:
    """
    解析合并后的词典正文（与 shixuehanying_imagery_extract.parse_sxhy 一致）。
    返回 headings 列表 + 子词→父标题映射。
    """
    headings: list[dict] = []
    subterm_map: dict[str, str] = {}

    current_cat = ""
    current_h1 = ""
    current_head = ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if is_skip_line(line):
            continue
        if line.startswith("# ======"):
            continue

        if line.startswith("## "):
            term = line[3:].strip()
            if not term or not _PURE_HAN.match(term):
                continue
            headings.append({
                "term": term, "level": 2, "category": current_cat, "parent_h1": current_h1,
            })
            current_head = term
        elif line.startswith("# "):
            raw_term = line[2:].strip()
            if not raw_term:
                continue
            if is_category_header(raw_term):
                current_cat = raw_term
                current_h1 = ""
                current_head = ""
            else:
                parts = [p for p in raw_term.split() if _PURE_HAN.match(p)]
                if not parts:
                    continue
                for term in parts:
                    headings.append({
                        "term": term, "level": 1, "category": current_cat, "parent_h1": "",
                    })
                current_h1 = parts[-1]
                current_head = parts[-1]
        else:
            if current_head:
                for ph in re.findall(r"[\u4e00-\u9fff]{2,5}", line):
                    if ph not in subterm_map:
                        subterm_map[ph] = current_head

    return headings, subterm_map


def build_match_vocab(headings: list[dict], subterm_map: dict[str, str]) -> dict[str, str]:
    vocab: dict[str, str] = {}
    for h in headings:
        t = h["term"]
        vocab[t] = t
    for ph, parent in subterm_map.items():
        if ph not in vocab:
            vocab[ph] = parent
    return vocab


def extract_imagery(
    poem_text: str,
    vocab: dict[str, str],
    single_char_terms: set[str],
    multi_char_terms: set[str],
) -> list[tuple[str, str]]:
    if not isinstance(poem_text, str):
        return []
    text = re.sub(r"[^\u4e00-\u9fff]", "", poem_text)
    n = len(text)
    if n == 0:
        return []

    hits = []
    covered = [False] * n
    i = 0
    while i < n:
        best_len = 0
        best_raw = None
        for length in range(min(5, n - i), 1, -1):
            chunk = text[i : i + length]
            if chunk in multi_char_terms:
                best_len = length
                best_raw = chunk
                break
        if best_raw:
            hits.append((best_raw, vocab[best_raw]))
            for k in range(i, i + best_len):
                covered[k] = True
            i += best_len
        else:
            i += 1

    for i, ch in enumerate(text):
        if not covered[i] and ch in single_char_terms:
            hits.append((ch, vocab[ch]))

    return hits


def heading_to_catalog_info(heading: str, catalog_df: pd.DataFrame) -> dict[str, str]:
    """将父标题意象（目录 leaf entry）映射到卷/类。"""
    if catalog_df.empty or not heading:
        return {"volume": "", "category": "", "top_header": ""}
    m = catalog_df[catalog_df["entry"] == heading]
    if m.empty:
        return {"volume": "", "category": "", "top_header": ""}
    r = m.iloc[0]
    return {
        "volume": str(r.get("volume", "")),
        "category": str(r.get("category", "")),
        "top_header": str(r.get("top_header", "")),
    }


def main() -> None:
    report_lines: list[str] = []

    # ── 1. 目录 ───────────────────────────────────────────────────
    if not CATALOG_TXT.exists():
        raise FileNotFoundError(CATALOG_TXT)
    cat_df = parse_catalog(CATALOG_TXT)
    cat_csv = OUT_DIR / "digi_sxhy_catalog_table.csv"
    cat_df.to_csv(cat_csv, index=False, encoding="utf-8-sig")
    report_lines.append(f"[1] 目录解析: {len(cat_df)} 条 → {cat_csv.name}")

    summary_lines = [
        "=" * 60,
        "《诗学含英》digi-sxhy 目录结构摘要",
        "=" * 60,
        f"输入: {CATALOG_TXT}",
        f"条目数: {len(cat_df)}",
    ]
    if not cat_df.empty:
        summary_lines.append(f"卷: {cat_df['volume'].nunique()}  类别数: {cat_df['category'].nunique()}")
        by_cat = (
            cat_df.groupby(["volume", "category"])["entry"]
            .count()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        summary_lines.extend(["", "各类别条目数 Top-15:"])
        for _, r in by_cat.head(15).iterrows():
            summary_lines.append(f"  {r['volume']}\t{r['category']}\t{r['n']}")
    (OUT_DIR / "digi_sxhy_catalog_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    # ── 2. 合并 clean ─────────────────────────────────────────────
    merge_clean_files(CLEAN_PARTS, MERGED_LEXICON)
    report_lines.append(f"[2] 合并词典正文 → {MERGED_LEXICON.relative_to(ROOT)}")

    # ── 3. 解析词典 + 匹配诗歌 ────────────────────────────────────
    headings, sub_map = parse_sxhy_lexicon(MERGED_LEXICON)
    vocab = build_match_vocab(headings, sub_map)
    multi_char_terms = {t for t in vocab if len(t) > 1}
    single_char_terms = {t for t in vocab if len(t) == 1}
    pd.DataFrame(headings)[["term", "level", "category", "parent_h1"]].to_csv(
        OUT_DIR / "digi_sxhy_vocab.csv", index=False, encoding="utf-8-sig"
    )
    report_lines.append(
        f"[3] 标题词条 {len(headings)}，子词映射 {len(sub_map)}，匹配词典 {len(vocab)}"
    )

    if not POEMS_CSV.exists():
        raise FileNotFoundError(POEMS_CSV)
    poems_df = pd.read_csv(POEMS_CSV)

    per_rows = []
    for idx, row in poems_df.iterrows():
        body = row.get("正文", "")
        hits = extract_imagery(body, vocab, single_char_terms, multi_char_terms)
        raw_words = [h[0] for h in hits]
        heads = [h[1] for h in hits]
        raw_cnt = Counter(raw_words)
        head_cnt = Counter(heads)
        unique_heads = sorted(head_cnt.keys(), key=lambda x: (-head_cnt[x], x))

        cat_paths = []
        for h in unique_heads:
            info = heading_to_catalog_info(h, cat_df)
            if info["volume"]:
                cat_paths.append(
                    f"{info['volume']}/{info['category']}/{h}"
                )
            else:
                cat_paths.append(f"?/{h}")

        per_rows.append({
            "ID": row.get("ID", idx),
            "诗名": row.get("诗名", ""),
            "朝代": row.get("朝代", ""),
            "花名": row.get("花名", ""),
            "sxhy_imagery_count": len(unique_heads),
            "sxhy_imagery_headings": "|".join(unique_heads),
            "sxhy_raw_words": "|".join(sorted(raw_cnt.keys(), key=lambda x: (-raw_cnt[x], x))),
            "sxhy_imagery_freq": "|".join(f"{k}:{v}" for k, v in head_cnt.most_common()),
            "sxhy_catalog_paths": "|".join(cat_paths),
        })

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(OUT_DIR / "digi_sxhy_imagery_per_poem.csv", index=False, encoding="utf-8-sig")

    freq_cnt = Counter()
    poem_cnt = Counter()
    for _, pr in per_df.iterrows():
        if isinstance(pr["sxhy_imagery_freq"], str) and pr["sxhy_imagery_freq"]:
            for entry in pr["sxhy_imagery_freq"].split("|"):
                if ":" in entry:
                    term, n = entry.rsplit(":", 1)
                    freq_cnt[term] += int(n)
        if isinstance(pr["sxhy_imagery_headings"], str) and pr["sxhy_imagery_headings"]:
            for term in pr["sxhy_imagery_headings"].split("|"):
                if term:
                    poem_cnt[term] += 1

    freq_df = pd.DataFrame(
        [
            {"imagery_heading": t, "total_hits": freq_cnt[t], "poem_count": poem_cnt.get(t, 0)}
            for t, _ in freq_cnt.most_common()
        ],
        columns=["imagery_heading", "total_hits", "poem_count"],
    )
    freq_df.to_csv(OUT_DIR / "digi_sxhy_imagery_frequency.csv", index=False, encoding="utf-8-sig")

    report_lines.append(f"[4] 诗歌 {len(per_df)} 首 → digi_sxhy_imagery_per_poem.csv")
    report_lines.append(f"[5] 意象频次表 → digi_sxhy_imagery_frequency.csv ({len(freq_df)} 条)")

    full_report = "\n".join(report_lines + ["", "完成。"])
    (OUT_DIR / "digi_sxhy_build_report.txt").write_text(full_report, encoding="utf-8")
    print(full_report)


if __name__ == "__main__":
    main()
