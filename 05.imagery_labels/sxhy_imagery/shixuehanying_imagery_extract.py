"""
step2c_imagery/shixuehanying_imagery_extract.py
================================================
以《诗学含英》为唯一词典来源，对 1075 首诗词进行意象提取，
并与手工词典（imagery_extract.py）的结果进行对比。

《诗学含英》词典解析策略
------------------------
  # 大类名（以"类"结尾）       → 类别标签，如"天文类"
  # 意象词（不以"类"结尾）     → 一级意象词条，如"斜阳"
  ## 意象词                    → 二级意象词条，如"初日"
  标题下的内容行                → 关联词语；2 字以上短语映射回父标题意象

匹配策略
--------
  1. 词典包含：所有标题词 + 标题下 2～5 字短语（指向父标题）
  2. 最长优先匹配（greedy, left-to-right），避免短字截断长词
  3. 单字标题（天/日/月/云 等）仅作为候补——仅当该位置未被多字词命中时计入
  4. 每首诗去重后输出命中意象列表（保留频次）

输出（output/）
--------------
  sxhy_vocab.csv              词典条目（标题词 + 来源分类 + 词条层级）
  sxhy_imagery_per_poem.csv   每首诗命中的意象列表
  sxhy_imagery_frequency.csv  全局意象词频排名
  sxhy_vs_manual_compare.csv  与手工词典提取结果的对比（共现 / 独有）
  sxhy_report.txt             汇总报告
"""

import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
ROOT      = BASE_DIR.parent.parent
SXHY_PATH = BASE_DIR / "诗学含英_simp_clean.txt"
DATA_CSV  = ROOT / "00.poems_dataset" / "poems_dataset_merged_done.csv"
OUT_DIR   = BASE_DIR / "output"
PREV_CSV  = OUT_DIR / "imagery_per_poem.csv"   # 手工词典结果，用于对比
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: 解析《诗学含英》构建词典 ─────────────────────────────

_PURE_HAN = re.compile(r"^[\u4e00-\u9fff]+$")

def is_skip_line(term: str) -> bool:
    """跳过书名/页码行（OCR 产生的噪声行，如"诗学含英 卷一 天文类"）"""
    return "诗学含英" in term or "诗 学 含 英" in term

def is_category_header(term: str) -> bool:
    """判断是否为章节标签（非意象词条）"""
    return term.endswith("类")


def parse_sxhy(path: Path):
    """
    解析《诗学含英》，返回：
      headings : list of dict {term, level(1|2), category}
      subterm_map : dict {2-5字短语 → 父意象词}  （仅2字以上）

    特殊处理：
      - OCR 页码行（含"诗学含英"）直接跳过
      - # 行含多个空格分隔的纯汉字词（如"# 步月 中庭 皓彩"）→ 拆成多个词条
    """
    headings    = []
    subterm_map = {}          # 2+字子词 → 父标题意象

    current_cat  = ""         # 最近的"类"标签
    current_h1   = ""         # 最近的 # 非类标题
    current_head = ""         # 最近的任意标题（用于子词归属）

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 跳过 OCR 页码 / 书名行
        if is_skip_line(line):
            continue

        if line.startswith("## "):
            term = line[3:].strip()
            if not term:
                continue
            # ## 行一般是单个词条，直接注册
            if _PURE_HAN.match(term):
                headings.append({"term": term, "level": 2, "category": current_cat,
                                 "parent_h1": current_h1})
                current_head = term

        elif line.startswith("# "):
            raw_term = line[2:].strip()
            if not raw_term:
                continue
            if is_category_header(raw_term):
                current_cat  = raw_term
                current_h1   = ""
                current_head = ""
            else:
                # 可能是多词合并行（如"# 步月 中庭 皓彩"），拆分
                parts = [p for p in raw_term.split() if _PURE_HAN.match(p)]
                if not parts:
                    continue
                for term in parts:
                    headings.append({"term": term, "level": 1, "category": current_cat,
                                     "parent_h1": ""})
                # 当前 head 取最后一个词
                current_h1   = parts[-1]
                current_head = parts[-1]

        else:
            # 内容行：提取 2-5 字汉字短语作为子词
            if current_head:
                phrases = re.findall(r"[\u4e00-\u9fff]{2,5}", line)
                for ph in phrases:
                    if ph not in subterm_map:
                        subterm_map[ph] = current_head

    return headings, subterm_map


print("解析《诗学含英》...")
headings, subterm_map = parse_sxhy(SXHY_PATH)
print(f"  标题词条数：{len(headings)}  （一级={sum(1 for h in headings if h['level']==1)}，"
      f"二级={sum(1 for h in headings if h['level']==2)}）")
print(f"  子词条数：{len(subterm_map)}")

# ── Step 2: 构建匹配词典 ──────────────────────────────────────────
# 词典 = {词语 → 最终归属的意象词（标题词）}

vocab: dict[str, str] = {}

# 先加标题词（自身即意象词）
for h in headings:
    t = h["term"]
    vocab[t] = t

# 再加子词（指向父标题），不覆盖已有标题词
for ph, parent in subterm_map.items():
    if ph not in vocab:
        vocab[ph] = parent

# 按词长降序排列，用于最长优先匹配
sorted_vocab = sorted(vocab.keys(), key=len, reverse=True)

# 单字词条（用于候补匹配）
single_char_terms = {t for t in vocab if len(t) == 1}
multi_char_terms  = {t for t in vocab if len(t) > 1}

print(f"  匹配词典总词数：{len(vocab)}  "
      f"（多字={len(multi_char_terms)}，单字={len(single_char_terms)}）")

# ── Step 3: 意象提取函数 ──────────────────────────────────────────

def extract_imagery(poem_text: str) -> list[tuple[str, str]]:
    """
    对单首诗正文执行最长优先意象匹配。
    返回 list of (命中原词, 父标题意象) 元组，含重复，保序。
    先做多字匹配（2+字），将命中位置标记；再对未标记位置做单字匹配。
    """
    if not isinstance(poem_text, str):
        return []
    text = re.sub(r"[^\u4e00-\u9fff]", "", poem_text)
    n    = len(text)
    if n == 0:
        return []

    hits    = []              # (原词, 父标题)
    covered = [False] * n

    # ─── 多字词匹配（最长优先，left-to-right greedy）───────────────
    i = 0
    while i < n:
        best_len = 0
        best_raw = None
        for length in range(min(5, n - i), 1, -1):
            chunk = text[i:i + length]
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

    # ─── 单字词候补匹配（仅未覆盖位置）──────────────────────────────
    for i, ch in enumerate(text):
        if not covered[i] and ch in single_char_terms:
            hits.append((ch, vocab[ch]))

    return hits


# ── Step 4: 读取诗词并批量提取 ────────────────────────────────────
print("\n读取诗词数据...")
poems_df = pd.read_csv(DATA_CSV)
print(f"  共 {len(poems_df)} 首诗")

print("提取意象...")
imagery_results = []
for _, row in poems_df.iterrows():
    imgs = extract_imagery(row["正文"])
    imagery_results.append(imgs)

# ── Step 5: 整理输出 ──────────────────────────────────────────────

# ─ 5a: sxhy_vocab.csv ─────────────────────────────────────────────
h_df = pd.DataFrame(headings)[["term","level","category","parent_h1"]]
h_df.to_csv(OUT_DIR / "sxhy_vocab.csv", index=False)
print(f"\nsxhy_vocab.csv 已保存（{len(h_df)} 条标题词）")

# ─ 5b: sxhy_imagery_per_poem.csv ─────────────────────────────────
per_poem_rows = []
for idx, row in poems_df.iterrows():
    hits = imagery_results[idx]   # list of (raw_word, parent_heading)
    raw_words = [h[0] for h in hits]
    headings_ = [h[1] for h in hits]

    raw_cnt  = Counter(raw_words)
    head_cnt = Counter(headings_)

    unique_heads = sorted(head_cnt.keys(), key=lambda x: -head_cnt[x])
    unique_raws  = sorted(raw_cnt.keys(),  key=lambda x: -raw_cnt[x])

    per_poem_rows.append({
        "ID":              row.get("ID", idx),
        "诗名":             row.get("诗名", ""),
        "朝代":             row.get("朝代", ""),
        "花名":             row.get("花名", ""),
        "imagery_count":   len(unique_heads),
        "imagery_list":    "|".join(unique_heads),     # 父标题意象（主输出）
        "raw_word_list":   "|".join(unique_raws),      # 原词（用于对比）
        "imagery_freq":    "|".join(f"{k}:{v}" for k, v in head_cnt.most_common()),
    })

per_poem_df = pd.DataFrame(per_poem_rows)
per_poem_df.to_csv(OUT_DIR / "sxhy_imagery_per_poem.csv", index=False)
print(f"sxhy_imagery_per_poem.csv 已保存（{len(per_poem_df)} 行）")

# ─ 5c: sxhy_imagery_frequency.csv ────────────────────────────────
# 从 per_poem_df 直接统计：总出现次数（含重复）和出现诗数（不去重）
freq_cnt     = Counter()    # 总频次（诗内重复计入）
poem_cnt_map = Counter()    # 出现诗数（每首诗只计一次）
raw_freq     = Counter()    # 原词频次

for _, pr in per_poem_df.iterrows():
    # 父标题意象
    if isinstance(pr["imagery_freq"], str) and pr["imagery_freq"]:
        for entry in pr["imagery_freq"].split("|"):
            if ":" in entry:
                term, n = entry.rsplit(":", 1)
                freq_cnt[term] += int(n)
    if isinstance(pr["imagery_list"], str) and pr["imagery_list"]:
        for term in pr["imagery_list"].split("|"):
            if term:
                poem_cnt_map[term] += 1
    # 原词
    if isinstance(pr["raw_word_list"], str) and pr["raw_word_list"]:
        for w in pr["raw_word_list"].split("|"):
            if w:
                raw_freq[w] += 1

freq_df = pd.DataFrame(
    [{"imagery": term, "count": cnt, "poem_count": poem_cnt_map.get(term, 0)}
     for term, cnt in freq_cnt.most_common()],
)
freq_df.to_csv(OUT_DIR / "sxhy_imagery_frequency.csv", index=False)
print(f"sxhy_imagery_frequency.csv 已保存（{len(freq_df)} 条意象）")

# ─ 5d: 与手工词典结果对比 ─────────────────────────────────────────
compare_rows = []
if PREV_CSV.exists():
    prev_df = pd.read_csv(PREV_CSV)
    # 手工词典结果列名（imagery_extract.py 输出为"物象列表"）
    for candidate in ("物象列表", "imagery_list"):
        if candidate in prev_df.columns:
            prev_col = candidate
            break
    else:
        prev_col = None

    for idx, row in per_poem_df.iterrows():
        # 诗学含英：父标题意象 + 原词
        new_heads = set(row["imagery_list"].split("|")) if row["imagery_list"] else set()
        new_raws  = set(row["raw_word_list"].split("|")) if row["raw_word_list"] else set()

        old_set = set()
        if prev_col and idx < len(prev_df):
            old_val = prev_df.iloc[idx][prev_col]
            if isinstance(old_val, str) and old_val:
                # 旧 CSV 用全角竖线 ｜(U+FF5C) 分隔
                old_set = set(
                    s.strip() for s in old_val.replace("｜", "|").replace("，", "|")
                                               .replace(",", "|").split("|")
                    if s.strip()
                )

        # 在原词层面与手工词典对比（两者都是具体词语）
        shared_raw = new_raws & old_set

        compare_rows.append({
            "ID":               row["ID"],
            "诗名":              row["诗名"],
            "花名":              row["花名"],
            "sxhy_heading":     row["imagery_list"],   # 父标题意象
            "sxhy_raw_words":   row["raw_word_list"],  # 诗学含英命中原词
            "manual_imagery":   "|".join(sorted(old_set)),
            "共有原词":           "|".join(sorted(shared_raw)),
            "仅诗学含英原词":      "|".join(sorted(new_raws - old_set)),
            "仅手工词典":          "|".join(sorted(old_set - new_raws)),
            "共有词数":            len(shared_raw),
            "sxhy_heading_数":  len(new_heads),
            "sxhy_raw词数":     len(new_raws),
            "manual词数":        len(old_set),
        })

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUT_DIR / "sxhy_vs_manual_compare.csv", index=False)
    print(f"sxhy_vs_manual_compare.csv 已保存（{len(compare_df)} 行）")
else:
    compare_df = None
    print("（未找到手工词典结果，跳过对比）")

# ── Step 6: 汇总报告 ──────────────────────────────────────────────
total_poems  = len(poems_df)
covered_cnt  = sum(1 for hits in imagery_results if hits)
avg_per_poem = sum(len({h[1] for h in hits}) for hits in imagery_results) / total_poems

# 按类别统计意象词
cat_to_terms = defaultdict(list)
for h in headings:
    cat_to_terms[h["category"]].append(h["term"])

lines = [
    "=" * 65,
    "《诗学含英》意象词典提取报告",
    "=" * 65,
    "",
    "── 词典统计 ──────────────────────────────────────────────",
    f"  标题意象词条数   : {len(headings)}",
    f"  子词（映射词）数 : {len(subterm_map)}",
    f"  匹配词典总词数   : {len(vocab)}",
    f"  单字词条数       : {len(single_char_terms)}",
    f"  多字词条数       : {len(multi_char_terms)}",
    "",
    "── 提取结果（1075首诗）─────────────────────────────────────",
    f"  有意象命中诗词数 : {covered_cnt} / {total_poems} ({covered_cnt/total_poems*100:.1f}%)",
    f"  平均每首独立意象 : {avg_per_poem:.2f} 个",
    f"  全局意象词种数   : {len(freq_cnt)}",
    "",
    "── Top-30 最高频意象 ────────────────────────────────────────",
    f"  {'意象':<8} {'总频次':>7}  {'出现诗数':>8}",
    "  " + "-" * 35,
]
for k, v in freq_cnt.most_common(30):
    pc = poem_cnt_map.get(k, 0)
    lines.append(f"  {k:<8} {v:>7}   {pc:>7}首")

lines += ["", "── 各大类意象词条数 ──────────────────────────────────────────"]
for cat, terms in sorted(cat_to_terms.items(), key=lambda x: -len(x[1])):
    lines.append(f"  {cat:<12}: {len(terms):>4} 条")

lines += ["", "── 各大类高频命中意象（Top-5）────────────────────────────────"]
for cat, terms in sorted(cat_to_terms.items(), key=lambda x: -len(x[1])):
    cat_freq = {t: freq_cnt.get(t, 0) for t in terms}
    top5 = sorted(cat_freq.items(), key=lambda x: -x[1])[:5]
    top5_str = "  ".join(f"{t}({n})" for t, n in top5 if n > 0)
    if top5_str:
        lines.append(f"  {cat:<12}: {top5_str}")

if compare_df is not None:
    avg_shared   = compare_df["共有词数"].mean()
    avg_sxhy_raw = compare_df["sxhy_raw词数"].mean()
    avg_sxhy_head = compare_df["sxhy_heading_数"].mean()
    avg_manual   = compare_df["manual词数"].mean()
    lines += [
        "",
        "── 与手工词典对比（每首诗平均，在原词层面比较）─────────────────",
        f"  诗学含英命中父标题意象数 : {avg_sxhy_head:.2f}",
        f"  诗学含英命中原词数       : {avg_sxhy_raw:.2f}",
        f"  手工词典命中词数         : {avg_manual:.2f}",
        f"  两者共有原词数           : {avg_shared:.2f}",
        f"  仅诗学含英独有           : {avg_sxhy_raw - avg_shared:.2f}",
        f"  仅手工词典独有           : {avg_manual - avg_shared:.2f}",
        "",
        "  说明：",
        "  - 诗学含英输出两层：原词（如'明月'）和父标题意象（如'夏夜'）",
        "  - 原词对比与手工词典同层级，可直接评估覆盖差异",
        "  - 父标题意象是诗学含英提供的语义分类，比原词更有概括性",
        "  - 两者共有原词 ≈ 手工词典和诗学含英都认可的核心意象",
    ]

report = "\n".join(lines)
print("\n" + report)
(OUT_DIR / "sxhy_report.txt").write_text(report, encoding="utf-8")

print("\n全部完成！输出文件：")
for f in ["sxhy_vocab.csv", "sxhy_imagery_per_poem.csv",
          "sxhy_imagery_frequency.csv", "sxhy_vs_manual_compare.csv",
          "sxhy_report.txt"]:
    print(f"  output/{f}")
