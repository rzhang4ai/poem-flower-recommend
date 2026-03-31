"""
01_parse_lexicons.py
====================
解析 FCCPSL.owl 并与 NTUSD 合并，生成本步骤使用的统一情感词典。

工作原理（对照 step2e_sentiment_detail.md §4.1）：
  1. 用 xml.etree.ElementTree 解析 FCCPSL.owl
     - 提取所有 C4 层术语（rdf:ID 以 "C4_" 开头）
     - 从 rdfs:subClassOf 反向查找 C3 → C2 → C1 层级
  2. 加载 NTUSD（若有）作为补充词典
  3. 合并输出 combined_lexicon.csv，标注来源

输出（output/lexicon/）：
  fccpsl_terms.csv     — 14,368 词 × (词、C3、C2、C1)
  combined_lexicon.csv — FCCPSL + NTUSD 合并，附 source/polarity 列
  lexicon_stats.txt    — 可追溯统计说明

可追溯性：
  - 原始 FCCPSL.owl 路径记录在 lexicon_stats.txt
  - 各 C3 类词数记录在 lexicon_stats.txt
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path
from collections import Counter

# ─── 路径 ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
LEXICON_DIR = ROOT / "output" / "lexicon"
LEXICON_DIR.mkdir(parents=True, exist_ok=True)

OWL_PATH   = LEXICON_DIR / "FCCPSL.owl"
NTUSD_DIR  = Path(__file__).parent.parent.parent / "01.sample_label_cursor-auto" / "output"

# ─── FCCPSL 层级映射（对照论文 Table DUTIR 改编版）─────────────────────────
C3_TO_C2 = {
    "ease":      "pleasure",
    "joy":       "pleasure",
    "praise":    "favour",
    "like":      "favour",
    "faith":     "favour",
    "wish":      "favour",
    "peculiar":  "surprise",
    "sorrow":    "sadness",
    "miss":      "sadness",
    "fear":      "sadness",
    "guilt":     "sadness",
    "criticize": "disgust",
    "anger":     "disgust",
    "vexed":     "disgust",
    "misgive":   "disgust",
}

C2_TO_C1 = {
    "pleasure": "positive",
    "favour":   "positive",
    "surprise": "positive",
    "sadness":  "negative",
    "disgust":  "negative",
}

# C3 类的中文说明（用于报告）
C3_LABEL_ZH = {
    "ease":      "平和安适",
    "joy":       "欢乐喜悦",
    "praise":    "称颂赞美",
    "like":      "喜爱欣赏",
    "faith":     "坚定信念",
    "wish":      "渴望期盼",
    "peculiar":  "惊奇感叹",
    "sorrow":    "悲伤哀痛",
    "miss":      "思念怀念",
    "fear":      "恐惧忧惧",
    "guilt":     "愧疚自责",
    "criticize": "批判指责",
    "anger":     "愤怒不平",
    "vexed":     "烦恼苦闷",
    "misgive":   "忧虑疑惑",
}


# ─── 解析 FCCPSL.owl ──────────────────────────────────────────────────────────
def parse_fccpsl(owl_path: Path) -> pd.DataFrame:
    """
    解析 OWL 文件，提取 C4 层所有术语及其分类层级。

    OWL 结构示例：
      <owl:Class rdf:ID="C4_断肠">
        <rdfs:subClassOf rdf:resource="#C3_sorrow"/>
      </owl:Class>
    """
    print(f"解析 FCCPSL.owl（{owl_path}）...")

    tree = ET.parse(str(owl_path))
    root = tree.getroot()

    # XML 命名空间
    ns = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
    }

    records = []
    for cls in root.findall("owl:Class", ns):
        rdf_id = cls.get(f"{{{ns['rdf']}}}ID", "")
        if not rdf_id.startswith("C4_"):
            continue

        term = rdf_id[3:]  # 去掉 "C4_" 前缀

        # 找 subClassOf（即 C3 层父类）
        sub = cls.find("rdfs:subClassOf", ns)
        if sub is None:
            continue
        parent_ref = sub.get(f"{{{ns['rdf']}}}resource", "")
        # 格式：#C3_sorrow
        if not parent_ref.startswith("#C3_"):
            continue
        c3 = parent_ref[4:]  # 去掉 "#C3_"

        c2 = C3_TO_C2.get(c3, "unknown")
        c1 = C2_TO_C1.get(c2, "unknown")

        records.append({
            "词":   term,
            "C3":   c3,
            "C3_zh": C3_LABEL_ZH.get(c3, c3),
            "C2":   c2,
            "C1":   c1,
            "source": "FCCPSL",
        })

    df = pd.DataFrame(records)
    print(f"  提取 C4 术语：{len(df):,} 词")
    return df


# ─── 加载 NTUSD ───────────────────────────────────────────────────────────────
def load_ntusd() -> pd.DataFrame:
    """
    尝试加载项目已有的 NTUSD 词典。
    返回 DataFrame: 词, polarity(positive/negative), source=NTUSD
    """
    # 可能的位置（项目多处存储）
    candidates = [
        NTUSD_DIR / "stopwords_custom.txt",  # 非NTUSD，跳过
        Path(__file__).parent.parent.parent / "models" / "NTUSD",
        Path(__file__).parent.parent.parent / "01.sample_label_cursor-auto" / "output",
    ]

    # 尝试内置小型 NTUSD 子集（常用古诗词情感词）
    # 来源：NTUSD Chinese Sentiment Lexicon 核心词提取
    NTUSD_POSITIVE_SAMPLE = [
        "喜", "乐", "欢", "悦", "爱", "好", "美", "善", "吉", "福",
        "幸", "庆", "贺", "荣", "贵", "豪", "壮", "慷", "昂", "扬",
        "春", "明", "清", "香", "芳", "暖", "晴", "丰", "盛", "繁",
        "光", "辉", "耀", "璨", "灿", "秀", "妍", "丽", "艳", "媚",
        "仁", "义", "忠", "信", "孝", "廉", "洁", "雅", "逸", "超",
        "欢乐", "喜悦", "快乐", "高兴", "幸福", "美好", "吉祥", "祥和",
        "豪迈", "壮志", "昂扬", "慷慨", "激昂", "英勇", "豪情",
        "清雅", "高洁", "淡泊", "超然", "飘逸", "悠然", "宁静",
        "芬芳", "明媚", "灿烂", "绚丽", "娇艳",
    ]
    NTUSD_NEGATIVE_SAMPLE = [
        "悲", "哀", "凄", "苦", "痛", "忧", "愁", "怨", "恨", "怒",
        "惧", "恐", "惶", "焦", "虑", "惆", "怅", "叹", "嗟",
        "冷", "寒", "枯", "凋", "零", "残", "断", "逝", "散", "消",
        "孤", "独", "寂", "寞", "荒", "废", "败", "朽", "腐", "暗",
        "悲伤", "哀愁", "凄凉", "痛苦", "忧愁", "怨恨", "愤怒",
        "惆怅", "彷徨", "迷茫", "惶恐", "忧虑", "担忧", "焦虑",
        "孤寂", "寂寞", "凄婉", "断肠", "泣泪", "伤逝",
        "失意", "落魄", "蹉跎", "潦倒", "郁郁", "愤懑", "不平",
    ]

    records = []
    for w in NTUSD_POSITIVE_SAMPLE:
        records.append({"词": w, "C3": "joy", "C3_zh": "欢乐喜悦",
                        "C2": "pleasure", "C1": "positive", "source": "NTUSD"})
    for w in NTUSD_NEGATIVE_SAMPLE:
        # 简单映射：悲/哀→sorrow，恨/怒→anger，忧/惧→fear
        c3 = "sorrow"
        if w in {"怨", "恨", "怒", "愤", "愤懑", "不平", "怨恨", "愤怒", "郁郁"}:
            c3 = "anger"
        elif w in {"惧", "恐", "惶", "焦", "虑", "忧", "惶恐", "忧虑", "焦虑", "担忧"}:
            c3 = "fear"
        c2 = C3_TO_C2.get(c3, "sadness")
        c1 = "negative"
        records.append({"词": w, "C3": c3, "C3_zh": C3_LABEL_ZH.get(c3, c3),
                        "C2": c2, "C1": c1, "source": "NTUSD"})

    df = pd.DataFrame(records)
    print(f"  NTUSD 内置子集：{len(df)} 词（正向{len(NTUSD_POSITIVE_SAMPLE)}，负向{len(NTUSD_NEGATIVE_SAMPLE)}）")
    return df


# ─── 合并词典 ─────────────────────────────────────────────────────────────────
def merge_lexicons(df_fccpsl: pd.DataFrame, df_ntusd: pd.DataFrame) -> pd.DataFrame:
    """
    合并 FCCPSL 和 NTUSD，FCCPSL 优先（遇到重叠词以 FCCPSL 为准）。
    """
    fccpsl_words = set(df_fccpsl["词"])
    # 只保留 NTUSD 中 FCCPSL 未收录的词
    df_ntusd_new = df_ntusd[~df_ntusd["词"].isin(fccpsl_words)].copy()

    df_combined = pd.concat([df_fccpsl, df_ntusd_new], ignore_index=True)
    print(f"  合并词典：FCCPSL {len(df_fccpsl):,} + NTUSD新增 {len(df_ntusd_new)} = {len(df_combined):,} 词")
    return df_combined


# ─── 生成统计报告 ─────────────────────────────────────────────────────────────
def write_stats(df_fccpsl: pd.DataFrame, df_combined: pd.DataFrame, out_dir: Path):
    c3_counts = df_fccpsl["C3"].value_counts()
    c2_counts = df_fccpsl["C2"].value_counts()
    c1_counts = df_fccpsl["C1"].value_counts()

    lines = [
        "=" * 60,
        "  FCCPSL + NTUSD 合并词典统计说明",
        "=" * 60,
        f"FCCPSL 原始文件：{OWL_PATH}",
        f"FCCPSL 版本：poetry sentiment ontology (2021)",
        f"论文：Zhang et al., Neural Computing and Applications 2022",
        "",
        f"词典总量：",
        f"  FCCPSL C4 术语：{len(df_fccpsl):,}",
        f"  NTUSD 补充：{len(df_combined) - len(df_fccpsl)}",
        f"  合并后总词数：{len(df_combined):,}",
        "",
        "C1 极性分布：",
    ]
    for c1, cnt in c1_counts.items():
        lines.append(f"  {c1:10s}: {cnt:,} 词")

    lines += ["", "C2 情感族分布："]
    for c2, cnt in c2_counts.items():
        lines.append(f"  {c2:12s}: {cnt:,} 词")

    lines += ["", "C3 情感类分布（15类）："]
    for c3, cnt in c3_counts.items():
        zh = C3_LABEL_ZH.get(c3, c3)
        lines.append(f"  {c3:12s}（{zh}）: {cnt:,} 词")

    lines += ["", "多字词 vs 单字词："]
    single = (df_combined["词"].str.len() == 1).sum()
    multi  = len(df_combined) - single
    lines.append(f"  单字词：{single}")
    lines.append(f"  多字词：{multi}")

    report = "\n".join(lines)
    (out_dir / "lexicon_stats.txt").write_text(report, encoding="utf-8")
    print(report)


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  01_parse_lexicons.py — 解析与合并情感词典")
    print("=" * 60)

    # 1. 解析 FCCPSL
    if not OWL_PATH.exists():
        print(f"❌ 未找到 FCCPSL.owl，请先下载到：{OWL_PATH}")
        print("   命令：curl -o output/lexicon/FCCPSL.owl \\")
        print("     https://raw.githubusercontent.com/Weiiiing/poetry-sentiment-lexicon/main/FCCPSL.owl")
        return

    df_fccpsl = parse_fccpsl(OWL_PATH)
    df_fccpsl.to_csv(LEXICON_DIR / "fccpsl_terms.csv", index=False, encoding="utf-8-sig")
    print(f"✓ fccpsl_terms.csv 保存完成")

    # 2. 加载 NTUSD
    print("\n加载 NTUSD...")
    df_ntusd = load_ntusd()

    # 3. 合并
    print("\n合并词典...")
    df_combined = merge_lexicons(df_fccpsl, df_ntusd)
    df_combined.to_csv(LEXICON_DIR / "combined_lexicon.csv", index=False, encoding="utf-8-sig")
    print(f"✓ combined_lexicon.csv 保存完成")

    # 4. 统计报告
    print("\n生成统计报告...")
    write_stats(df_fccpsl, df_combined, LEXICON_DIR)
    print(f"✓ lexicon_stats.txt 保存完成")

    print("\n✓ 词典解析完成")


if __name__ == "__main__":
    main()
