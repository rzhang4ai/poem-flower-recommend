"""
extract_fccpsl_terms.py
=======================
从 Weiiiing 的 FCCPSL.owl 中提取所有词汇及分类标签（C3/C2/C1）。

输入：
  output/lexicon/FCCPSL.owl

输出：
  output/lexicon/fccpsl_terms_only.csv
  output/lexicon/fccpsl_terms_only_stats.txt

运行：
  source flower_env/bin/activate
  cd 02.sample_label_phase2/step2e_sentiment
  python extract_fccpsl_terms.py
"""

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LEXICON_DIR = BASE_DIR / "output" / "lexicon"
LEXICON_DIR.mkdir(parents=True, exist_ok=True)

OWL_PATH = LEXICON_DIR / "FCCPSL.owl"
OUT_CSV = LEXICON_DIR / "fccpsl_terms_only.csv"
OUT_STATS = LEXICON_DIR / "fccpsl_terms_only_stats.txt"

# 与论文调整后的类别保持一致
C3_TO_C2 = {
    "ease": "pleasure",
    "joy": "pleasure",
    "praise": "favour",
    "like": "favour",
    "faith": "favour",
    "wish": "favour",
    "peculiar": "surprise",
    "sorrow": "sadness",
    "miss": "sadness",
    "fear": "sadness",
    "guilt": "sadness",
    "criticize": "disgust",
    "anger": "disgust",
    "vexed": "disgust",
    "misgive": "disgust",
}

C2_TO_C1 = {
    "pleasure": "positive",
    "favour": "positive",
    "surprise": "positive",
    "sadness": "negative",
    "disgust": "negative",
}

C3_LABEL_ZH = {
    "ease": "平和安适",
    "joy": "欢乐喜悦",
    "praise": "称颂赞美",
    "like": "喜爱欣赏",
    "faith": "坚定信念",
    "wish": "渴望期盼",
    "peculiar": "惊奇感叹",
    "sorrow": "悲伤哀痛",
    "miss": "思念怀念",
    "fear": "恐惧忧惧",
    "guilt": "愧疚自责",
    "criticize": "批判指责",
    "anger": "愤怒不平",
    "vexed": "烦恼苦闷",
    "misgive": "忧虑疑惑",
}


def parse_fccpsl_terms(owl_path: Path) -> pd.DataFrame:
    tree = ET.parse(str(owl_path))
    root = tree.getroot()
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
    }

    rows = []
    for cls in root.findall("owl:Class", ns):
        rdf_id = cls.get(f"{{{ns['rdf']}}}ID", "")
        if not rdf_id.startswith("C4_"):
            continue

        term = rdf_id[3:]  # 去掉 C4_
        sub = cls.find("rdfs:subClassOf", ns)
        if sub is None:
            continue
        parent_ref = sub.get(f"{{{ns['rdf']}}}resource", "")
        if not parent_ref.startswith("#C3_"):
            continue

        c3 = parent_ref[4:]  # 去掉 #C3_
        c2 = C3_TO_C2.get(c3, "unknown")
        c1 = C2_TO_C1.get(c2, "unknown")
        rows.append(
            {
                "词": term,
                "C3": c3,
                "C3_zh": C3_LABEL_ZH.get(c3, c3),
                "C2": c2,
                "C1": c1,
            }
        )

    df = pd.DataFrame(rows)
    # 稳定排序：先按 C1/C2/C3，再按词长和字典序
    df["词长"] = df["词"].astype(str).str.len()
    df = df.sort_values(
        by=["C1", "C2", "C3", "词长", "词"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    return df


def write_stats(df: pd.DataFrame, owl_path: Path, out_path: Path):
    c1_cnt = Counter(df["C1"])
    c2_cnt = Counter(df["C2"])
    c3_cnt = Counter(df["C3"])
    single = int((df["词长"] == 1).sum())
    multi = int(len(df) - single)

    lines = [
        "=" * 70,
        "FCCPSL 词汇与分类标签导出统计",
        "=" * 70,
        f"来源文件: {owl_path}",
        f"总词数(C4): {len(df)}",
        f"单字词: {single}",
        f"多字词: {multi}",
        "",
        "C1 分布:",
    ]
    for k, v in sorted(c1_cnt.items()):
        lines.append(f"  {k:<10} {v:>6}")

    lines += ["", "C2 分布:"]
    for k, v in sorted(c2_cnt.items()):
        lines.append(f"  {k:<10} {v:>6}")

    lines += ["", "C3 分布:"]
    for k, v in sorted(c3_cnt.items()):
        lines.append(f"  {k:<12} {v:>6}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not OWL_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {OWL_PATH}\n"
            "请先将 FCCPSL.owl 放到 output/lexicon/ 目录。"
        )

    df = parse_fccpsl_terms(OWL_PATH)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_stats(df, OWL_PATH, OUT_STATS)

    print(f"✅ 已输出: {OUT_CSV}")
    print(f"✅ 已输出: {OUT_STATS}")
    print(f"   词条数: {len(df)}")


if __name__ == "__main__":
    main()
