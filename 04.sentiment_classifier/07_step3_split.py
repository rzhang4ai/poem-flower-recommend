"""
07_step3_split.py
=================
生成并保存全实验共用的固定数据集切分（seed=42 锁定，不可更改）。

输出：
  output/splits/fspc_split.csv     FSPC 5000首 + split(train/test) 列
  output/splits/golden_split.csv   Golden 1896首 + split(train/test) 列
  output/splits/split_stats.txt    切分统计报告

重要说明：
  - FSPC split 用于 5极性分类对比（SVM vs 微调BERT）
  - Golden split 用于 C2/C3 SVM 训练与评估
  - ⚠ 微调BERT 是用全量 FSPC 训练的，在 fspc_test 上评估存在数据泄露
    因此与 SVM 的对比为"近似比较"，结论仅供参考
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================================================================
# 配置
# ===========================================================================
SEED            = 42
FSPC_TEST_RATIO = 0.20
GOLD_TEST_RATIO = 0.20

_SCRIPT_DIR = Path(__file__).resolve().parent
FSPC_PATH   = _SCRIPT_DIR / "output" / "lexicon"      / "FSPC_V1.0.json"
GOLDEN_PATH = _SCRIPT_DIR / "output" / "pseudo_label" / "golden_dataset_step1.csv"
OUT_DIR     = _SCRIPT_DIR / "output" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FSPC_POL_MAP = {
    "1": "Negative",
    "2": "Implicit Negative",
    "3": "Neutral",
    "4": "Implicit Positive",
    "5": "Positive",
}

POL_ORDER = ["Negative", "Implicit Negative", "Neutral", "Implicit Positive", "Positive"]


# ===========================================================================
# 加载 FSPC
# ===========================================================================
def load_fspc(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            poem = d.get("poem", "").replace("|", "")
            text = "".join(c for c in poem if "\u4e00" <= c <= "\u9fff")
            hol  = str(d.get("setiments", {}).get("holistic", "3"))
            rows.append({
                "poet":     d.get("poet", ""),
                "title":    d.get("title", ""),
                "dynasty":  d.get("dynasty", ""),
                "poem_raw": poem,
                "text":     text,
                "polarity": FSPC_POL_MAP.get(hol, "Neutral"),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    lines: list[str] = []
    sep = "=" * 60

    # ── FSPC split ────────────────────────────────────────────────
    print(f"加载 FSPC: {FSPC_PATH}")
    df_fspc = load_fspc(FSPC_PATH)

    idx = np.arange(len(df_fspc))
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=FSPC_TEST_RATIO,
        random_state=SEED,
        stratify=df_fspc["polarity"],
    )
    df_fspc["split"] = "train"
    df_fspc.iloc[idx_te, df_fspc.columns.get_loc("split")] = "test"

    out_fspc = OUT_DIR / "fspc_split.csv"
    df_fspc.to_csv(out_fspc, index=False, encoding="utf-8-sig")

    lines += [sep, "  FSPC 数据集切分", sep]
    lines.append(f"  总量:  {len(df_fspc)}")
    lines.append(f"  train: {(df_fspc['split']=='train').sum()}")
    lines.append(f"  test:  {(df_fspc['split']=='test').sum()}")
    lines.append("\n  极性分布（train / test）：")
    lines.append(f"  {'极性':<22s} {'train':>6} {'test':>6}")
    for pol in POL_ORDER:
        n_tr = ((df_fspc["split"] == "train") & (df_fspc["polarity"] == pol)).sum()
        n_te = ((df_fspc["split"] == "test")  & (df_fspc["polarity"] == pol)).sum()
        lines.append(f"  {pol:<22s} {n_tr:>6} {n_te:>6}")

    # ── Golden split ──────────────────────────────────────────────
    print(f"加载 Golden: {GOLDEN_PATH}")
    df_gold = pd.read_csv(GOLDEN_PATH)

    # 确保每个 pseudo_label 类别都有足够样本做分层切分
    # 样本数<2 的类别暂时全放 train
    label_counts = df_gold["pseudo_label"].value_counts()
    can_stratify = label_counts[label_counts >= 2].index.tolist()
    mask_ok   = df_gold["pseudo_label"].isin(can_stratify)
    df_ok     = df_gold[mask_ok].reset_index(drop=True)
    df_rare   = df_gold[~mask_ok].copy()
    df_rare["split"] = "train"

    idx_g = np.arange(len(df_ok))
    idx_g_tr, idx_g_te = train_test_split(
        idx_g,
        test_size=GOLD_TEST_RATIO,
        random_state=SEED,
        stratify=df_ok["pseudo_label"],
    )
    df_ok["split"] = "train"
    df_ok.iloc[idx_g_te, df_ok.columns.get_loc("split")] = "test"

    df_gold_split = pd.concat([df_ok, df_rare], ignore_index=True)
    out_gold = OUT_DIR / "golden_split.csv"
    df_gold_split.to_csv(out_gold, index=False, encoding="utf-8-sig")

    lines += ["", sep, "  Golden 数据集切分", sep]
    lines.append(f"  总量:  {len(df_gold_split)}")
    lines.append(f"  train: {(df_gold_split['split']=='train').sum()}")
    lines.append(f"  test:  {(df_gold_split['split']=='test').sum()}")
    lines.append("\n  C3 标签分布（train / test）：")
    lines.append(f"  {'C3标签':<14s} {'train':>6} {'test':>6}")
    for lbl in df_gold["pseudo_label"].value_counts().index:
        n_tr = ((df_gold_split["split"] == "train") & (df_gold_split["pseudo_label"] == lbl)).sum()
        n_te = ((df_gold_split["split"] == "test")  & (df_gold_split["pseudo_label"] == lbl)).sum()
        lines.append(f"  {lbl:<14s} {n_tr:>6} {n_te:>6}")

    lines += [
        "",
        "⚠ 注意：微调BERT 在全量 FSPC 训练，fspc_test 对它不是真正盲测。",
        "  SVM vs 微调BERT 的比较为'近似比较'，结论仅供方向性参考。",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_stats = OUT_DIR / "split_stats.txt"
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\n切分文件已保存:")
    print(f"  {out_fspc}")
    print(f"  {out_gold}")
    print(f"  {out_stats}")


if __name__ == "__main__":
    main()
