"""
将 step2b（CCPoem token）、step2c（诗学含英意象）、step2e（情感分层）三张表按 ID 合并。

输出：本目录下 poems_structured_merged.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

PATH_CCPOEM = ROOT / "02.sample_label_phase2/step2b_bert/output/ccpoem_token_importance.csv"
PATH_SXHY = ROOT / "02.sample_label_phase2/step2c_imagery/output/digi_sxhy_imagery_per_poem.csv"
PATH_SENT = ROOT / "02.sample_label_phase2/step2e_sentiment/output/results/sentiment_final_predictions.csv"

OUT_CSV = OUT_DIR / "poems_structured_merged.csv"


def main() -> None:
    for p in (PATH_CCPOEM, PATH_SXHY, PATH_SENT):
        if not p.exists():
            raise FileNotFoundError(f"缺失: {p}")

    cc = pd.read_csv(PATH_CCPOEM)
    sx = pd.read_csv(PATH_SXHY)
    se = pd.read_csv(PATH_SENT)

    # 元信息以情感表为准（含 作者、月份、全文）；其余表去掉重复列只保留 ID + 特有列
    meta_cols = {"诗名", "作者", "朝代", "花名", "月份"}
    cc_only = [c for c in cc.columns if c not in meta_cols or c == "ID"]
    sx_only = [c for c in sx.columns if c not in meta_cols or c == "ID"]

    cc_sub = cc[cc_only]
    sx_sub = sx[sx_only]

    merged = se.merge(cc_sub, on="ID", how="outer", validate="one_to_one")
    merged = merged.merge(sx_sub, on="ID", how="outer", validate="one_to_one")

    merged = merged.sort_values("ID").reset_index(drop=True)

    # 列顺序：标识与正文 → BERT token → 意象 → 情感
    priority = [
        "ID", "诗名", "作者", "朝代", "花名", "月份",
        "text",
        "正文_preview",
        "ccpoem_top5", "ccpoem_top10", "ccpoem_top10_with_score", "ccpoem_cls_l2",
        "sxhy_imagery_count", "sxhy_imagery_headings", "sxhy_raw_words",
        "sxhy_imagery_freq", "sxhy_catalog_paths",
        "l1_polarity", "l1_polarity_zh",
        "l2_c2", "l2_c2_zh", "l3_c3", "l3_c3_zh",
        "prob_l1_negative", "prob_l1_implicit_negative", "prob_l1_neutral",
        "prob_l1_implicit_positive", "prob_l1_positive",
    ]
    rest = [c for c in merged.columns if c not in priority]
    merged = merged[[c for c in priority if c in merged.columns] + rest]

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"行数: {len(merged)}  列数: {len(merged.columns)}")
    print(f"已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
