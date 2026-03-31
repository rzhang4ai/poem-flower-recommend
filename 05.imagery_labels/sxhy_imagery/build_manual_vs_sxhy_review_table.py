"""
build_manual_vs_sxhy_review_table.py
====================================
生成“手工意象词典 vs 诗学含英目录词条”的人工审核对照表。

输出列包含：
  - 手工类
  - 诗学含英类
  - 是否同类
  - 冲突说明

输出文件：
  output/manual_vs_sxhy_review_table.csv
  output/manual_vs_sxhy_review_summary.txt
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_SCRIPT = BASE_DIR / "imagery_extract.py"
SXHY_CSV = OUT_DIR / "sxhy_catalog_category_table.csv"

OUT_CSV = OUT_DIR / "manual_vs_sxhy_review_table.csv"
OUT_SUMMARY = OUT_DIR / "manual_vs_sxhy_review_summary.txt"


def load_manual_vocab_from_script(script_path: Path) -> dict[str, str]:
    """
    从 imagery_extract.py 读取 IMAGERY_VOCAB，构建 词->手工类 映射。
    """
    spec = importlib.util.spec_from_file_location("imagery_extract_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "IMAGERY_VOCAB"):
        raise AttributeError("imagery_extract.py 未定义 IMAGERY_VOCAB")
    vocab = getattr(module, "IMAGERY_VOCAB")
    if not isinstance(vocab, dict):
        raise TypeError("IMAGERY_VOCAB 不是 dict")

    word_to_cat: dict[str, str] = {}
    for cat, words in vocab.items():
        if not isinstance(words, list):
            continue
        for w in words:
            w = str(w).strip()
            if not w:
                continue
            # 同词若多类，保留首次定义，避免覆盖
            if w not in word_to_cat:
                word_to_cat[w] = str(cat)
    return word_to_cat


def to_coarse_manual(cat: str) -> str:
    mapping = {
        "天象时令": "自然时空",
        "山水地理": "自然地理",
        "植物花卉": "植物生物",
        "动物禽鸟": "动物生物",
        "器物人文": "器物人事",
        "人物意象": "人物人伦",
        "情感抽象": "情志评价",
        "时空行为": "时空行为",
    }
    return mapping.get(cat, "其他")


def to_coarse_sxhy(cat: str) -> str:
    # 规则映射：把《诗学含英》类别映射到便于比较的大类
    if any(k in cat for k in ["天文", "时令", "节序"]):
        return "自然时空"
    if any(k in cat for k in ["地舆", "游眺"]):
        return "自然地理"
    if any(k in cat for k in ["百花", "百草", "树木", "竹木", "百谷", "蔬菜", "百果"]):
        return "植物生物"
    if any(k in cat for k in ["飞禽", "走兽", "鳞介", "昆虫"]):
        return "动物生物"
    if any(k in cat for k in ["器用", "珍宝", "服饰", "饮食", "音乐", "书画", "宫室"]):
        return "器物人事"
    if any(k in cat for k in ["人伦", "师友", "丽人", "百官", "仕进", "君道", "臣道"]):
        return "人物人伦"
    if any(k in cat for k in ["志气", "庆吊", "祖饯", "仕宦", "释老", "文学", "人品", "人事", "谢惠"]):
        return "情志评价"
    return "其他"


def judge_conflict(manual_cat: str, sxhy_cat: str, same_coarse: bool) -> tuple[str, str]:
    """
    返回：
      是否同类（是/否/待定）
      冲突说明
    """
    if not manual_cat and not sxhy_cat:
        return "待定", "两侧都缺失类别"
    if manual_cat and not sxhy_cat:
        return "待定", "仅手工词典命中，诗学含英未收录该词"
    if sxhy_cat and not manual_cat:
        return "待定", "仅诗学含英收录，手工词典未收录该词"
    if same_coarse:
        if manual_cat == sxhy_cat:
            return "是", "细类一致"
        return "是", "细类不同但大类一致"
    return "否", "大类不一致，建议人工复核语境"


def main():
    if not SXHY_CSV.exists():
        raise FileNotFoundError(f"缺少文件: {SXHY_CSV}")

    manual_word_to_cat = load_manual_vocab_from_script(MANUAL_SCRIPT)
    sxhy_df = pd.read_csv(SXHY_CSV)
    sxhy_df = sxhy_df[["entry", "category", "volume", "emotion_hint"]].copy()
    sxhy_df = sxhy_df.rename(columns={"entry": "词条", "category": "诗学含英类", "volume": "卷次"})

    # 同一词条在诗学含英可能归多个类，先聚合
    agg = (
        sxhy_df.groupby("词条", as_index=False)
        .agg({
            "诗学含英类": lambda x: "|".join(sorted(set(map(str, x)))),
            "卷次": lambda x: "|".join(sorted(set(map(str, x)))),
            "emotion_hint": lambda x: "|".join(sorted(set(map(str, x)))),
        })
    )

    all_terms = sorted(set(agg["词条"]).union(set(manual_word_to_cat.keys())))
    rows = []
    for term in all_terms:
        manual_cat = manual_word_to_cat.get(term, "")
        rec = agg[agg["词条"] == term]
        sxhy_cat = rec["诗学含英类"].iloc[0] if len(rec) else ""
        volume = rec["卷次"].iloc[0] if len(rec) else ""
        emo = rec["emotion_hint"].iloc[0] if len(rec) else ""

        manual_coarse = to_coarse_manual(manual_cat) if manual_cat else ""
        # 诗学含英可能多个细类，按“任一大类匹配”判同类
        sxhy_cats = [c for c in sxhy_cat.split("|") if c]
        sxhy_coarse_set = {to_coarse_sxhy(c) for c in sxhy_cats} if sxhy_cats else set()
        same_coarse = bool(manual_coarse and (manual_coarse in sxhy_coarse_set))

        same_flag, conflict_note = judge_conflict(manual_cat, sxhy_cat, same_coarse)

        rows.append({
            "词条": term,
            "手工类": manual_cat,
            "诗学含英类": sxhy_cat,
            "是否同类": same_flag,
            "冲突说明": conflict_note,
            "手工大类": manual_coarse,
            "诗学含英大类": "|".join(sorted(sxhy_coarse_set)) if sxhy_coarse_set else "",
            "卷次": volume,
            "诗学含英情感倾向初判": emo,
        })

    out_df = pd.DataFrame(rows)
    # 审核优先：先看“否”，再看“待定”，最后“是”
    order_map = {"否": 0, "待定": 1, "是": 2}
    out_df["_rank"] = out_df["是否同类"].map(order_map).fillna(9)
    out_df = out_df.sort_values(by=["_rank", "词条"], ascending=[True, True]).drop(columns=["_rank"])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    cnt = out_df["是否同类"].value_counts(dropna=False).to_dict()
    summary_lines = [
        "=" * 70,
        "手工词典 vs 诗学含英 词级人工审核对照摘要",
        "=" * 70,
        f"总词条数: {len(out_df)}",
        f"同类(是): {cnt.get('是', 0)}",
        f"不同类(否): {cnt.get('否', 0)}",
        f"待定: {cnt.get('待定', 0)}",
        "",
        f"输出表: {OUT_CSV}",
        "说明：建议优先人工检查“是否同类=否”的条目。",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"✅ 已输出: {OUT_CSV}")
    print(f"✅ 已输出: {OUT_SUMMARY}")
    print(f"  条目总数={len(out_df)}  是={cnt.get('是',0)} 否={cnt.get('否',0)} 待定={cnt.get('待定',0)}")


if __name__ == "__main__":
    main()
