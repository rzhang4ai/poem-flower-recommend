"""
解析 sxhy_catalog_paths，生成 9 维语义维度计数 + IDF 加权归一化向量；
计算 ccpoem_top10 与 sxhy_raw_words 的双重确认意象 confirmed_imagery；
合并原始数据集中的 月份数字。

9 个维度：
  0 dim_nature    天文类
  1 dim_season    时令类 + 节序类
  2 dim_space     地舆类 + 宫室类 + 游眺类
  3 dim_people    人伦类 + 师友类 + 丽人类 + 人品类 + 君道类 + 臣道类 + 百官类
  4 dim_virtue    志气类 + 仕进类 + 仕宦类 + 释老类
  5 dim_artifact  器用类上 + 器用类下 + 服饰类 + 珍宝类 + 饮食类 + 音乐类 + 书画类
  6 dim_biota     树木类 + 竹木类 + 百草类 + 飞禽类 + 走兽类 + 昆虫类 + 鳞介类 + 百果类 + 百谷类 + 蔬菜类
  7 dim_social    人事类 + 文学类 + 庆吊类 + 祖饯类 + 谢惠类
  8 dim_flower    百花类（单独拆出，不再与其他生物类混叠）

输入：同目录 poems_structured_merged.csv（或 --input 指定）
       00.poems_dataset/poems_dataset_merged_done.csv（取 月份数字）
输出：同目录 poems_structured_with_dims.csv

注：6 维气氛向量（atm_*）已移除，气氛匹配由推荐系统直接使用情感分类器输出。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# 百花类单独为 dim_flower（dim 8），不再归入 dim_biota
CATEGORY_TO_DIM: dict[str, int] = {}
for c in ["天文类"]:
    CATEGORY_TO_DIM[c] = 0
for c in ["时令类", "节序类"]:
    CATEGORY_TO_DIM[c] = 1
for c in ["地舆类", "宫室类", "游眺类"]:
    CATEGORY_TO_DIM[c] = 2
for c in ["人伦类", "师友类", "丽人类", "人品类", "君道类", "臣道类", "百官类"]:
    CATEGORY_TO_DIM[c] = 3
for c in ["志气类", "仕进类", "仕宦类", "释老类"]:
    CATEGORY_TO_DIM[c] = 4
for c in ["器用类上", "器用类下", "服饰类", "珍宝类", "饮食类", "音乐类", "书画类"]:
    CATEGORY_TO_DIM[c] = 5
for c in ["树木类", "竹木类", "百草类", "飞禽类", "走兽类", "昆虫类", "鳞介类", "百果类", "百谷类", "蔬菜类"]:
    CATEGORY_TO_DIM[c] = 6
for c in ["人事类", "文学类", "庆吊类", "祖饯类", "谢惠类"]:
    CATEGORY_TO_DIM[c] = 7
for c in ["百花类"]:
    CATEGORY_TO_DIM[c] = 8

DIM_NAMES = [
    "dim_nature",    # 0
    "dim_season",    # 1
    "dim_space",     # 2
    "dim_people",    # 3
    "dim_virtue",    # 4
    "dim_artifact",  # 5
    "dim_biota",     # 6
    "dim_social",    # 7
    "dim_flower",    # 8 ← 新增，百花类单独
]


def _parse_catalog_paths(path_str: str | float) -> list[int]:
    """解析 sxhy_catalog_paths，返回 9 维整数计数向量。"""
    counts = [0] * 9
    if pd.isna(path_str) or not str(path_str).strip():
        return counts
    for seg in str(path_str).split("|"):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split("/")
        if len(parts) >= 3:
            cat = parts[1]
            idx = CATEGORY_TO_DIM.get(cat)
            if idx is not None:
                counts[idx] += 1
    return counts


def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s <= 0:
        return [0.0] * len(vec)
    return [x / s for x in vec]


def _tokens_from_ccpoem_top10(s: str | float) -> set[str]:
    if pd.isna(s) or not str(s).strip():
        return set()
    parts = re.split(r"[\s\u3000]+", str(s).strip())
    return {p.strip() for p in parts if p.strip()}


def _words_from_sxhy_raw(s: str | float) -> set[str]:
    if pd.isna(s) or not str(s).strip():
        return set()
    return {p.strip() for p in str(s).split("|") if p.strip()}


def _confirmed_imagery(row) -> str:
    t10 = _tokens_from_ccpoem_top10(row.get("ccpoem_top10"))
    raw = _words_from_sxhy_raw(row.get("sxhy_raw_words"))
    inter = sorted(t10 & raw)
    return "|".join(inter)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 9 维语义矩阵 + IDF 加权向量")
    ap.add_argument("--input", type=Path, default=HERE / "poems_structured_merged.csv")
    ap.add_argument("--meta", type=Path,
                    default=ROOT / "00.poems_dataset/poems_dataset_merged_done.csv")
    ap.add_argument("--output", type=Path, default=HERE / "poems_structured_with_dims.csv")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input)

    # ── 第一遍：收集原始计数 ─────────────────────────────────────────────
    dim_rows: list[list[int]] = []
    confirmed: list[str] = []
    for _, row in df.iterrows():
        dim_rows.append(_parse_catalog_paths(row.get("sxhy_catalog_paths")))
        confirmed.append(_confirmed_imagery(row))

    # ── 计算每维度的 IDF ──────────────────────────────────────────────────
    N = len(dim_rows)
    # df_counts[i] = 该维度 count > 0 的诗数
    df_counts = [sum(1 for r in dim_rows if r[i] > 0) for i in range(9)]
    # 加 1 平滑，避免 log(0)
    idfs = [math.log(N / max(1, dfc)) for dfc in df_counts]

    print("=== 9 维 IDF 权重 ===")
    for name, idf, dfc in zip(DIM_NAMES, idfs, df_counts):
        print(f"  {name}: df={dfc} ({dfc/N*100:.0f}%), IDF={idf:.3f}")

    # ── 写入原始计数列 ────────────────────────────────────────────────────
    for i, name in enumerate(DIM_NAMES):
        df[name] = [r[i] for r in dim_rows]
    df["confirmed_imagery"] = confirmed

    # ── 生成两种 dim_vector ────────────────────────────────────────────────
    # dim_vector:     L2 归一化原始计数（保留，兼容旧代码）
    # dim_vector_idf: IDF 加权后 L2 归一化（推荐使用）
    raw_vecs: list[str] = []
    idf_vecs: list[str] = []
    for r in dim_rows:
        raw_vecs.append(json.dumps(_l2_normalize([float(x) for x in r])))
        weighted = [r[i] * idfs[i] for i in range(9)]
        idf_vecs.append(json.dumps(_l2_normalize(weighted)))

    df["dim_vector"] = raw_vecs
    df["dim_vector_idf"] = idf_vecs

    # ── 合并月份数字 ──────────────────────────────────────────────────────
    month_num_col = "月份数字"
    if args.meta.exists():
        meta = pd.read_csv(args.meta, usecols=lambda c: c in ("ID", month_num_col))
        df = df.merge(meta, on="ID", how="left")
    else:
        df[month_num_col] = pd.NA

    # ── 列排序 ────────────────────────────────────────────────────────────
    base_cols = list(df.columns)
    new_cols = DIM_NAMES + ["dim_vector", "dim_vector_idf", "confirmed_imagery", month_num_col]
    for c in new_cols:
        if c in base_cols:
            base_cols.remove(c)
    insert_after = "sxhy_catalog_paths"
    if insert_after in base_cols:
        idx = base_cols.index(insert_after) + 1
    else:
        idx = len(base_cols)
    ordered = base_cols[:idx] + new_cols + base_cols[idx:]
    seen: set[str] = set()
    final_order = []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            final_order.append(c)
    df = df[[c for c in final_order if c in df.columns]]

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n行数: {len(df)}  列数: {len(df.columns)}")
    print(f"已保存: {args.output}")
    with_conf = sum(1 for x in confirmed if x)
    print(f"含 confirmed_imagery 的诗: {with_conf}/{N}")


if __name__ == "__main__":
    main()
