#!/usr/bin/env python3
"""
poem_product_match_v1.py — 诗词花名 vs 供应商花材匹配（Brighten HK）
====================================================================
目标：
1) 统计诗词数据集中的花名，有多少能在供应商商品中匹配到（可买到）。
2) 统计供应商商品中有哪些“品种/花材”诗词数据集中没有出现（缺口）。
3) 生成可视化 HTML 报告 + CSV 结果文件，便于评估是否需要扩大诗词规模或引入当代文本。

输入：
- poems_dataset_merged.csv（默认：../poems_dataset/poems_dataset_merged.csv，字段：花名）
- flower_supply.db（默认：./data/flower_supply.db，表：product）

输出（默认到 ./match_v1/ ）：
- poem_product_match_v1_report.html
- poem_flowers_coverage.csv
- matched_products.csv
- unmatched_products.csv
- supplier_only_varieties.csv

说明：
- 供应商站点为香港繁体页面，本脚本会尝试使用 OpenCC 做繁简转换。
- 本项目已将 opencc-python-reimplemented 安装到 ./ .deps（避免系统权限问题）。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SCRIPT_DIR / "data" / "flower_supply.db"
DEFAULT_POEM_CSV = (SCRIPT_DIR.parent / "poems_dataset" / "poems_dataset_merged.csv")
OUT_DIR_DEFAULT = SCRIPT_DIR / "match_v1"


def _try_opencc():
    """
    尝试导入 OpenCC。
    - 优先从本项目的 ./ .deps 导入（已安装 opencc-python-reimplemented）
    - 若失败则返回 (None, None)
    """
    try:
        from opencc import OpenCC  # type: ignore
        return OpenCC("t2s"), OpenCC("s2t")
    except Exception:
        deps = SCRIPT_DIR / ".deps"
        if deps.exists():
            import sys
            sys.path.insert(0, str(deps))
            try:
                from opencc import OpenCC  # type: ignore
                return OpenCC("t2s"), OpenCC("s2t")
            except Exception:
                return None, None
        return None, None


T2S, S2T = _try_opencc()


def t2s(text: str) -> str:
    if not text:
        return ""
    if T2S is None:
        return text
    try:
        return T2S.convert(text)
    except Exception:
        return text


def s2t(text: str) -> str:
    if not text:
        return ""
    if S2T is None:
        return text
    try:
        return S2T.convert(text)
    except Exception:
        return text


def norm_text(s: str) -> str:
    """用于匹配：去空白与常见标点、统一全角括号等。"""
    s = (s or "").strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[·•・,，。;；:：!！?？\"“”'‘’]", "", s)
    return s


def base_variety_from_product_name(name: str) -> str:
    """
    从供应商商品名称中抽取“品种/花材主名”。
    示例：
      '葉材 - 五針松' -> '五針松'
      '大理花 - 綠野仙蹤' -> '大理花'
      '維也納菊（可選色）' -> '維也納菊'
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    # 去括号备注
    raw = re.split(r"[（(]", raw, maxsplit=1)[0].strip()
    # 去前缀类目
    raw = re.sub(r"^(鮮花|葉材|配花|觀果|乾花|永生花|花器|花泥|包裝|工具)\s*[-－]\s*", "", raw)
    # 若包含 ' - ' / '－'：取左侧作为主类（如 大理花 - 梅根）
    if " - " in raw:
        raw = raw.split(" - ", 1)[0].strip()
    if "－" in raw:
        raw = raw.split("－", 1)[0].strip()
    return raw.strip()


def build_flower_tokens(poem_flowers: list[str]) -> dict[str, set[str]]:
    """
    为每个诗词花名生成一组可匹配 token：
    - 原名（简体）
    - 去/补“花”后缀
    - 转繁体、转简体（若可用）
    """
    out: dict[str, set[str]] = {}
    for f in poem_flowers:
        f = (f or "").strip()
        if not f:
            continue
        tokens = set()
        tokens.add(f)
        if f.endswith("花"):
            tokens.add(f[:-1])
        else:
            tokens.add(f + "花")
        # 繁简互转（不依赖则原样）
        more = set()
        for t in list(tokens):
            more.add(t2s(t))
            more.add(s2t(t))
        tokens |= more
        # 归一化
        tokens = {norm_text(x) for x in tokens if len(norm_text(x)) >= 2}
        out[f] = tokens
    return out


@dataclass
class ProductRow:
    external_id: str
    name: str
    price: float | None
    currency: str
    product_url: str
    category_id: int | None


def load_poem_flowers(poem_csv: Path) -> tuple[list[str], Counter]:
    with open(poem_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        flowers = []
        for r in reader:
            flowers.append((r.get("花名") or "").strip())
    cnt = Counter([x for x in flowers if x])
    uniq = sorted(cnt.keys(), key=lambda x: (-cnt[x], x))
    return uniq, cnt


def load_products(db_path: Path) -> list[ProductRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT external_id, name, price, currency, product_url, category_id FROM product ORDER BY id"
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            ProductRow(
                external_id=str(r["external_id"] or ""),
                name=str(r["name"] or ""),
                price=r["price"],
                currency=str(r["currency"] or "HKD"),
                product_url=str(r["product_url"] or ""),
                category_id=r["category_id"],
            )
        )
    conn.close()
    return rows


def match_products(poem_tokens: dict[str, set[str]], products: list[ProductRow]):
    # 预构建：每个 token -> 归属的 poem flower（token 可能对应多个花名，取并集）
    token_to_flowers: dict[str, set[str]] = defaultdict(set)
    for flower, toks in poem_tokens.items():
        for t in toks:
            token_to_flowers[t].add(flower)

    poem_flowers = list(poem_tokens.keys())
    poem_matched_counts = Counter()
    poem_to_products: dict[str, list[str]] = defaultdict(list)

    matched_products = []
    unmatched_products = []

    # 为避免 O(N*M) 太慢：先把 token 按长度降序，命中较长 token 优先
    all_tokens = sorted(token_to_flowers.keys(), key=len, reverse=True)

    for p in products:
        name_norm = norm_text(p.name)
        name_s = norm_text(t2s(p.name))
        name_t = norm_text(s2t(p.name))
        variety = base_variety_from_product_name(p.name)
        variety_norm = norm_text(variety)
        variety_s = norm_text(t2s(variety))
        variety_t = norm_text(s2t(variety))

        haystacks = {name_norm, name_s, name_t, variety_norm, variety_s, variety_t}

        matched = set()
        for tok in all_tokens:
            if len(tok) < 2:
                continue
            # 任一 haystack 含 token 即视作命中
            if any(tok and (tok in h) for h in haystacks if h):
                matched |= token_to_flowers[tok]

        if matched:
            for f in matched:
                poem_matched_counts[f] += 1
                poem_to_products[f].append(p.external_id)
            matched_products.append((p, variety, sorted(matched)))
        else:
            unmatched_products.append((p, variety))

    return poem_matched_counts, poem_to_products, matched_products, unmatched_products


def escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def svg_bar_horizontal(labels, values, max_width=520, bar_height=20, color="#3498db"):
    if not labels:
        return "<p>无数据</p>"
    max_val = max(values) or 1
    h = len(labels) * (bar_height + 6) + 30
    w = max_width + 240
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = 20 + i * (bar_height + 6)
        width = (val / max_val) * max_width
        lines.append(f'<rect x="0" y="{y}" width="{width}" height="{bar_height}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{max_width + 8}" y="{y + bar_height - 5}" font-size="12" fill="#333">{escape(lab)}</text>')
        lines.append(f'<text x="{max_width - 6}" y="{y + bar_height - 5}" font-size="11" fill="#333" text-anchor="end">{val}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def svg_bar_vertical(labels, values, max_height=260, bar_w=26, color="#2ecc71"):
    if not labels:
        return "<p>无数据</p>"
    max_val = max(values) or 1
    w = len(labels) * (bar_w + 6) + 80
    h = max_height + 60
    base_y = max_height + 20
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = 40 + i * (bar_w + 6)
        bh = (val / max_val) * max_height
        lines.append(f'<rect x="{x}" y="{base_y - bh}" width="{bar_w}" height="{bh}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{x + bar_w/2}" y="{base_y + 18}" font-size="10" fill="#333" text-anchor="middle">{escape(lab)}</text>')
        lines.append(f'<text x="{x + bar_w/2}" y="{base_y - bh - 4}" font-size="9" fill="#333" text-anchor="middle">{val}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="诗词花名 vs 供应商花材匹配（v1）")
    ap.add_argument("--poem-csv", default=str(DEFAULT_POEM_CSV), help="诗词合并 CSV（包含列：花名）")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="供应商 SQLite DB（flower_supply.db）")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT), help="输出目录")
    ap.add_argument("--top-n", type=int, default=25, help="可视化展示 Top N")
    args = ap.parse_args()

    poem_csv = Path(args.poem_csv)
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not poem_csv.exists():
        raise SystemExit(f"找不到 poem CSV: {poem_csv}")
    if not db_path.exists():
        raise SystemExit(f"找不到 DB: {db_path}")

    poem_flowers, poem_counts = load_poem_flowers(poem_csv)
    products = load_products(db_path)

    poem_tokens = build_flower_tokens(poem_flowers)
    matched_counts, poem_to_products, matched_products, unmatched_products = match_products(poem_tokens, products)

    poem_available = sorted([f for f in poem_flowers if matched_counts.get(f, 0) > 0])
    poem_missing = sorted([f for f in poem_flowers if matched_counts.get(f, 0) == 0])

    # 供应商“品种”统计：以 variety_guess 聚合
    variety_counter = Counter()
    variety_matched = Counter()
    variety_samples = defaultdict(list)
    for p, variety, mflowers in matched_products:
        v = variety or ""
        variety_counter[v] += 1
        variety_matched[v] += 1
        if len(variety_samples[v]) < 3:
            variety_samples[v].append((p.external_id, p.name, p.product_url))
    for p, variety in unmatched_products:
        v = variety or ""
        variety_counter[v] += 1
        if len(variety_samples[v]) < 3:
            variety_samples[v].append((p.external_id, p.name, p.product_url))

    supplier_only_varieties = [v for v in variety_counter.keys() if variety_matched.get(v, 0) == 0]

    # 输出 CSV：诗词花名覆盖
    cov_rows = []
    for f in poem_flowers:
        ex = poem_to_products.get(f, [])[:8]
        cov_rows.append(
            {
                "poem_flower": f,
                "poem_count": poem_counts.get(f, 0),
                "matched_products_count": matched_counts.get(f, 0),
                "example_product_external_ids": " | ".join(ex),
            }
        )
    write_csv(
        out_dir / "poem_flowers_coverage.csv",
        ["poem_flower", "poem_count", "matched_products_count", "example_product_external_ids"],
        cov_rows,
    )

    # 输出 CSV：匹配/未匹配商品
    mp_rows = []
    for p, variety, mflowers in matched_products:
        mp_rows.append(
            {
                "external_id": p.external_id,
                "name": p.name,
                "variety_guess": variety,
                "price": p.price,
                "currency": p.currency,
                "matched_poem_flowers": " | ".join(mflowers),
                "product_url": p.product_url,
            }
        )
    write_csv(
        out_dir / "matched_products.csv",
        ["external_id", "name", "variety_guess", "price", "currency", "matched_poem_flowers", "product_url"],
        mp_rows,
    )

    up_rows = []
    for p, variety in unmatched_products:
        up_rows.append(
            {
                "external_id": p.external_id,
                "name": p.name,
                "variety_guess": variety,
                "price": p.price,
                "currency": p.currency,
                "product_url": p.product_url,
            }
        )
    write_csv(
        out_dir / "unmatched_products.csv",
        ["external_id", "name", "variety_guess", "price", "currency", "product_url"],
        up_rows,
    )

    # 输出 CSV：供应商独有品种（未在诗词花名中命中）
    sv_rows = []
    for v in sorted(supplier_only_varieties, key=lambda x: (-variety_counter[x], x)):
        samples = variety_samples.get(v, [])
        sv_rows.append(
            {
                "supplier_variety": v,
                "product_count": variety_counter.get(v, 0),
                "sample_products": " | ".join([f"{sid}:{sname}" for sid, sname, _ in samples]),
            }
        )
    write_csv(out_dir / "supplier_only_varieties.csv", ["supplier_variety", "product_count", "sample_products"], sv_rows)

    # HTML 报告
    poem_total = len(poem_flowers)
    poem_available_n = len(poem_available)
    poem_missing_n = len(poem_missing)
    prod_total = len(products)
    prod_matched = len(matched_products)
    prod_unmatched = len(unmatched_products)

    top_n = max(5, int(args.top_n))
    # Top 诗词花名：按“命中的商品数”排序
    top_poem = sorted(poem_flowers, key=lambda f: (-matched_counts.get(f, 0), -poem_counts.get(f, 0), f))[:top_n]
    top_poem_vals = [matched_counts.get(f, 0) for f in top_poem]
    # Top 供应商独有品种
    top_sup_only = sorted(supplier_only_varieties, key=lambda v: (-variety_counter.get(v, 0), v))[:top_n]
    top_sup_only_vals = [variety_counter.get(v, 0) for v in top_sup_only]

    opencc_status = "可用（已做繁简互转）" if T2S is not None else "不可用（仅做原文/简单后缀匹配，覆盖率可能偏低）"

    html = []
    html.append(
        f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>诗词花名 vs 供应商花材匹配报告（v1）</title>
<style>
 body {{ font-family: "PingFang SC","Microsoft YaHei",sans-serif; margin: 24px; background:#fafafa; color:#222; max-width:1200px; }}
 h1 {{ border-bottom:2px solid #3498db; padding-bottom:8px; color:#2c3e50; }}
 h2 {{ margin-top:28px; color:#34495e; }}
 .meta, .card {{ background:#fff; border-radius:8px; padding:14px 16px; box-shadow:0 1px 4px rgba(0,0,0,0.06); margin:14px 0; }}
 .meta p {{ margin:6px 0; }}
 table {{ border-collapse:collapse; width:100%; background:#fff; }}
 th, td {{ border:1px solid #ddd; padding:8px 10px; font-size:13px; vertical-align:top; }}
 th {{ background:#3498db; color:#fff; text-align:left; }}
 .tag {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#ecf0f1; margin-right:6px; }}
 .small {{ color:#666; font-size:12px; }}
 a {{ color:#1f6feb; text-decoration:none; }}
 a:hover {{ text-decoration:underline; }}
 .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
 @media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
 code {{ background:#eee; padding:1px 6px; border-radius:4px; }}
 .footer {{ margin-top:30px; color:#777; font-size:12px; }}
 .warn {{ background:#fff3cd; border:1px solid #ffe69c; padding:10px 12px; border-radius:8px; }}
</style></head><body>
<h1>诗词花名 vs 供应商花材匹配报告（v1）</h1>
<div class="meta">
  <p><span class="tag">诗词数据</span> <code>{escape(str(poem_csv))}</code></p>
  <p><span class="tag">供应商数据</span> <code>{escape(str(db_path))}</code></p>
  <p><span class="tag">繁简转换</span> {escape(opencc_status)}</p>
  <p><strong>诗词花名（去重）</strong>: {poem_total}；<strong>可在供应商买到</strong>: {poem_available_n}；<strong>买不到（未命中）</strong>: {poem_missing_n}</p>
  <p><strong>供应商商品</strong>: {prod_total}；<strong>命中诗词花名</strong>: {prod_matched}；<strong>未命中</strong>: {prod_unmatched}</p>
</div>
"""
    )
    if T2S is None:
        html.append('<div class="warn"><strong>提示</strong>：当前未检测到 OpenCC，繁简差异会显著影响匹配结果。建议在 flower_supply 目录确保存在 <code>.deps/opencc</code> 或安装 opencc。</div>')

    html.append("<h2>一、诗词花名覆盖率（Top）</h2>")
    html.append('<div class="card"><p class="small">按“命中到的供应商商品数”排序。更适合评估商业闭环覆盖。</p>')
    html.append(svg_bar_horizontal(top_poem, top_poem_vals, color="#e67e22"))
    html.append("</div>")

    html.append("<h2>二、供应商独有花材（Top）</h2>")
    html.append('<div class="card"><p class="small">这些“品种主名”在本次匹配规则下未命中任何诗词花名，可用来评估是否引入当代文本或扩大诗词花名表。</p>')
    html.append(svg_bar_horizontal(top_sup_only, top_sup_only_vals, color="#8e44ad"))
    html.append("</div>")

    html.append("<h2>三、明细表</h2>")
    html.append('<div class="cols">')

    # 可买到：诗词花名 -> 商品示例
    html.append('<div class="card">')
    html.append("<h3>可在供应商买到的诗词花名</h3>")
    html.append('<table><tr><th>花名</th><th>诗词条数</th><th>命中商品数</th><th>示例商品</th></tr>')
    for f in sorted(poem_available, key=lambda x: (-matched_counts.get(x, 0), x)):
        ex_ids = poem_to_products.get(f, [])[:5]
        ex_links = []
        # 从 matched_products 里取名称与链接
        id_to_prod = {p.external_id: p for p, _, _ in matched_products}
        for eid in ex_ids:
            p = id_to_prod.get(eid)
            if not p:
                continue
            ex_links.append(f'<a href="{escape(p.product_url)}" target="_blank">{escape(p.name)}</a>')
        html.append(
            f"<tr><td>{escape(f)}</td>"
            f"<td>{poem_counts.get(f,0)}</td>"
            f"<td>{matched_counts.get(f,0)}</td>"
            f"<td>{'<br/>'.join(ex_links) if ex_links else ''}</td></tr>"
        )
    html.append("</table></div>")

    # 买不到：未命中的诗词花名
    html.append('<div class="card">')
    html.append("<h3>诗词中出现但供应商未命中的花名</h3>")
    html.append('<table><tr><th>花名</th><th>诗词条数</th></tr>')
    for f in sorted(poem_missing, key=lambda x: (-poem_counts.get(x, 0), x)):
        html.append(f"<tr><td>{escape(f)}</td><td>{poem_counts.get(f,0)}</td></tr>")
    html.append("</table></div>")

    html.append("</div>")  # cols

    # 供应商未命中商品示例
    html.append("<h2>四、供应商未命中商品（抽样）</h2>")
    html.append('<div class="card"><p class="small">用于人工检查是否需要新增别名规则（例如某些花名在商品里用英文/俗名/缩写）。</p>')
    html.append('<table><tr><th>external_id</th><th>商品名</th><th>variety_guess</th><th>价格</th><th>链接</th></tr>')
    for p, variety in unmatched_products[:50]:
        price = f"{p.price} {p.currency}" if p.price is not None else "-"
        html.append(
            f"<tr><td>{escape(p.external_id)}</td><td>{escape(p.name)}</td><td>{escape(variety)}</td>"
            f"<td>{escape(price)}</td><td><a href=\"{escape(p.product_url)}\" target=\"_blank\">打开</a></td></tr>"
        )
    html.append("</table></div>")

    html.append(
        f'<div class="footer">输出目录：<code>{escape(str(out_dir))}</code>（已生成 CSV 与本报告 HTML）</div>'
    )
    html.append("</body></html>")

    report_path = out_dir / "poem_product_match_v1_report.html"
    report_path.write_text("".join(html), encoding="utf-8")

    print("✅ 已生成匹配报告：", report_path)
    print("✅ CSV 输出：", out_dir)
    print(f"  - 诗词花名可买到：{poem_available_n}/{poem_total}")
    print(f"  - 供应商商品命中：{prod_matched}/{prod_total}")


if __name__ == "__main__":
    main()

