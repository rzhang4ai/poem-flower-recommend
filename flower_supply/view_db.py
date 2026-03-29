#!/usr/bin/env python3
"""
查看 flower_supply.db 中的数据摘要与样本，或导出为 CSV。
用法：
  python view_db.py                    # 打印摘要 + 前 20 条商品
  python view_db.py --limit 50         # 前 50 条
  python view_db.py --export products.csv   # 导出全部商品为 CSV
  python view_db.py --no-preview --export products.csv   # 只导出不打印
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "data" / "flower_supply.db"


def run():
    parser = argparse.ArgumentParser(description="查看 flower_supply 数据库")
    parser.add_argument("--limit", type=int, default=20, help="预览时显示的商品条数，0 表示不预览")
    parser.add_argument("--export", metavar="FILE", help="导出商品表到 CSV 文件")
    parser.add_argument("--no-preview", action="store_true", help="与 --export 同用时只导出不打印预览")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # 确保新列存在（与 crawl 的 init_db 迁移一致）
    cur.execute("PRAGMA table_info(product)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("color", "unit", "size", "origin", "care_instructions"):
        if col not in existing:
            cur.execute(f"ALTER TABLE product ADD COLUMN {col} TEXT")
    conn.commit()

    # 摘要
    cur.execute("SELECT COUNT(*) FROM product")
    n_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM product WHERE image_url IS NOT NULL AND image_url != ''")
    n_with_image_url = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM product WHERE image_local_path IS NOT NULL AND image_local_path != ''")
    n_with_local_image = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM product WHERE price IS NOT NULL")
    n_with_price = cur.fetchone()[0]

    cur.execute(
        "SELECT c.name, COUNT(p.id) AS cnt FROM product p JOIN category c ON p.category_id = c.id GROUP BY p.category_id"
    )
    by_category = cur.fetchall()

    print("=" * 60)
    print("flower_supply 数据摘要")
    print("=" * 60)
    print(f"商品总数:     {n_total}")
    print(f"有价格:       {n_with_price}")
    print(f"有图片 URL:   {n_with_image_url}")
    print(f"已下载图片:   {n_with_local_image}")
    print("\n按分类:")
    for row in by_category:
        print(f"  {row['name']}: {row['cnt']} 条")
    print("=" * 60)

    # 预览
    if args.limit > 0 and not args.no_preview:
        cur.execute(
            """SELECT p.external_id, p.name, p.price, p.currency, p.spec_text, p.color, p.unit, p.size, p.origin,
                      p.image_url, p.image_local_path, p.product_url, p.fetched_at
               FROM product p ORDER BY p.id LIMIT ?""",
            (args.limit,),
        )
        rows = cur.fetchall()
        print(f"\n前 {len(rows)} 条商品预览:\n")
        for r in rows:
            name = (r["name"] or "")[:32] + ("..." if len(r["name"] or "") > 32 else "")
            price = f"{r['price']} {r['currency']}" if r["price"] is not None else "-"
            img = "有" if (r["image_url"] and r["image_url"].strip()) else "无"
            local = "已下载" if (r["image_local_path"] and r["image_local_path"].strip()) else "-"
            unit_val = r["unit"] if "unit" in r.keys() else None
            color_val = r["color"] if "color" in r.keys() else None
            unit = (unit_val or "")[:16] if unit_val else "-"
            color = (color_val or "")[:12] if color_val else "-"
            print(f"  ID {r['external_id']:>6} | {price:>10} | 单位:{unit} 颜色:{color} | 图:{img} {local} | {name}")
            if r["product_url"]:
                print(f"            {r['product_url'][:70]}...")
        print()

    # 导出 CSV
    if args.export:
        out_path = Path(args.export)
        if not out_path.is_absolute():
            out_path = SCRIPT_DIR / out_path
        cur.execute(
            """SELECT external_id, name, price, currency, spec_text, color, unit, size, origin, care_instructions,
                      image_url, image_local_path, product_url, stock_status, fetched_at
               FROM product ORDER BY id"""
        )
        rows = cur.fetchall()
        fieldnames = list(rows[0].keys()) if rows else ["external_id", "name", "price", "currency", "spec_text", "color", "unit", "size", "origin", "care_instructions", "image_url", "image_local_path", "product_url", "stock_status", "fetched_at"]
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])
        print(f"已导出 {len(rows)} 条 → {out_path}")

    conn.close()


if __name__ == "__main__":
    run()
