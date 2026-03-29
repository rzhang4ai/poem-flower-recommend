#!/usr/bin/env python3
"""
仅根据数据库中已有的 image_url 下载图片到本地，并更新 image_local_path。
适合先跑爬虫时用了 --no-images，查看数据后再补下图片。

用法：
  python download_images_only.py              # 下载所有尚未有本地图片的商品
  python download_images_only.py --limit 50   # 最多下载 50 张
  python download_images_only.py --delay 0.5  # 请求间隔 0.5 秒
"""

import argparse
import sqlite3
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
IMAGES_DIR = SCRIPT_DIR / "images" / "brighten_hk"
DB_PATH = DATA_DIR / "flower_supply.db"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def main():
    parser = argparse.ArgumentParser(description="仅下载商品图片（依据 DB 中的 image_url）")
    parser.add_argument("--limit", type=int, default=0, help="最多下载张数，0 表示不限制")
    parser.add_argument("--delay", type=float, default=0.3, help="每张之间的间隔（秒）")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """SELECT id, external_id, image_url FROM product
           WHERE image_url IS NOT NULL AND TRIM(image_url) != ''
           AND (image_local_path IS NULL OR TRIM(image_local_path) = '')"""
    )
    rows = cur.fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]
    conn.close()

    if not rows:
        print("没有需要下载图片的商品（均已下载或无 image_url）。")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for pid, external_id, image_url in rows:
        try:
            path = IMAGES_DIR / f"{external_id}.jpg"
            req = urllib.request.Request(image_url.strip(), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                path.write_bytes(resp.read())
            rel = str(path.relative_to(SCRIPT_DIR))
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE product SET image_local_path = ? WHERE id = ?", (rel, pid))
            conn.commit()
            conn.close()
            ok += 1
            print(f"  [{ok + fail}/{len(rows)}] {external_id} OK")
        except Exception as e:
            fail += 1
            print(f"  [{ok + fail}/{len(rows)}] {external_id} 失败: {e}")
        time.sleep(args.delay)

    print(f"\n完成: 成功 {ok}, 失败 {fail}。")


if __name__ == "__main__":
    main()
