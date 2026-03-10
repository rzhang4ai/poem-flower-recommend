#!/usr/bin/env python3
"""
繽紛 Brighten-Mall 鲜切花爬虫
从 https://www.brighten.hk/page/030 等分类页抓取商品链接，再进入详情页抓取：
品种（标题）、规格、价格、图片、链接等，写入 SQLite，可选下载图片。

依赖：pip install playwright && playwright install chromium
用法：
  python crawl_brighten_hk.py
  python crawl_brighten_hk.py --category-url "https://www.brighten.hk/v2/official/SalePageCategory/9795" --max-products 20
  python crawl_brighten_hk.py --no-images --delay 2
"""

import argparse
import os
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path

# 项目内 data 目录
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
IMAGES_DIR = SCRIPT_DIR / "images" / "brighten_hk"
DB_PATH = DATA_DIR / "flower_supply.db"

# 默认入口（鲜切花）
DEFAULT_CATEGORY_URL = "https://www.brighten.hk/page/030"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 列表页采集控制参数默认值（可通过 CLI 覆盖）
DEFAULT_MAX_SCROLLS = 40  # 每页最多下拉次数
DEFAULT_MAX_PAGES = 5     # 最多翻页次数（Next）

# Brighten 分类映射（用于写入数据库 category 表，与网站层级对应）
# 说明：external_id 对应 URL 中的分类 ID（SalePageCategory/{id}）或 page/xxx 的 xxx
BRIGHTEN_CATEGORIES = {
    "030": {
        "name": "鲜切花",
        "full_path": "首頁/鲜切花",
        "parent_external_id": None,
        "level": 1,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/page/030",
    },
    "9795": {
        "name": "鮮花花材葉材",
        "full_path": "首頁/鮮花花材葉材",
        "parent_external_id": None,
        "level": 1,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/9795",
    },
    "13027": {
        "name": "次日送",
        "full_path": "首頁/鮮花花材葉材/次日送",
        "parent_external_id": "9795",
        "level": 2,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/13027",
    },
    "9809": {
        "name": "雲南鮮花",
        "full_path": "首頁/鮮花花材葉材/雲南鮮花",
        "parent_external_id": "9795",
        "level": 2,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/9809",
    },
    "9796": {
        "name": "荷蘭鮮花",
        "full_path": "首頁/鮮花花材葉材/荷蘭鮮花",
        "parent_external_id": "9795",
        "level": 2,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/9796",
    },
    "14878": {
        "name": "外國鮮花",
        "full_path": "首頁/鮮花花材葉材/外國鮮花",
        "parent_external_id": "9795",
        "level": 2,
        "category_type": "cut_flower",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/14878",
    },
    "9864": {
        "name": "盆栽植物",
        "full_path": "首頁/盆栽植物",
        "parent_external_id": None,
        "level": 1,
        "category_type": "potted",
        "url": "https://www.brighten.hk/v2/official/SalePageCategory/9864",
    },
}


def extract_category_external_id(category_url: str) -> str:
    """
    从分类 URL 提取 external_id：
    - /v2/official/SalePageCategory/9809?... -> 9809
    - /page/030 -> 030
    """
    m = re.search(r"/SalePageCategory/(\d+)", category_url)
    if m:
        return m.group(1)
    m = re.search(r"/page/(\d+)", category_url)
    if m:
        return m.group(1)
    return "unknown"


def ensure_category(conn: sqlite3.Connection, supplier_id: int, external_id: str, category_url: str) -> int:
    """
    确保 category 表中存在该 external_id，并返回 category_id（自增主键）。
    若 external_id 在 BRIGHTEN_CATEGORIES 中，则写入/更新 name/full_path/parent/level。
    """
    meta = BRIGHTEN_CATEGORIES.get(external_id, {})
    name = meta.get("name") or f"Category {external_id}"
    full_path = meta.get("full_path")
    parent_external_id = meta.get("parent_external_id")
    level = meta.get("level")
    category_type = meta.get("category_type")

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO category (supplier_id, external_id, name, full_path, parent_external_id, level, category_type, url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(supplier_id, external_id) DO UPDATE SET
          name=excluded.name,
          full_path=COALESCE(excluded.full_path, category.full_path),
          parent_external_id=COALESCE(excluded.parent_external_id, category.parent_external_id),
          level=COALESCE(excluded.level, category.level),
          category_type=COALESCE(excluded.category_type, category.category_type),
          url=excluded.url
        """,
        (supplier_id, external_id, name, full_path, parent_external_id, level, category_type, category_url),
    )
    conn.commit()
    cur.execute("SELECT id FROM category WHERE supplier_id=? AND external_id=?", (supplier_id, external_id))
    row = cur.fetchone()
    return int(row[0])


def init_db():
    """初始化数据库与 supplier/category 记录，并迁移已有表增加新列。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    schema_path = SCRIPT_DIR / "schema.sql"
    conn = sqlite3.connect(DB_PATH)
    if schema_path.exists():
        conn.executescript(schema_path.read_text(encoding="utf-8"))
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO supplier (id, name, slug, base_url) VALUES (1, '繽紛 Brighten-Mall', 'brighten_hk', 'https://www.brighten.hk')"
    )
    # 迁移 category 表新增列（若不存在）
    cur.execute("PRAGMA table_info(category)")
    cat_cols = {row[1] for row in cur.fetchall()}
    for col in ("full_path", "parent_external_id", "level", "category_type"):
        if col not in cat_cols:
            if col == "level":
                cur.execute("ALTER TABLE category ADD COLUMN level INTEGER")
            elif col == "category_type":
                cur.execute("ALTER TABLE category ADD COLUMN category_type TEXT")
            else:
                cur.execute(f"ALTER TABLE category ADD COLUMN {col} TEXT")

    # 初始化常用分类（与网站结构对应）
    for ext_id, meta in BRIGHTEN_CATEGORIES.items():
        cur.execute(
            """
            INSERT OR IGNORE INTO category (supplier_id, external_id, name, full_path, parent_external_id, level, category_type, url, created_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                ext_id,
                meta.get("name"),
                meta.get("full_path"),
                meta.get("parent_external_id"),
                meta.get("level"),
                meta.get("category_type"),
                meta.get("url"),
            ),
        )
    # 为已有数据库添加新列（若不存在）
    cur.execute("PRAGMA table_info(product)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("color", "unit", "size", "origin", "care_instructions"):
        if col not in existing:
            cur.execute(f"ALTER TABLE product ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()


def get_product_links_from_page(page, base_url: str) -> list[str]:
    """从当前已加载的页面中提取所有商品详情页链接（去重）。"""
    els = page.query_selector_all('a[href*="/SalePage/Index/"]')
    out = []
    seen = set()
    for el in els:
        try:
            h = el.get_attribute("href")
            if not h:
                continue
            if h.startswith("/"):
                h = urllib.parse.urljoin(base_url, h)
            if h in seen or "/SalePage/Index/" not in h:
                continue
            seen.add(h)
            out.append(h)
        except Exception:
            continue
    return out


def collect_all_product_links(page, category_url: str, max_scrolls: int = DEFAULT_MAX_SCROLLS) -> list[str]:
    """
    在分类页上滚动加载更多商品（若有无限滚动），并收集所有商品链接。
    若链接数量在连续两次滚动后不变则停止。
    """
    base_url = urllib.parse.urljoin(category_url, "/")
    seen_count = -1
    stable_rounds = 0
    all_links = []

    for _ in range(max_scrolls):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        all_links = get_product_links_from_page(page, base_url)
        if len(all_links) == seen_count:
            stable_rounds += 1
            if stable_rounds >= 2:
                break
        else:
            stable_rounds = 0
        seen_count = len(all_links)
    return all_links


def collect_links_with_paging(
    page,
    category_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_scrolls: int = DEFAULT_MAX_SCROLLS,
) -> list[str]:
    """
    处理「滚动加载 + 多页」的分类：
    - 对当前页反复下拉，直到链接数量稳定或达到 max_scrolls。
    - 若存在「下一頁 / Next」按钮，则点击进入下一页，重复上述步骤，直到到达 max_pages 或无下一页。
    """
    all_links: list[str] = []
    visited_pages = 0

    while True:
        visited_pages += 1
        page_links = collect_all_product_links(page, category_url, max_scrolls=max_scrolls)
        # 累积所有页面的商品链接
        merged = list({*all_links, *page_links})
        merged.sort()
        all_links = merged

        if visited_pages >= max_pages:
            break

        # 尝试找到“下一頁/Next”按钮
        next_clicked = False
        # 1) 常见 aria/rel 选择器
        for sel in [
            "a[rel='next']",
            "a[aria-label='Next']",
            "button[aria-label='Next']",
            "a[aria-label*='下一']",
            "button[aria-label*='下一']",
        ]:
            btn = page.query_selector(sel)
            if btn:
                try:
                    btn.click()
                    page.wait_for_timeout(3000)
                    next_clicked = True
                    break
                except Exception:
                    continue
        # 2) 退而求其次，用文本匹配“下一頁/下一页”
        if not next_clicked:
            try:
                loc = page.get_by_text("下一", exact=False)
                if loc.count() > 0:
                    el = loc.first
                    txt = (el.inner_text() or "").strip()
                    if "頁" in txt or "页" in txt:
                        el.click()
                        page.wait_for_timeout(3000)
                        next_clicked = True
            except Exception:
                pass

        if not next_clicked:
            break

    return all_links


def extract_product_id(url: str) -> str | None:
    m = re.search(r"/SalePage/Index/(\d+)", url)
    return m.group(1) if m else None


def _extract_label_value(page, label_patterns: list[str], max_chars: int = 500) -> str:
    """
    从详情页中提取「标签：值」。label_patterns 如 ["顏色", "颜色"]，返回第一个匹配到的值。
    尝试：1) 页面文本正则 2) 含标签的单元格/下一单元格 3) 含标签元素的后继文本
    """
    # 1) 整页或主内容区文本 + 正则
    try:
        body = page.locator("body").inner_text()
        for label in label_patterns:
            # 匹配 "标签：" 或 "标签:" 后的内容，到换行或下一常见标签
            m = re.search(
                re.escape(label) + r"\s*[：:]\s*([^\n]+?)(?=\n|$|顏色|颜色|單位|单位|規格|规格|尺寸|呎吋|產地|产地|收花)",
                body,
                re.DOTALL,
            )
            if m:
                return m.group(1).strip()[:max_chars]
    except Exception:
        pass
    # 2) 表格或列表：找含标签的 td/th/dt，取下一格或同格内冒号后
    try:
        for label in label_patterns:
            loc = page.get_by_text(label, exact=False)
            if loc.count() == 0:
                continue
            el = loc.first.element_handle()
            if not el:
                continue
            # 同元素内 "标签：值"
            text = el.evaluate("el => el.innerText || ''")
            if isinstance(text, str) and label in text and "：" in text:
                parts = re.split(r"\s*[：:]\s*", text, 1)
                if len(parts) == 2:
                    return parts[1].strip()[:max_chars]
            # 下一兄弟节点（如下一 td）
            next_text = el.evaluate(
                "el => el.nextElementSibling ? (el.nextElementSibling.innerText || '').trim() : ''"
            )
            if isinstance(next_text, str) and next_text:
                return next_text[:max_chars]
            # 父的下一兄弟（如 tr 下一 td）
            parent_next = el.evaluate(
                "el => el.closest('tr') ? (el.closest('tr').querySelector('td:last-child, th:last-child')?.innerText || '').trim() : ''"
            )
            if isinstance(parent_next, str) and parent_next:
                return parent_next[:max_chars]
    except Exception:
        pass
    return ""


def scrape_product_detail(page, url: str) -> dict | None:
    """
    进入商品详情页，等待主要内容出现后提取：name, price, spec_text, image_url, product_url。
    若页面异常或超时则返回 None。
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3000)
    except Exception as e:
        print(f"  访问失败 {url}: {e}")
        return None

    data = {
        "external_id": extract_product_id(url) or "",
        "name": "",
        "price": None,
        "currency": "HKD",
        "spec_text": "",
        "color": "",
        "unit": "",
        "size": "",
        "origin": "",
        "care_instructions": "",
        "image_url": "",
        "product_url": url,
        "stock_status": "",
    }
    if not data["external_id"]:
        return None

    # 标题：常见选择器
    for sel in [
        "h1",
        "[class*='sale-page-title']",
        "[class*='product-title']",
        "meta[property='og:title']",
    ]:
        try:
            if sel.startswith("meta"):
                el = page.query_selector(sel)
                if el:
                    data["name"] = (el.get_attribute("content") or "").strip()
            else:
                el = page.query_selector(sel)
                if el:
                    data["name"] = (el.inner_text() or "").strip()
            if data["name"] and len(data["name"]) < 200:
                break
        except Exception:
            continue
    if not data["name"]:
        data["name"] = f"商品 {data['external_id']}"

    # 价格：含数字与小数/逗号的文本
    try:
        price_els = page.query_selector_all("[class*='price'], [class*='Price'], .amount, [itemprop='price']")
        for el in price_els:
            text = (el.inner_text() or "").replace(",", "").strip()
            m = re.search(r"[\d,]+\.?\d*", text)
            if m:
                data["price"] = float(m.group(0).replace(",", ""))
                if "USD" in text or "US$" in text:
                    data["currency"] = "USD"
                break
    except Exception:
        pass

    # 主图
    try:
        img = page.query_selector(
            "img[src*='cdn.91app'], img[src*='brighten'], .main-image img, [class*='product'] img, [class*='sale'] img"
        )
        if img:
            data["image_url"] = (img.get_attribute("src") or "").strip()
    except Exception:
        pass

    # 规格/单位/尺寸/颜色/产地/收花後護理方法：按标签提取（香港网页以繁体为主）
    data["color"] = _extract_label_value(page, ["顏色", "颜色"], max_chars=200)
    data["unit"] = _extract_label_value(page, ["單位", "单位", "規格", "规格"], max_chars=200)
    data["size"] = _extract_label_value(page, ["尺寸", "呎吋"], max_chars=200)
    data["origin"] = _extract_label_value(page, ["產地", "产地"], max_chars=200)
    data["care_instructions"] = _extract_label_value(page, ["收花後護理方法", "收花后护理方法"], max_chars=2000)
    # 整块「商品規格」作为 spec_text（若上述 unit 已取到枝数等，这里作补充）
    try:
        for block in page.query_selector_all("div, section, td, li"):
            text = (block.inner_text() or "").strip()
            if "商品規格" in text or ("規格" in text and len(text) < 800):
                data["spec_text"] = text[:500]
                break
        if not data["spec_text"] and data["unit"]:
            data["spec_text"] = data["unit"]
    except Exception:
        pass

    return data


def save_product(conn: sqlite3.Connection, category_id: int, supplier_id: int, row: dict):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO product (category_id, supplier_id, external_id, name, description, price, currency, spec_text, color, unit, size, origin, care_instructions, image_url, product_url, stock_status, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(supplier_id, external_id) DO UPDATE SET
          name=excluded.name, price=excluded.price, currency=excluded.currency,
          spec_text=excluded.spec_text, color=excluded.color, unit=excluded.unit, size=excluded.size,
          origin=excluded.origin, care_instructions=excluded.care_instructions,
          image_url=excluded.image_url, product_url=excluded.product_url,
          stock_status=excluded.stock_status, fetched_at=datetime('now')
        """,
        (
            category_id,
            supplier_id,
            row["external_id"],
            row["name"],
            None,
            row["price"],
            row["currency"],
            row["spec_text"] or None,
            row.get("color") or None,
            row.get("unit") or None,
            row.get("size") or None,
            row.get("origin") or None,
            row.get("care_instructions") or None,
            row["image_url"] or None,
            row["product_url"],
            row["stock_status"] or None,
        ),
    )
    conn.commit()


def download_image(image_url: str, external_id: str, image_dir: Path | None = None) -> str | None:
    """下载图片到 images/brighten_hk/{category_external_id}/{external_id}.jpg，返回相对路径或 None。"""
    if not image_url or not external_id:
        return None
    try:
        import urllib.request
        if image_dir is None:
            image_dir = IMAGES_DIR
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{external_id}.jpg"
        req = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            path.write_bytes(resp.read())
        return str(path.relative_to(SCRIPT_DIR))
    except Exception as e:
        print(f"  图片下载失败 {external_id}: {e}")
        return None


def run_crawl(
    category_url: str = DEFAULT_CATEGORY_URL,
    max_products: int = 50,
    delay_seconds: float = 2.0,
    download_images: bool = True,
    max_scrolls: int = DEFAULT_MAX_SCROLLS,
    max_pages: int = DEFAULT_MAX_PAGES,
):
    init_db()
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        import sys
        print("加载 playwright 失败，请在本机当前使用的 Python 环境中安装：")
        print(f"  当前 Python: {sys.executable}")
        print(f"  错误信息: {type(e).__name__}: {e}")
        print("  执行：")
        print("    pip install playwright")
        print("    playwright install chromium")
        print("  若已安装仍报错，请在终端执行以下命令排查：")
        print(f"    {sys.executable} -c \"from playwright.sync_api import sync_playwright; print('ok')\"")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        conn = sqlite3.connect(DB_PATH)

        all_links: list[str] = []
        try:
            page.goto(category_url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(4000)
            # 支持「滚动加载 + 多页」的分类（如：雲南鮮花 9809）
            all_links = collect_links_with_paging(
                page,
                category_url,
                max_pages=max_pages,
                max_scrolls=max_scrolls,
            )
        except Exception as e:
            print(f"列表页加载失败: {e}")
        all_links = all_links[: max_products] if max_products else all_links
        print(f"共收集到 {len(all_links)} 个商品链接，开始抓取详情…")

        # 根据 URL 识别分类 external_id，并写入/更新 category 表（与网站层级结构对应）
        category_external_id = extract_category_external_id(category_url)
        category_id = ensure_category(conn, supplier_id=1, external_id=category_external_id, category_url=category_url)
        image_dir = IMAGES_DIR / category_external_id
        for i, url in enumerate(all_links):
            print(f"  [{i+1}/{len(all_links)}] {url}")
            row = scrape_product_detail(page, url)
            if row:
                save_product(conn, category_id=category_id, supplier_id=1, row=row)
                if download_images and row.get("image_url"):
                    local = download_image(row["image_url"], row["external_id"], image_dir=image_dir)
                    if local:
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE product SET image_local_path=? WHERE supplier_id=1 AND external_id=?",
                            (local, row["external_id"]),
                        )
                        conn.commit()
            time.sleep(delay_seconds)

        conn.close()
        browser.close()

    print("抓取完成。数据已写入:", DB_PATH)


def main():
    parser = argparse.ArgumentParser(description="繽紛 Brighten-Mall 鲜切花爬虫")
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL, help="分类页 URL")
    parser.add_argument("--max-products", type=int, default=0, help="最多抓取商品数，0 表示不限制（抓取当前列表页全部）")
    parser.add_argument("--delay", type=float, default=2.0, help="请求间隔（秒）")
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=DEFAULT_MAX_SCROLLS,
        help="单页最大下拉次数（处理无限滚动）",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="最大翻页次数（处理下一頁按钮）",
    )
    parser.add_argument("--no-images", action="store_true", help="不下载图片")
    args = parser.parse_args()

    run_crawl(
        category_url=args.category_url,
        max_products=args.max_products,
        delay_seconds=args.delay,
        download_images=not args.no_images,
        max_scrolls=args.max_scrolls,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
