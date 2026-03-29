# flower_supply — 香港本地花卉供应商数据

用于存储与同步香港本地花卉供应商的可售花材数据（品种、规格、数量、价格、图片等），支撑「诗花雅送」从诗歌推荐到花材购买的商业闭环。

## 目录结构

- **data/** — SQLite 数据库与导出文件
- **images/** — 按供应商与商品缓存的图片（可选）
- **schema.sql** — 数据库表结构
- **crawl_brighten_hk.py** — 繽紛 Brighten-Mall 鲜切花爬虫
- **DESIGN.md** — 网页结构分析、存储设计与爬虫策略

## 数据来源（当前）

| 供应商 | 说明 | 入口链接 |
|--------|------|----------|
| 繽紛 Brighten-Mall | 香港花墟大型门店，鲜切花分类 | https://www.brighten.hk/page/030 |

## Brighten 分类（已支持）

- **次日送**（鮮花花材葉材/次日送，切花）：`https://www.brighten.hk/v2/official/SalePageCategory/13027?sortMode=Curator`
- **雲南鮮花**（鮮花花材葉材/雲南鮮花，切花）：`https://www.brighten.hk/v2/official/SalePageCategory/9809?sortMode=PageView`
- **荷蘭鮮花**（鮮花花材葉材/荷蘭鮮花，切花）：`https://www.brighten.hk/v2/official/SalePageCategory/9796?sortMode=PageView`
- **外國鮮花**（鮮花花材葉材/外國鮮花，切花）：`https://www.brighten.hk/v2/official/SalePageCategory/14878?sortMode=Curator`
- **盆栽植物**（單獨作為盆栽類，與切花分開）：`https://www.brighten.hk/v2/official/SalePageCategory/9864?sortMode=PageView`

## 使用方式

1. 安装依赖：`pip install playwright && playwright install chromium`
2. 运行爬虫：`python crawl_brighten_hk.py [--category-url URL] [--max-products N] [--no-images]`
3. 数据写入 `data/flower_supply.db`，图片可选保存到 `images/brighten_hk/`
4. **查看数据**：`python view_db.py`（摘要 + 前 20 条预览），或 `python view_db.py --export products.csv` 导出 CSV
5. **仅补下图片**：若爬虫用了 `--no-images`，确认数据后再运行 `python download_images_only.py`

详见 **DESIGN.md** 与各脚本内 `--help`。
