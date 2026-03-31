# Brighten-Mall 网页结构与爬虫设计

## 一、网页结构分析

### 1. 技术栈

- **前端**：AngularJS（可见 `ng-bind`、`{{ }}`、`SalePageIndexCtrl` 等），内容为**客户端渲染**，首屏 HTML 几乎无商品数据。
- **CDN**：`cms-static.cdn.91app.hk` 存放图片等静态资源。
- **路由**：
  - 分类/列表：`/page/030`（鲜切花）、`/v2/official/SalePageCategory/9795`（鮮花花材葉材）等。
  - 商品详情：`/SalePage/Index/{id}`，例如 `/SalePage/Index/513986`（洋甘菊）。

### 2. 页面内容结构

| 页面类型 | URL 示例 | 内容 |
|----------|----------|------|
| 分类列表 | /page/030, /v2/official/SalePageCategory/9795 | 商品卡片列表，需 JS 渲染后才有链接与价格 |
| 商品详情 | /SalePage/Index/513986 | 标题、价格、商品規格(spec)、图片、配送方式等 |

详情页模板中可见的字段含义（渲染后可抓）：

- **标题**：`SalePageIndexCtrl` 相关标题、商品名稱
- **价格**：`item.Price \| preferredCurrency`、`ng-bind="item.Price | preferredCurrency"`
- **规格**：`商品規格`、`spec.ContentList`
- **图片**：主图多来自 CDN，需从 DOM 中取 `img` 的 `src`
- **商品编号**：商品編號、SalePageModel
- **库存/状态**：SoldQty、開賣時間等

列表页需在浏览器中等待接口或渲染完成，再从 DOM 中抓取指向 `/SalePage/Index/` 的链接及卡片上的名称、价格（若有）。

### 3. 数据如何存储

- **supplier**：当前仅一条，繽紛 Brighten-Mall，`base_url = https://www.brighten.hk`。
- **category**：按站点分类建表，如「鲜切花(page/030)」「鮮花花材葉材(9795)」等，便于按类爬取与展示。
- **product**：每款商品一条记录，字段包括：`external_id`（站点商品 ID）、`name`、`price`、`currency`、`spec_text`（规格说明）、`image_url`/`image_local_path`、`product_url`、`stock_status`、`fetched_at` 等，见 `schema.sql`。
- **product_spec**（可选）：若将规格拆成「规格名 + 规格值」，可写入此表，便于筛选（如按长度、数量）。

## 二、友好爬虫策略

1. **识别与节制**
   - User-Agent 使用常见浏览器标识，不伪装成搜索引擎。
   - 请求间隔：列表页与详情页之间、连续详情页之间建议 **1.5–3 秒**，避免短时间大量请求。
   - 仅抓取商品与分类相关页面，不抓会员、结账、个人资料等。

2. **遵守站点规则**
   - 爬取前查看 `https://www.brighten.hk/robots.txt`（若可访问），避免抓取标明 Disallow 的路径。
   - 不绕过登录或反爬机制；若需登录才能看到价格，则考虑只抓公开信息或放弃该部分。

3. **数据使用与更新**
   - 数据仅用于项目内「诗歌推荐 → 花材展示 → 跳转购买」闭环，不对外再分发原始数据。
   - 建议定期（如每周）增量更新价格与上下架状态，避免长期不更新造成误导。

4. **实现方式**
   - 因列表与详情均为 **JS 渲染**，采用 **Playwright**（或 Selenium）拉取真实渲染后的 HTML，再解析 DOM 获取链接、标题、价格、规格、图片。
   - 可选：在浏览器开发者工具中抓包列表/详情接口（如 XHR），若为公开接口且返回 JSON，可改为直接请求接口，减少对页面的请求量，更友好。

## 三、爬虫流程（当前实现）

1. **初始化**：创建/连接 SQLite（`data/flower_supply.db`），执行 `schema.sql`，确保存在 `supplier` 与 `category` 记录。
2. **入口**：默认从 `https://www.brighten.hk/page/030` 进入（可配置为其他分类 URL）。
3. **列表页**：Playwright 打开入口 URL，等待包含 `/SalePage/Index/` 的链接出现，收集所有商品详情页 URL（去重）。
4. **分页**：若列表有「下一页」，循环进入下一页，重复收集链接，直到达到设定的最大页数或无下一页。
5. **详情页**：逐条访问商品 URL，间隔 1.5–3 秒，从渲染后的 DOM 中提取：标题、价格、规格文本、主图 URL、商品编号、库存/状态；写入 `product` 表（存在则更新 `price`、`spec_text`、`image_url`、`fetched_at` 等）。
6. **图片**：可选将主图下载到 `images/brighten_hk/{external_id}.jpg`，并写回 `image_local_path`。
7. **日志**：输出已抓取数量、失败 URL、简单错误信息，便于排查与限速调整。

以上设计保证：**结构清晰、可维护、对目标站友好、数据可直接用于后续推荐与跳转购买**。

## 四、与网站结构的映射（分类层级 & 文件夹管理）

Brighten-Mall 的分类 URL 常见形式：

- `https://www.brighten.hk/v2/official/SalePageCategory/{id}?sortMode=...`
- `https://www.brighten.hk/page/{id}`

### 1) 数据库 `category` 表如何对应网站层级

为了让数据结构与网站面包屑一致，`category` 表新增：

- `full_path`: 例如 `首頁/鮮花花材葉材/雲南鮮花`
- `parent_external_id`: 例如 `雲南鮮花(9809)` 的父为 `鮮花花材葉材(9795)`
- `level`: 根=0，一级=1，二级=2

当前预置映射（可继续扩展）：

- `030`：首頁/鲜切花
- `9795`：首頁/鮮花花材葉材
- `13027`：首頁/鮮花花材葉材/次日送
- `9809`：首頁/鮮花花材葉材/雲南鮮花

爬虫运行时会从 `--category-url` 自动解析 `external_id` 并 `INSERT/UPDATE` 到 `category` 表。

### 2) 图片文件夹如何对应分类

图片按分类分目录存放，便于管理与缓存：

- `images/brighten_hk/{category_external_id}/{product_external_id}.jpg`

例如：

- 次日送：`images/brighten_hk/13027/552916.jpg`
- 雲南鮮花：`images/brighten_hk/9809/xxxxxx.jpg`
