-- flower_supply 数据表结构（SQLite）
-- 用于存储香港本地供应商的花材品种、规格、价格等，便于与诗歌推荐打通购买闭环

-- 供应商
CREATE TABLE IF NOT EXISTS supplier (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  base_url TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

-- 分类（如：鲜切花、配花、云南鲜花）
CREATE TABLE IF NOT EXISTS category (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  supplier_id INTEGER NOT NULL REFERENCES supplier(id),
  external_id TEXT,
  name TEXT NOT NULL,
  full_path TEXT,          -- 对应网站层级路径（如：首頁/鮮花花材葉材/雲南鮮花）
  parent_external_id TEXT, -- 父分类 external_id（便于映射网站结构）
  level INTEGER,           -- 层级（根=0）
  category_type TEXT,      -- cut_flower / potted / other 等，用于业务区分
  url TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  UNIQUE(supplier_id, external_id)
);

-- 商品（单 SKU 或主商品）
CREATE TABLE IF NOT EXISTS product (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id INTEGER REFERENCES category(id),
  supplier_id INTEGER NOT NULL REFERENCES supplier(id),
  external_id TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  price REAL,
  currency TEXT DEFAULT 'HKD',
  spec_text TEXT,
  color TEXT,
  unit TEXT,
  size TEXT,
  origin TEXT,
  care_instructions TEXT,
  image_url TEXT,
  image_local_path TEXT,
  product_url TEXT,
  stock_status TEXT,
  fetched_at TEXT DEFAULT (datetime('now')),
  UNIQUE(supplier_id, external_id)
);

-- 规格明细（可选：把 spec_text 拆成结构化）
CREATE TABLE IF NOT EXISTS product_spec (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL REFERENCES product(id),
  spec_name TEXT NOT NULL,
  spec_value TEXT,
  UNIQUE(product_id, spec_name)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_product_supplier ON product(supplier_id);
CREATE INDEX IF NOT EXISTS idx_product_category ON product(category_id);
CREATE INDEX IF NOT EXISTS idx_product_name ON product(name);
CREATE INDEX IF NOT EXISTS idx_category_supplier ON category(supplier_id);
