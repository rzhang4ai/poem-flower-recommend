"""
build_dataset_v2.py — 诗花雅送 · 改进版数据提取脚本
修复内容：
  1. 朝代·作者 误匹配 → 使用 KNOWN_DYNASTIES 白名单 + 末尾匹配（处理"词牌+朝代·作者"格式）
  2. 标题层级不一致 → 统一处理带#和不带#的朝代·作者行
  3. 正文粘连作者名 → 剔除首行中朝代·作者前缀
  4. 赏析提取启发式 → 改用段落长度 + 多关键词综合判断
  5. 花名/月份检测 → 改用内容特征 + 结构位置双重验证
  新增：月份字段提取
"""

import os, re, sys
import pandas as pd

# ── 常量 ──────────────────────────────────────────────────────────────────────

KNOWN_DYNASTIES = [
    '先秦','两汉','东汉','西汉','东晋','西晋',
    '南朝梁','南朝陈','南朝宋','南朝齐',
    '北朝周','北朝齐','北朝魏',
    '汉','魏','晋','隋','唐','五代','宋','辽','金','元','明','清',
    '近代','现代','当代'
]
# 按长度降序，优先匹配长的（如"南朝梁"优先于"梁"）
KNOWN_DYNASTIES.sort(key=len, reverse=True)

# 月份正则
MONTH_RE = re.compile(
    r'^(正月|二月|三月|四月|五月|六月|七月|八月|九月|十月|十一月|十二月)'
    r'(花卉|花|时令)?'
)
MONTH_TO_INT = {
    '正月': 1, '二月': 2, '三月': 3, '四月': 4,
    '五月': 5, '六月': 6, '七月': 7, '八月': 8,
    '九月': 9, '十月': 10, '十一月': 11, '十二月': 12,
}

# 干扰性标题（跳过）
SKIP_TITLE_RE = re.compile(r'[《》〈〉【】]')
SKIP_KEYWORDS = {'花月令', '瓶史', '月表', '按：', '例：', '凡例'}

# 赏析段落判断（仅作备用）
ANALYSIS_STARTS = ('这', '此', '（', '(', '全', '诗', '词', '作', '本', '上', '下',
                   '首', '该', '其', '以', '从', '在', '由', '对', '通', '结')

# ── 花名白名单（来源：原书花名索引）────────────────────────────────────────────
FLOWER_WHITELIST_PRIMARY = {
    '梅花', '红梅', '蜡梅', '迎春花', '樱桃花',
    '玉兰', '月季', '杏花', '李花', '桃花', '棠梨花', '林檎花', '郁李花',
    '剪春罗', '素馨', '莱花', '绣球', '绣球花', '木兰',
    '玫瑰', '蔷薇', '酴醿', '芍药', '牡丹',
    '合欢', '玉蕊花', '玉簪花', '含笑', '辛夷', '木香',
    '山茶花', '山茶', '瑞香', '兰花', '秋兰',
    '石竹', '杨花', '木棉花', '紫荆', '琼花', '萱草', '棣棠', '棣花',
    '金沙', '木芙蓉', '莲花', '荷花', '茉莉',
    '凤仙花', '向日葵', '木槿', '金银花',
    '秋海棠', '芦花', '桂花', '菊花', '甘菊', '野菊',
    '金钱花', '雁来红', '秋葵', '山丹', '栀子花',
    '紫薇', '牵牛花', '蜀葵', '鸡冠花',
    '水仙花', '水仙', '蓼花', '滴滴金', '蘋花',
    '山枇杷', '杜鹃', '桐花', '凌霄花', '榴花', '罂粟',
    '曼陀罗', '虞美人',
}

FLOWER_ALIAS = {
    '荷花': '莲花', '木莲': '辛夷', '木芍药': '牡丹',
    '水芙蓉': '莲花', '芙蕖': '莲花', '芙蓉': '木芙蓉',
    '粉梅': '红梅', '绿萼梅': '梅花',
    '蔷薇花': '蔷薇', '月月红': '月季', '长春花': '月季',
    '岩桂': '桂花', '木犀': '桂花',
    '朱槿': '木槿', '拒霜': '木芙蓉',
    '百日红': '紫薇', '怕痒花': '紫薇',
    '老少年': '雁来红',
    '八仙花': '绣球花',
    '安石榴': '榴花', '石榴': '榴花',
    '金凤仙': '凤仙花', '旱珍珠': '凤仙花',
    '夜合': '合欢', '合昏': '合欢', '马缨花': '合欢',
    '红荆': '紫荆', '满条红': '紫荆',
    '丽春花': '虞美人', '满园春': '虞美人',
    '忘忧花': '萱草',
    '忍冬': '金银花', '金钥股': '金银花',
    '抹丽': '茉莉', '孽华': '茉莉',
    '菝葡': '栀子花', '栀子': '栀子花',
    '茶蘼': '酴醿',
    '旋覆花': '滴滴金',
    '米囊花': '罂粟', '端午花': '蜀葵', '一丈红': '蜀葵',
    '黄蜀葵': '秋葵', '侧金盏': '秋葵',
    '喇叭花': '牵牛花',
    '来禽花': '林檎花', '花红': '林檎花', '沙果': '林檎花',
    '含桃': '樱桃花',
    '英雄树': '木棉花', '攀枝花': '木棉花',
    '红百合': '山丹', '连珠': '山丹',
}

def normalize_flower(text: str):
    """将标题标准化为白名单主条目，匹配失败返回 None。"""
    tc = text.strip().replace(' ', '').replace('\u3000', '')
    for f in FLOWER_WHITELIST_PRIMARY:
        if tc == f.replace(' ', ''):
            return f
    for alias, canonical in FLOWER_ALIAS.items():
        if tc == alias.replace(' ', ''):
            return canonical
    return None

# ── 核心函数 ──────────────────────────────────────────────────────────────────

def parse_dynasty_author(text: str):
    """
    从一行文本中提取（朝代, 作者）。
    支持：
      - 格式A: 朝代·作者        e.g. '宋·苏轼'
      - 格式B: 词牌+朝代·作者   e.g. '梅花清·郑侠如', '回文宋·苏轼'
    无法解析时返回 (None, None)。
    """
    text = text.strip().lstrip('#').strip()

    # 格式A：朝代在开头
    for d in KNOWN_DYNASTIES:
        if text.startswith(d + '·') or text.startswith(d + '·'):
            author = text[len(d) + 1:].strip()
            if 1 <= len(author) <= 8 and not re.search(r'[，。！？、；：《》]', author):
                return d, author

    # 格式B：朝代在末尾（词牌主题+朝代·作者）
    m = re.search(
        r'(' + '|'.join(re.escape(d) for d in KNOWN_DYNASTIES) + r')[··](.{1,8}?)$',
        text
    )
    if m:
        dynasty = m.group(1)
        author  = m.group(2).strip()
        if 1 <= len(author) <= 8 and not re.search(r'[，。！？、；：《》]', author):
            return dynasty, author

    return None, None


def is_flower_name(text: str) -> bool:
    """判断标题是否是白名单花名（精确匹配）。"""
    return normalize_flower(text) is not None


def is_poem_line(text: str) -> bool:
    """
    判断一行是诗文（True）还是赏析开始（False）。
    修复原脚本以40字为阈值导致多句合并的诗文被误判为赏析的问题。
    诗文特征：标点分段后每个片段字数较短（avg<=8, max<=12），无书名号，无分析性开头词。
    """
    text = text.strip()
    if not text:
        return False
    # 明确赏析特征
    if re.search(r'[《》<>〈〉]', text):
        return False
    if text.startswith(('这', '此', '（', '(', '全诗', '诗人', '作者', '词人',
                        '上片', '下片', '上阕', '下阕', '首联', '颔联', '颈联', '尾联')):
        return False
    # 按标点/空白切分，计算每段字数
    segments = re.split(r'[，。？！、；\s]+', text)
    segments = [s for s in segments if s]
    if not segments:
        return False
    avg_len = sum(len(s) for s in segments) / len(segments)
    max_len = max(len(s) for s in segments)
    # 诗文：每段平均<=8字，最长段<=12字
    if max_len <= 12 and avg_len <= 8:
        return True
    return False


def classify_line(line: str) -> str:
    """兼容旧调用接口。"""
    return 'poem' if is_poem_line(line) else 'analysis'


# ── 主提取逻辑 ────────────────────────────────────────────────────────────────

def build_dataset(md_path: str, out_csv: str):
    print("🚀 读取 Markdown 文件...")
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip() for l in f.readlines()]

    # ── 第一遍：收集所有标题行（带#的）和朝代·作者行（带/不带#）
    heading_lines = []   # (行号, 文本)  所有 # 开头的行
    da_lines      = []   # (行号, 朝代, 作者)  朝代·作者锚点

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith('#'):
            text = stripped.lstrip('#').strip()
            heading_lines.append((i, text))
            d, a = parse_dynasty_author(text)
            if d:
                da_lines.append((i, d, a, True))   # True = 带#
        else:
            if stripped:
                d, a = parse_dynasty_author(stripped)
                if d:
                    da_lines.append((i, d, a, False))  # False = 不带#

    total = len(da_lines)
    print(f"🎯 共定位到 {total} 首诗词（朝代·作者锚点）\n")
    print("⏳ 提取中...")

    # 建立辅助索引
    heading_set = {i for i, _ in heading_lines}

    dataset = []
    current_flower = '未知花卉'
    current_month  = '未知月份'

    for idx, (da_line_no, dynasty, author, has_hash) in enumerate(da_lines):
        progress = (idx + 1) / total * 100
        sys.stdout.write(f"\r🔄 [{idx+1:04d}/{total:04d}] {progress:.1f}%")
        sys.stdout.flush()

        # ── 1. 找诗名：向上找最近的标题行（不是朝代·作者行本身）
        title = '未知诗名'
        title_line_no = da_line_no
        for j in range(da_line_no - 1, -1, -1):
            if j in heading_set:
                cand = lines[j].lstrip('#').strip()
                d2, _ = parse_dynasty_author(cand)
                if d2:
                    continue   # 跳过：这是另一个朝代·作者行
                if any(k in cand for k in SKIP_KEYWORDS) or SKIP_TITLE_RE.search(cand):
                    continue   # 跳过干扰标题
                title = cand
                title_line_no = j
                break

        # ── 2. 找花名：从诗名行向上找
        for j in range(title_line_no - 1, -1, -1):
            if j in heading_set:
                cand = lines[j].lstrip('#').strip()
                flower = normalize_flower(cand)
                if flower:
                    current_flower = flower
                    break
                elif MONTH_RE.match(cand):
                    break   # 到月份层了，停止

        # ── 3. 找月份：继续向上找月份标题
        for j in range(title_line_no - 1, -1, -1):
            if j in heading_set:
                cand = lines[j].lstrip('#').strip()
                m = MONTH_RE.match(cand)
                if m:
                    current_month = m.group(1)  # 只取"X月"部分
                    break

        # ── 4. 确定本诗范围：到下一首朝代·作者行之前
        next_da_line = da_lines[idx + 1][0] if idx + 1 < total else len(lines)

        # 找下一首诗的标题行（如果有#标题的话），作为结束边界
        end_line = next_da_line
        for j in range(next_da_line - 1, da_line_no, -1):
            if j in heading_set:
                end_line = j
                break

        # ── 5. 提取正文和赏析
        poem_lines     = []
        analysis_lines = []
        in_analysis    = False

        # 处理朝代·作者行本身可能粘连的诗文（格式C）
        raw_da = lines[da_line_no].strip().lstrip('#').strip()
        leftover = raw_da[raw_da.index('·') + 1 + len(author):].strip() if author in raw_da else ''
        # 去掉可能粘连在作者名后的诗文（格式B的词牌部分已在parse中处理）
        # 只留真正的诗文（不是朝代）
        if leftover and not any(leftover.startswith(d) for d in KNOWN_DYNASTIES):
            if leftover and not re.match(r'^[，。！？]', leftover):
                poem_lines.append(leftover)

        for j in range(da_line_no + 1, end_line):
            raw_line = lines[j].strip()
            if not raw_line:
                continue

            # 内容行有时带#号（MD格式不规范），去掉#后当正文处理
            is_hash_content = raw_line.startswith('#') and not any(
                (raw_line.lstrip('#').strip()).startswith(d + '·') or
                (raw_line.lstrip('#').strip()).startswith(d + '·')
                for d in KNOWN_DYNASTIES
            )
            if raw_line.startswith('#') and not is_hash_content:
                break   # 真正的下一个标题，停止
            content_line = raw_line.lstrip('#').strip() if is_hash_content else raw_line

            if is_poem_line(content_line):
                # 诗文行：无论 in_analysis 状态如何，若之前赏析行很少（<2行），
                # 认为是词序后的正文，归入诗文
                if in_analysis and len(analysis_lines) < 3:
                    poem_lines.extend(analysis_lines)
                    analysis_lines = []
                    in_analysis = False
                if not in_analysis:
                    poem_lines.append(content_line)
                else:
                    analysis_lines.append(content_line)
            else:
                in_analysis = True
                analysis_lines.append(content_line)

        dataset.append({
            'ID'  : idx + 1,
            '月份' : current_month,
            '月份数字': MONTH_TO_INT.get(current_month, 0),
            '花名' : current_flower,
            '诗名' : title,
            '朝代' : dynasty,
            '作者' : author,
            '正文' : '\n'.join(poem_lines).strip(),
            '赏析' : '\n'.join(analysis_lines).strip(),
        })

    print(f"\n\n💾 写入 CSV...")
    df = pd.DataFrame(dataset)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 完成！共 {len(df)} 首诗词 → {out_csv}")
    return df


# ── 数据质量报告 ──────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 数据质量报告")
    print("=" * 60)

    total = len(df)
    print(f"总条目数: {total}")
    print()

    # 空值
    print("── 空值统计 ──")
    for col in ['月份', '花名', '诗名', '朝代', '作者', '正文', '赏析']:
        empty = df[col].fillna('').apply(lambda x: x.strip() == '' or x in ('未知诗名','未知花卉','未知月份'))
        print(f"  {col}: {empty.sum()} 空/未知 ({empty.sum()/total*100:.1f}%)")

    print()
    print("── 字段长度异常 ──")
    df['正文长度'] = df['正文'].fillna('').apply(len)
    df['赏析长度'] = df['赏析'].fillna('').apply(len)
    short_poem = df[df['正文长度'] < 5]
    short_ana  = df[df['赏析长度'] < 20]
    print(f"  正文 < 5字: {len(short_poem)} 条")
    print(f"  赏析 < 20字: {len(short_ana)} 条")

    print()
    print("── 朝代分布 ──")
    print(df['朝代'].value_counts().to_string())

    print()
    print("── 月份分布 ──")
    print(df['月份'].value_counts().to_string())

    print()
    flower_count = df['花名'].nunique()
    print(f"── 花名唯一值: {flower_count} 个 ──")
    print(df['花名'].value_counts().head(20).to_string())

    print()
    dup = df.duplicated(subset=['诗名', '作者'])
    print(f"── 重复条目 (诗名+作者相同): {dup.sum()} 条 ──")
    if dup.sum():
        print(df[dup][['诗名','朝代','作者']].head(10).to_string())

    print("=" * 60)


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 允许命令行传参，否则用默认路径
    md_path  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, 'flower_poems.md')
    csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(project_root, 'poems_dataset_v2.csv')

    df = build_dataset(md_path, csv_path)
    quality_report(df)
