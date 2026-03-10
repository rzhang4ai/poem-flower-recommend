"""
poems_dataset_eda.py — 诗花雅送 · 合并数据集可视化探索分析
================================================================
基于项目规划（Poetic Flora Advisor）对 poems_dataset_merged.csv 做初步数据探索，
生成丰富详细的可视化报告，便于了解数据分布与质量。
仅使用 Python 标准库，无需 pandas/matplotlib。

输出：poems_dataset_eda_report.html（内嵌 SVG 图表与表格）
用法：python3 poems_dataset_eda.py
"""

import csv
import os
import re
from collections import Counter, defaultdict

MONTH_ORDER = {
    '正月': 1, '二月': 2, '三月': 3, '四月': 4, '五月': 5, '六月': 6,
    '七月': 7, '八月': 8, '九月': 9, '十月': 10, '十一月': 11, '十二月': 12,
}
MONTHS_SORTED = sorted(MONTH_ORDER.keys(), key=lambda x: MONTH_ORDER[x])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'poems_dataset_merged.csv')
OUTPUT_HTML = os.path.join(SCRIPT_DIR, 'poems_dataset_eda_report.html')


def load_rows():
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030'):
        try:
            with open(CSV_PATH, encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                return rows
        except (UnicodeDecodeError, Exception):
            continue
    with open(CSV_PATH, encoding='utf-8-sig', errors='replace') as f:
        return list(csv.DictReader(f))


def str_val(r, key, default=''):
    v = r.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ''):
        return default
    return str(v).strip()


def escape(s):
    if s is None:
        return ''
    s = str(s)
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def svg_bar_horizontal(labels, values, max_width=400, bar_height=22, font_size=12, color='#3498db'):
    """生成水平条形图 SVG。"""
    n = len(labels)
    if n == 0:
        return '<p>无数据</p>'
    max_val = max(values) or 1
    h = n * (bar_height + 4) + 30
    w = max_width + 180
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 20 + i * (bar_height + 4)
        width = (val / max_val) * max_width if max_val else 0
        lines.append(f'<rect x="0" y="{y}" width="{width}" height="{bar_height}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{max_width + 8}" y="{y + bar_height - 5}" font-size="{font_size}" fill="#333">{escape(label)}</text>')
        lines.append(f'<text x="{max_width - 5}" y="{y + bar_height - 5}" font-size="{font_size - 1}" fill="#333" text-anchor="end">{val}</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


def svg_bar_vertical(labels, values, max_height=280, bar_width=28, color='#2ecc71'):
    """生成垂直条形图 SVG。"""
    n = len(labels)
    if n == 0:
        return '<p>无数据</p>'
    max_val = max(values) or 1
    w = n * (bar_width + 6) + 80
    h = max_height + 60
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    base_y = max_height + 20
    for i, (label, val) in enumerate(zip(labels, values)):
        x = 40 + i * (bar_width + 6)
        bh = (val / max_val) * max_height if max_val else 0
        lines.append(f'<rect x="{x}" y="{base_y - bh}" width="{bar_width}" height="{bh}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{x + bar_width/2}" y="{base_y + 18}" font-size="10" fill="#333" text-anchor="middle">{escape(label)}</text>')
        lines.append(f'<text x="{x + bar_width/2}" y="{base_y - bh - 4}" font-size="9" fill="#333" text-anchor="middle">{val}</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


def svg_heatmap(matrix_rows, row_labels, col_labels, max_cell=80, colors=('#fff', '#ffeda0', '#f03b20')):
    """简单热力图：matrix_rows 为二维列表，行=row_labels，列=col_labels。"""
    if not matrix_rows or not row_labels or not col_labels:
        return '<p>无数据</p>'
    max_val = max(max(r) for r in matrix_rows) or 1
    cell = max_cell
    w = len(col_labels) * cell + 120
    h = len(row_labels) * (cell // 2) + 40
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for j, cl in enumerate(col_labels):
        lines.append(f'<text x="{120 + j * cell + cell/2}" y="20" font-size="9" text-anchor="middle">{escape(cl)}</text>')
    for i, rl in enumerate(row_labels):
        lines.append(f'<text x="4" y="{40 + i * (cell//2) + (cell//2)/2 + 4}" font-size="9">{escape(rl)}</text>')
    for i, row in enumerate(matrix_rows):
        for j, v in enumerate(row):
            t = v / max_val if max_val else 0
            if t <= 0:
                fill = colors[0]
            elif t < 0.5:
                fill = colors[1]
            else:
                fill = colors[2]
            x, y = 120 + j * cell, 40 + i * (cell // 2)
            lines.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{(cell//2)-2}" fill="{fill}" stroke="#ddd"/>')
            if v > 0:
                lines.append(f'<text x="{x+cell/2-2}" y="{y+(cell//2)/2+4}" font-size="9" text-anchor="middle">{v}</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


# 堆叠柱状图用色（花名分段）
STACK_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
    '#e67e22', '#34495e', '#c0392b', '#16a085', '#7f8c8d', '#95a5a6',
]


def svg_stacked_bar(month_labels, segment_labels, data_matrix, colors=None):
    """堆叠柱状图：每月一根柱，按花名分段。data_matrix[i][j]=月份i中花j的数量。"""
    if not month_labels or not segment_labels or not data_matrix:
        return '<p>无数据</p>'
    colors = colors or STACK_COLORS
    n_months = len(month_labels)
    n_seg = len(segment_labels)
    max_total = max(sum(row) for row in data_matrix) or 1
    bar_w = 36
    gap = 8
    max_bar_h = 220
    margin_l, margin_r = 50, 50
    margin_t, margin_b = 50, 55
    legend_h = 22 * ((n_seg + 4) // 5) + 10
    w = margin_l + n_months * (bar_w + gap) - gap + margin_r
    h = margin_t + max_bar_h + margin_b + legend_h
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    base_y = margin_t + max_bar_h
    for i, row in enumerate(data_matrix):
        x = margin_l + i * (bar_w + gap)
        y_cur = base_y
        total = sum(row)
        for j, val in enumerate(row):
            if val <= 0:
                continue
            seg_h = (val / max_total) * max_bar_h
            y_cur -= seg_h
            fill = colors[j % len(colors)]
            lines.append(f'<rect x="{x}" y="{y_cur}" width="{bar_w}" height="{seg_h}" fill="{fill}" stroke="#fff" stroke-width="0.5"/>')
        if total > 0:
            lines.append(f'<text x="{x + bar_w/2}" y="{base_y + 16}" font-size="10" text-anchor="middle">{total}</text>')
        lines.append(f'<text x="{x + bar_w/2}" y="{base_y + 32}" font-size="9" text-anchor="middle">{escape(month_labels[i])}</text>')
    # 图例
    ly = base_y + 44
    for k in range(0, n_seg, 5):
        for j in range(k, min(k + 5, n_seg)):
            lx = margin_l + (j - k) * 100
            fill = colors[j % len(colors)]
            lines.append(f'<rect x="{lx}" y="{ly}" width="14" height="12" fill="{fill}"/>')
            lines.append(f'<text x="{lx + 18}" y="{ly + 10}" font-size="10">{escape(segment_labels[j])}</text>')
        ly += 20
    lines.append('</svg>')
    return '\n'.join(lines)


def run_eda():
    rows = load_rows()
    if not rows:
        print('❌ 无法读取或数据为空')
        return

    n_total = len(rows)
    # 统一字段
    for r in rows:
        for k in list(r.keys()):
            r[k] = str_val(r, k)
        r['正文'] = r.get('正文', '') or ''
        r['赏析'] = r.get('赏析', '') or ''

    # 统计
    months = Counter(r['月份'] for r in rows)
    flowers = Counter(r['花名'] for r in rows)
    dynasties = Counter(r['朝代'] for r in rows)
    authors = Counter(r['作者'] for r in rows)
    reviewers = Counter(r['审核员'] for r in rows)
    statuses = Counter(r['审核状态'] for r in rows)

    # 非空计数
    key_cols = ['月份', '花名', '诗名', '朝代', '作者', '正文', '赏析', '审核状态', '审核员']
    completeness = [(c, sum(1 for r in rows if str_val(r, c))) for c in key_cols]

    # 冲突 ID
    conflict_count = sum(1 for r in rows if re.match(r'^.+\-\d{2}$', str(r.get('ID', ''))))
    conflict_base = Counter()
    for r in rows:
        m = re.match(r'^(\d+)\-\d{2}$', str(r.get('ID', '')))
        if m:
            conflict_base[m.group(1)] += 1

    # 正文/赏析长度
    body_lens = [len(r['正文']) for r in rows]
    comment_lens = [len(r['赏析']) for r in rows]

    def stats(lst):
        if not lst:
            return 0, 0, 0, 0
        s = sorted(lst)
        n = len(s)
        return min(s), max(s), sum(s) / n, s[n // 2] if n else 0

    body_min, body_max, body_mean, body_med = stats(body_lens)
    comm_min, comm_max, comm_mean, comm_med = stats(comment_lens)

    # 月份×花名（前15花 + 前30花用于大热力图）
    top_flowers = [f for f, _ in flowers.most_common(15)]
    top_flowers_30 = [f for f, _ in flowers.most_common(30)]
    month_flower = defaultdict(lambda: Counter())
    for r in rows:
        if r['花名'] in top_flowers and r['月份'] in MONTH_ORDER:
            month_flower[r['月份']][r['花名']] += 1
    heatmap_rows = []
    for m in MONTHS_SORTED:
        heatmap_rows.append([month_flower[m].get(f, 0) for f in top_flowers])

    # 每月花种数 + 每月全部花名及数量（用于「每月有什么花」+ 堆叠图）
    month_flower_count = {}
    month_flower_full = defaultdict(lambda: Counter())  # 每月 -> 花名 -> 数量
    for r in rows:
        m, f = r['月份'], r['花名']
        if m in MONTH_ORDER and f:
            month_flower_full[m][f] += 1
    # 堆叠柱状图数据：全球 top10 花 + 其他，每月一段
    top10_flowers = [f for f, _ in flowers.most_common(10)]
    segment_labels_stacked = top10_flowers + ['其他']
    stacked_matrix = []
    for m in MONTHS_SORTED:
        cnt = month_flower_full[m]
        row = [cnt.get(f, 0) for f in top10_flowers]
        other = sum(cnt.values()) - sum(row)
        row.append(max(0, other))
        stacked_matrix.append(row)
    # 月份×花名 大热力图（Top 30）
    heatmap_rows_30 = []
    for m in MONTHS_SORTED:
        heatmap_rows_30.append([month_flower_full[m].get(f, 0) for f in top_flowers_30])
    for m in MONTHS_SORTED:
        month_flower_count[m] = len(month_flower_full[m])

    # 正文为空的条目 ID
    empty_body_ids = [str_val(r, 'ID') for r in rows if not str_val(r, '正文')]

    # 朝代×月份 + 朝代×花名
    top_dynasties = [d for d, _ in dynasties.most_common(8)]
    top_dynasties_10 = [d for d, _ in dynasties.most_common(10)]
    top_flowers_25 = [f for f, _ in flowers.most_common(25)]
    dynasty_month = defaultdict(lambda: Counter())
    dynasty_month_all = defaultdict(lambda: Counter())
    dynasty_flower = defaultdict(lambda: Counter())  # 朝代 -> 花名 -> 数量
    for r in rows:
        d, m, f = r['朝代'], r['月份'], r['花名']
        if m in MONTH_ORDER and d:
            dynasty_month_all[d][m] += 1
        if d and f:
            dynasty_flower[d][f] += 1
        if d in top_dynasties:
            dynasty_month[d][m] += 1
    # 朝代×花名 矩阵（Top10 朝代 × Top25 花）
    dynasty_flower_rows = [[dynasty_flower[d].get(f, 0) for f in top_flowers_25] for d in top_dynasties_10]

    n_flowers = len(flowers)
    n_authors = len(authors)
    n_dynasties = len(dynasties)

    # ── 生成 HTML ───────────────────────────────────────────────────────
    html = []
    html.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>诗花雅送 · 合并数据集探索分析报告</title>
<style>
  body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; background: #fafafa; color: #222; max-width: 1200px; }
  h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
  h2 { color: #34495e; margin-top: 32px; }
  .meta { background: #ecf0f1; padding: 14px 18px; border-radius: 8px; margin: 16px 0; }
  .meta p { margin: 6px 0; }
  .fig-wrap { margin: 24px 0; padding: 16px; background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .fig-wrap h3 { margin-top: 0; color: #555; font-size: 1.05em; }
  table.stats-table { border-collapse: collapse; margin: 12px 0; font-size: 14px; }
  table.stats-table th, table.stats-table td { border: 1px solid #bdc3c7; padding: 8px 12px; text-align: right; }
  table.stats-table th { background: #3498db; color: #fff; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }
  .footer { margin-top: 40px; color: #7f8c8d; font-size: 12px; }
</style>
</head>
<body>
<h1>🌺 诗花雅送 · 合并数据集探索分析报告</h1>
<p>基于 <code>poems_dataset_merged.csv</code> 的初步数据探索，用于了解数据分布与质量，支撑后续 Poetic Flora Advisor 推荐系统与 LDA/情感分析等任务。</p>
<div class="meta">
  <p><strong>数据文件：</strong> poems_dataset_merged.csv</p>
  <p><strong>总条数：</strong> ''' + str(n_total) + '''</p>
  <p><strong>花名种类数：</strong> ''' + str(n_flowers) + ''' &nbsp;|&nbsp; <strong>作者数：</strong> ''' + str(n_authors) + ''' &nbsp;|&nbsp; <strong>朝代数：</strong> ''' + str(n_dynasties) + '''</p>
  <p><strong>冲突拆分 ID 条数：</strong> ''' + str(conflict_count) + '''（合并时同一 ID 多来源保留为 xxx-01, xxx-02）</p>
  <p><strong>正文为空的条目（''' + str(len(empty_body_ids)) + ''' 条）：</strong> ''' + '，'.join(escape(i) for i in sorted(empty_body_ids, key=lambda x: (len(x), x))) + '''</p>
</div>
''')

    # 一、数据概览
    html.append('<h2>一、数据概览与完整性</h2>')
    html.append('<div class="fig-wrap"><h3>各字段非空数量</h3><table class="stats-table"><tr><th>字段</th><th>非空条数</th></tr>')
    for c, cnt in completeness:
        html.append(f'<tr><td>{escape(c)}</td><td>{cnt}</td></tr>')
    html.append('</table></div>')
    html.append('<div class="fig-wrap"><h3>ID 类型</h3><p>普通 ID：' + str(n_total - conflict_count) + ' 条 &nbsp;|&nbsp; 冲突拆分 ID（含 -01/-02）：' + str(conflict_count) + ' 条</p></div>')

    # 二、月份分布
    html.append('<h2>二、月份分布</h2>')
    m_labels = [m for m in MONTHS_SORTED if months.get(m)]
    m_vals = [months.get(m, 0) for m in m_labels]
    html.append('<div class="fig-wrap"><h3>按月份诗作条数（正月–十二月）</h3>' + svg_bar_vertical(m_labels, m_vals, color='#3498db') + '</div>')

    # 三、花名分布 Top 30
    html.append('<h2>三、花名分布</h2>')
    top30 = flowers.most_common(30)
    html.append('<div class="fig-wrap"><h3>花名诗作条数 Top 30</h3>' + svg_bar_horizontal([f for f, _ in top30], [c for _, c in top30], color='coral') + '</div>')

    # 四、朝代分布 Top 20
    html.append('<h2>四、朝代分布</h2>')
    top20d = dynasties.most_common(20)
    html.append('<div class="fig-wrap"><h3>朝代诗作条数 Top 20</h3>' + svg_bar_vertical([d for d, _ in top20d], [c for _, c in top20d], color='teal') + '</div>')

    # 五、作者分布 Top 25
    html.append('<h2>五、作者分布</h2>')
    top25a = authors.most_common(25)
    html.append('<div class="fig-wrap"><h3>作者诗作条数 Top 25</h3>' + svg_bar_horizontal([a for a, _ in top25a], [c for _, c in top25a], color='mediumpurple') + '</div>')

    # 七、正文与赏析长度
    html.append('<h2>七、正文与赏析长度</h2>')
    html.append('<div class="fig-wrap"><h3>正文与赏析长度统计（字符数）</h3>')
    html.append('<table class="stats-table"><tr><th></th><th>最小值</th><th>最大值</th><th>平均值</th><th>中位数</th></tr>')
    html.append(f'<tr><td>正文</td><td>{body_min}</td><td>{body_max}</td><td>{body_mean:.1f}</td><td>{body_med}</td></tr>')
    html.append(f'<tr><td>赏析</td><td>{comm_min}</td><td>{comm_max}</td><td>{comm_mean:.1f}</td><td>{comm_med}</td></tr></table></div>')
    # 长度分布简表：分段计数
    bins_body = [0, 50, 100, 200, 500, 1000, 5000, 1000000]
    bins_comm = [0, 100, 500, 1000, 2000, 5000, 1000000]
    body_dist = Counter()
    for L in body_lens:
        for i in range(len(bins_body) - 1):
            if bins_body[i] <= L < bins_body[i + 1]:
                label = f'{bins_body[i]}+' if bins_body[i + 1] >= 1000000 else f'{bins_body[i]}-{bins_body[i+1]}'
                body_dist[label] += 1
                break
    comm_dist = Counter()
    for L in comment_lens:
        for i in range(len(bins_comm) - 1):
            if bins_comm[i] <= L < bins_comm[i + 1]:
                label = f'{bins_comm[i]}+' if bins_comm[i + 1] >= 1000000 else f'{bins_comm[i]}-{bins_comm[i+1]}'
                comm_dist[label] += 1
                break
    html.append('<div class="fig-wrap"><h3>正文字符数分段分布</h3><table class="stats-table"><tr><th>区间</th><th>条数</th></tr>')
    def sort_bin_key(x):
        m = re.match(r'^(\d+)', x)
        return int(m.group(1)) if m else 0
    for k in sorted(body_dist.keys(), key=sort_bin_key):
        html.append(f'<tr><td>{k}</td><td>{body_dist[k]}</td></tr>')
    html.append('</table></div>')
    html.append('<div class="fig-wrap"><h3>赏析字符数分段分布</h3><table class="stats-table"><tr><th>区间</th><th>条数</th></tr>')
    for k in sorted(comm_dist.keys(), key=sort_bin_key):
        html.append(f'<tr><td>{k}</td><td>{comm_dist[k]}</td></tr>')
    html.append('</table></div>')

    # 八、月份×花名热力图与可视化
    html.append('<h2>八、月份与花名关系</h2>')
    html.append('<div class="fig-wrap"><h3>每月花名构成（堆叠柱状图）</h3><p>每根柱 = 一个月份，按花名分段；Top 10 花名单独色块，其余归为「其他」。可一眼看出各月诗作总量及花种构成。</p>' + svg_stacked_bar(MONTHS_SORTED, segment_labels_stacked, stacked_matrix) + '</div>')
    html.append('<div class="fig-wrap"><h3>月份 × 花名 数量热力图（Top 30 花名）</h3><p>行=月份，列=花名，颜色深浅=该月该花诗作条数。</p>' + svg_heatmap(heatmap_rows_30, MONTHS_SORTED, top_flowers_30, max_cell=36) + '</div>')
    html.append('<div class="fig-wrap"><h3>月份 × 花名 数量热力图（Top 15 花名，精简）</h3>' + svg_heatmap(heatmap_rows, MONTHS_SORTED, top_flowers, max_cell=52) + '</div>')
    mfc_labels = list(month_flower_count.keys())
    mfc_vals = list(month_flower_count.values())
    html.append('<div class="fig-wrap"><h3>各月份涉及的花名种类数</h3>' + svg_bar_vertical(mfc_labels, mfc_vals, color='slateblue') + '</div>')
    # 每月花名及数量明细（表格）
    html.append('<div class="fig-wrap"><h3>每个月份的花名及数量（明细表）</h3>')
    for m in MONTHS_SORTED:
        cnt = month_flower_full[m]
        if not cnt:
            continue
        html.append(f'<p><strong>{escape(m)}</strong>（共 {sum(cnt.values())} 条）</p>')
        html.append('<table class="stats-table"><tr><th>花名</th><th>数量</th></tr>')
        for f, n in cnt.most_common():
            html.append(f'<tr><td>{escape(f)}</td><td>{n}</td></tr>')
        html.append('</table>')
    html.append('</div>')

    # 九、朝代×月份 + 朝代×花名
    html.append('<h2>九、朝代与月份、朝代与花名</h2>')
    html.append('<p>下面用两张热力图分别呈现：<strong>朝代×月份</strong>（各朝各月诗作数量）、<strong>朝代×花名</strong>（各朝各花诗作数量），便于同时从时间与花种两个维度看分布。</p>')
    dm_rows = [[dynasty_month[d].get(m, 0) for m in MONTHS_SORTED] for d in top_dynasties]
    html.append('<div class="fig-wrap"><h3>主要朝代 × 月份 数量热力图（Top 8 朝代）</h3>' + svg_heatmap(dm_rows, top_dynasties, MONTHS_SORTED, max_cell=48) + '</div>')
    html.append('<div class="fig-wrap"><h3>主要朝代 × 花名 数量热力图（Top 10 朝代 × Top 25 花名）</h3><p>行=朝代，列=花名，单元格=该朝该花诗作条数。</p>' + svg_heatmap(dynasty_flower_rows, top_dynasties_10, top_flowers_25, max_cell=32) + '</div>')
    # 全表：所有朝代 × 每月诗作数量
    all_dynasties = sorted(dynasty_month_all.keys(), key=lambda d: -sum(dynasty_month_all[d].values()))
    html.append('<div class="fig-wrap"><h3>每朝代每月诗作数量（全表）</h3><p>行=朝代，列=月份，单元格=该朝代在该月的诗作条数。</p>')
    html.append('<table class="stats-table"><tr><th>朝代</th>' + ''.join(f'<th>{escape(m)}</th>' for m in MONTHS_SORTED) + '<th>合计</th></tr>')
    for d in all_dynasties:
        row_vals = [dynasty_month_all[d].get(m, 0) for m in MONTHS_SORTED]
        total = sum(row_vals)
        html.append('<tr><td>' + escape(d) + '</td>' + ''.join(f'<td>{v}</td>' for v in row_vals) + f'<td><strong>{total}</strong></td></tr>')
    html.append('</table></div>')

    # 十、冲突 ID
    html.append('<div class="footer">报告由 poems_dataset_eda.py 自动生成 · 诗花雅送 Poetic Flora Advisor 数据探索</div>')
    html.append('</body></html>')

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(''.join(html))

    print('✅ 报告已生成: poems_dataset_eda_report.html')
    print(f'   总条数: {n_total}  花名种类: {n_flowers}  作者数: {n_authors}  朝代数: {n_dynasties}')
    return OUTPUT_HTML


if __name__ == '__main__':
    run_eda()
