"""
poems_dataset_merge.py — 三人审核结果按月份合并
=================================================
从三份审核文件中按月份提取数据并合并：
  - poems_dataset_v5_reviewed_Zhang Rui_revised.csv  → 1–4 月（正月、二月、三月、四月）
  - poems_dataset_v5_reviewed_Wang Donnie.csv       → 5–8 月（五月、六月、七月、八月）
  - poems_dataset_v5_reviewed_Shi Zhigang.csv      → 9–12 月（九月、十月、十一月、十二月）

ID 冲突时：全部保留，将同一 ID 的多条记录依次改为 xxx-01、xxx-02、……
合并完成后输出 poems_dataset_merged.csv，并打印合并报告（总条数、冲突 ID 列表）。

用法：
    python3 poems_dataset_merge.py
    python3 poems_dataset_merge.py --dry-run   # 只打印报告，不写文件
"""

import argparse
import csv
import os
from collections import defaultdict

MONTHS_1_4 = ('正月', '二月', '三月', '四月')
MONTHS_5_8 = ('五月', '六月', '七月', '八月')
MONTHS_9_12 = ('九月', '十月', '十一月', '十二月')

SOURCES = [
    ('poems_dataset_v5_reviewed_Zhang Rui_revised.csv', MONTHS_1_4, 'Zhang Rui (1-4月)'),
    ('poems_dataset_v5_reviewed_Wang Donnie.csv', MONTHS_5_8, 'Wang Donnie (5-8月)'),
    ('poems_dataset_v5_reviewed_Shi Zhigang.csv', MONTHS_9_12, 'Shi Zhigang (9-12月)'),
]

OUTPUT_FILE = 'poems_dataset_merged.csv'


def load_rows(path: str, months: tuple) -> list[dict]:
    """读取 CSV，只保留月份在 months 内的行。优先 UTF-8，失败则尝试 GBK。"""
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030'):
        try:
            with open(path, encoding=enc) as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = [r for r in reader if (r.get('月份') or '').strip() in months]
            return fieldnames, rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if (r.get('月份') or '').strip() in months]
    return fieldnames, rows


def merge_with_id_resolution(all_rows: list[dict]) -> tuple[list[dict], dict]:
    """
    合并所有行。对重复出现的 ID 重命名为 xxx-01, xxx-02, ...
    返回 (合并后的 rows, 冲突信息 {原ID: [新ID列表]})。
    """
    id_to_indices = defaultdict(list)
    for i, row in enumerate(all_rows):
        raw_id = (row.get('ID') or '').strip()
        id_to_indices[raw_id].append(i)

    conflicts = {}  # original_id -> [new_id, new_id, ...]
    for raw_id, indices in id_to_indices.items():
        if len(indices) <= 1:
            continue
        suffixes = [f'{raw_id}-{str(j + 1).zfill(2)}' for j in range(len(indices))]
        conflicts[raw_id] = suffixes
        for j, idx in enumerate(indices):
            all_rows[idx]['ID'] = suffixes[j]

    return all_rows, conflicts


def main():
    parser = argparse.ArgumentParser(description='诗花雅送 · 三人审核结果按月份合并')
    parser.add_argument('--dry-run', action='store_true', help='只打印报告，不写文件')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_rows = []
    fieldnames = None

    for filename, months, label in SOURCES:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f'❌ 找不到文件: {filename}')
            return
        fn, rows = load_rows(path, months)
        if fieldnames is None:
            fieldnames = fn
        all_rows.extend(rows)
        print(f'✅ {label}: 从 {filename} 提取 {len(rows)} 条')

    if not all_rows:
        print('❌ 没有可合并的数据')
        return

    merged, conflicts = merge_with_id_resolution(all_rows)

    # ── 报告内容 ──
    report_lines = [
        '诗花雅送 · 合并报告',
        '=' * 60,
        f'总数据条数: {len(merged)}',
        '',
        f'冲突 ID 数量: {len(conflicts)} 个' if conflicts else '冲突 ID: 无',
    ]
    if conflicts:
        report_lines.append('冲突 ID 及新 ID 对应:')
        def _sort_key(x):
            s = x[0]
            try:
                return (0, int(s))
            except (ValueError, TypeError):
                return (1, str(s))
        for orig, new_ids in sorted(conflicts.items(), key=_sort_key):
            report_lines.append(f'  {orig} → {", ".join(new_ids)}')
    report_lines.append('=' * 60)
    report_text = '\n'.join(report_lines)

    print('\n' + report_text)

    if args.dry_run:
        print('\n[--dry-run] 未写入文件')
        return

    out_path = os.path.join(base_dir, OUTPUT_FILE)
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    report_path = os.path.join(base_dir, 'poems_dataset_merge_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f'\n✅ 已写入: {OUTPUT_FILE}')
    print(f'✅ 报告已保存: poems_dataset_merge_report.txt')


if __name__ == '__main__':
    main()
