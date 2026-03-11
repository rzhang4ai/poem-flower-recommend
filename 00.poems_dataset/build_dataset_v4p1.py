"""
build_dataset_v4p1.py  — Patch 1
===================================
问题：ID 7–166 的条目月份被误标为「十一月」，实际应属「正月」。
操作：读取 poems_dataset_v4.csv，批量修正月份字段，输出 poems_dataset_v5.csv。

用法：
    python3 build_dataset_v4p1.py
    python3 build_dataset_v4p1.py --input poems_dataset_v4.csv --output poems_dataset_v5.csv
    python3 build_dataset_v4p1.py --dry-run   # 只打印会改动哪些行，不写文件
"""

import argparse
import csv
import os
import shutil

MONTH_TO_INT = {
    '正月': 1,  '二月': 2,  '三月': 3,  '四月': 4,
    '五月': 5,  '六月': 6,  '七月': 7,  '八月': 8,
    '九月': 9,  '十月': 10, '十一月': 11, '十二月': 12,
}

# ── 修正规则 ──────────────────────────────────────────────────────────────────
# 每条规则：(ID起始, ID结束含, 错误月份, 正确月份)
# 可以在这里继续追加后续 patch 规则
PATCH_RULES = [
    (7, 166, '十一月', '正月'),
]

def apply_patches(rows: list[dict], dry_run: bool = False) -> tuple[list[dict], int]:
    """对 rows 应用所有 patch 规则，返回（修改后的rows，修改条数）"""
    changed = 0
    for row in rows:
        try:
            rid = int(row.get('ID', -1))
        except ValueError:
            continue
        for (id_from, id_to, wrong_month, correct_month) in PATCH_RULES:
            if id_from <= rid <= id_to and row.get('月份') == wrong_month:
                if dry_run:
                    print(f"  [DRY-RUN] ID={rid:4d}  月份: {wrong_month} → {correct_month}"
                          f"  诗名: {row.get('诗名','')[:20]}")
                else:
                    row['月份']     = correct_month
                    row['月份数字'] = str(MONTH_TO_INT.get(correct_month, 0))
                changed += 1
    return rows, changed


def main():
    parser = argparse.ArgumentParser(description='诗花雅送数据集 Patch 1')
    parser.add_argument('--input',   default='poems_dataset_v4.csv')
    parser.add_argument('--output',  default='poems_dataset_v5.csv')
    parser.add_argument('--dry-run', action='store_true',
                        help='只预览改动，不写文件')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'❌ 找不到输入文件: {args.input}')
        return

    # 读取
    with open(args.input, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f'✅ 读取 {len(rows)} 条  来源: {args.input}')

    # 验证：打印 patch 范围内改前的月份分布，方便确认
    affected = [r for r in rows
                if any(fr <= int(r.get('ID',-1) or -1) <= to
                       for fr,to,_,__ in PATCH_RULES)]
    months_before = {}
    for r in affected:
        m = r.get('月份','')
        months_before[m] = months_before.get(m, 0) + 1
    print(f'\n📋 Patch 范围内（ID 7–166）当前月份分布:')
    for m, cnt in sorted(months_before.items(), key=lambda x: MONTH_TO_INT.get(x[0], 99)):
        print(f'   {m}: {cnt} 条')

    if args.dry_run:
        print(f'\n--- DRY-RUN 预览（不写文件）---')
        _, changed = apply_patches(rows, dry_run=True)
        print(f'\n共将修改 {changed} 条')
        return

    # 应用
    rows, changed = apply_patches(rows, dry_run=False)
    print(f'\n✏️  已修改 {changed} 条（月份 → 正月，月份数字 → 1）')

    # 写出
    if os.path.exists(args.output):
        shutil.copy2(args.output, args.output + '.bak')
        print(f'💾 已备份旧文件: {args.output}.bak')

    with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ 输出: {args.output}')

    # 验证结果
    jan = sum(1 for r in rows if r.get('月份') == '正月')
    nov = sum(1 for r in rows if r.get('月份') == '十一月')
    print(f'\n📊 修改后月份统计:')
    print(f'   正月: {jan} 条')
    print(f'   十一月: {nov} 条')


if __name__ == '__main__':
    main()
