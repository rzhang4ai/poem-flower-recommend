"""
build_dataset_v5p1.py  — 审核后花名修正
=========================================
在 poems_dataset_v5_reviewed_Zhang Rui.csv 基础上，批量修正指定条目的花名字段，
输出 poems_dataset_v5_reviewed_Zhang Rui_revised.csv。

花名修正规则（审核时无法修改的条目）：
  二月：ID 179–225, 1005–1007  樱桃花 → 海棠
  二月：ID 304–322, 1017       李花   → 梨花
  二月：ID 352–355            棠梨   → 菜花
  三月：ID 527                郁李花 → 楝花

用法：
    python3 build_dataset_v5p1.py
    python3 build_dataset_v5p1.py --dry-run   # 只预览改动，不写文件
"""

import argparse
import csv
import os
import shutil

# ── 花名修正规则 ──────────────────────────────────────────────────────────────
# 每条规则：(ID 集合, 错误花名或花名元组, 正确花名)
# 棠梨/棠梨花 均改为菜花
FLOWER_PATCH_RULES = [
    (set(range(179, 226)) | {1005, 1006, 1007}, '樱桃花', '海棠'),
    (set(range(304, 323)) | {1017}, '李花', '梨花'),
    (set(range(352, 356)), ('棠梨', '棠梨花'), '菜花'),  # 棠梨或棠梨花 → 菜花
    ({527}, '郁李花', '楝花'),
]


def _wrong_flower_matches(row_flower: str, wrong_flower) -> bool:
    if isinstance(wrong_flower, str):
        return (row_flower or '').strip() == wrong_flower
    return (row_flower or '').strip() in wrong_flower


def apply_flower_patches(rows: list[dict], dry_run: bool = False) -> tuple[list[dict], int]:
    """对 rows 应用所有花名修正规则，返回（修改后的 rows，修改条数）"""
    changed = 0
    for row in rows:
        try:
            rid = int(row.get('ID', -1))
        except (ValueError, TypeError):
            continue
        current = (row.get('花名') or '').strip()
        for id_set, wrong_flower, correct_flower in FLOWER_PATCH_RULES:
            if rid not in id_set:
                continue
            if not _wrong_flower_matches(current, wrong_flower):
                continue
            if dry_run:
                print(f"  [DRY-RUN] ID={rid:4d}  花名: {current} → {correct_flower}  诗名: {(row.get('诗名') or '')[:24]}")
            else:
                row['花名'] = correct_flower
            changed += 1
            break
    return rows, changed


def main():
    default_input = 'poems_dataset_v5_reviewed_Zhang Rui.csv'
    default_output = 'poems_dataset_v5_reviewed_Zhang Rui_revised.csv'

    parser = argparse.ArgumentParser(description='诗花雅送 · 审核后花名修正')
    parser.add_argument('--input', default=default_input, help='审核后的 CSV')
    parser.add_argument('--output', default=default_output, help='修正后输出 CSV')
    parser.add_argument('--dry-run', action='store_true', help='只预览改动，不写文件')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'❌ 找不到输入文件: {args.input}')
        return

    with open(args.input, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f'✅ 读取 {len(rows)} 条  来源: {args.input}')

    if args.dry_run:
        print('\n--- DRY-RUN 预览（不写文件）---')
        _, changed = apply_flower_patches(rows, dry_run=True)
        print(f'\n共将修改 {changed} 条')
        return

    rows, changed = apply_flower_patches(rows, dry_run=False)
    print(f'\n✏️  已修改花名 {changed} 条')

    if os.path.exists(args.output):
        shutil.copy2(args.output, args.output + '.bak')
        print(f'💾 已备份旧文件: {args.output}.bak')

    with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ 输出: {args.output}')


if __name__ == '__main__':
    main()
