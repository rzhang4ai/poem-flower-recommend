"""
step0_sample/sample_200.py
===========================
分层抽样：从 poems_dataset_v5.csv 中抽取200条，
保证月份、花名、朝代的多样性覆盖。

用法：
    python3 sample_200.py
    python3 sample_200.py --input ../../poems_dataset_v5.csv --n 200
"""

import argparse
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTH_ORDER = {
    '正月':1,'二月':2,'三月':3,'四月':4,'五月':5,'六月':6,
    '七月':7,'八月':8,'九月':9,'十月':10,'十一月':11,'十二月':12
}

def sample_200(df: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    分层抽样策略：
    1. 按月份均匀抽（每月约 n/12 条）→ 保证季节覆盖
    2. 优先选赏析最长的条目（信息量最大）
    3. 补充抽取覆盖更多花名和朝代
    4. 加入正文最短的极端case（边界测试）
    """
    rng = np.random.default_rng(seed)
    selected_ids = set()

    # ── 第一层：按月份分层，每月按赏析长度优先 ──────────────────────────────
    df['赏析长度'] = df['赏析'].fillna('').str.len()
    df['正文长度'] = df['正文'].fillna('').str.len()
    df['月份数字'] = df['月份'].map(MONTH_ORDER).fillna(0).astype(int)

    per_month = max(1, n // 12)
    for month, group in df.groupby('月份'):
        # 按赏析长度降序，优先选信息量大的
        top = group.sort_values('赏析长度', ascending=False)
        take = min(per_month, len(top))
        selected_ids.update(top.head(take).index.tolist())

    # ── 第二层：补充覆盖更多花名（每个花名至少1条）──────────────────────────
    flowers_covered = df.loc[list(selected_ids), '花名'].unique()
    flowers_missing = df[~df['花名'].isin(flowers_covered)]['花名'].unique()
    for flower in flowers_missing:
        cands = df[df['花名'] == flower].sort_values('赏析长度', ascending=False)
        if len(cands) > 0 and len(selected_ids) < n:
            selected_ids.add(cands.index[0])

    # ── 第三层：补充覆盖更多朝代 ─────────────────────────────────────────────
    dynasties_covered = df.loc[list(selected_ids), '朝代'].unique()
    dynasties_missing = df[~df['朝代'].isin(dynasties_covered)]['朝代'].unique()
    for dynasty in dynasties_missing:
        cands = df[df['朝代'] == dynasty].sort_values('赏析长度', ascending=False)
        if len(cands) > 0 and len(selected_ids) < n:
            selected_ids.add(cands.index[0])

    # ── 第四层：加入正文极短的边界case ──────────────────────────────────────
    short_poem = df[df['正文长度'] < 20].sort_values('正文长度')
    for idx in short_poem.index:
        if len(selected_ids) >= n:
            break
        selected_ids.add(idx)

    # ── 第五层：随机补足至 n 条 ──────────────────────────────────────────────
    remaining = df[~df.index.isin(selected_ids)]
    if len(selected_ids) < n and len(remaining) > 0:
        fill_n = min(n - len(selected_ids), len(remaining))
        fill_idx = rng.choice(remaining.index.tolist(), size=fill_n, replace=False)
        selected_ids.update(fill_idx.tolist())

    result = df.loc[list(selected_ids)].copy()
    result = result.sort_values('月份数字').reset_index(drop=True)
    result['sample_id'] = range(1, len(result) + 1)
    return result


def print_stats(df: pd.DataFrame, sample: pd.DataFrame):
    print("\n── 原始数据集 ─────────────────────────────")
    print(f"  总条数:   {len(df)}")
    print(f"  花名种类: {df['花名'].nunique()}")
    print(f"  朝代数:   {df['朝代'].nunique()}")
    print(f"  月份数:   {df['月份'].nunique()}")

    print("\n── 抽样结果 ────────────────────────────────")
    print(f"  抽取条数: {len(sample)}")
    print(f"  花名覆盖: {sample['花名'].nunique()} / {df['花名'].nunique()}")
    print(f"  朝代覆盖: {sample['朝代'].nunique()} / {df['朝代'].nunique()}")
    print(f"  月份覆盖: {sample['月份'].nunique()} / 12")

    print("\n── 各月份分布 ──────────────────────────────")
    month_dist = sample.groupby('月份').size().reset_index(name='count')
    month_dist['月份数字'] = month_dist['月份'].map(MONTH_ORDER)
    month_dist = month_dist.sort_values('月份数字')
    for _, row in month_dist.iterrows():
        bar = '█' * row['count']
        print(f"  {row['月份']:5s}: {row['count']:3d} {bar}")

    print("\n── 赏析长度分布 ─────────────────────────────")
    print(f"  平均: {sample['赏析长度'].mean():.0f} 字")
    print(f"  最短: {sample['赏析长度'].min()} 字")
    print(f"  最长: {sample['赏析长度'].max()} 字")
    print(f"  赏析为空: {(sample['赏析'].fillna('') == '').sum()} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='../../poems_dataset_v5.csv')
    parser.add_argument('--n',      type=int, default=200)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), args.input)
    if not os.path.exists(csv_path):
        # 兼容直接放在同目录的情况
        csv_path = args.input
    if not os.path.exists(csv_path):
        print(f"❌ 找不到输入文件: {csv_path}")
        print("   请将 poems_dataset_v5.csv 放在项目根目录，或用 --input 指定路径")
        return

    print(f"✅ 读取数据集: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 过滤掉正文和赏析都为空的条目（无法参与NLP分析）
    df = df[~((df['正文'].fillna('') == '') & (df['赏析'].fillna('') == ''))].copy()

    sample = sample_200(df, n=args.n, seed=args.seed)
    print_stats(df, sample)

    out_path = os.path.join(OUTPUT_DIR, "sample_200.csv")
    sample.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 已保存: {out_path}")

    # 同时保存抽样统计报告
    stats_path = os.path.join(OUTPUT_DIR, "sample_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"抽样参数: n={args.n}, seed={args.seed}\n")
        f.write(f"输入文件: {csv_path}\n")
        f.write(f"抽取条数: {len(sample)}\n")
        f.write(f"花名覆盖: {sample['花名'].nunique()}\n")
        f.write(f"朝代覆盖: {sample['朝代'].nunique()}\n\n")
        f.write("各月份分布:\n")
        for m, cnt in sample.groupby('月份').size().items():
            f.write(f"  {m}: {cnt}\n")
    print(f"📊 统计报告: {stats_path}")


if __name__ == '__main__':
    main()
