"""按「当前月」前后各一月」窗口过滤（跨年处理）。"""

from __future__ import annotations

import pandas as pd


def month_window_nums(month: int) -> set[int]:
    if month < 1 or month > 12:
        raise ValueError("month 须在 1–12")
    prev_m = 12 if month == 1 else month - 1
    next_m = 1 if month == 12 else month + 1
    return {prev_m, month, next_m}


def filter_by_month(df: pd.DataFrame, target_month: int | None) -> pd.DataFrame:
    """target_month 为 None 时不筛选。"""
    if target_month is None:
        return df
    if "月份数字" not in df.columns:
        raise ValueError("数据表缺少 月份数字 列")
    win = month_window_nums(target_month)
    s = df["月份数字"]
    return df[s.isin(win)].copy()


def month_match_score(poem_month: float | int, target: int) -> float:
    """用于排序：当月完全匹配权重高于邻月。"""
    try:
        m = int(poem_month)
    except (TypeError, ValueError):
        return 0.0
    if m == target:
        return 2.0
    if m in month_window_nums(target):
        return 1.0
    return 0.0
