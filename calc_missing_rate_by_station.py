#!/usr/bin/env python3
"""
按站点统计缺失率脚本。
默认将 -99.00 视为缺失值，同时也会统计 NaN 为空缺。
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计每个站点的数据缺失率")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("over/STGNP/data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv"),
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("over/STGNP/data/dataset/SM_NQ/SM_NQ-30-minutes_05cm_missing_rate_by_station.csv"),
        help="输出统计结果 CSV 路径",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=-99.0,
        help="用于表示缺失值的占位符（默认 -99.0）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    df = pd.read_csv(args.input)

    # 时间字段不参与站点缺失率统计
    time_cols = {"yyyy", "mm", "dd", "HH", "MM", "SS", "datetime", "date", "time", "timestamp"}
    station_cols = [
        c
        for c in df.columns
        if str(c) not in time_cols and not str(c).lower().startswith("unnamed")
    ]

    if not station_cols:
        raise ValueError("未检测到站点列，请检查输入文件列名")

    station_df = df[station_cols]

    # 缺失定义：等于 missing_value 或原本就是 NaN
    missing_mask = station_df.isna() | station_df.eq(args.missing_value)
    missing_count = missing_mask.sum(axis=0)

    total_rows = len(df)
    result = pd.DataFrame(
        {
            "station": station_cols,
            "missing_count": missing_count.values,
            "missing_rate_percent": (missing_count.values / total_rows * 100),
            "non_missing_count": (total_rows - missing_count.values),
        }
    ).sort_values(["missing_rate_percent", "station"], ascending=[False, True])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"输入文件: {args.input}")
    print(f"总行数: {total_rows}")
    print(f"站点数: {len(station_cols)}")
    print(f"缺失标记值: {args.missing_value}")
    print(f"平均缺失率(%): {result['missing_rate_percent'].mean():.4f}")
    print(f"最大缺失率(%): {result['missing_rate_percent'].max():.4f}")
    print(f"最小缺失率(%): {result['missing_rate_percent'].min():.4f}")
    print(f"有缺失站点数: {(result['missing_count'] > 0).sum()}")
    print(f"无缺失站点数: {(result['missing_count'] == 0).sum()}")
    print("\n缺失率最高的前 10 个站点:")
    print(result.head(10).to_string(index=False))
    print(f"\n统计结果已保存: {args.output}")


if __name__ == "__main__":
    main()
