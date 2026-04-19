#!/usr/bin/env python3
"""Analyze missing rates across multiple SM_NQ depth CSV files.

Outputs:
- Long-form missing statistics by station and depth.
- Pivot table (station x depth).
- Station ranking for multi-depth modeling.
- Markdown report with readable tables and recommendations.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


TIME_COLS = {"yyyy", "mm", "dd", "HH", "MM", "SS", "datetime", "date", "time", "timestamp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze station missing rates across SM_NQ depths")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dataset/SM_NQ_depths"),
        help="Directory containing SM_NQ-30-minutes_*cm.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset/SM_NQ_depths"),
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="SM_NQ-30-minutes_*cm.csv",
        help="CSV filename pattern",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=-99.0,
        help="Missing placeholder value in CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Top-K recommended stations to output in markdown",
    )
    return parser.parse_args()


def extract_depth_label(filename: str) -> Tuple[int, str]:
    match = re.search(r"_(\d+)cm\.csv$", filename)
    if not match:
        return 9999, "unknown"
    depth_num = int(match.group(1))
    return depth_num, f"{depth_num}cm"


def station_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name in TIME_COLS:
            continue
        if name.lower().startswith("unnamed"):
            continue
        cols.append(name)
    return cols


def analyze_one_depth(csv_path: Path, missing_value: float) -> pd.DataFrame:
    depth_num, depth_label = extract_depth_label(csv_path.name)
    df = pd.read_csv(csv_path)

    st_cols = station_columns(df)
    if not st_cols:
        raise ValueError(f"No station columns found in {csv_path}")

    station_df = df[st_cols].apply(pd.to_numeric, errors="coerce")
    missing_mask = station_df.isna() | station_df.eq(missing_value)
    missing_count = missing_mask.sum(axis=0)

    total_rows = len(df)
    out = pd.DataFrame(
        {
            "station_id": st_cols,
            "depth_cm": depth_label,
            "depth_num": depth_num,
            "total_rows": total_rows,
            "missing_count": missing_count.values,
            "non_missing_count": (total_rows - missing_count.values),
            "missing_rate": (missing_count.values / total_rows),
            "missing_rate_percent": (missing_count.values / total_rows * 100.0),
            "source_csv": csv_path.name,
        }
    )
    return out


def build_station_ranking(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby("station_id", as_index=False).agg(
        depth_count=("depth_cm", "nunique"),
        avg_missing_rate_percent=("missing_rate_percent", "mean"),
        max_missing_rate_percent=("missing_rate_percent", "max"),
        min_missing_rate_percent=("missing_rate_percent", "min"),
        median_missing_rate_percent=("missing_rate_percent", "median"),
        std_missing_rate_percent=("missing_rate_percent", "std"),
        avg_non_missing_count=("non_missing_count", "mean"),
    )

    grouped["std_missing_rate_percent"] = grouped["std_missing_rate_percent"].fillna(0.0)
    grouped = grouped.sort_values(
        ["avg_missing_rate_percent", "max_missing_rate_percent", "std_missing_rate_percent", "station_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return grouped


def build_recommendation(ranking_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tier_a = ranking_df[
        (ranking_df["avg_missing_rate_percent"] <= 3.0)
        & (ranking_df["max_missing_rate_percent"] <= 5.0)
    ].copy()

    tier_b = ranking_df[
        (ranking_df["avg_missing_rate_percent"] <= 5.0)
        & (ranking_df["max_missing_rate_percent"] <= 10.0)
    ].copy()

    return {
        "tier_a": tier_a,
        "tier_b": tier_b,
    }


def format_table(df: pd.DataFrame, float_digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float"]).columns
    for col in float_cols:
        out[col] = out[col].map(lambda x: f"{x:.{float_digits}f}")
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "（空表）"

    text_df = df.astype(str)
    headers = list(text_df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    depth_summary: pd.DataFrame,
    pivot_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    recommendation: Dict[str, pd.DataFrame],
    top_k: int,
) -> None:
    top_df = ranking_df.head(top_k)
    tier_a = recommendation["tier_a"].head(top_k)
    tier_b = recommendation["tier_b"].head(top_k)

    depth_summary_display = format_table(depth_summary).rename(
        columns={
            "depth_num": "深度编号",
            "depth_cm": "深度",
            "station_count": "站点数",
            "avg_missing_rate_percent": "平均缺失率(%)",
            "median_missing_rate_percent": "中位缺失率(%)",
            "min_missing_rate_percent": "最小缺失率(%)",
            "max_missing_rate_percent": "最大缺失率(%)",
            "stations_le_1pct": "缺失率<=1%站点数",
            "stations_le_5pct": "缺失率<=5%站点数",
        }
    )

    rank_cols = [
        "rank",
        "station_id",
        "avg_missing_rate_percent",
        "max_missing_rate_percent",
        "std_missing_rate_percent",
    ]

    top_display = format_table(top_df[rank_cols]).rename(
        columns={
            "rank": "排名",
            "station_id": "站点ID",
            "avg_missing_rate_percent": "平均缺失率(%)",
            "max_missing_rate_percent": "最大缺失率(%)",
            "std_missing_rate_percent": "缺失率波动(标准差)",
        }
    )

    tier_a_display = format_table(tier_a[rank_cols]).rename(
        columns={
            "rank": "排名",
            "station_id": "站点ID",
            "avg_missing_rate_percent": "平均缺失率(%)",
            "max_missing_rate_percent": "最大缺失率(%)",
            "std_missing_rate_percent": "缺失率波动(标准差)",
        }
    )

    tier_b_display = format_table(tier_b[rank_cols]).rename(
        columns={
            "rank": "排名",
            "station_id": "站点ID",
            "avg_missing_rate_percent": "平均缺失率(%)",
            "max_missing_rate_percent": "最大缺失率(%)",
            "std_missing_rate_percent": "缺失率波动(标准差)",
        }
    )

    matrix_display = format_table(pivot_df.reset_index()).rename(columns={"station_id": "站点ID"})

    lines: List[str] = []
    lines.append("# SM_NQ 多深度站点缺失率分析报告")
    lines.append("")
    lines.append("## 1. 各深度总体缺失情况")
    lines.append("")
    lines.append(dataframe_to_markdown(depth_summary_display))
    lines.append("")

    lines.append("## 2. 推荐站点（按跨深度缺失率从低到高）")
    lines.append("")
    lines.append(f"跨 4 个深度综合平均缺失率最低的前 {top_k} 个站点：")
    lines.append("")
    lines.append(dataframe_to_markdown(top_display))
    lines.append("")

    lines.append("Tier-A 候选站点（平均缺失率 <= 3%，且最大缺失率 <= 5%）：")
    lines.append("")
    if tier_a.empty:
        lines.append("没有站点满足 Tier-A 条件。")
    else:
        lines.append(dataframe_to_markdown(tier_a_display))
    lines.append("")

    lines.append("Tier-B 候选站点（平均缺失率 <= 5%，且最大缺失率 <= 10%）：")
    lines.append("")
    if tier_b.empty:
        lines.append("没有站点满足 Tier-B 条件。")
    else:
        lines.append(dataframe_to_markdown(tier_b_display))
    lines.append("")

    lines.append("## 3. 站点 x 深度缺失率矩阵（%）")
    lines.append("")
    lines.append("行是站点 ID，列是不同深度。")
    lines.append("")
    lines.append(dataframe_to_markdown(matrix_display))
    lines.append("")

    lines.append("## 4. 选站建议")
    lines.append("")
    lines.append("- 优先选择平均缺失率和最大缺失率都较低的站点。")
    lines.append("- 做多深度预测时，优先选择跨深度缺失率波动小（标准差低）的站点。")
    lines.append("- 如果模型对缺失较敏感，建议先用 Tier-A 站点建模，再逐步扩展到 Tier-B。")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    csv_files = sorted(args.input_dir.glob(args.pattern), key=lambda p: extract_depth_label(p.name)[0])
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched pattern '{args.pattern}' under {args.input_dir}")

    depth_results: List[pd.DataFrame] = []
    for csv_path in csv_files:
        depth_df = analyze_one_depth(csv_path, args.missing_value)
        depth_results.append(depth_df)

    long_df = pd.concat(depth_results, axis=0, ignore_index=True)
    long_df = long_df.sort_values(["depth_num", "station_id"]).reset_index(drop=True)

    pivot_df = long_df.pivot(index="station_id", columns="depth_cm", values="missing_rate_percent")
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns, key=lambda x: int(re.sub(r"cm", "", x))), axis=1)
    pivot_df = pivot_df.sort_index()

    ranking_df = build_station_ranking(long_df)
    recommendation = build_recommendation(ranking_df)

    depth_summary = (
        long_df.groupby(["depth_num", "depth_cm"], as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_missing_rate_percent=("missing_rate_percent", "mean"),
            median_missing_rate_percent=("missing_rate_percent", "median"),
            min_missing_rate_percent=("missing_rate_percent", "min"),
            max_missing_rate_percent=("missing_rate_percent", "max"),
            stations_le_1pct=("missing_rate_percent", lambda s: int((s <= 1.0).sum())),
            stations_le_5pct=("missing_rate_percent", lambda s: int((s <= 5.0).sum())),
        )
        .sort_values("depth_num")
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_csv = args.output_dir / "SM_NQ_depths_missing_rate_long.csv"
    pivot_csv = args.output_dir / "SM_NQ_depths_missing_rate_matrix.csv"
    ranking_csv = args.output_dir / "SM_NQ_depths_station_ranking.csv"
    tier_a_csv = args.output_dir / "SM_NQ_depths_recommend_tier_a.csv"
    tier_b_csv = args.output_dir / "SM_NQ_depths_recommend_tier_b.csv"
    summary_csv = args.output_dir / "SM_NQ_depths_missing_rate_summary.csv"
    report_md = args.output_dir / "SM_NQ_depths_missing_rate_report.md"

    long_df.to_csv(long_csv, index=False, encoding="utf-8")
    pivot_df.reset_index().to_csv(pivot_csv, index=False, encoding="utf-8")
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8")
    recommendation["tier_a"].to_csv(tier_a_csv, index=False, encoding="utf-8")
    recommendation["tier_b"].to_csv(tier_b_csv, index=False, encoding="utf-8")
    depth_summary.to_csv(summary_csv, index=False, encoding="utf-8")

    write_markdown_report(
        output_path=report_md,
        depth_summary=depth_summary,
        pivot_df=pivot_df,
        ranking_df=ranking_df,
        recommendation=recommendation,
        top_k=args.top_k,
    )

    print("Analysis completed.")
    print(f"Input files: {len(csv_files)}")
    for f in csv_files:
        print(f"  - {f}")
    print("Generated outputs:")
    print(f"  - {long_csv}")
    print(f"  - {pivot_csv}")
    print(f"  - {ranking_csv}")
    print(f"  - {tier_a_csv}")
    print(f"  - {tier_b_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
