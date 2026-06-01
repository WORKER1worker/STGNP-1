#!/usr/bin/env python3
"""汇总多个 checkpoint 的站点指标并绘制空间分布图。

运行示例:
    python plot_station_metric_map_aggregate.py \
        --checkpoints checkpoints/SM/hierarchical_SM_20260507T142853,checkpoints/SM/hierarchical_SM_20260507T144943,checkpoints/SM/hierarchical_SM_20260507T150939,checkpoints/SM/hierarchical_SM_20260507T153005,checkpoints/SM/hierarchical_SM_20260507T154953 \
        --station-meta data/dataset/SM_NQ/Stations_information_NAQU.csv \
        --metric MAE \
        --output checkpoints/SM/hierarchical_SM_20260507T154953/analysis/station_metric_map_MAE_aggregate.png \
        --csv-output checkpoints/SM/hierarchical_SM_20260507T154953/analysis/station_metric_map_MAE_aggregate.csv
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总站点指标并绘制空间分布图")
    parser.add_argument("--checkpoints", type=str, required=True, help="checkpoint 目录列表（逗号分隔）")
    parser.add_argument("--station-meta", type=str, required=True, help="站点元数据 CSV 路径")
    parser.add_argument("--metric", type=str, default="MAE", help="要汇总的指标列名（如 MAE/RMSE/R2）")
    parser.add_argument("--output", type=str, default="", help="输出 PNG 路径")
    parser.add_argument("--csv-output", type=str, default="", help="输出 CSV 路径")
    parser.add_argument("--title", type=str, default="", help="图标题")
    parser.add_argument("--cmap", type=str, default="viridis", help="颜色映射")
    parser.add_argument("--annotate", action="store_true", help="在图上标注 station_id")
    args = parser.parse_args()

    metric = args.metric.strip()
    # 解析 checkpoint 列表
    ckpts = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    if not ckpts:
        raise ValueError("No checkpoints provided")

    frames = []
    for ckpt in ckpts:
        metrics_path = os.path.join(ckpt, "analysis", "station_metrics.csv")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(f"Missing station_metrics.csv: {metrics_path}")
        df = pd.read_csv(metrics_path)
        if metric not in df.columns:
            raise ValueError(f"Metric column not found in {metrics_path}: {metric}")
        df = df[["station_id", metric]].copy()
        df["station_id"] = df["station_id"].astype(str)
        df.rename(columns={metric: "metric"}, inplace=True)
        df["checkpoint"] = ckpt
        frames.append(df)

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    # 对同名站点做均值/标准差/次数统计
    agg = all_df.groupby("station_id")["metric"].agg(["mean", "std", "count"]).reset_index()

    meta = pd.read_csv(args.station_meta)
    for col in ["station_id", "lon", "lat"]:
        if col not in meta.columns:
            raise ValueError(f"Missing column in station meta: {col}")
    meta["station_id"] = meta["station_id"].astype(str)

    # 关联经纬度
    merged = agg.merge(meta[["station_id", "lon", "lat"]], on="station_id", how="inner")
    if merged.empty:
        raise RuntimeError("No matching stations between metrics and metadata")

    # 输出汇总 CSV
    csv_path = args.csv_output.strip() or os.path.join(os.getcwd(), f"station_metric_aggregate_{metric}.csv")
    merged.to_csv(csv_path, index=False)

    # 输出图片
    out_path = args.output.strip() or os.path.join(os.getcwd(), f"station_metric_map_aggregate_{metric}.png")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(
        merged["lon"],
        merged["lat"],
        c=merged["mean"],
        cmap=args.cmap,
        s=60,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(args.title or f"Station Metric Mean Map ({metric})")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(f"{metric} (mean)")

    if args.annotate:
        for _, row in merged.iterrows():
            ax.text(row["lon"], row["lat"], row["station_id"], fontsize=7, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved map: {out_path}")


if __name__ == "__main__":
    main()
