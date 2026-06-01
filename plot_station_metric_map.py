#!/usr/bin/env python3
"""Plot station metric values on a lon/lat map."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot station metric spatial distribution")
    parser.add_argument("--metrics-path", type=str, required=True, help="Path to station_metrics.csv")
    parser.add_argument("--station-meta", type=str, required=True, help="Path to station metadata CSV")
    parser.add_argument("--metric", type=str, default="MAE", help="Metric column to plot (MAE/RMSE/R2)")
    parser.add_argument("--output", type=str, default="", help="Output PNG path")
    parser.add_argument("--title", type=str, default="", help="Plot title")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--annotate", action="store_true", help="Annotate station_id on points")
    args = parser.parse_args()

    metric = args.metric.strip()
    metrics_df = pd.read_csv(args.metrics_path)
    meta_df = pd.read_csv(args.station_meta)

    for col in ["station_id", "lon", "lat"]:
        if col not in meta_df.columns:
            raise ValueError(f"Missing column in station meta: {col}")
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric column not found: {metric}")

    metrics_df["station_id"] = metrics_df["station_id"].astype(str)
    meta_df["station_id"] = meta_df["station_id"].astype(str)

    merged = metrics_df.merge(meta_df[["station_id", "lon", "lat"]], on="station_id", how="inner")
    if merged.empty:
        raise RuntimeError("No matching stations between metrics and metadata")

    out_path = args.output.strip() or os.path.join(os.path.dirname(args.metrics_path), f"station_metric_map_{metric}.png")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(
        merged["lon"],
        merged["lat"],
        c=merged[metric],
        cmap=args.cmap,
        s=60,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(args.title or f"Station Metric Map ({metric})")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(metric)

    if args.annotate:
        for _, row in merged.iterrows():
            ax.text(row["lon"], row["lat"], row["station_id"], fontsize=7, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
