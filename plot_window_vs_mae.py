#!/usr/bin/env python3
"""Draw a publication-quality window-vs-metric chart from a summary CSV."""

import argparse
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw window vs MAE plot")
    parser.add_argument("--input", type=str, required=True, help="Input results_summary_mean.csv")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default="Window Length vs MAE (mean ± std)")
    parser.add_argument("--metric", type=str, default="overall_mae_mean", help="Mean metric column")
    parser.add_argument("--metric-std", type=str, default="overall_mae_std", help="Std metric column")
    parser.add_argument("--best-metric", type=str, default="overall_r2_mean", help="Column to locate best point")
    parser.add_argument("--best-mode", type=str, default="max", choices=["max", "min"], help="Best metric mode")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    required = ["window", args.metric, args.metric_std]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.sort_values("window")
    x = df["window"].to_numpy()
    y = df[args.metric].to_numpy()
    yerr = df[args.metric_std].to_numpy()

    best_col = args.best_metric if args.best_metric in df.columns else args.metric
    best_vals = df[best_col].to_numpy()
    best_idx = int(best_vals.argmax()) if args.best_mode == "max" else int(best_vals.argmin())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_color = "#2E5EAA"
    best_color = "#D1495B"

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=line_color,
        linewidth=2.7,
        marker="o",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        capsize=5,
        elinewidth=1.5,
        alpha=0.7,
        zorder=3,
    )

    ax.scatter([x[best_idx]], [y[best_idx]], color=best_color, s=70, zorder=5)
    ax.annotate(
        "Best",
        (x[best_idx], y[best_idx]),
        textcoords="offset points",
        xytext=(8, -16),
        fontsize=12,
        color=best_color,
    )

    ax.set_title(args.title, pad=12)
    ax.set_xlabel("Window (hours)")
    ax.set_ylabel(args.metric.replace("_mean", "").replace("overall_", "").upper())

    ax.yaxis.grid(True, linestyle="--", color="#c8c8c8", alpha=0.5)
    ax.xaxis.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_base, output_ext = os.path.splitext(args.output)
    png_path = args.output if output_ext.lower() == ".png" else output_base + ".png"
    pdf_path = output_base + ".pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
