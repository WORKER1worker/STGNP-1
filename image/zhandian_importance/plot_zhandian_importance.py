#!/usr/bin/env python3
"""Recreate station-importance and Top-K reconstruction figures from CSV results."""

from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

CONTRIBUTION_CSV = (
    ROOT
    / "STGNP/experiments/ratio_holdout_multi/contribution_20260528T161547/contribution_ranking.csv"
)
TOPK_MEAN_CSV = (
    ROOT
    / "STGNP/experiments/topk_key_stations/20260528T162352/results_summary_mean.csv"
)
STATION_INFO_CSV = ROOT / "STGNP/dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv"
TOP_STATION_COUNT = 16


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 14,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        path = OUT_DIR / f"{stem}{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 160
        fig.savefig(path, **kwargs)
    plt.close(fig)


def draw_station_importance(ax: plt.Axes, show_title: bool = True) -> None:
    contrib = pd.read_csv(CONTRIBUTION_CSV)
    stations = pd.read_csv(STATION_INFO_CSV)
    top = (
        contrib.sort_values("rank")
        .head(TOP_STATION_COUNT)
        .merge(stations[["station_id", "lon", "lat"]], on="station_id", how="left")
        .iloc[::-1]
        .reset_index(drop=True)
    )
    top["plot_delta_mae"] = top["delta_mae"].clip(lower=0.0)

    y = np.arange(len(top))
    colors = mpl.colormaps["viridis"](np.linspace(1.0, 0.08, len(top)))
    bars = ax.barh(y, top["plot_delta_mae"], color=colors, edgecolor="white", linewidth=1.0)

    max_score = float(top["plot_delta_mae"].max())
    ax.set_xlim(0, max_score * 1.20)
    ax.set_yticks(y)
    ax.set_yticklabels(top["station_id"], fontsize=12)
    ax.set_xlabel(r"Contribution Score ($\Delta$MAE)", fontsize=15)
    if show_title:
        ax.set_title(
            f"Figure 10a: Top-{TOP_STATION_COUNT} Key Stations Contribution Ranking",
            fontsize=17,
            pad=8,
        )
    ax.grid(axis="x", color="#cfcfcf", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    for bar, (_, row) in zip(bars, top.iterrows()):
        value = float(row["plot_delta_mae"])
        y_mid = bar.get_y() + bar.get_height() / 2
        if value > max_score * 0.18:
            label_x = value - max_score * 0.012
            label_color = "white"
            label_ha = "right"
        else:
            label_x = value + max_score * 0.018
            label_color = "#111111"
            label_ha = "left"
        ax.text(
            label_x,
            y_mid,
            f"{value:.4f}",
            ha=label_ha,
            va="center",
            fontsize=10,
            color=label_color,
        )
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            coord_x = max(value + max_score * 0.085, max_score * 0.16)
            ax.text(
                coord_x,
                y_mid,
                f"({row['lat']:.1f}N, {row['lon']:.1f}E)",
                ha="left",
                va="center",
                fontsize=8,
                color="#505050",
            )


def plot_station_importance() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    draw_station_importance(ax, show_title=True)
    fig.tight_layout()
    save_figure(fig, "站点重要性")


def annotate_series(
    ax: plt.Axes, x: pd.Series, y: pd.Series, labels, offsets, fontsize: float = 10.0
) -> None:
    for xi, yi, label, (dx, dy) in zip(x, y, labels, offsets):
        if not label:
            continue
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#111111",
        )


def smooth_curve(x: pd.Series, y: pd.Series, points: int = 320) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    x_dense = np.linspace(x_arr.min(), x_arr.max(), points)
    y_dense = PchipInterpolator(x_arr, y_arr)(x_dense)
    return x_dense, y_dense


def smooth_band(
    x: pd.Series, mean: pd.Series, std: pd.Series, points: int = 320
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = mean - std
    upper = mean + std
    x_dense, lower_s = smooth_curve(x, lower, points)
    _, upper_s = smooth_curve(x, upper, points)
    return x_dense, lower_s, upper_s


def draw_topk_reconstruction(ax: plt.Axes, show_title: bool = True) -> plt.Axes:
    df = pd.read_csv(TOPK_MEAN_CSV).sort_values("top_k").reset_index(drop=True)
    x = df["top_k"]
    mae = df["overall_mae_mean"]
    rmse = df["overall_rmse_mean"]
    r2 = df["overall_r2_mean"]

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(1.1)

    blue = "#2C7FB8"
    orange = "#F28E2B"
    green = "#43A047"

    x_mae, mae_s = smooth_curve(x, mae)
    x_rmse, rmse_s = smooth_curve(x, rmse)
    x_r2, r2_s = smooth_curve(x, r2)
    x_mae_band, mae_lower, mae_upper = smooth_band(x, mae, df["overall_mae_std"])
    x_rmse_band, rmse_lower, rmse_upper = smooth_band(x, rmse, df["overall_rmse_std"])
    x_r2_band, r2_lower, r2_upper = smooth_band(x, r2, df["overall_r2_std"])

    ax.fill_between(x_mae_band, mae_lower, mae_upper, color=blue, alpha=0.18, zorder=1)
    ax.plot(x_mae, mae_s, color=blue, linewidth=2.4, label="MAE", zorder=3)
    ax.scatter(x, mae, s=28, facecolor="white", edgecolor=blue, linewidth=1.1, zorder=4)

    ax.fill_between(x_rmse_band, rmse_lower, rmse_upper, color=orange, alpha=0.20, zorder=1)
    ax.plot(x_rmse, rmse_s, color=orange, linewidth=2.4, label="RMSE", zorder=3)
    ax.scatter(x, rmse, s=28, facecolor="white", edgecolor=orange, linewidth=1.1, zorder=4)

    ax2.fill_between(x_r2_band, r2_lower, r2_upper, color=green, alpha=0.18, zorder=1)
    ax2.plot(x_r2, r2_s, color=green, linestyle="--", linewidth=2.4, label=r"$R^2$", zorder=3)
    ax2.scatter(x, r2, s=28, facecolor="white", edgecolor=green, linewidth=1.1, zorder=4)

    best_k = int(df.loc[df["overall_mae_mean"].idxmin(), "top_k"])
    ax.axvline(best_k, color="#777777", linestyle=":", linewidth=2.0)
    ax.text(
        best_k - 0.18,
        ax.get_ylim()[1] - 0.004,
        f"Recommended\n(Top-{best_k})",
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.92},
    )

    ax.set_xlim(float(x.min()) - 0.6, float(x.max()) + 1.0)
    ax.set_xticks(x)
    ax.set_ylim(0.045, 0.14)
    ax2.set_ylim(-0.30, 0.78)
    ax.set_xlabel(r"Number of Retained Stations $K$", fontsize=15)
    ax.set_ylabel("Error (MAE / RMSE)", fontsize=15)
    ax2.set_ylabel(r"$R^2$", fontsize=15)
    if show_title:
        ax.set_title("Figure 10b: Top-K Reconstruction Performance", fontsize=18, pad=8)
    ax.grid(axis="y", color="#b8b8b8", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)

    annotate_series(
        ax,
        x,
        mae,
        [f"{v:.4f}" for v in mae],
        [(0, 5), (0, 5), (0, 5), (0, 5), (-2, 5), (0, -15), (-16, 5)],
    )
    annotate_series(
        ax,
        x,
        rmse,
        [f"{v:.4f}" for v in rmse],
        [(0, 5), (0, 5), (0, 5), (0, 5), (-2, 5), (0, 5), (-16, 5)],
    )
    annotate_series(
        ax2,
        x,
        r2,
        [f"{v:.3f}" for v in r2[:-1]] + [""],
        [(0, 5), (0, 5), (0, 5), (0, -23), (12, 6), (-12, -18), (-64, -18)],
    )

    label_x = float(x.max()) + 0.24
    ax.text(label_x, float(mae.iloc[-1]), "MAE", color=blue, va="center", ha="left", fontsize=11)
    ax.text(label_x, float(rmse.iloc[-1]), "RMSE", color=orange, va="center", ha="left", fontsize=11)
    ax2.text(label_x, float(r2.iloc[-1]), r"$R^2$", color=green, va="center", ha="left", fontsize=11)
    return ax2


def plot_topk_reconstruction() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    draw_topk_reconstruction(ax, show_title=True)
    fig.tight_layout()
    save_figure(fig, "top-k对比")


def plot_combined_figure() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.6, 6.4),
        gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.30},
    )
    draw_station_importance(axes[0], show_title=False)
    draw_topk_reconstruction(axes[1], show_title=False)

    axes[0].text(
        -0.13,
        1.03,
        "a",
        transform=axes[0].transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axes[1].text(
        -0.12,
        1.03,
        "b",
        transform=axes[1].transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.96, wspace=0.32)
    save_figure(fig, "站点重要性_top-k组合")


def plot_combined_figure_vertical() -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 10.2),
        gridspec_kw={"height_ratios": [1.10, 1.0], "hspace": 0.24},
    )
    draw_station_importance(axes[0], show_title=False)
    draw_topk_reconstruction(axes[1], show_title=False)

    axes[0].text(
        -0.11,
        1.02,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axes[1].text(
        -0.11,
        1.02,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.07, top=0.97, hspace=0.24)
    save_figure(fig, "站点重要性_top-k上下组合")


def main() -> None:
    plot_station_importance()
    plot_topk_reconstruction()
    plot_combined_figure()
    plot_combined_figure_vertical()


if __name__ == "__main__":
    main()
