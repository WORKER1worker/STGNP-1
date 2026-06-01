#!/usr/bin/env python3
"""Plot station-level prediction accuracy from the 80 key-site holdout reports."""

from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

SUMMARY_CSVS = [
    ROOT / "STGNP/experiments/ratio_holdout_multi/20260527T151211/results_summary_by_seed.csv",
    ROOT / "STGNP/experiments/ratio_holdout_multi/20260528T112911/results_summary_by_seed.csv",
    ROOT / "STGNP/experiments/ratio_holdout_multi/20260528T152428/results_summary_by_seed.csv",
]
STATION_INFO_CSV = ROOT / "STGNP/dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_report_table(report_path: Path) -> pd.DataFrame:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("station_id") and "MAE" in line and "RMSE" in line:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Cannot find station metric table in {report_path}")

    rows = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("="):
            break
        parts = stripped.split()
        if len(parts) != 7:
            continue
        rows.append(parts)

    if not rows:
        raise ValueError(f"Empty station metric table in {report_path}")

    df = pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "target_station_index",
            "original_node_index",
            "sample_count",
            "MAE",
            "RMSE",
            "R2",
        ],
    )
    for col in ["target_station_index", "original_node_index", "sample_count"]:
        df[col] = pd.to_numeric(df[col])
    for col in ["MAE", "RMSE", "R2"]:
        df[col] = pd.to_numeric(df[col])
    return df


def load_80_run_metrics() -> pd.DataFrame:
    frames = []
    for summary_csv in SUMMARY_CSVS:
        summary = pd.read_csv(summary_csv)
        for _, row in summary.iterrows():
            analysis_dir = Path(str(row["analysis_dir"]))
            report_path = analysis_dir / "analysis_report.txt"
            df = read_report_table(report_path)
            df["analysis_dir"] = str(analysis_dir)
            df["holdout_ratio"] = row["holdout_ratio"]
            df["seed"] = row["seed"]
            df["holdout_seed"] = row["holdout_seed"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_station_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = (
        metrics.groupby("station_id")
        .agg(
            mae_mean=("MAE", "mean"),
            mae_std=("MAE", "std"),
            rmse_mean=("RMSE", "mean"),
            r2_mean=("R2", "mean"),
            eval_count=("MAE", "count"),
            sample_count_total=("sample_count", "sum"),
        )
        .reset_index()
    )
    stations = pd.read_csv(STATION_INFO_CSV)
    stations["station_id"] = stations["station_id"].astype(str)
    agg["station_id"] = agg["station_id"].astype(str)
    merged = agg.merge(stations[["station_id", "lon", "lat"]], on="station_id", how="left")
    if merged[["lon", "lat"]].isna().any().any():
        missing = merged.loc[merged[["lon", "lat"]].isna().any(axis=1), "station_id"].tolist()
        raise ValueError(f"Missing coordinates for stations: {missing}")
    return merged


def degree_formatter(axis: str):
    suffix = "E" if axis == "lon" else "N"
    return FuncFormatter(lambda value, _: f"{value:.1f}°{suffix}")


def label_offsets(station_id: str) -> Tuple[int, int, str, str]:
    custom = {
        "BC02": (7, -7, "left", "top"),
        "BC03": (7, 3, "left", "bottom"),
        "BC04": (7, -2, "left", "center"),
        "BC05": (7, -4, "left", "center"),
        "BC06": (7, 5, "left", "bottom"),
        "BC07": (7, 1, "left", "center"),
        "BC08": (7, -5, "left", "center"),
        "CD01": (7, -2, "left", "center"),
        "CD03": (7, 4, "left", "bottom"),
        "CD07": (7, 0, "left", "center"),
        "C1": (-16, 8, "right", "bottom"),
        "C2": (7, 2, "left", "bottom"),
        "F3": (7, 6, "left", "bottom"),
        "F4": (7, 1, "left", "center"),
        "F5": (7, -7, "left", "top"),
        "P1": (-18, 6, "right", "bottom"),
        "P2": (7, 6, "left", "bottom"),
        "P10": (7, 5, "left", "bottom"),
        "P11": (7, -7, "left", "top"),
        "MS3494": (7, -8, "left", "top"),
        "MS3501": (7, 2, "left", "center"),
        "MS3513": (7, -2, "left", "center"),
        "MS3518": (7, -3, "left", "center"),
        "MS3523": (7, 0, "left", "center"),
        "MS3527": (7, -1, "left", "center"),
        "MS3533": (7, 0, "left", "center"),
    }
    return custom.get(station_id, (7, 1, "left", "center"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 220
        fig.savefig(OUT_DIR / f"{stem}{suffix}", **kwargs)
    plt.close(fig)


def plot_accuracy_map(agg: pd.DataFrame) -> None:
    plot_df = agg.sort_values("mae_mean", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8.9, 7.6))

    size = 45 + plot_df["eval_count"] * 4.0
    norm = mpl.colors.Normalize(vmin=0.03, vmax=max(0.15, float(plot_df["mae_mean"].max())))
    sc = ax.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=plot_df["mae_mean"],
        s=size,
        cmap="YlOrRd",
        norm=norm,
        edgecolors="black",
        linewidths=0.75,
        zorder=3,
    )

    high_error = set(plot_df.head(7)["station_id"])
    low_error = set(plot_df.tail(3)["station_id"])
    for _, row in plot_df.iterrows():
        dx, dy, ha, va = label_offsets(row["station_id"])
        emphasized = row["station_id"] in high_error or row["station_id"] in low_error
        ax.annotate(
            row["station_id"],
            xy=(row["lon"], row["lat"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2 if emphasized else 7.0,
            fontweight="bold" if emphasized else "normal",
            color="#111111" if emphasized else "#444444",
            ha=ha,
            va=va,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68 if emphasized else 0.45,
            },
            zorder=4,
        )

    ax.set_xlim(91.45, 92.62)
    ax.set_ylim(31.02, 31.96)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(degree_formatter("lon"))
    ax.yaxis.set_major_formatter(degree_formatter("lat"))
    ax.tick_params(top=True, right=True, labeltop=True, labelright=False, direction="in", length=4)
    ax.grid(color="#d0d0d0", linewidth=0.7, alpha=0.65, zorder=0)

    cbar = fig.colorbar(sc, ax=ax, pad=0.018, fraction=0.046)
    cbar.set_label("Mean MAE (lower is better)")

    legend_counts = [20, 30, 40]
    handles = [
        ax.scatter(
            [],
            [],
            s=45 + c * 4.0,
            facecolor="#eeeeee",
            edgecolor="black",
            linewidth=0.75,
            label=f"{c} runs",
        )
        for c in legend_counts
    ]
    leg = ax.legend(
        handles=handles,
        title="Evaluated as\nholdout target",
        loc="lower left",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    leg.get_frame().set_edgecolor("#999999")
    leg.get_frame().set_facecolor("white")

    ax.text(
        0.02,
        0.975,
        "80 random station-removal experiments",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
    )

    fig.tight_layout()
    save_figure(fig, "站点预测精度空间分布")


def main() -> None:
    metrics = load_80_run_metrics()
    agg = aggregate_station_metrics(metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "station_accuracy_80runs_long.csv", index=False)
    agg.sort_values("mae_mean", ascending=False).to_csv(
        OUT_DIR / "station_accuracy_80runs_summary.csv", index=False
    )
    plot_accuracy_map(agg)


if __name__ == "__main__":
    main()
