#!/usr/bin/env python3
"""Reproduce the temporal coverage heatmap for NAQU stations.

The reference image uses missing values as white gaps and a continuous blue-green
field for observed records. The default mode reproduces that appearance from the
raw 30-minute soil-moisture values. Use ``--mode coverage`` for a literal daily
availability-ratio heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


TIME_COLS = ["yyyy", "mm", "dd", "HH", "MM", "SS"]

# Top-to-bottom order observed in the selected-35 reference figure.
SELECTED_35_ORDER = [
    "MS3603",
    "MS3627",
    "MS3533",
    "MS3518",
    "MS3523",
    "MS3494",
    "BC08",
    "BC03",
    "BC04",
    "MS3513",
    "MS3527",
    "MS3614",
    "CD01",
    "MSBJ",
    "MS3482",
    "BC05",
    "MS3620",
    "MSNQRW",
    "C1",
    "BC02",
    "MS3576",
    "CD07",
    "BC07",
    "MS3488",
    "BC06",
    "MS3501",
    "P10",
    "P1",
    "P11",
    "F4",
    "F5",
    "P2",
    "F3",
    "CD03",
    "C2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 0-1 temporal data-availability heatmap for NAQU stations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv"),
        help="Input SM_NQ 30-minute CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("image/temporal_coverage_heatmap_all_stations"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--split-date",
        default="2018-01-01",
        help="Date used for the illustrative train/test split marker.",
    )
    parser.add_argument(
        "--mode",
        choices=["reference", "coverage"],
        default="reference",
        help=(
            "'reference' matches the provided figure appearance from raw values; "
            "'coverage' plots daily valid-record ratios."
        ),
    )
    parser.add_argument(
        "--station-scope",
        choices=["all", "selected35"],
        default="all",
        help="'all' uses every station column in the CSV; 'selected35' reproduces the old 35-station figure.",
    )
    parser.add_argument(
        "--missing-summary",
        type=Path,
        default=None,
        help="Optional output CSV for station missing-rate summary. Defaults to <output-prefix>_missing_rate.csv.",
    )
    return parser.parse_args()


def station_columns(csv_path: Path, station_scope: str) -> list[str]:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing_time = [col for col in TIME_COLS if col not in columns]
    if missing_time:
        raise ValueError(f"Input CSV is missing required time columns: {missing_time}")

    all_stations = [col for col in columns if col not in TIME_COLS]
    if station_scope == "selected35":
        missing = [col for col in SELECTED_35_ORDER if col not in all_stations]
        if missing:
            raise ValueError(f"Input CSV is missing selected station columns: {missing}")
        return SELECTED_35_ORDER
    return all_stations


def load_source(csv_path: Path, station_scope: str) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    stations = station_columns(csv_path, station_scope)
    frame = pd.read_csv(csv_path, usecols=TIME_COLS + stations)

    time = pd.to_datetime(
        {
            "year": frame["yyyy"],
            "month": frame["mm"],
            "day": frame["dd"],
            "hour": frame["HH"],
            "minute": frame["MM"],
            "second": frame["SS"],
        },
        errors="raise",
    )

    values = frame[stations].replace(-99, np.nan)
    values.index = time
    return time, values


def missing_summary(values: pd.DataFrame) -> pd.DataFrame:
    missing = values.isna()
    total_count = len(values)
    summary = pd.DataFrame(
        {
            "station": values.columns,
            "total_count": total_count,
            "missing_count": missing.sum(axis=0).astype(int).to_numpy(),
        }
    )
    summary["non_missing_count"] = summary["total_count"] - summary["missing_count"]
    summary["missing_rate_percent"] = summary["missing_count"] / summary["total_count"] * 100
    summary["non_missing_rate_percent"] = (
        summary["non_missing_count"] / summary["total_count"] * 100
    )
    summary = summary.sort_values(["missing_rate_percent", "station"], ascending=[True, True])
    summary.insert(0, "plot_order", range(1, len(summary) + 1))
    return summary[
        [
            "plot_order",
            "station",
            "total_count",
            "missing_count",
            "missing_rate_percent",
            "non_missing_count",
            "non_missing_rate_percent",
        ]
    ].reset_index(drop=True)


def build_daily_coverage(values: pd.DataFrame) -> pd.DataFrame:
    valid = values.notna()
    coverage = valid.astype("float32")

    daily = coverage.resample("D").mean()
    full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_days).fillna(0.0)
    return daily


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def reference_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "reference_coverage",
        [
            (0.00, "#ffffff"),
            (0.25, "#c9e8d1"),
            (0.50, "#63c0c5"),
            (0.75, "#2b82bd"),
            (1.00, "#0c3e7b"),
        ],
    )


def build_plot_matrix(
    values: pd.DataFrame, station_order: list[str], mode: str
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if mode == "coverage":
        daily = build_daily_coverage(values)
        return daily[station_order].to_numpy(dtype=float).T, daily.index

    raw = values[station_order].to_numpy(dtype=float).T
    # The reference figure's non-missing field behaves like a reversed
    # soil-moisture color scale while retaining the availability-style legend.
    matrix = 1.0 - np.clip(raw / 0.8, 0.0, 1.0)
    return np.ma.masked_invalid(matrix), values.index


def plot_heatmap(
    values: pd.DataFrame, station_order: list[str], output_prefix: Path, split_date: str, mode: str
) -> None:
    configure_matplotlib()

    matrix, time_index = build_plot_matrix(values, station_order, mode)
    x0 = mdates.date2num(time_index[0])
    x1 = mdates.date2num(time_index[-1])
    if mode == "coverage":
        x1 = mdates.date2num(time_index[-1] + pd.Timedelta(days=1))
    n_stations = len(station_order)

    cmap = reference_colormap()
    cmap.set_bad("white")

    fig_height = 8.16 if n_stations <= 35 else max(10.8, 0.19 * n_stations)
    fig, ax = plt.subplots(figsize=(13.12, fig_height), dpi=100)
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[x0, x1, n_stations - 0.5, -0.5],
    )

    for y in np.arange(0.5, n_stations - 0.5, 1.0):
        ax.axhline(y, color="white", lw=0.35, alpha=0.55)

    split_dt = pd.Timestamp(split_date)
    ax.axvline(mdates.date2num(split_dt), color="black", linestyle="--", lw=1.8)
    ax.text(
        mdates.date2num(split_dt - pd.DateOffset(years=1)),
        1.4,
        "Train/Test split\n(illustrative)",
        ha="center",
        va="center",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.75", alpha=0.88),
    )

    ax.set_yticks(np.arange(n_stations))
    ax.set_yticklabels(station_order, fontsize=11 if n_stations <= 35 else 8)

    tick_dates = [
        time_index[0],
        pd.Timestamp("2012-01-01"),
        pd.Timestamp("2014-01-01"),
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2020-01-01"),
        time_index[-1],
    ]
    ax.set_xticks([mdates.date2num(d) for d in tick_dates])
    ax.set_xticklabels(["2010", "2012", "2014", "2016", "2018", "2020", "2021"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(n_stations - 0.5, -0.5)

    ax.set_title(
        f"Figure 1: Data Temporal Coverage of {n_stations} NAQU Stations (2010-2021)",
        fontsize=16,
        pad=26,
    )

    colorbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.047)
    colorbar.ax.set_title("Data availability", fontsize=12, pad=8)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    colorbar.set_label(
        "1 (available, dark blue)\n\n0 (missing, white)",
        rotation=90,
        labelpad=18,
        fontsize=12,
    )

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.png", dpi=100, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _, values = load_source(args.data, args.station_scope)
    summary = missing_summary(values)
    station_order = summary["station"].tolist()
    if args.station_scope == "selected35":
        station_order = SELECTED_35_ORDER
        summary["plot_order"] = summary["station"].map(
            {station: i for i, station in enumerate(station_order, start=1)}
        )
        summary = summary.sort_values("plot_order").reset_index(drop=True)

    plot_heatmap(values, station_order, args.output_prefix, args.split_date, args.mode)

    summary_out = args.missing_summary
    if summary_out is None:
        summary_out = args.output_prefix.with_name(f"{args.output_prefix.name}_missing_rate.csv")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False, float_format="%.6f")

    print(f"Wrote {args.output_prefix}.png")
    print(f"Wrote {args.output_prefix}.svg")
    print(f"Wrote {args.output_prefix}.pdf")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
