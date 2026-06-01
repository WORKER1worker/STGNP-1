#!/usr/bin/env python3
"""Time window ablation for SM strict holdout inference.

This script runs:
1) Build a fixed split with strict holdout stations.
2) Train for each time window (t_len = hours * 2 for 30-min data).
3) Run holdout inference + analysis.
4) Summarize metrics and draw comparison plots.
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STATIONS = [
    "C2", "CD03", "F3", "P2", "F5", "F4", "P11", "P1", "P10", "MS3501",
    "BC06", "MS3488", "BC07", "CD07", "MS3576", "BC02", "C1", "MSNQRW", "MS3620", "BC05",
    "MS3482", "MSBJ", "CD01", "MS3614", "MS3527", "MS3513", "BC04", "BC03", "BC08", "MS3494",
    "MS3523", "MS3518", "MS3533", "MS3603", "MS3627",
]

DEFAULT_HOLDOUT: List[str] = []
DEFAULT_WINDOWS = [1, 2, 3, 4, 5, 6, 8, 12]
DEFAULT_HOLDOUT_EXCLUDE = ["BC02", "CD01", "MS3482", "CD03"]


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        val = item.strip()
        if val and val not in seen:
            out.append(val)
            seen.add(val)
    return out


def parse_seed_list(seeds_arg: str, base_seed: int, num_seeds: int) -> List[int]:
    seeds_raw = seeds_arg.strip() if seeds_arg else ""
    if seeds_raw:
        values = [int(x.strip()) for x in re.split(r"[\n,]+", seeds_raw) if x.strip()]
        if not values:
            raise ValueError("--seeds provided but no valid values parsed")
        return values
    if num_seeds <= 1:
        return [int(base_seed)]
    return [int(base_seed) + i for i in range(int(num_seeds))]


def select_holdout_stations(
    stations: List[str],
    holdout_ratio: float,
    seed: int,
    exclude: List[str],
) -> List[str]:
    if not stations:
        raise ValueError("No stations available for holdout selection")
    exclude_set = {s.strip() for s in exclude if s.strip()}
    pool = [s for s in stations if s not in exclude_set]
    if len(pool) < 2:
        raise ValueError("Holdout pool too small after exclusions")
    count = int(round(len(stations) * float(holdout_ratio)))
    count = max(1, min(count, len(pool) - 1))
    rng = random.Random(int(seed))
    return sorted(rng.sample(pool, count))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv_line(path: str, values: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def run_cmd(cmd: List[str], cwd: str, log_path: str) -> str:
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        collected = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            collected.append(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")
        return "".join(collected)


def write_window_marker(checkpoint_dir: str, analysis_dir: str, info: Dict[str, object]) -> None:
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        marker_path = os.path.join(checkpoint_dir, "window_info.json")
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    if analysis_dir and os.path.isdir(analysis_dir):
        marker_path = os.path.join(analysis_dir, "window_info.json")
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def parse_file_time(train_output: str, model: str, pred_attr: str) -> str:
    m = re.search(r"^\s*file_time:\s*([0-9T]+)", train_output, re.MULTILINE)
    if m:
        return m.group(1).strip()
    pred_name = pred_attr.replace("_Concentration", "")
    pattern = rf"{re.escape(model)}_{re.escape(pred_name)}_([0-9T]{{15}})"
    m = re.search(pattern, train_output)
    if m:
        return m.group(1).strip()
    raise RuntimeError("Unable to parse file_time from train output")


def parse_overall_metrics(report_path: str) -> Dict[str, float]:
    out = {"overall_mae": float("nan"), "overall_rmse": float("nan"), "overall_r2": float("nan")}
    if not os.path.isfile(report_path):
        return out
    text = open(report_path, "r", encoding="utf-8").read()
    patterns = {
        "overall_mae": r"总体 MAE:\s*([0-9.]+)",
        "overall_rmse": r"总体 RMSE:\s*([0-9.]+)",
        "overall_r2": r"总体 R2:\s*([0-9.\-]+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            out[key] = float(match.group(1))
    return out


def compute_stability_and_extremes(
    csv_path: str,
    holdout_stations: List[str],
    extreme_quantile: float = 0.95,
) -> Dict[str, float]:
    out = {
        "stability_std": float("nan"),
        "extreme_mae": float("nan"),
        "extreme_rmse": float("nan"),
    }
    if not os.path.isfile(csv_path):
        return out

    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "station_id", "true_value", "pred_value"],
    )
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(holdout_stations)]
    if df.empty:
        return out

    agg = (
        df.groupby(["station_id", "timestamp"], as_index=False)[["true_value", "pred_value"]]
        .mean()
        .sort_values(["station_id", "timestamp"])
    )
    err = agg["pred_value"] - agg["true_value"]
    out["stability_std"] = float(err.std(ddof=0))

    agg["delta_true"] = agg.groupby("station_id")["true_value"].diff().abs()
    deltas = agg["delta_true"].dropna()
    if deltas.empty:
        return out

    threshold = float(deltas.quantile(extreme_quantile))
    extreme_df = agg[agg["delta_true"] >= threshold]
    if extreme_df.empty:
        return out

    extreme_err = extreme_df["pred_value"] - extreme_df["true_value"]
    out["extreme_mae"] = float(extreme_err.abs().mean())
    out["extreme_rmse"] = float(np.sqrt(np.mean(np.square(extreme_err))))
    return out


def sample_abs_errors(
    csv_path: str,
    holdout_stations: List[str],
    max_samples: int = 20000,
) -> List[float]:
    if not os.path.isfile(csv_path):
        return []
    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "station_id", "true_value", "pred_value"],
    )
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(holdout_stations)]
    if df.empty:
        return []

    agg = (
        df.groupby(["station_id", "timestamp"], as_index=False)[["true_value", "pred_value"]]
        .mean()
        .sort_values(["station_id", "timestamp"])
    )
    abs_err = (agg["pred_value"] - agg["true_value"]).abs().to_numpy()
    if abs_err.size <= max_samples:
        return abs_err.tolist()

    rng = np.random.default_rng(20260509)
    idx = rng.choice(abs_err.size, size=max_samples, replace=False)
    return abs_err[idx].tolist()


def collect_station_series(
    csv_path: str,
    station_ids: List[str],
    max_points: int = 500,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not os.path.isfile(csv_path):
        return out

    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "station_id", "true_value", "pred_value"],
    )
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(station_ids)]
    if df.empty:
        return out

    agg = df.groupby(["station_id", "timestamp"], as_index=False)[["true_value", "pred_value"]].mean()
    for sid in station_ids:
        sub = agg[agg["station_id"] == sid].sort_values("timestamp")
        if sub.empty:
            continue
        if len(sub) > max_points:
            stride = int(np.ceil(len(sub) / max_points))
            sub = sub.iloc[::stride]
        out[sid] = sub
    return out


def plot_metric_curve(
    df: pd.DataFrame,
    y_mean_col: str,
    y_std_col: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    if y_std_col and y_std_col in df.columns:
        plt.errorbar(
            df["window"],
            df[y_mean_col],
            yerr=df[y_std_col],
            marker="o",
            capsize=3,
        )
    else:
        plt.plot(df["window"], df[y_mean_col], marker="o")
    plt.xlabel("Window (hours)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_distribution(
    windows: List[int],
    abs_errors: Dict[int, List[float]],
    out_path: str,
) -> None:
    data = [abs_errors.get(w, []) for w in windows]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=[str(w) for w in windows], showfliers=False)
    plt.xlabel("Window (hours)")
    plt.ylabel("Absolute Error")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trend_comparison(
    station_id: str,
    series_by_window: Dict[int, pd.DataFrame],
    out_path: str,
) -> None:
    if not series_by_window:
        return

    windows_sorted = sorted(series_by_window.keys())
    base_df = series_by_window[windows_sorted[0]]
    times = pd.to_datetime(base_df["timestamp"], unit="s")

    plt.figure(figsize=(10, 4))
    plt.plot(times, base_df["true_value"], color="black", linewidth=1.5, label="Truth")

    cmap = plt.get_cmap("tab10")
    for idx, window in enumerate(windows_sorted):
        df = series_by_window[window]
        t = pd.to_datetime(df["timestamp"], unit="s")
        color = cmap(idx % 10)
        plt.plot(t, df["pred_value"], color=color, alpha=0.7, label=f"Pred {window}h")

    plt.title(f"Trend Comparison - {station_id}")
    plt.xlabel("Time")
    plt.ylabel("SM")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time window ablation for SM holdout")
    parser.add_argument("--workspace", type=str, default=".", help="Project root (STGNP)")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs for train/test")
    parser.add_argument("--seed", type=int, default=3, help="Base random seed")
    parser.add_argument("--seeds", type=str, default="", help="Comma/newline separated seed list")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of seeds when --seeds is empty")
    parser.add_argument("--model", type=str, default="hierarchical", help="Model name")
    parser.add_argument("--pred-attr", type=str, default="SM", help="Prediction attribute")
    parser.add_argument("--config", type=str, default="SM_config1", help="Model config name")
    parser.add_argument("--n-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--n-epochs-decay", type=int, default=0, help="Decay epochs")
    parser.add_argument("--training-strategy", type=str, default="pmts", choices=["tts", "mts", "pmts"],
                        help="context/target split strategy")
    parser.add_argument("--pmts-update-freq", type=int, default=1,
                        help="PMTS update frequency in epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--eval-stride", type=int, default=1, help="Eval stride (1 = sliding)")
    parser.add_argument("--windows", type=str, default=",".join(str(w) for w in DEFAULT_WINDOWS),
                        help="Comma-separated window hours")
    parser.add_argument("--holdout-stations", type=str, default=",".join(DEFAULT_HOLDOUT),
                        help="Optional holdout stations override (comma/newline)")
    parser.add_argument("--holdout-ratio", type=float, default=(1 / 7),
                        help="Holdout ratio when holdout list is empty (default=1/7 for 6:1)")
    parser.add_argument("--holdout-seed", type=int, default=-1,
                        help="Seed for holdout station sampling (defaults to --seed)")
    parser.add_argument("--holdout-exclude", type=str, default=",".join(DEFAULT_HOLDOUT_EXCLUDE),
                        help="Stations excluded from holdout sampling (comma/newline)")
    parser.add_argument("--stations", type=str, default=",".join(DEFAULT_STATIONS),
                        help="Comma/newline separated candidate stations")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/SM_timeWindow_PMTS",
                        help="Checkpoint root (will append dataset mode)")
    parser.add_argument("--experiment-name", type=str, default="time_window_ablation",
                        help="Output folder name under experiments")
    parser.add_argument("--skip-build", action="store_true", help="Skip building split artifacts")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-test", action="store_true", help="Skip holdout inference")
    args = parser.parse_args()

    root = os.path.abspath(args.workspace)
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    if not windows:
        raise ValueError("No valid windows provided")

    stations = dedup_keep_order(re.split(r"[\n,]+", args.stations))
    holdout_seed = args.seed if args.holdout_seed < 0 else args.holdout_seed
    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)

    holdout_raw = dedup_keep_order(re.split(r"[\n,]+", args.holdout_stations))
    holdout_exclude = dedup_keep_order(re.split(r"[\n,]+", args.holdout_exclude))
    if holdout_raw:
        holdout = holdout_raw
    else:
        holdout = select_holdout_stations(stations, args.holdout_ratio, holdout_seed, holdout_exclude)

    for sid in holdout:
        if sid not in stations:
            raise ValueError(f"Holdout station {sid} is not in candidate stations")

    selected = [s for s in stations if s not in set(holdout)]
    if len(selected) == 0:
        raise ValueError("No training stations left after holdout exclusion")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_root = os.path.join(root, "experiments", args.experiment_name, run_tag)
    ensure_dir(exp_root)

    split_dir = os.path.join(exp_root, "split_fixed")
    logs_root = os.path.join(exp_root, "logs")
    ensure_dir(split_dir)
    ensure_dir(logs_root)

    selected_file = os.path.join(split_dir, "selected_stations.txt")
    test_file = os.path.join(split_dir, "test_stations.txt")
    holdout_file = os.path.join(split_dir, "holdout_stations.txt")
    write_csv_line(selected_file, selected)
    write_csv_line(test_file, [])
    write_csv_line(holdout_file, holdout)

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            "data/dataset/build_sm_nq_split.py",
            "--location-path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
            "--data-path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
            "--output-dir", split_dir,
            "--selected-stations-file", selected_file,
            "--test-stations-file", test_file,
            "--holdout-station-id", ",".join(holdout),
        ]
        run_cmd(build_cmd, cwd=root, log_path=os.path.join(logs_root, "build.log"))

    summary_rows: List[Dict[str, object]] = []
    abs_error_samples: Dict[int, List[float]] = {}
    trend_series_by_window: Dict[int, Dict[str, pd.DataFrame]] = {}
    first_seed = seeds[0] if seeds else args.seed

    pred_name = args.pred_attr.replace("_Concentration", "")

    for window in windows:
        t_len = int(window) * 2
        window_tag = f"window{window}h"
        for seed in seeds:
            logs_dir = os.path.join(logs_root, window_tag, f"seed{seed}")
            ensure_dir(logs_dir)

            file_time = ""
            if not args.skip_train:
                train_cmd = [
                    sys.executable,
                    "train.py",
                    "--model", args.model,
                    "--dataset_mode", "SM",
                    "--pred_attr", args.pred_attr,
                    "--config", args.config,
                    "--gpu_ids", args.gpu_ids,
                    "--seed", str(seed),
                    "--batch_size", str(args.batch_size),
                    "--t_len", str(t_len),
                    "--eval_stride", str(args.eval_stride),
                    "--training_strategy", str(args.training_strategy),
                    "--pmts_update_freq", str(args.pmts_update_freq),
                    "--enable_val",
                    "--save_best",
                    "--n_epochs", str(args.n_epochs),
                    "--n_epochs_decay", str(args.n_epochs_decay),
                    "--checkpoints_dir", args.checkpoints_dir,
                    "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
                    "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
                    "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
                    "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
                    "--sm_holdout_station_id", ",".join(holdout),
                    "--sm_holdout_location_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
                    "--sm_holdout_data_path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
                    "--sm_eval_target_mode", "holdout",
                ]
                train_output = run_cmd(train_cmd, cwd=root, log_path=os.path.join(logs_dir, "train.log"))
                file_time = parse_file_time(train_output, args.model, args.pred_attr)

            checkpoint_dir = ""
            analysis_dir = ""
            if file_time:
                checkpoint_dir = os.path.join(
                    root, args.checkpoints_dir, "SM", f"{args.model}_{pred_name}_{file_time}"
                )
                analysis_dir = os.path.join(checkpoint_dir, "analysis")

            if file_time:
                marker_info = {
                    "window": window,
                    "t_len": t_len,
                    "seed": seed,
                    "file_time": file_time,
                    "eval_stride": args.eval_stride,
                    "training_strategy": args.training_strategy,
                    "pmts_update_freq": args.pmts_update_freq,
                    "holdout_ratio": args.holdout_ratio,
                    "holdout_seed": holdout_seed,
                    "holdout_exclude": holdout_exclude,
                    "holdout_stations": holdout,
                }
                write_window_marker(checkpoint_dir, analysis_dir, marker_info)

            if (not args.skip_test) and file_time:
                test_cmd = [
                    sys.executable,
                    "test_and_analyze.py",
                    "--model", args.model,
                    "--dataset_mode", "SM",
                    "--pred_attr", args.pred_attr,
                    "--config", args.config,
                    "--phase", "test",
                    "--gpu_ids", args.gpu_ids,
                    "--file_time", file_time,
                    "--epoch", "best",
                    "--t_len", str(t_len),
                    "--eval_stride", str(args.eval_stride),
                    "--checkpoints_dir", args.checkpoints_dir,
                    "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
                    "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
                    "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
                    "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
                    "--sm_holdout_station_id", ",".join(holdout),
                    "--sm_holdout_location_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
                    "--sm_holdout_data_path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
                    "--sm_eval_target_mode", "holdout",
                    "--station_meta_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
                ]
                run_cmd(test_cmd, cwd=root, log_path=os.path.join(logs_dir, "test_holdout.log"))

            report_path = os.path.join(analysis_dir, "analysis_report.txt") if analysis_dir else ""
            metrics = parse_overall_metrics(report_path) if report_path else {}

            csv_path = os.path.join(analysis_dir, "predictions_structured.csv") if analysis_dir else ""
            stability_ext = compute_stability_and_extremes(csv_path, holdout)

            abs_error_samples.setdefault(window, []).extend(sample_abs_errors(csv_path, holdout))
            if seed == first_seed:
                trend_series_by_window[window] = collect_station_series(csv_path, holdout)

            summary_rows.append({
                "window": window,
                "t_len": t_len,
                "seed": seed,
                "overall_mae": metrics.get("overall_mae"),
                "overall_rmse": metrics.get("overall_rmse"),
                "overall_r2": metrics.get("overall_r2"),
                "stability_std": stability_ext.get("stability_std"),
                "extreme_mae": stability_ext.get("extreme_mae"),
                "extreme_rmse": stability_ext.get("extreme_rmse"),
                "file_time": file_time,
                "checkpoint_dir": checkpoint_dir,
                "analysis_dir": analysis_dir,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["window", "seed"], ascending=True).reset_index(drop=True)
    summary_by_seed_csv = os.path.join(exp_root, "results_summary_by_seed.csv")
    summary_df.to_csv(summary_by_seed_csv, index=False)

    metric_cols = [
        "overall_mae",
        "overall_rmse",
        "overall_r2",
        "stability_std",
        "extreme_mae",
        "extreme_rmse",
    ]
    mean_df = summary_df.groupby(["window", "t_len"], as_index=False)[metric_cols].mean()
    std_df = summary_df.groupby(["window", "t_len"], as_index=False)[metric_cols].std(ddof=0)
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in metric_cols})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in metric_cols})
    summary_mean_df = mean_df.merge(std_df, on=["window", "t_len"], how="left")

    summary_csv = os.path.join(exp_root, "results_summary_mean.csv")
    summary_mean_df.to_csv(summary_csv, index=False)

    latest_dir = os.path.join(root, "experiments", args.experiment_name)
    ensure_dir(latest_dir)
    summary_mean_df.to_csv(os.path.join(latest_dir, "results_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(latest_dir, "results_summary_by_seed.csv"), index=False)

    plots_dir = os.path.join(exp_root, "plots")
    ensure_dir(plots_dir)

    plot_metric_curve(summary_mean_df, "overall_mae_mean", "overall_mae_std", "MAE", os.path.join(plots_dir, "window_vs_mae.png"))
    plot_metric_curve(summary_mean_df, "overall_rmse_mean", "overall_rmse_std", "RMSE", os.path.join(plots_dir, "window_vs_rmse.png"))
    plot_metric_curve(summary_mean_df, "overall_r2_mean", "overall_r2_std", "R2", os.path.join(plots_dir, "window_vs_r2.png"))
    plot_metric_curve(summary_mean_df, "stability_std_mean", "stability_std_std", "Error Std", os.path.join(plots_dir, "window_vs_stability.png"))
    plot_metric_curve(summary_mean_df, "extreme_mae_mean", "extreme_mae_std", "Extreme MAE", os.path.join(plots_dir, "window_vs_extreme_mae.png"))
    plot_metric_curve(summary_mean_df, "extreme_rmse_mean", "extreme_rmse_std", "Extreme RMSE", os.path.join(plots_dir, "window_vs_extreme_rmse.png"))

    plot_error_distribution(windows, abs_error_samples, os.path.join(plots_dir, "error_distribution_boxplot.png"))

    trend_dir = os.path.join(plots_dir, "trend_compare")
    ensure_dir(trend_dir)
    for sid in holdout:
        series_by_window = {}
        for window in windows:
            series = trend_series_by_window.get(window, {})
            if sid in series:
                series_by_window[window] = series[sid]
        plot_trend_comparison(sid, series_by_window, os.path.join(trend_dir, f"trend_compare_{sid}.png"))

    run_info = {
        "exp_root": exp_root,
        "split_dir": split_dir,
        "holdout_stations": holdout,
        "windows": windows,
        "t_len_per_window": {str(w): int(w) * 2 for w in windows},
        "summary_mean_csv": summary_csv,
        "summary_by_seed_csv": summary_by_seed_csv,
        "seeds": seeds,
        "training_strategy": args.training_strategy,
        "pmts_update_freq": args.pmts_update_freq,
        "holdout_ratio": args.holdout_ratio,
        "holdout_seed": holdout_seed,
        "holdout_exclude": holdout_exclude,
        "plots_dir": plots_dir,
    }
    with open(os.path.join(exp_root, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Time window ablation finished")
    print(f"Summary (mean) CSV: {summary_csv}")
    print(f"Summary (by seed) CSV: {summary_by_seed_csv}")
    print(f"Plots dir : {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
