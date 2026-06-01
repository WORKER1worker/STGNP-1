#!/usr/bin/env python3
"""Run holdout-ratio experiments with multiple unknown stations.

For each ratio r, randomly select r * total_stations as holdout,
train on the remaining stations, then run holdout inference.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STATIONS = [
    "C2", "CD03", "F3", "P2", "F5", "F4", "P11", "P1", "P10", "MS3501",
    "BC06", "MS3488", "BC07", "CD07", "MS3576", "BC02", "C1", "MSNQRW", "MS3620", "BC05",
    "MS3482", "MSBJ", "CD01", "MS3614", "MS3527", "MS3513", "BC04", "BC03", "BC08", "MS3494",
    "MS3523", "MS3518", "MS3533", "MS3603", "MS3627",
]


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
    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            out[k] = float(m.group(1))
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
            df["holdout_ratio"],
            df[y_mean_col],
            yerr=df[y_std_col],
            marker="o",
            capsize=3,
        )
    else:
        plt.plot(df["holdout_ratio"], df[y_mean_col], marker="o")
    plt.xlabel("Holdout Ratio")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_station_importance(summary_df: pd.DataFrame, stations: List[str]) -> pd.DataFrame:
    rows = []
    total_runs = len(summary_df)
    for station in stations:
        removed_mask = summary_df["holdout_stations"].apply(lambda s: station in s)
        removed = summary_df[removed_mask]
        present = summary_df[~removed_mask]

        removed_count = len(removed)
        present_count = len(present)

        removed_mae = removed["overall_mae"].mean() if removed_count else float("nan")
        present_mae = present["overall_mae"].mean() if present_count else float("nan")
        removed_rmse = removed["overall_rmse"].mean() if removed_count else float("nan")
        present_rmse = present["overall_rmse"].mean() if present_count else float("nan")
        removed_r2 = removed["overall_r2"].mean() if removed_count else float("nan")
        present_r2 = present["overall_r2"].mean() if present_count else float("nan")

        rows.append({
            "station_id": station,
            "total_runs": total_runs,
            "removed_count": removed_count,
            "present_count": present_count,
            "removed_ratio": removed_count / total_runs if total_runs else float("nan"),
            "overall_mae_removed_mean": removed_mae,
            "overall_mae_present_mean": present_mae,
            "overall_rmse_removed_mean": removed_rmse,
            "overall_rmse_present_mean": present_rmse,
            "overall_r2_removed_mean": removed_r2,
            "overall_r2_present_mean": present_r2,
            "importance_mae": removed_mae - present_mae,
            "importance_rmse": removed_rmse - present_rmse,
            "importance_r2": present_r2 - removed_r2,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="importance_mae", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def plot_station_metric_map(
    metrics_df: pd.DataFrame,
    station_meta_path: str,
    metric_col: str,
    out_path: str,
    title: str,
    cmap: str = "viridis",
    annotate: bool = False,
) -> None:
    meta_df = pd.read_csv(station_meta_path)
    for col in ["station_id", "lon", "lat"]:
        if col not in meta_df.columns:
            raise ValueError(f"Missing column in station meta: {col}")
    if metric_col not in metrics_df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")

    metrics_df = metrics_df.copy()
    metrics_df["station_id"] = metrics_df["station_id"].astype(str)
    meta_df["station_id"] = meta_df["station_id"].astype(str)

    merged = metrics_df.merge(meta_df[["station_id", "lon", "lat"]], on="station_id", how="inner")
    if merged.empty:
        raise RuntimeError("No matching stations between metrics and metadata")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(
        merged["lon"],
        merged["lat"],
        c=merged[metric_col],
        cmap=cmap,
        s=60,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(metric_col)

    if annotate:
        for _, row in merged.iterrows():
            ax.text(row["lon"], row["lat"], row["station_id"], fontsize=7, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run holdout-ratio experiments")
    parser.add_argument("--workspace", type=str, default=".", help="Project root (STGNP)")
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6", help="Holdout ratios")
    parser.add_argument("--seed", type=int, default=20260507, help="Base random seed")
    parser.add_argument("--seeds", type=str, default="", help="Comma/newline separated seed list")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds when --seeds is empty")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs for train/test")
    parser.add_argument("--model", type=str, default="hierarchical", help="Model name")
    parser.add_argument("--pred-attr", type=str, default="SM", help="Prediction attribute")
    parser.add_argument("--config", type=str, default="SM_config1", help="Model config name")
    parser.add_argument("--n-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--n-epochs-decay", type=int, default=0, help="Decay epochs")
    parser.add_argument("--training-strategy", type=str, default="mts", choices=["tts", "mts", "pmts"],
                        help="Context/target split strategy for training")
    parser.add_argument("--pmts-update-freq", type=int, default=1,
                        help="PMTS update frequency in epochs")
    parser.add_argument("--window-hours", type=int, default=4,
                        help="Time window in hours (30-min data => t_len = hours * 2)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/SM_keySite",
                        help="Checkpoint root (will append dataset mode)")
    parser.add_argument("--experiment-name", type=str, default="ratio_holdout_multi", help="Output folder name")
    parser.add_argument("--stations", type=str, default=",".join(DEFAULT_STATIONS), help="Candidate stations")
    parser.add_argument("--station-meta-path", type=str, default="data/dataset/SM_NQ/Stations_information_NAQU.csv",
                        help="Station metadata path for spatial plots")
    parser.add_argument("--skip-train", action="store_true", help="Only build splits")
    parser.add_argument("--skip-test", action="store_true", help="Skip holdout inference")
    args = parser.parse_args()

    root = os.path.abspath(args.workspace)
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    if not ratios:
        raise ValueError("No valid ratios provided")

    stations = dedup_keep_order(re.split(r"[\n,]+", args.stations))
    if len(stations) < 3:
        raise ValueError("Not enough stations provided")

    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)
    t_len = int(args.window_hours) * 2
    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_root = os.path.join(root, "experiments", args.experiment_name, run_tag)
    ensure_dir(exp_root)

    pred_name = args.pred_attr.replace("_Concentration", "")
    summary_rows = []

    for ratio_idx, ratio in enumerate(ratios):
        ratio_tag = f"r{int(round(ratio * 100)):02d}"
        for seed in seeds:
            split_dir = os.path.join(exp_root, f"split_{ratio_tag}", f"seed{seed}")
            logs_dir = os.path.join(exp_root, "logs", ratio_tag, f"seed{seed}")
            ensure_dir(split_dir)
            ensure_dir(logs_dir)

            holdout_seed = seed + ratio_idx * 1000
            rng = random.Random(holdout_seed)
            holdout_cnt = max(1, int(round(len(stations) * ratio)))
            holdout_cnt = min(holdout_cnt, len(stations) - 1)
            holdout_stations = sorted(rng.sample(stations, holdout_cnt))
            selected = sorted([s for s in stations if s not in holdout_stations])
            if len(selected) == 0:
                raise RuntimeError("No training stations left after holdout selection")

            selected_file = os.path.join(split_dir, "selected_stations.txt")
            test_file = os.path.join(split_dir, "test_stations.txt")
            holdout_file = os.path.join(split_dir, "holdout_stations.txt")
            write_csv_line(selected_file, selected)
            write_csv_line(test_file, [])
            write_csv_line(holdout_file, holdout_stations)

            build_cmd = [
                sys.executable,
                "data/dataset/build_sm_nq_split.py",
                "--location-path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
                "--data-path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
                "--output-dir", split_dir,
                "--selected-stations-file", selected_file,
                "--test-stations-file", test_file,
                "--holdout-station-id", ",".join(holdout_stations),
            ]
            run_cmd(build_cmd, cwd=root, log_path=os.path.join(logs_dir, "build.log"))

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
                    "--t_len", str(t_len),
                    "--enable_val",
                    "--save_best",
                    "--n_epochs", str(args.n_epochs),
                    "--n_epochs_decay", str(args.n_epochs_decay),
                    "--training_strategy", args.training_strategy,
                    "--pmts_update_freq", str(args.pmts_update_freq),
                    "--checkpoints_dir", args.checkpoints_dir,
                    "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
                    "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
                    "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
                    "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
                    "--sm_holdout_station_id", ",".join(holdout_stations),
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
                    "--checkpoints_dir", args.checkpoints_dir,
                    "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
                    "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
                    "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
                    "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
                    "--sm_holdout_station_id", ",".join(holdout_stations),
                    "--sm_holdout_location_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
                    "--sm_holdout_data_path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
                    "--sm_eval_target_mode", "holdout",
                    "--station_meta_path", args.station_meta_path,
                ]
                run_cmd(test_cmd, cwd=root, log_path=os.path.join(logs_dir, "test_holdout.log"))

            metrics = parse_overall_metrics(os.path.join(analysis_dir, "analysis_report.txt")) if analysis_dir else {}
            summary_rows.append({
                "holdout_ratio": ratio,
                "ratio_tag": ratio_tag,
                "seed": seed,
                "holdout_seed": holdout_seed,
                "window_hours": int(args.window_hours),
                "t_len": t_len,
                "holdout_count": len(holdout_stations),
                "selected_count": len(selected),
                "holdout_stations": holdout_stations,
                "holdout_stations_str": ",".join(holdout_stations),
                "file_time": file_time,
                "split_dir": split_dir,
                "checkpoint_dir": checkpoint_dir,
                "analysis_dir": analysis_dir,
                "overall_mae": metrics.get("overall_mae"),
                "overall_rmse": metrics.get("overall_rmse"),
                "overall_r2": metrics.get("overall_r2"),
            })

            run_info = {
                "split_dir": split_dir,
                "logs_dir": logs_dir,
                "holdout_ratio": ratio,
                "ratio_tag": ratio_tag,
                "seed": seed,
                "holdout_seed": holdout_seed,
                "window_hours": int(args.window_hours),
                "t_len": t_len,
                "holdout_count": len(holdout_stations),
                "holdout_stations": holdout_stations,
                "selected_count": len(selected),
                "file_time": file_time,
                "checkpoint_dir": checkpoint_dir,
                "analysis_dir": analysis_dir,
                "training_strategy": args.training_strategy,
                "pmts_update_freq": int(args.pmts_update_freq),
            }
            with open(os.path.join(split_dir, "run_info.json"), "w", encoding="utf-8") as f:
                json.dump(run_info, f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["holdout_ratio", "seed"], ascending=True).reset_index(drop=True)
    summary_by_seed_csv = os.path.join(exp_root, "results_summary_by_seed.csv")
    summary_csv_df = summary_df.copy()
    summary_csv_df["holdout_stations"] = summary_csv_df["holdout_stations_str"]
    summary_csv_df = summary_csv_df.drop(columns=["holdout_stations_str"])
    summary_csv_df.to_csv(summary_by_seed_csv, index=False)

    metric_cols = ["overall_mae", "overall_rmse", "overall_r2"]
    mean_df = summary_df.groupby(["holdout_ratio", "window_hours", "t_len"], as_index=False)[metric_cols].mean()
    std_df = summary_df.groupby(["holdout_ratio", "window_hours", "t_len"], as_index=False)[metric_cols].std(ddof=0)
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in metric_cols})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in metric_cols})
    summary_mean_df = mean_df.merge(std_df, on=["holdout_ratio"], how="left")
    summary_mean_df = summary_mean_df.sort_values(by="holdout_ratio", ascending=True).reset_index(drop=True)

    summary_mean_csv = os.path.join(exp_root, "results_summary_mean.csv")
    summary_mean_df.to_csv(summary_mean_csv, index=False)
    summary_compat_csv = os.path.join(exp_root, "summary_holdout_ratio.csv")
    summary_mean_df.to_csv(summary_compat_csv, index=False)

    latest_dir = os.path.join(root, "experiments", args.experiment_name)
    ensure_dir(latest_dir)
    summary_mean_df.to_csv(os.path.join(latest_dir, "results_summary.csv"), index=False)
    summary_csv_df.to_csv(os.path.join(latest_dir, "results_summary_by_seed.csv"), index=False)

    plots_dir = os.path.join(exp_root, "plots")
    ensure_dir(plots_dir)
    plot_metric_curve(summary_mean_df, "overall_mae_mean", "overall_mae_std", "MAE", os.path.join(plots_dir, "ratio_vs_mae.png"))
    plot_metric_curve(summary_mean_df, "overall_rmse_mean", "overall_rmse_std", "RMSE", os.path.join(plots_dir, "ratio_vs_rmse.png"))
    plot_metric_curve(summary_mean_df, "overall_r2_mean", "overall_r2_std", "R2", os.path.join(plots_dir, "ratio_vs_r2.png"))

    importance_df = compute_station_importance(summary_df, stations)
    importance_csv = os.path.join(exp_root, "station_importance_summary.csv")
    importance_df.to_csv(importance_csv, index=False)

    importance_map = os.path.join(plots_dir, "station_importance_map_mae.png")
    plot_station_metric_map(
        importance_df,
        args.station_meta_path,
        "importance_mae",
        importance_map,
        "Station Importance (MAE)",
    )

    meta = {
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "pred_attr": args.pred_attr,
        "ratios": ratios,
        "seeds": seeds,
        "window_hours": int(args.window_hours),
        "t_len": t_len,
        "training_strategy": args.training_strategy,
        "pmts_update_freq": int(args.pmts_update_freq),
        "checkpoints_dir": args.checkpoints_dir,
        "summary_mean_csv": summary_mean_csv,
        "summary_by_seed_csv": summary_by_seed_csv,
        "importance_csv": importance_csv,
        "plots_dir": plots_dir,
    }
    with open(os.path.join(exp_root, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== Holdout-ratio experiment completed ===")
    print(f"Output root: {exp_root}")
    print(f"Summary (mean) CSV: {summary_mean_csv}")
    print(f"Summary (by seed) CSV: {summary_by_seed_csv}")
    print(f"Importance CSV: {importance_csv}")


if __name__ == "__main__":
    main()
