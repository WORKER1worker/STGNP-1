#!/usr/bin/env python3
"""Batch ratio experiment for fixed holdout station MS3603.

This script runs end-to-end experiments:
1) Build split artifacts for each station ratio
2) Train model from scratch
3) Run holdout inference for MS3603
4) Collect metrics and draw comparison plots

Default candidate stations are the 35 stations provided by the user.
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
from typing import Dict, List, Tuple

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


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        val = item.strip()
        if val and val not in seen:
            out.append(val)
            seen.add(val)
    return out


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


def parse_file_time(train_output: str) -> str:
    # From options dump in train.py: "file_time: 20260427T171842"
    m = re.search(r"^\s*file_time:\s*([0-9T]+)", train_output, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback to checkpoint name pattern in output.
    m = re.search(r"hierarchical_SM_([0-9T]{15})", train_output)
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


def sample_split(
    known_pool: List[str],
    ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    selected_cnt = max(2, int(round(len(known_pool) * ratio)))
    selected_cnt = min(selected_cnt, len(known_pool))
    selected = rng.sample(known_pool, selected_cnt)

    test_cnt = max(1, int(round(len(selected) * test_ratio)))
    test_cnt = min(test_cnt, len(selected) - 1)
    test_stations = rng.sample(selected, test_cnt)
    return sorted(selected), sorted(test_stations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ratio experiments for holdout station MS3603")
    parser.add_argument("--workspace", type=str, default=".", help="Project root (STGNP)")
    parser.add_argument("--holdout-station", type=str, default="MS3603", help="Fixed unknown station")
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.5,0.8", help="Comma-separated removal fractions (e.g. 0.1 means remove 10% of known stations)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio within selected stations")
    parser.add_argument("--seed", type=int, default=20260427, help="Base random seed")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs for train/test")
    parser.add_argument("--config", type=str, default="SM_config1", help="Model config name")
    parser.add_argument("--n-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--n-epochs-decay", type=int, default=0, help="Decay epochs")
    parser.add_argument("--experiment-name", type=str, default="ratio_ms3603", help="Output folder name under experiments")
    parser.add_argument(
        "--stations",
        type=str,
        default=",".join(DEFAULT_STATIONS),
        help="Comma/newline separated candidate stations",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.workspace)
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    if not ratios:
        raise ValueError("No valid ratios provided")

    stations = dedup_keep_order(re.split(r"[\n,]+", args.stations))
    if args.holdout_station not in stations:
        raise ValueError(f"Holdout station {args.holdout_station} is not in candidate stations")

    known_pool = [s for s in stations if s != args.holdout_station]
    if len(known_pool) < 4:
        raise ValueError("Known station pool is too small")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_root = os.path.join(root, "experiments", args.experiment_name, run_tag)
    ensure_dir(exp_root)

    summary_rows = []

    for idx, ratio in enumerate(ratios):
        ratio_tag = f"r{int(round(ratio * 100)):02d}"
        split_dir = os.path.join(exp_root, f"split_{ratio_tag}")
        logs_dir = os.path.join(exp_root, "logs", ratio_tag)
        ensure_dir(split_dir)
        ensure_dir(logs_dir)

        # Here `ratio` is treated as removal fraction: remove this fraction of known_pool randomly
        removed_cnt = max(1, int(round(len(known_pool) * ratio)))
        rng = random.Random(args.seed + idx)
        removed = sorted(rng.sample(known_pool, removed_cnt))
        selected = sorted([s for s in known_pool if s not in removed])
        # Ensure at least one test station
        test_cnt = max(1, int(round(len(selected) * args.test_ratio)))
        test_cnt = min(test_cnt, len(selected) - 1) if len(selected) > 1 else 0
        test_stations = sorted(rng.sample(selected, test_cnt)) if test_cnt > 0 else []

        selected_file = os.path.join(split_dir, "selected_stations.txt")
        test_file = os.path.join(split_dir, "test_stations.txt")
        holdout_file = os.path.join(split_dir, "holdout_stations.txt")
        write_csv_line(selected_file, selected)
        write_csv_line(test_file, test_stations)
        write_csv_line(holdout_file, [args.holdout_station])

        build_cmd = [
            sys.executable,
            "data/dataset/build_sm_nq_split.py",
            "--location-path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
            "--data-path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
            "--output-dir", split_dir,
            "--selected-stations-file", selected_file,
            "--test-stations-file", test_file,
            "--holdout-station-id", args.holdout_station,
        ]
        run_cmd(build_cmd, cwd=root, log_path=os.path.join(logs_dir, "build.log"))

        train_cmd = [
            sys.executable,
            "train.py",
            "--model", "hierarchical",
            "--dataset_mode", "SM",
            "--pred_attr", "SM",
            "--config", args.config,
            "--gpu_ids", args.gpu_ids,
            "--seed", str(args.seed + idx),
            "--enable_val",
            "--save_best",
            "--n_epochs", str(args.n_epochs),
            "--n_epochs_decay", str(args.n_epochs_decay),
            "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
            "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
            "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
            "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
            "--sm_holdout_station_id", args.holdout_station,
            "--sm_holdout_location_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
            "--sm_holdout_data_path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
        ]
        train_output = run_cmd(train_cmd, cwd=root, log_path=os.path.join(logs_dir, "train.log"))
        file_time = parse_file_time(train_output)

        test_cmd = [
            sys.executable,
            "test_and_analyze.py",
            "--model", "hierarchical",
            "--dataset_mode", "SM",
            "--pred_attr", "SM",
            "--config", args.config,
            "--phase", "test",
            "--gpu_ids", args.gpu_ids,
            "--file_time", file_time,
            "--epoch", "best",
            "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
            "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
            "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
            "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
            "--sm_holdout_station_id", args.holdout_station,
            "--sm_holdout_location_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
            "--sm_holdout_data_path", "data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
            "--sm_eval_target_mode", "holdout",
            "--station_meta_path", "data/dataset/SM_NQ/Stations_information_NAQU.csv",
        ]
        run_cmd(test_cmd, cwd=root, log_path=os.path.join(logs_dir, "test_holdout.log"))

        analysis_dir = os.path.join(root, "checkpoints", "SM", f"hierarchical_SM_{file_time}", "analysis")
        station_metrics_path = os.path.join(analysis_dir, "station_metrics.csv")
        if not os.path.isfile(station_metrics_path):
            raise FileNotFoundError(f"Missing station metrics: {station_metrics_path}")

        station_df = pd.read_csv(station_metrics_path)
        row = station_df[station_df["station_id"].astype(str) == args.holdout_station]
        if row.empty:
            raise RuntimeError(f"Holdout station {args.holdout_station} not found in station_metrics.csv")
        row = row.iloc[0]

        overall = parse_overall_metrics(os.path.join(analysis_dir, "analysis_report.txt"))
        summary_rows.append({
            "ratio": ratio,
            "ratio_tag": ratio_tag,
            "seed": args.seed + idx,
            "file_time": file_time,
            "selected_count": len(selected),
            "test_count": len(test_stations),
            "holdout_station": args.holdout_station,
            "holdout_sample_count": int(row.get("sample_count", 0)),
            "holdout_mae": float(row["MAE"]),
            "holdout_rmse": float(row["RMSE"]),
            "holdout_r2": float(row["R2"]),
            "overall_mae": overall["overall_mae"],
            "overall_rmse": overall["overall_rmse"],
            "overall_r2": overall["overall_r2"],
            "split_dir": split_dir,
            "analysis_dir": analysis_dir,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="ratio", ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(exp_root, "summary_ms3603.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Plot holdout metrics against ratio.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    xs = summary_df["ratio"].to_list()

    axes[0].plot(xs, summary_df["holdout_mae"], marker="o", linewidth=2)
    axes[0].set_title("MS3603 MAE vs Station Ratio")
    axes[0].set_xlabel("Selected Station Ratio")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, summary_df["holdout_rmse"], marker="o", linewidth=2)
    axes[1].set_title("MS3603 RMSE vs Station Ratio")
    axes[1].set_xlabel("Selected Station Ratio")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.3)

    axes[2].plot(xs, summary_df["holdout_r2"], marker="o", linewidth=2)
    axes[2].set_title("MS3603 R2 vs Station Ratio")
    axes[2].set_xlabel("Selected Station Ratio")
    axes[2].set_ylabel("R2")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Holdout Station Comparison: {args.holdout_station}", fontsize=13)
    fig.tight_layout()
    plot_png = os.path.join(exp_root, "comparison_ms3603.png")
    fig.savefig(plot_png, dpi=160)
    plt.close(fig)

    meta = {
        "created_at": datetime.now().isoformat(),
        "holdout_station": args.holdout_station,
        "ratios": ratios,
        "test_ratio": args.test_ratio,
        "seed_base": args.seed,
        "summary_csv": summary_csv,
        "comparison_plot": plot_png,
        "rows": len(summary_df),
    }
    with open(os.path.join(exp_root, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== Ratio experiment completed ===")
    print(f"Output root: {exp_root}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Comparison plot: {plot_png}")


if __name__ == "__main__":
    main()
