#!/usr/bin/env python3
"""Run sensor ratio + training strategy experiments for SM strict holdout."""

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


DEFAULT_RATIO_STR = "1:6,2:5,3:4,4:3,5:2,6:1"


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


def load_station_list(path: str) -> List[str]:
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find station list file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    values = [s.strip() for s in re.split(r"[\n,]+", raw) if s.strip()]
    return dedup_keep_order(values)


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


def load_completed_runs(checkpoints_dir: str) -> Dict[tuple, str]:
    completed = {}
    if not os.path.isdir(checkpoints_dir):
        return completed
    for root, _, files in os.walk(checkpoints_dir):
        if "analysis_report.txt" not in files:
            continue
        analysis_dir = root
        marker_path = os.path.join(analysis_dir, "strategy_ratio_info.json")
        if not os.path.isfile(marker_path):
            marker_path = os.path.join(os.path.dirname(analysis_dir), "strategy_ratio_info.json")
        if not os.path.isfile(marker_path):
            continue
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except Exception:
            continue
        strategy = str(marker.get("strategy", "")).lower().strip()
        observed = marker.get("observed_count")
        seed = marker.get("seed")
        if strategy and observed is not None and seed is not None:
            completed[(strategy, int(observed), int(seed))] = analysis_dir
    return completed


def resolve_exp_root(root: str, experiment_name: str, resume: bool, resume_run_dir: str) -> str:
    base_dir = os.path.join(root, "experiments", experiment_name)
    if resume_run_dir:
        exp_root = resume_run_dir
        if not os.path.isabs(exp_root):
            exp_root = os.path.join(root, exp_root)
        ensure_dir(exp_root)
        return exp_root

    if resume and os.path.isdir(base_dir):
        candidates = [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_root = os.path.join(base_dir, run_tag)
    ensure_dir(exp_root)
    return exp_root


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


def parse_ratio_list(raw: str, total: int) -> List[Dict[str, object]]:
    items = []
    seen = set()
    for token in re.split(r"[\n,]+", raw or ""):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [p.strip() for p in token.split(":")]
            if len(parts) != 2:
                raise ValueError(f"Invalid ratio token: {token}")
            obs = int(parts[0])
            unobs = int(parts[1])
            if obs <= 0 or unobs <= 0:
                raise ValueError(f"Ratio must be positive: {token}")
            observed = int(round(total * (obs / float(obs + unobs))))
            observed = max(1, min(observed, total - 1))
            holdout = total - observed
            label = f"{obs}:{unobs}"
        else:
            ratio = float(token)
            holdout = int(round(total * ratio))
            holdout = max(1, min(holdout, total - 1))
            observed = total - holdout
            label = f"{observed}:{holdout}"

        key = (observed, holdout)
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "ratio_label": label,
            "observed_count": observed,
            "holdout_count": holdout,
            "ratio_tag": f"obs{observed}_holdout{holdout}",
        })
    if not items:
        raise ValueError("No valid ratios provided")
    return items


def write_marker(checkpoint_dir: str, analysis_dir: str, info: Dict[str, object]) -> None:
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        marker_path = os.path.join(checkpoint_dir, "strategy_ratio_info.json")
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    if analysis_dir and os.path.isdir(analysis_dir):
        marker_path = os.path.join(analysis_dir, "strategy_ratio_info.json")
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def plot_strategy_curve(
    summary_mean_df: pd.DataFrame,
    ratio_labels: List[str],
    strategies: List[str],
    out_path: str,
) -> None:
    plt.figure(figsize=(7.5, 4))
    x = np.arange(len(ratio_labels))
    for strategy in strategies:
        sub = summary_mean_df[summary_mean_df["strategy"] == strategy]
        sub = sub.set_index("ratio_label")
        y = []
        yerr = []
        for label in ratio_labels:
            if label in sub.index:
                row = sub.loc[label]
                y.append(row.get("overall_mae_mean", float("nan")))
                yerr.append(row.get("overall_mae_std", float("nan")))
            else:
                y.append(float("nan"))
                yerr.append(float("nan"))
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=strategy.upper())

    plt.xticks(x, ratio_labels)
    plt.xlabel("Observed:Unmonitored Ratio")
    plt.ylabel("MAE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy + ratio experiments for SM holdout")
    parser.add_argument("--workspace", type=str, default=".", help="Project root (STGNP)")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs for train/test")
    parser.add_argument("--seed", type=int, default=20260515, help="Base random seed")
    parser.add_argument("--seeds", type=str, default="", help="Comma/newline separated seed list")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds when --seeds is empty")
    parser.add_argument("--model", type=str, default="hierarchical", help="Model name")
    parser.add_argument("--pred-attr", type=str, default="SM", help="Prediction attribute")
    parser.add_argument("--config", type=str, default="SM_config1", help="Model config name")
    parser.add_argument("--n-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--n-epochs-decay", type=int, default=0, help="Decay epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--window-hours", type=int, default=4, help="Time window in hours (30-min data => t_len = hours * 2)")
    parser.add_argument("--eval-stride", type=int, default=1, help="Eval stride (1 = sliding)")
    parser.add_argument("--ratios", type=str, default=DEFAULT_RATIO_STR, help="Observed:unmonitored ratios")
    parser.add_argument("--strategies", type=str, default="orig,tts,mts,pmts", help="Comma-separated strategies")
    parser.add_argument("--num-train-target", type=int, default=4, help="num_train_target for training")
    parser.add_argument("--orig-num-train-target", type=int, default=3,
                        help="num_train_target for original training mode")
    parser.add_argument("--pmts-update-freq", type=int, default=1, help="PMTS update frequency in epochs")

    parser.add_argument("--station-file", type=str, default="dataset/SM_NQ_selected35/selected_stations.txt",
                        help="Candidate station list file")
    parser.add_argument("--location-path", type=str,
                        default="dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv",
                        help="Station metadata path for split building")
    parser.add_argument("--data-path", type=str,
                        default="dataset/SM_NQ_selected35/SM_NQ-30-minutes_05cm_selected35.csv",
                        help="Soil moisture data path for split building")
    parser.add_argument("--holdout-location-path", type=str,
                        default="dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv",
                        help="Full location path for strict holdout inference")
    parser.add_argument("--holdout-data-path", type=str,
                        default="dataset/SM_NQ_selected35/SM_NQ-30-minutes_05cm_selected35.csv",
                        help="Full data path for strict holdout inference")
    parser.add_argument("--station-meta-path", type=str,
                        default="dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv",
                        help="Station metadata path for analysis plots")

    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/SM_trainingStrategy",
                        help="Checkpoint root (will append dataset mode)")
    parser.add_argument("--experiment-name", type=str, default="training_strategy_ratio",
                        help="Output folder name under experiments")
    parser.add_argument("--resume", action="store_true", help="Resume and skip completed runs")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume (force a new run dir)")
    parser.add_argument("--resume-run-dir", type=str, default="",
                        help="Explicit experiment run directory to resume")

    parser.add_argument("--skip-build", action="store_true", help="Skip building split artifacts")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-test", action="store_true", help="Skip holdout inference")
    args = parser.parse_args()

    root = os.path.abspath(args.workspace)
    station_file = args.station_file
    if not os.path.isabs(station_file):
        station_file = os.path.join(root, station_file)
    stations = load_station_list(station_file)
    if len(stations) < 3:
        raise ValueError("Not enough stations provided")

    ratios = parse_ratio_list(args.ratios, total=len(stations))
    strategies = [s.lower() for s in dedup_keep_order(re.split(r"[\n,]+", args.strategies))]
    if not strategies:
        raise ValueError("No strategies provided")

    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)
    t_len = int(args.window_hours) * 2

    resume_enabled = True
    if args.no_resume:
        resume_enabled = False
    elif args.resume:
        resume_enabled = True

    exp_root = resolve_exp_root(root, args.experiment_name, resume_enabled, args.resume_run_dir)

    completed = {}
    if resume_enabled:
        checkpoints_root = os.path.join(root, args.checkpoints_dir, "SM")
        completed = load_completed_runs(checkpoints_root)
        print(f"[Resume] Found {len(completed)} completed runs in {checkpoints_root}")

    pred_name = args.pred_attr.replace("_Concentration", "")
    summary_rows: List[Dict[str, object]] = []

    for ratio_idx, ratio_item in enumerate(ratios):
        ratio_label = ratio_item["ratio_label"]
        observed_count = int(ratio_item["observed_count"])
        holdout_count = int(ratio_item["holdout_count"])
        ratio_tag = str(ratio_item["ratio_tag"])

        for seed in seeds:
            if resume_enabled:
                all_done = True
                for strategy in strategies:
                    key = (strategy, observed_count, seed)
                    if key not in completed:
                        all_done = False
                        break
                if all_done:
                    print(f"[Resume] Skip ratio={ratio_label}, seed={seed} (all strategies done)")
                    continue

            split_dir = os.path.join(exp_root, f"split_{ratio_tag}", f"seed{seed}")
            logs_root = os.path.join(exp_root, "logs", ratio_tag, f"seed{seed}")
            ensure_dir(split_dir)
            ensure_dir(logs_root)

            holdout_seed = seed + ratio_idx * 1000
            rng = random.Random(holdout_seed)
            holdout_stations = sorted(rng.sample(stations, holdout_count))
            selected = [s for s in stations if s not in set(holdout_stations)]
            if len(selected) == 0:
                raise RuntimeError("No training stations left after holdout selection")

            selected_file = os.path.join(split_dir, "selected_stations.txt")
            test_file = os.path.join(split_dir, "test_stations.txt")
            holdout_file = os.path.join(split_dir, "holdout_stations.txt")
            write_csv_line(selected_file, selected)
            write_csv_line(test_file, [])
            write_csv_line(holdout_file, holdout_stations)

            if not args.skip_build:
                build_cmd = [
                    sys.executable,
                    "data/dataset/build_sm_nq_split.py",
                    "--location-path", args.location_path,
                    "--data-path", args.data_path,
                    "--output-dir", split_dir,
                    "--selected-stations-file", selected_file,
                    "--test-stations-file", test_file,
                    "--holdout-station-id", ",".join(holdout_stations),
                ]
                run_cmd(build_cmd, cwd=root, log_path=os.path.join(logs_root, "build.log"))

            for strategy in strategies:
                if strategy not in ["orig", "tts", "mts", "pmts"]:
                    raise ValueError(f"Unknown strategy: {strategy}")

                if resume_enabled and (strategy, observed_count, seed) in completed:
                    print(f"[Resume] Skip strategy={strategy}, ratio={ratio_label}, seed={seed}")
                    continue

                training_strategy = strategy
                num_train_target = int(args.num_train_target)
                if strategy == "orig":
                    # Original training behavior maps to random split per sample.
                    training_strategy = "mts"
                    num_train_target = int(args.orig_num_train_target)

                logs_dir = os.path.join(logs_root, strategy)
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
                        "--enable_val",
                        "--save_best",
                        "--n_epochs", str(args.n_epochs),
                        "--n_epochs_decay", str(args.n_epochs_decay),
                        "--checkpoints_dir", args.checkpoints_dir,
                        "--training_strategy", training_strategy,
                        "--pmts_update_freq", str(args.pmts_update_freq),
                        "--num_train_target", str(num_train_target),
                        "--sm_location_path", os.path.join(split_dir, "Stations_information_NAQU_subset.csv"),
                        "--sm_data_path", os.path.join(split_dir, "SM_NQ-30-minutes_05cm_subset.csv"),
                        "--sm_test_nodes_path", os.path.join(split_dir, "test_nodes.npy"),
                        "--sm_holdout_nodes_path", os.path.join(split_dir, "holdout_nodes.npy"),
                        "--sm_holdout_station_id", ",".join(holdout_stations),
                        "--sm_holdout_location_path", args.holdout_location_path,
                        "--sm_holdout_data_path", args.holdout_data_path,
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
                        "ratio_label": ratio_label,
                        "observed_count": observed_count,
                        "holdout_count": holdout_count,
                        "strategy": strategy,
                        "seed": seed,
                        "file_time": file_time,
                        "t_len": t_len,
                        "window_hours": int(args.window_hours),
                        "num_train_target": int(num_train_target),
                        "pmts_update_freq": int(args.pmts_update_freq),
                        "holdout_seed": holdout_seed,
                        "holdout_stations": holdout_stations,
                    }
                    write_marker(checkpoint_dir, analysis_dir, marker_info)

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
                        "--sm_holdout_station_id", ",".join(holdout_stations),
                        "--sm_holdout_location_path", args.holdout_location_path,
                        "--sm_holdout_data_path", args.holdout_data_path,
                        "--sm_eval_target_mode", "holdout",
                        "--station_meta_path", args.station_meta_path,
                    ]
                    run_cmd(test_cmd, cwd=root, log_path=os.path.join(logs_dir, "test_holdout.log"))

                report_path = os.path.join(analysis_dir, "analysis_report.txt") if analysis_dir else ""
                metrics = parse_overall_metrics(report_path) if report_path else {}

                summary_rows.append({
                    "ratio_label": ratio_label,
                    "observed_count": observed_count,
                    "holdout_count": holdout_count,
                    "strategy": strategy,
                    "seed": seed,
                    "holdout_seed": holdout_seed,
                    "holdout_stations": holdout_stations,
                    "holdout_stations_str": ",".join(holdout_stations),
                    "window_hours": int(args.window_hours),
                    "t_len": t_len,
                    "num_train_target": int(num_train_target),
                    "pmts_update_freq": int(args.pmts_update_freq),
                    "file_time": file_time,
                    "checkpoint_dir": checkpoint_dir,
                    "analysis_dir": analysis_dir,
                    "overall_mae": metrics.get("overall_mae"),
                    "overall_rmse": metrics.get("overall_rmse"),
                    "overall_r2": metrics.get("overall_r2"),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_by_seed_csv = os.path.join(exp_root, "results_summary_by_seed.csv")
    summary_df.to_csv(summary_by_seed_csv, index=False)

    metric_cols = ["overall_mae", "overall_rmse", "overall_r2"]
    mean_df = summary_df.groupby(["ratio_label", "observed_count", "holdout_count", "strategy"], as_index=False)[metric_cols].mean()
    std_df = summary_df.groupby(["ratio_label", "observed_count", "holdout_count", "strategy"], as_index=False)[metric_cols].std(ddof=0)
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in metric_cols})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in metric_cols})
    summary_mean_df = mean_df.merge(std_df, on=["ratio_label", "observed_count", "holdout_count", "strategy"], how="left")

    summary_csv = os.path.join(exp_root, "results_summary_mean.csv")
    summary_mean_df.to_csv(summary_csv, index=False)

    latest_dir = os.path.join(root, "experiments", args.experiment_name)
    ensure_dir(latest_dir)
    summary_mean_df.to_csv(os.path.join(latest_dir, "results_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(latest_dir, "results_summary_by_seed.csv"), index=False)

    plots_dir = os.path.join(exp_root, "plots")
    ensure_dir(plots_dir)

    ratio_labels = [item["ratio_label"] for item in ratios]
    plot_strategy_curve(
        summary_mean_df,
        ratio_labels,
        [s.lower() for s in strategies],
        os.path.join(plots_dir, "ratio_vs_mae_by_strategy.png"),
    )

    run_info = {
        "exp_root": exp_root,
        "ratios": ratios,
        "strategies": strategies,
        "seeds": seeds,
        "t_len": t_len,
        "window_hours": int(args.window_hours),
        "num_train_target": int(args.num_train_target),
        "orig_num_train_target": int(args.orig_num_train_target),
        "pmts_update_freq": int(args.pmts_update_freq),
        "summary_mean_csv": summary_csv,
        "summary_by_seed_csv": summary_by_seed_csv,
        "plots_dir": plots_dir,
    }
    with open(os.path.join(exp_root, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Strategy + ratio experiment finished")
    print(f"Summary (mean) CSV: {summary_csv}")
    print(f"Summary (by seed) CSV: {summary_by_seed_csv}")
    print(f"Plots dir : {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
