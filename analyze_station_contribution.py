#!/usr/bin/env python3
"""Compute station contribution excluding self-error.

Delta_i = E[M_-i | i removed] - E[M_-i | i kept]
where M_-i is the mean metric over all holdout stations excluding i.
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def parse_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[\n,]+", raw) if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_result_files(root: str) -> List[str]:
    base_dir = os.path.join(root, "experiments", "ratio_holdout_multi")
    if not os.path.isdir(base_dir):
        return []
    results = []
    for dirpath, _, filenames in os.walk(base_dir):
        if dirpath == base_dir:
            continue
        if "merged_" in os.path.basename(dirpath):
            continue
        if "results_summary_by_seed.csv" in filenames:
            results.append(os.path.join(dirpath, "results_summary_by_seed.csv"))
    results.sort()
    return results


def read_station_metrics(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "station_id" not in df.columns:
        return None
    return df


def compute_m_minus_i(df: pd.DataFrame, station_id: str, metric: str) -> Optional[float]:
    if metric not in df.columns:
        return None
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    if station_id in set(df["station_id"].tolist()):
        others = df[df["station_id"] != station_id]
    else:
        others = df
    if others.empty:
        return None
    return float(others[metric].mean())


def ratio_weight(ratio: float, mode: str, count: int) -> float:
    if mode == "count":
        return float(count)
    if mode == "inv_ratio":
        return 1.0 / float(ratio) if ratio > 0 else 0.0
    return 1.0


def compute_weighted_delta(
    per_ratio: Dict[float, Dict[str, List[float]]],
    mode: str,
    higher_is_better: bool,
) -> Tuple[float, int, int, int]:
    weighted_sum = 0.0
    weight_total = 0.0
    ratio_count = 0
    removed_count = 0
    kept_count = 0

    for ratio, bucket in per_ratio.items():
        removed_vals = bucket.get("removed", [])
        kept_vals = bucket.get("kept", [])
        removed_count += len(removed_vals)
        kept_count += len(kept_vals)
        if not removed_vals or not kept_vals:
            continue
        removed_mean = float(np.mean(removed_vals))
        kept_mean = float(np.mean(kept_vals))
        delta = kept_mean - removed_mean if higher_is_better else removed_mean - kept_mean
        w = ratio_weight(ratio, mode, len(removed_vals) + len(kept_vals))
        weighted_sum += delta * w
        weight_total += w
        ratio_count += 1

    if weight_total == 0:
        return float("nan"), removed_count, kept_count, ratio_count
    return weighted_sum / weight_total, removed_count, kept_count, ratio_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze station contribution excluding self-error")
    parser.add_argument("--workspace", type=str, default=".", help="Project root (STGNP)")
    parser.add_argument("--inputs", type=str, default="",
                        help="Comma/newline-separated results_summary_by_seed.csv paths")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for contribution results")
    parser.add_argument("--stations", type=str, default=",".join(DEFAULT_STATIONS),
                        help="Candidate stations list")
    parser.add_argument("--ratio-weight", type=str, default="uniform",
                        choices=["uniform", "count", "inv_ratio"],
                        help="How to weight ratios when averaging deltas")
    args = parser.parse_args()

    root = os.path.abspath(args.workspace)
    inputs = dedup_keep_order(parse_list(args.inputs))
    if not inputs:
        inputs = discover_result_files(root)
    if not inputs:
        raise FileNotFoundError("No results_summary_by_seed.csv inputs found")

    stations = dedup_keep_order(parse_list(args.stations))
    if not stations:
        raise ValueError("No stations provided")

    rows = []
    for path in inputs:
        df = pd.read_csv(path)
        if "holdout_ratio" not in df.columns or "holdout_stations" not in df.columns:
            continue
        if "analysis_dir" not in df.columns:
            continue
        df = df.copy()
        df["holdout_list"] = df["holdout_stations"].astype(str).apply(
            lambda s: [x for x in s.split(",") if x]
        )
        df["source_run"] = os.path.basename(os.path.dirname(path))
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid experiment rows loaded")

    all_df = pd.concat(rows, ignore_index=True)

    per_station_runs: Dict[str, List[Dict[str, object]]] = {s: [] for s in stations}
    self_metrics: Dict[str, Dict[str, List[float]]] = {s: {"MAE": [], "RMSE": [], "R2": []} for s in stations}

    for _, row in all_df.iterrows():
        ratio = float(row["holdout_ratio"])
        holdout_set = set(row["holdout_list"])
        analysis_dir = str(row["analysis_dir"])
        metrics_path = os.path.join(analysis_dir, "station_metrics.csv")
        df = read_station_metrics(metrics_path)
        if df is None or df.empty:
            continue
        df["station_id"] = df["station_id"].astype(str)
        station_rows = df.set_index("station_id")

        for station_id in stations:
            m_mae = compute_m_minus_i(df, station_id, "MAE")
            m_rmse = compute_m_minus_i(df, station_id, "RMSE")
            m_r2 = compute_m_minus_i(df, station_id, "R2")
            if m_mae is None or m_rmse is None or m_r2 is None:
                continue
            per_station_runs[station_id].append({
                "ratio": ratio,
                "removed": station_id in holdout_set,
                "mae": m_mae,
                "rmse": m_rmse,
                "r2": m_r2,
            })

            if station_id in station_rows.index:
                self_metrics[station_id]["MAE"].append(float(station_rows.loc[station_id, "MAE"]))
                self_metrics[station_id]["RMSE"].append(float(station_rows.loc[station_id, "RMSE"]))
                self_metrics[station_id]["R2"].append(float(station_rows.loc[station_id, "R2"]))

    contrib_rows = []
    for station_id, records in per_station_runs.items():
        per_ratio: Dict[float, Dict[str, List[float]]] = {}
        for rec in records:
            ratio = float(rec["ratio"])
            per_ratio.setdefault(ratio, {"removed": [], "kept": []})
            key = "removed" if rec["removed"] else "kept"
            per_ratio[ratio][key].append(float(rec["mae"]))

        delta_mae, removed_count, kept_count, ratio_count = compute_weighted_delta(
            per_ratio, args.ratio_weight, higher_is_better=False
        )

        per_ratio = {}
        for rec in records:
            ratio = float(rec["ratio"])
            per_ratio.setdefault(ratio, {"removed": [], "kept": []})
            key = "removed" if rec["removed"] else "kept"
            per_ratio[ratio][key].append(float(rec["rmse"]))
        delta_rmse, _, _, _ = compute_weighted_delta(per_ratio, args.ratio_weight, higher_is_better=False)

        per_ratio = {}
        for rec in records:
            ratio = float(rec["ratio"])
            per_ratio.setdefault(ratio, {"removed": [], "kept": []})
            key = "removed" if rec["removed"] else "kept"
            per_ratio[ratio][key].append(float(rec["r2"]))
        delta_r2, _, _, _ = compute_weighted_delta(per_ratio, args.ratio_weight, higher_is_better=True)

        contrib_rows.append({
            "station_id": station_id,
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "delta_r2": delta_r2,
            "removed_count": removed_count,
            "kept_count": kept_count,
            "ratio_count": ratio_count,
            "total_runs": len(records),
        })

    contrib_df = pd.DataFrame(contrib_rows)
    contrib_df = contrib_df.sort_values("delta_mae", ascending=False).reset_index(drop=True)
    contrib_df["rank"] = contrib_df.index + 1

    pred_rows = []
    for station_id, metrics in self_metrics.items():
        mae_vals = metrics["MAE"]
        rmse_vals = metrics["RMSE"]
        r2_vals = metrics["R2"]
        pred_rows.append({
            "station_id": station_id,
            "self_mae_mean": float(np.mean(mae_vals)) if mae_vals else float("nan"),
            "self_mae_std": float(np.std(mae_vals)) if mae_vals else float("nan"),
            "self_rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "self_rmse_std": float(np.std(rmse_vals)) if rmse_vals else float("nan"),
            "self_r2_mean": float(np.mean(r2_vals)) if r2_vals else float("nan"),
            "self_r2_std": float(np.std(r2_vals)) if r2_vals else float("nan"),
            "self_count": len(mae_vals),
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df = pred_df.sort_values("self_mae_mean", ascending=False).reset_index(drop=True)

    out_dir = args.output_dir.strip()
    if not out_dir:
        tag = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = os.path.join(root, "experiments", "ratio_holdout_multi", f"contribution_{tag}")
    ensure_dir(out_dir)

    contrib_path = os.path.join(out_dir, "contribution_ranking.csv")
    pred_path = os.path.join(out_dir, "predictability_ranking.csv")
    contrib_df.to_csv(contrib_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    meta = {
        "created_at": datetime.now().isoformat(),
        "inputs": inputs,
        "ratio_weight": args.ratio_weight,
        "contribution_csv": contrib_path,
        "predictability_csv": pred_path,
    }
    with open(os.path.join(out_dir, "contribution_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Contribution analysis completed")
    print(f"Output dir: {out_dir}")
    print(f"Contribution ranking: {contrib_path}")
    print(f"Predictability ranking: {pred_path}")


if __name__ == "__main__":
    main()
