#!/usr/bin/env python3
import argparse
import math
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import spfs_model
import spfs_utils


def downsample_index(n_points, max_points):
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points)
    stride = int(math.ceil(n_points / float(max_points)))
    return np.arange(0, n_points, stride)


def calc_station_metrics(pred, truth):
    mask = truth > 0
    valid_count = int(np.sum(mask))
    if valid_count == 0:
        return {
            "valid_count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
        }

    err = pred[mask] - truth[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    mape_mask = truth > 10
    if np.sum(mape_mask) == 0:
        mape_mask = mask
    mape = float(np.mean(np.abs(pred[mape_mask] - truth[mape_mask]) / (truth[mape_mask] + 1e-10)))

    denom = float(np.mean((truth[mask] - np.mean(truth[mask])) ** 2))
    if denom <= 1e-12:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.mean(err ** 2) / denom)

    return {
        "valid_count": valid_count,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }


def evaluate(model, x_gp, x_fs, gp, fs, te, y, h, device):
    model.eval()
    preds = []

    num_chunk = x_gp.shape[1] // h
    with torch.no_grad():
        for i in range(num_chunk):
            s, e = i * h, (i + 1) * h
            batch_x_gp = spfs_utils.to_device(x_gp[:, s:e], device, torch.float32)
            batch_x_fs = spfs_utils.to_device(x_fs[:, s:e], device, torch.float32)
            batch_te = spfs_utils.to_device(te[:, s:e], device, torch.long)
            batch_gp = spfs_utils.to_device(gp, device, torch.float32)
            batch_fs = spfs_utils.to_device(fs, device, torch.float32)

            pred = model(batch_x_gp, batch_x_fs, batch_gp, batch_fs, batch_te)
            preds.append(pred.cpu().numpy())

        pred_all = np.concatenate(preds, axis=1) if len(preds) > 0 else np.zeros_like(y[:, :0])

        num_res = y.shape[1] - pred_all.shape[1]
        if num_res > 0:
            batch_x_gp = spfs_utils.to_device(x_gp[:, -h:], device, torch.float32)
            batch_x_fs = spfs_utils.to_device(x_fs[:, -h:], device, torch.float32)
            batch_te = spfs_utils.to_device(te[:, -h:], device, torch.long)
            batch_gp = spfs_utils.to_device(gp, device, torch.float32)
            batch_fs = spfs_utils.to_device(fs, device, torch.float32)
            pred_tail = model(batch_x_gp, batch_x_fs, batch_gp, batch_fs, batch_te).cpu().numpy()
            pred_tail = pred_tail[:, -num_res:]
            pred_all = np.concatenate([pred_all, pred_tail], axis=1)

    return pred_all


def load_station_ids(station_file, test_file):
    station_df = pd.read_csv(station_file)
    if "station_id" not in station_df.columns:
        raise RuntimeError("station_file must contain 'station_id' column")

    station_df = station_df.copy()
    station_df["station_id"] = station_df["station_id"].astype(str)
    station_df = station_df.sort_values("station_id")
    all_station_ids = station_df["station_id"].tolist()

    test_nodes = np.asarray(np.load(test_file)).reshape(-1).astype(int)

    mapped_ids = []
    for node_idx in test_nodes:
        if 0 <= node_idx < len(all_station_ids):
            mapped_ids.append(all_station_ids[node_idx])
        else:
            mapped_ids.append(f"NODE_{node_idx:03d}")
    return mapped_ids, test_nodes


def safe_station_filename(station_id):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(station_id))
    return safe or "unknown_station"


def make_station_plots(pred, truth, station_ids, out_dir, max_points):
    os.makedirs(out_dir, exist_ok=True)
    n_station = min(len(station_ids), pred.shape[0], truth.shape[0])

    for i in range(n_station):
        sid = station_ids[i]
        x = np.arange(pred.shape[1])
        y_pred = pred[i, :, 0]
        y_true = truth[i, :, 0]

        idx = downsample_index(len(x), max_points)
        x_plot = x[idx]
        pred_plot = y_pred[idx]
        true_plot = y_true[idx]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_plot, true_plot, label="Ground Truth", linewidth=1.4)
        ax.plot(x_plot, pred_plot, label="Prediction", linewidth=1.4, alpha=0.9)
        ax.set_title(f"Station {sid}: Prediction vs Ground Truth")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"station_{safe_station_filename(sid)}_pred_vs_true.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def build_predictions_csv(pred, truth, station_ids, test_nodes, csv_path):
    rows = []
    n_station = min(len(station_ids), pred.shape[0], truth.shape[0])
    n_time = min(pred.shape[1], truth.shape[1])

    for i in range(n_station):
        sid = station_ids[i]
        node_idx = int(test_nodes[i]) if i < len(test_nodes) else i
        for t in range(n_time):
            y_true = float(truth[i, t, 0])
            y_pred = float(pred[i, t, 0])
            rows.append(
                {
                    "station_id": sid,
                    "original_node_index": node_idx,
                    "target_station_index": i,
                    "time_index": t,
                    "true_value": y_true,
                    "pred_value": y_pred,
                    "abs_error": abs(y_pred - y_true),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df


def write_report(report_path, overall_metrics, station_metrics_df, output_dir, elapsed_sec):
    lines = []
    lines.append("=" * 80)
    lines.append("SPFS Test Predict Analysis Report")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Elapsed time: {elapsed_sec:.2f} sec")
    lines.append("")

    lines.append("[Overall Metrics]")
    lines.append(f"RMSE: {overall_metrics[0]:.6f}")
    lines.append(f"MAE : {overall_metrics[1]:.6f}")
    lines.append(f"MAPE: {overall_metrics[2]:.6f}")
    lines.append(f"R2  : {overall_metrics[3]:.6f}")
    lines.append("")

    lines.append("[Station Metrics Top 20 by MAE]")
    if station_metrics_df.empty:
        lines.append("No station metrics")
    else:
        lines.append(station_metrics_df.sort_values("MAE", ascending=False).head(20).to_string(index=False))
    lines.append("")

    lines.append("[Output Directory]")
    lines.append(output_dir)
    lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(args):
    start = time.time()
    spfs_utils.set_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "station_figures")
    os.makedirs(figures_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, args.log_name)
    log = open(log_file, "w")

    spfs_utils.log_string(log, "loading data...")
    (
        _train_x_gp,
        _train_gp,
        _train_x_fs,
        _train_fs,
        _train_te,
        _train_y,
        _val_x_gp,
        _val_gp,
        _val_x_fs,
        _val_fs,
        _val_te,
        _val_y,
        test_x_gp,
        test_gp,
        test_x_fs,
        test_fs,
        test_te,
        test_y,
    ) = spfs_utils.load_data(args)

    spfs_utils.log_string(log, f"device: {device}")
    spfs_utils.log_string(log, f"test_x: {test_x_gp.shape}\ttest_y: {test_y.shape}")

    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Cannot find model checkpoint: {args.model_file}")

    ckpt = torch.load(args.model_file, map_location=device)
    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    std = std if std > 0 else 1.0

    model = spfs_model.SPFSModel(T=args.T, d=args.d, mean=mean, std=std).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spfs_utils.log_string(log, f"trainable parameters: {num_params:,}")
    spfs_utils.log_string(log, "model restored!")

    test_pred = evaluate(model, test_x_gp, test_x_fs, test_gp, test_fs, test_te, test_y, args.h, device)

    station_ids, test_nodes = load_station_ids(args.station_file, args.test_file)

    overall_metrics = spfs_utils.metric(test_pred, test_y)
    spfs_utils.log_string(
        log,
        f"overall test_rmse: {overall_metrics[0]:.3f}, test_mae: {overall_metrics[1]:.3f}, "
        f"test_mape: {overall_metrics[2]:.3f}, test_r2: {overall_metrics[3]:.3f}",
    )

    station_rows = []
    n_station = min(len(station_ids), test_pred.shape[0], test_y.shape[0])
    for i in range(n_station):
        sid = station_ids[i]
        node_idx = int(test_nodes[i]) if i < len(test_nodes) else i
        m = calc_station_metrics(test_pred[i, :, 0], test_y[i, :, 0])
        station_rows.append(
            {
                "station_id": sid,
                "original_node_index": node_idx,
                "target_station_index": i,
                **m,
            }
        )

    station_metrics_df = pd.DataFrame(station_rows)
    station_metrics_path = os.path.join(args.output_dir, args.station_metrics_name)
    station_metrics_df.to_csv(station_metrics_path, index=False, encoding="utf-8")

    predictions_csv_path = os.path.join(args.output_dir, args.predictions_csv_name)
    _ = build_predictions_csv(test_pred, test_y, station_ids, test_nodes, predictions_csv_path)

    make_station_plots(
        pred=test_pred,
        truth=test_y,
        station_ids=station_ids,
        out_dir=figures_dir,
        max_points=args.max_plot_points_per_station,
    )

    report_path = os.path.join(args.output_dir, args.report_name)
    elapsed = time.time() - start
    write_report(
        report_path=report_path,
        overall_metrics=overall_metrics,
        station_metrics_df=station_metrics_df,
        output_dir=args.output_dir,
        elapsed_sec=elapsed,
    )

    spfs_utils.log_string(log, f"station metrics saved: {station_metrics_path}")
    spfs_utils.log_string(log, f"predictions csv saved: {predictions_csv_path}")
    spfs_utils.log_string(log, f"station figures dir: {figures_dir}")
    spfs_utils.log_string(log, f"analysis report saved: {report_path}")
    spfs_utils.log_string(log, f"total time: {elapsed / 60:.1f}min")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", default=24, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--T", default=48, type=int)
    parser.add_argument("--d", default=64, type=int)
    parser.add_argument("--seed", default=2026, type=int)

    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--gpu_id", default=-1, type=int, help="CUDA device id, -1 means auto")

    parser.add_argument(
        "--data_file",
        default="data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv",
        type=str,
    )
    parser.add_argument(
        "--station_file",
        default="data/dataset/SM_NQ/Stations_information_NAQU.csv",
        type=str,
    )
    parser.add_argument("--test_file", default="dataset/SM_NQ/test_nodes.npy", type=str)
    parser.add_argument("--max_intervals", default=0, type=int)

    parser.add_argument("--model_file", required=True, type=str)
    parser.add_argument("--output_dir", default="spfs_predict_analysis", type=str)
    parser.add_argument("--log_name", default="predict_log.txt", type=str)
    parser.add_argument("--report_name", default="analysis_report.txt", type=str)
    parser.add_argument("--station_metrics_name", default="station_metrics.csv", type=str)
    parser.add_argument("--predictions_csv_name", default="predictions_structured.csv", type=str)
    parser.add_argument("--max_plot_points_per_station", default=2000, type=int)

    main(parser.parse_args())
