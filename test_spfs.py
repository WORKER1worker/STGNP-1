import argparse
import os
import time

import numpy as np
import torch

import spfs_model
import spfs_utils


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


def main(args):
    spfs_utils.set_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu_id {args.gpu_id}. Available GPU count: {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    log = open(args.log_file, "w")
    start = time.time()
    spfs_utils.log_string(log, f"device: {device}")

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

    spfs_utils.log_string(log, f"test_x: {test_x_gp.shape}\ttest_y: {test_y.shape}")

    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Cannot find model checkpoint: {args.model_file}")

    ckpt = torch.load(args.model_file, map_location=device)
    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    std = std if std > 0 else 1.0

    model = spfs_model.SPFSModel(T=args.T, d=args.d, mean=mean, std=std).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spfs_utils.log_string(log, f"trainable parameters: {num_params:,}")
    spfs_utils.log_string(log, "model restored!")

    test_pred = evaluate(model, test_x_gp, test_x_fs, test_gp, test_fs, test_te, test_y, args.h, device)
    test_rmse, test_mae, test_mape, test_r2 = spfs_utils.metric(test_pred, test_y)

    spfs_utils.log_string(
        log,
        f"test_rmse: {test_rmse:.3f}, test_mae: {test_mae:.3f}, test_mape: {test_mape:.3f}, test_r2: {test_r2:.3f}",
    )

    end = time.time()
    spfs_utils.log_string(log, f"total time: {(end - start) / 60:.1f}min")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", default=24, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--T", default=48, type=int)
    parser.add_argument("--d", default=64, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu_id", default=0, type=int)

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
    parser.add_argument("--log_file", default="spfs_test_log.txt", type=str)

    main(parser.parse_args())
