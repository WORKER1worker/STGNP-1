import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

import spfs_model
import spfs_utils


def evaluate(model, x_gp, x_fs, gp, fs, te, y, h, device):
    model.eval()
    losses = []
    preds = []

    num_chunk = x_gp.shape[1] // h
    with torch.no_grad():
        for i in range(num_chunk):
            s, e = i * h, (i + 1) * h
            batch_x_gp = spfs_utils.to_device(x_gp[:, s:e], device, torch.float32)
            batch_x_fs = spfs_utils.to_device(x_fs[:, s:e], device, torch.float32)
            batch_te = spfs_utils.to_device(te[:, s:e], device, torch.long)
            batch_y = spfs_utils.to_device(y[:, s:e], device, torch.float32)
            batch_gp = spfs_utils.to_device(gp, device, torch.float32)
            batch_fs = spfs_utils.to_device(fs, device, torch.float32)

            pred = model(batch_x_gp, batch_x_fs, batch_gp, batch_fs, batch_te)
            loss = spfs_model.masked_mse_loss(pred, batch_y)
            losses.append(float(loss.item()))
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

    return float(np.mean(losses)) if len(losses) > 0 else np.nan, pred_all


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

    save_dir = spfs_utils.prepare_output_dir(args.checkpoints_dir, args.dataset_name, args.exp_name)
    model_file = os.path.join(save_dir, "best_model.pt")
    log_file = os.path.join(save_dir, "train_log.txt")
    args_file = os.path.join(save_dir, "train_opt.txt")

    with open(args_file, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    log = open(log_file, "w")
    start = time.time()
    spfs_utils.log_string(log, f"device: {device}")

    spfs_utils.log_string(log, "loading data...")
    (
        train_x_gp,
        train_gp,
        train_x_fs,
        train_fs,
        train_te,
        train_y,
        val_x_gp,
        val_gp,
        val_x_fs,
        val_fs,
        val_te,
        val_y,
        test_x_gp,
        test_gp,
        test_x_fs,
        test_fs,
        test_te,
        test_y,
    ) = spfs_utils.load_data(args)

    valid_train = train_y[train_y > 0]
    mean = float(np.mean(valid_train)) if valid_train.size > 0 else 0.0
    std = float(np.std(valid_train)) if valid_train.size > 0 else 1.0
    std = std if std > 0 else 1.0

    spfs_utils.log_string(log, f"train_x: {train_x_gp.shape}\ttrain_y: {train_y.shape}")
    spfs_utils.log_string(log, f"val_x:   {val_x_gp.shape}\tval_y:   {val_y.shape}")
    spfs_utils.log_string(log, f"test_x:  {test_x_gp.shape}\ttest_y:  {test_y.shape}")
    spfs_utils.log_string(log, f"mean: {mean:.4f}, std: {std:.4f}")

    model = spfs_model.SPFSModel(T=args.T, d=args.d, mean=mean, std=std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spfs_utils.log_string(log, f"total trainable parameters: {num_params:,}")
    spfs_utils.log_string(log, "**** training model ****")

    train_time, val_time = [], []
    val_loss_min = np.inf
    wait = 0

    for epoch in range(args.epochs):
        if wait >= args.patience:
            spfs_utils.log_string(log, f"early stop at epoch: {epoch}")
            break

        model.train()
        train_loss = []

        start_idx = np.random.choice(range(args.h))
        num_train = (train_x_gp.shape[1] - start_idx) // args.h
        end_idx = start_idx + num_train * args.h

        train_x_gp_epoch = train_x_gp[:, start_idx:end_idx]
        train_x_fs_epoch = train_x_fs[:, start_idx:end_idx]
        train_x_gp_epoch = np.reshape(train_x_gp_epoch, newshape=(-1, num_train, args.h, args.K, 1))
        train_x_fs_epoch = np.reshape(train_x_fs_epoch, newshape=(-1, num_train, args.h, args.K, 1))

        train_te_epoch = train_te[:, start_idx:end_idx]
        train_te_epoch = np.reshape(train_te_epoch, newshape=(1, num_train, args.h))

        train_y_epoch = train_y[:, start_idx:end_idx]
        train_y_epoch = np.reshape(train_y_epoch, newshape=(-1, num_train, args.h, 1))

        permutation = np.random.permutation(num_train)
        train_x_gp_epoch = train_x_gp_epoch[:, permutation]
        train_x_fs_epoch = train_x_fs_epoch[:, permutation]
        train_te_epoch = train_te_epoch[:, permutation]
        train_y_epoch = train_y_epoch[:, permutation]

        t1 = time.time()
        for i in range(num_train):
            batch_x_gp = spfs_utils.to_device(train_x_gp_epoch[:, i], device, torch.float32)
            batch_x_fs = spfs_utils.to_device(train_x_fs_epoch[:, i], device, torch.float32)
            batch_te = spfs_utils.to_device(train_te_epoch[:, i], device, torch.long)
            batch_y = spfs_utils.to_device(train_y_epoch[:, i], device, torch.float32)
            batch_gp = spfs_utils.to_device(train_gp, device, torch.float32)
            batch_fs = spfs_utils.to_device(train_fs, device, torch.float32)

            optimizer.zero_grad()
            pred = model(batch_x_gp, batch_x_fs, batch_gp, batch_fs, batch_te)
            loss = spfs_model.masked_mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss.append(float(loss.item()))

        train_time.append(time.time() - t1)
        train_loss = float(np.mean(train_loss)) if len(train_loss) > 0 else np.nan

        t1 = time.time()
        val_loss, _ = evaluate(model, val_x_gp, val_x_fs, val_gp, val_fs, val_te, val_y, args.h, device)
        val_time.append(time.time() - t1)

        spfs_utils.log_string(
            log,
            (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | epoch: {epoch + 1:03d}/{args.epochs}, "
                f"train_time: {train_time[-1]:.2f}s, train_loss: {train_loss:.4f}, "
                f"val_time: {val_time[-1]:.2f}s, val_loss: {val_loss:.4f}"
            ),
        )

        if val_loss <= val_loss_min:
            spfs_utils.log_string(
                log,
                f"val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {model_file}",
            )
            val_loss_min = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                    "epoch": epoch,
                },
                model_file,
            )
            wait = 0
        else:
            wait += 1

    spfs_utils.log_string(
        log,
        (
            f"training finished, average train time: {np.mean(train_time):.3f}s, "
            f"average val time: {np.mean(val_time):.3f}s, min val loss: {val_loss_min:.4f}"
        ),
    )

    spfs_utils.log_string(log, "**** testing model ****")
    ckpt = torch.load(model_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_pred = evaluate(model, test_x_gp, test_x_fs, test_gp, test_fs, test_te, test_y, args.h, device)
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
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=10, type=int)
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

    parser.add_argument("--checkpoints_dir", default="checkpoints", type=str)
    parser.add_argument("--dataset_name", default="SM", type=str)
    parser.add_argument("--exp_name", default="SM_NQ", type=str)

    main(parser.parse_args())
