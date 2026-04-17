import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def log_string(log, string):
    log.write(string + "\n")
    log.flush()
    print(string)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric(pred, truth):
    idx = truth > 0
    if np.sum(idx) == 0:
        return np.nan, np.nan, np.nan, np.nan

    rmse = np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2))
    mae = np.mean(np.abs(pred[idx] - truth[idx]))
    denom = np.mean((truth[idx] - np.mean(truth[idx])) ** 2)
    r2 = 1 - np.mean((pred[idx] - truth[idx]) ** 2) / (denom + 1e-10)

    idx_mape = truth > 10
    if np.sum(idx_mape) == 0:
        # For low-value targets (e.g., soil moisture), fall back to valid positive values.
        idx_mape = idx
    mape = np.mean(np.abs(pred[idx_mape] - truth[idx_mape]) / (truth[idx_mape] + 1e-10))

    return rmse, mae, mape, r2


def haversine(lon1, lat1, lon2, lat2):
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def heterogeneous_relations(A, train_node, K):
    relation = np.zeros_like(A, dtype=np.float32) - 1.0
    relation[:, train_node] = A[:, train_node]
    np.fill_diagonal(relation, val=-1.0)

    neighbor = np.argsort(-relation, axis=1)
    neighbor = neighbor[:, :K]

    relation = -np.sort(-relation, axis=-1)
    relation = relation[:, :K]
    relation[relation < 0] = 0
    relation_sum = np.sum(relation, axis=1, keepdims=True)
    relation = relation / (relation_sum + 1e-10)
    return relation.astype(np.float32), neighbor.astype(np.int64)


def _read_sm_nq_dataframe(data_file):
    df = pd.read_csv(data_file)
    dt = pd.to_datetime(
        df[["yyyy", "mm", "dd", "HH", "MM", "SS"]].rename(
            columns={
                "yyyy": "year",
                "mm": "month",
                "dd": "day",
                "HH": "hour",
                "MM": "minute",
                "SS": "second",
            }
        )
    )
    return df, dt


def _build_station_order(station_file, df):
    station_df = pd.read_csv(station_file)
    station_df = station_df[["station_id", "lon", "lat"]].copy()
    station_df["station_id"] = station_df["station_id"].astype(str)
    station_df = station_df.sort_values("station_id")

    station_ids = [sid for sid in station_df["station_id"].tolist() if sid in df.columns]
    if len(station_ids) == 0:
        raise RuntimeError("No station columns matched between station file and data file.")

    station_df = station_df[station_df["station_id"].isin(station_ids)].copy()
    station_df = station_df.set_index("station_id").loc[station_ids].reset_index()
    return station_ids, station_df


def _build_distance_matrix(station_df):
    n = len(station_df)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lon_i, lat_i = float(station_df.iloc[i]["lon"]), float(station_df.iloc[i]["lat"])
        for j in range(n):
            lon_j, lat_j = float(station_df.iloc[j]["lon"]), float(station_df.iloc[j]["lat"])
            dist[i, j] = haversine(lon_i, lat_i, lon_j, lat_j)
    return dist


def _build_te(dt, T):
    secs = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    te = (secs // (24 * 3600 / T)).astype(np.int32)
    te = np.array(te, dtype=np.int32)[np.newaxis]
    return te


def _fill_and_mask(x_raw):
    x = x_raw.copy()
    missing = x <= -90

    global_valid = x[~missing]
    global_mean = float(np.mean(global_valid)) if global_valid.size > 0 else 0.0

    for i in range(x.shape[0]):
        valid = x[i][~missing[i]]
        station_mean = float(np.mean(valid)) if valid.size > 0 else global_mean
        x[i, missing[i]] = station_mean

    y_truth = x_raw.copy()
    y_truth[missing] = 0.0
    return x.astype(np.float32), y_truth.astype(np.float32), missing


def load_data(args):
    df, dt = _read_sm_nq_dataframe(args.data_file)
    station_ids, station_df = _build_station_order(args.station_file, df)

    x_raw = df[station_ids].to_numpy(dtype=np.float32).T
    max_intervals = int(getattr(args, "max_intervals", 0))
    if max_intervals > 0:
        max_intervals = min(max_intervals, x_raw.shape[1])
        x_raw = x_raw[:, :max_intervals]
        dt = dt.iloc[:max_intervals]

    N, num_interval = x_raw.shape

    x, y_truth_all, _ = _fill_and_mask(x_raw)
    TE = _build_te(dt, args.T)

    test_node = np.load(args.test_file)
    all_node = np.arange(N)
    train_node = np.setdiff1d(all_node, test_node)
    np.random.shuffle(train_node)

    dist_mx = _build_distance_matrix(station_df)
    dist_train = dist_mx[np.ix_(train_node, train_node)]
    std_gp = np.std(dist_train)
    if std_gp <= 0:
        std_gp = 1.0
    A_gp = np.exp(-(dist_mx ** 2) / (std_gp ** 2)).astype(np.float32)
    gp, neighbor_gp = heterogeneous_relations(A_gp, train_node, args.K)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(x)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    fs, neighbor_fs = heterogeneous_relations(corr.astype(np.float32), train_node, args.K)

    num_train = int(0.7 * num_interval)
    num_val = int(0.2 * num_train)
    num_train -= num_val
    num_test = num_interval - num_train - num_val

    train_TE = TE[:, :num_train]
    val_TE = TE[:, num_train : num_train + num_val]
    test_TE = TE[:, -num_test:]

    train_x_gp = np.transpose(
        x[neighbor_gp[train_node], :num_train, np.newaxis], axes=(0, 2, 1, 3)
    )
    train_x_fs = np.transpose(
        x[neighbor_fs[train_node], :num_train, np.newaxis], axes=(0, 2, 1, 3)
    )
    train_y = y_truth_all[train_node, :num_train, np.newaxis]

    val_x_gp = np.transpose(
        x[neighbor_gp[train_node], num_train : num_train + num_val, np.newaxis], axes=(0, 2, 1, 3)
    )
    val_x_fs = np.transpose(
        x[neighbor_fs[train_node], num_train : num_train + num_val, np.newaxis], axes=(0, 2, 1, 3)
    )
    val_y = y_truth_all[train_node, num_train : num_train + num_val, np.newaxis]

    test_x_gp = np.transpose(
        x[neighbor_gp[test_node], -num_test:, np.newaxis], axes=(0, 2, 1, 3)
    )
    test_x_fs = np.transpose(
        x[neighbor_fs[test_node], -num_test:, np.newaxis], axes=(0, 2, 1, 3)
    )
    test_y = y_truth_all[test_node, -num_test:, np.newaxis]

    train_gp = gp[train_node, np.newaxis, np.newaxis]
    train_fs = fs[train_node, np.newaxis, np.newaxis]
    val_gp = gp[train_node, np.newaxis, np.newaxis]
    val_fs = fs[train_node, np.newaxis, np.newaxis]
    test_gp = gp[test_node, np.newaxis, np.newaxis]
    test_fs = fs[test_node, np.newaxis, np.newaxis]

    return (
        train_x_gp.astype(np.float32),
        train_gp.astype(np.float32),
        train_x_fs.astype(np.float32),
        train_fs.astype(np.float32),
        train_TE.astype(np.int64),
        train_y.astype(np.float32),
        val_x_gp.astype(np.float32),
        val_gp.astype(np.float32),
        val_x_fs.astype(np.float32),
        val_fs.astype(np.float32),
        val_TE.astype(np.int64),
        val_y.astype(np.float32),
        test_x_gp.astype(np.float32),
        test_gp.astype(np.float32),
        test_x_fs.astype(np.float32),
        test_fs.astype(np.float32),
        test_TE.astype(np.int64),
        test_y.astype(np.float32),
    )


def prepare_output_dir(base_dir, dataset_name, exp_name=""):
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    if exp_name:
        run_name = f"spfs_{dataset_name}_{exp_name}_{ts}"
    else:
        run_name = f"spfs_{dataset_name}_{ts}"
    save_dir = os.path.join(base_dir, dataset_name, run_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def to_device(arr, device, dtype=None):
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)
