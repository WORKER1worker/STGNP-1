#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整的模型测试与结果分析脚本。

功能覆盖：
1. 加载测试数据集与训练好的模型，并执行批量推理
2. 缓存并保存预测结果
3. 生成结构化 CSV（UTF-8）
4. 绘制趋势图与误差分布图（matplotlib，不依赖 seaborn）
5. 计算总体与分站点指标（MAE、RMSE）
6. 基于 test_nodes.npy 构建站点映射，支持 fallback
7. 输出终端摘要与 TXT 分析报告
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


@dataclass
class AnalysisArgs:
    output_dir: str
    csv_name: str
    report_name: str
    station_metrics_name: str
    trend_plot_name: str
    error_plot_name: str
    max_plot_points: int
    max_trend_points_per_station: int
    trend_top_k_stations: int
    csv_chunk_size: int
    progress_every_batches: int
    test_nodes_path: str
    station_meta_path: str
    keep_missing: bool
    true_min: Optional[float]
    true_max: Optional[float]
    pred_min: Optional[float]
    pred_max: Optional[float]
    abs_error_max: Optional[float]


def parse_args() -> Tuple[AnalysisArgs, object, dict]:
    """先解析分析参数，再把剩余参数交给 TestOptions。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", type=str, default="", help="分析结果输出目录，默认保存到当前 checkpoint 目录下")
    parser.add_argument("--csv_name", type=str, default="predictions_structured.csv", help="结构化 CSV 文件名")
    parser.add_argument("--report_name", type=str, default="analysis_report.txt", help="分析报告文件名")
    parser.add_argument("--station_metrics_name", type=str, default="station_metrics.csv", help="分站点指标文件名")
    parser.add_argument("--trend_plot_name", type=str, default="trend_prediction_vs_truth.png", help="趋势图文件名")
    parser.add_argument("--error_plot_name", type=str, default="error_analysis.png", help="误差分析图文件名")

    parser.add_argument("--max_plot_points", type=int, default=5000, help="误差图最大采样点数")
    parser.add_argument("--max_trend_points_per_station", type=int, default=1200, help="每个站点趋势图最大点数")
    parser.add_argument("--trend_top_k_stations", type=int, default=6, help="趋势图最多展示站点数")
    parser.add_argument("--csv_chunk_size", type=int, default=50000, help="CSV 分块写入行数")
    parser.add_argument("--progress_every_batches", type=int, default=20, help="推理日志打印间隔（batch）")

    parser.add_argument("--test_nodes_path", type=str, default="", help="test_nodes.npy 路径（可选）")
    parser.add_argument("--station_meta_path", type=str, default="", help="站点元数据路径（CSV/Excel，可选）")

    parser.add_argument("--keep_missing", action="store_true", help="保留缺失样本（默认会过滤缺失样本）")
    parser.add_argument("--true_min", type=float, default=None, help="过滤真实值下界")
    parser.add_argument("--true_max", type=float, default=None, help="过滤真实值上界")
    parser.add_argument("--pred_min", type=float, default=None, help="过滤预测值下界")
    parser.add_argument("--pred_max", type=float, default=None, help="过滤预测值上界")
    parser.add_argument("--abs_error_max", type=float, default=None, help="过滤绝对误差上限")

    analysis_ns, remaining_argv = parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining_argv
        opt, model_config = TestOptions().parse()
    finally:
        sys.argv = old_argv

    analysis_args = AnalysisArgs(
        output_dir=analysis_ns.output_dir,
        csv_name=analysis_ns.csv_name,
        report_name=analysis_ns.report_name,
        station_metrics_name=analysis_ns.station_metrics_name,
        trend_plot_name=analysis_ns.trend_plot_name,
        error_plot_name=analysis_ns.error_plot_name,
        max_plot_points=max(0, analysis_ns.max_plot_points),
        max_trend_points_per_station=max(50, analysis_ns.max_trend_points_per_station),
        trend_top_k_stations=max(1, analysis_ns.trend_top_k_stations),
        csv_chunk_size=max(1000, analysis_ns.csv_chunk_size),
        progress_every_batches=max(1, analysis_ns.progress_every_batches),
        test_nodes_path=analysis_ns.test_nodes_path,
        station_meta_path=analysis_ns.station_meta_path,
        keep_missing=analysis_ns.keep_missing,
        true_min=analysis_ns.true_min,
        true_max=analysis_ns.true_max,
        pred_min=analysis_ns.pred_min,
        pred_max=analysis_ns.pred_max,
        abs_error_max=analysis_ns.abs_error_max,
    )
    return analysis_args, opt, model_config


def set_chinese_font() -> bool:
    """配置中文字体，找不到时自动回退。

    Returns:
        bool: 是否成功启用中文字体。
    """
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


def tr(cn_text: str, en_text: str, use_cn_font: bool) -> str:
    """根据字体可用性选择中文或英文文案。"""
    if use_cn_font:
        return cn_text
    return en_text


def infer_test_nodes_candidates(dataset_mode: str) -> List[str]:
    mode = dataset_mode or ""
    candidates = [
        f"dataset/{mode}/test_nodes.npy",
        f"dataset/{mode.lower()}/test_nodes.npy",
        f"dataset/{mode.upper()}/test_nodes.npy",
        f"data/dataset/{mode}/test_nodes.npy",
        f"data/dataset/{mode}/NP/test_nodes.npy",
        f"data/dataset/{mode.lower()}/test_nodes.npy",
        f"data/dataset/{mode.lower()}/NP/test_nodes.npy",
    ]
    # 对常见数据集给出额外候选
    if mode.lower() == "sm":
        candidates.extend([
            "dataset/SM/test_nodes.npy",
            "data/dataset/SM/test_nodes.npy",
            "data/dataset/Pali_SM/NP/test_nodes.npy",
            "data/dataset/Pali_ST/NP/test_nodes.npy",
        ])
    # 去重并保持顺序
    seen = set()
    dedup = []
    for p in candidates:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def load_test_nodes(
    analysis_args: AnalysisArgs,
    opt,
    raw_dataset,
    target_station_count: int,
) -> Tuple[np.ndarray, str]:
    """加载 test_nodes.npy，失败时回退到顺序映射。"""
    # 显式指定路径优先，便于复现实验。
    if analysis_args.test_nodes_path:
        if os.path.isfile(analysis_args.test_nodes_path):
            arr = np.load(analysis_args.test_nodes_path)
            arr = np.asarray(arr).reshape(-1).astype(int)
            print(f"[映射] 使用显式 test_nodes 文件: {analysis_args.test_nodes_path}，长度={len(arr)}")
            return arr, analysis_args.test_nodes_path
        print(f"[映射] 显式 test_nodes 文件不存在: {analysis_args.test_nodes_path}，将继续自动推断")

    # 自动模式下，优先与当前推理数据保持一致，避免路径猜测与数据加载来源不一致。
    ds = getattr(raw_dataset, "dataset", None)
    if ds is not None and hasattr(ds, "test_node_index") and ds.test_node_index is not None:
        arr = np.asarray(ds.test_node_index).reshape(-1).astype(int)
        print(f"[映射] 使用数据集对象 test_node_index，长度={len(arr)}")
        return arr, "<dataset.test_node_index>"

    paths = infer_test_nodes_candidates(opt.dataset_mode)
    for path in paths:
        if os.path.isfile(path):
            arr = np.load(path)
            arr = np.asarray(arr).reshape(-1).astype(int)
            print(f"[映射] 使用 test_nodes 文件: {path}，长度={len(arr)}")
            return arr, path

    arr = np.arange(target_station_count, dtype=int)
    print("[映射] test_nodes.npy 未找到，回退为顺序映射 0..N-1")
    return arr, "<fallback_sequential>"


def infer_station_meta_candidates(dataset_mode: str) -> List[str]:
    mode = dataset_mode.lower()
    candidates = []
    if mode == "sm":
        candidates.extend([
            "data/dataset/SM_NQ/Stations_information_NAQU.csv",
            "data/dataset/SM/Pali-Stations.csv",
            "data/dataset/Pali_ST/Stations.csv",
            "data/dataset/Pali_ST/Stations.xlsx",
            "data/dataset/Pali_SM/Stations.csv",
            "data/dataset/Pali_SM/Stations.xlsx",
        ])
    candidates.extend([
        f"data/dataset/{dataset_mode}/Stations.csv",
        f"data/dataset/{dataset_mode}/Stations.xlsx",
    ])
    seen = set()
    dedup = []
    for p in candidates:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def read_station_meta(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def extract_station_ids(meta_df: pd.DataFrame) -> List[str]:
    if meta_df.empty:
        return []
    preferred_cols = [
        "station_id",
        "station",
        "station_name",
        "name",
        "站点编号",
        "站点ID",
        "站点名称",
    ]
    col_name = None
    cols_lower = {c.lower(): c for c in meta_df.columns}
    for col in preferred_cols:
        if col in meta_df.columns:
            col_name = col
            break
        lower = col.lower()
        if lower in cols_lower:
            col_name = cols_lower[lower]
            break
    if col_name is None:
        col_name = meta_df.columns[0]

    values = meta_df[col_name].astype(str).fillna("").tolist()
    return [v.strip() if v.strip() else f"NODE_{i:03d}" for i, v in enumerate(values)]


def load_station_id_table(analysis_args: AnalysisArgs, opt, raw_dataset, max_index: int) -> Tuple[List[str], str]:
    paths = []
    if analysis_args.station_meta_path:
        paths.append(analysis_args.station_meta_path)

    # 自动模式下优先使用数据集内部站点顺序，保证与推理时节点顺序一致。
    ds = getattr(raw_dataset, "dataset", None)
    if ds is not None and hasattr(ds, "station_ids") and ds.station_ids is not None:
        station_ids = [str(x) for x in list(ds.station_ids)]
        if len(station_ids) > 0:
            print(f"[映射] 使用数据集对象 station_ids，总站点={len(station_ids)}")
            return station_ids, "<dataset.station_ids>"

    paths.extend(infer_station_meta_candidates(opt.dataset_mode))

    for path in paths:
        if os.path.isfile(path):
            try:
                df = read_station_meta(path)
                station_ids = extract_station_ids(df)
                if station_ids:
                    print(f"[映射] 使用站点元数据: {path}，总站点={len(station_ids)}")
                    return station_ids, path
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[映射] 读取站点元数据失败: {path}，错误: {exc}")

    fallback = [f"NODE_{i:03d}" for i in range(max_index + 1)]
    print("[映射] 站点元数据未找到，回退为 NODE_xxx")
    return fallback, "<fallback_node_id>"


def build_target_station_mapping(
    test_nodes: np.ndarray,
    station_ids: Sequence[str],
    target_station_count: int,
) -> List[Dict[str, object]]:
    mapping: List[Dict[str, object]] = []
    for target_idx in range(target_station_count):
        if target_idx < len(test_nodes):
            original_idx = int(test_nodes[target_idx])
        else:
            original_idx = target_idx

        if 0 <= original_idx < len(station_ids):
            station_id = station_ids[original_idx]
        else:
            station_id = f"NODE_{original_idx:03d}"

        mapping.append(
            {
                "target_station_index": target_idx,
                "original_node_index": original_idx,
                "station_id": station_id,
            }
        )
    return mapping


def to_numpy_3d(arr: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    if x.ndim != 3:
        raise ValueError(f"{name} 维度应为 2 或 3，当前为 {x.ndim}，shape={x.shape}")
    return x


def to_numpy_2d_or_3d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    x = np.asarray(arr)
    if x.ndim == 2 or x.ndim == 3:
        return x
    raise ValueError(f"missing_target 维度应为 2 或 3，当前为 {x.ndim}，shape={x.shape}")


def format_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def downsample_by_stride(values: np.ndarray, max_points: int) -> np.ndarray:
    if len(values) <= max_points:
        return values
    stride = int(math.ceil(len(values) / float(max_points)))
    return values[::stride]


def reservoir_add(
    reservoir: List[dict],
    row: dict,
    max_size: int,
    seen_count: int,
    rng: np.random.Generator,
) -> int:
    if max_size <= 0:
        return seen_count
    seen_count += 1
    if len(reservoir) < max_size:
        reservoir.append(row)
    else:
        j = int(rng.integers(0, seen_count))
        if j < max_size:
            reservoir[j] = row
    return seen_count


def is_missing_at(missing_target: Optional[np.ndarray], t_idx: int, s_idx: int, f_idx: int) -> bool:
    if missing_target is None:
        return False
    if missing_target.ndim == 2:
        return bool(missing_target[t_idx, s_idx] > 0.5)
    return bool(missing_target[t_idx, s_idx, f_idx] > 0.5)


def should_keep_record(
    true_val: float,
    pred_val: float,
    abs_error: float,
    is_missing: bool,
    analysis_args: AnalysisArgs,
) -> bool:
    if np.isnan(true_val) or np.isnan(pred_val):
        return False
    if (not analysis_args.keep_missing) and is_missing:
        return False
    if analysis_args.true_min is not None and true_val < analysis_args.true_min:
        return False
    if analysis_args.true_max is not None and true_val > analysis_args.true_max:
        return False
    if analysis_args.pred_min is not None and pred_val < analysis_args.pred_min:
        return False
    if analysis_args.pred_max is not None and pred_val > analysis_args.pred_max:
        return False
    if analysis_args.abs_error_max is not None and abs_error > analysis_args.abs_error_max:
        return False
    return True


def calc_r2_from_stats(ss_res: float, y_sum: float, y_sq_sum: float, n: int) -> float:
    """基于聚合统计量计算 R2，避免保存全量数组。"""
    if n <= 1:
        return float("nan")
    ss_tot = y_sq_sum - (y_sum * y_sum) / float(n)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def generate_csv_and_stats(
    y_pred: np.ndarray,
    y_target: np.ndarray,
    time_arr: np.ndarray,
    missing_target: Optional[np.ndarray],
    station_mapping: Sequence[Dict[str, object]],
    csv_path: str,
    analysis_args: AnalysisArgs,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """按循环分块构建 DataFrame 并写入 CSV，同时完成总体/分站点统计。"""
    if os.path.exists(csv_path):
        os.remove(csv_path)

    y_pred = to_numpy_3d(y_pred, "y_pred")
    y_target = to_numpy_3d(y_target, "y_target")
    missing_target = to_numpy_2d_or_3d(missing_target)

    n_time = min(len(time_arr), y_pred.shape[0], y_target.shape[0])
    n_station = min(len(station_mapping), y_pred.shape[1], y_target.shape[1])
    n_feat = min(y_pred.shape[2], y_target.shape[2])

    print(f"[CSV] 数据维度：time={n_time}, station={n_station}, feature={n_feat}")

    chunk_rows: List[dict] = []
    header_written = False

    total_count = 0
    total_abs_sum = 0.0
    total_sq_sum = 0.0
    total_true_sum = 0.0
    total_true_sq_sum = 0.0

    dropped_count = 0
    kept_count = 0

    station_stats = defaultdict(
        lambda: {
            "count": 0,
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "true_sum": 0.0,
            "true_sq_sum": 0.0,
            "target_idx": -1,
            "node_idx": -1,
        }
    )

    rng = np.random.default_rng(2026)
    sample_rows: List[dict] = []
    sample_seen = 0

    for t_idx in range(n_time):
        timestamp = float(time_arr[t_idx])
        time_str = format_timestamp(timestamp)

        for s_idx in range(n_station):
            m = station_mapping[s_idx]
            station_id = str(m["station_id"])
            node_idx = int(m["original_node_index"])

            for f_idx in range(n_feat):
                true_val = float(y_target[t_idx, s_idx, f_idx])
                pred_val = float(y_pred[t_idx, s_idx, f_idx])
                abs_error = abs(pred_val - true_val)
                sq_error = (pred_val - true_val) ** 2
                missing_flag = is_missing_at(missing_target, t_idx, s_idx, min(f_idx, (missing_target.shape[2] - 1)) if missing_target is not None and missing_target.ndim == 3 else 0)

                keep = should_keep_record(true_val, pred_val, abs_error, missing_flag, analysis_args)
                if not keep:
                    dropped_count += 1
                    continue

                row = {
                    "timestamp": timestamp,
                    "formatted_time": time_str,
                    "target_station_index": int(s_idx),
                    "original_node_index": node_idx,
                    "station_id": station_id,
                    "feature_index": int(f_idx),
                    "true_value": true_val,
                    "pred_value": pred_val,
                    "abs_error": abs_error,
                    "sq_error": sq_error,
                }
                chunk_rows.append(row)

                kept_count += 1
                total_count += 1
                total_abs_sum += abs_error
                total_sq_sum += sq_error
                total_true_sum += true_val
                total_true_sq_sum += true_val * true_val

                st = station_stats[station_id]
                st["count"] += 1
                st["abs_sum"] += abs_error
                st["sq_sum"] += sq_error
                st["true_sum"] += true_val
                st["true_sq_sum"] += true_val * true_val
                st["target_idx"] = int(s_idx)
                st["node_idx"] = int(node_idx)

                sample_seen = reservoir_add(sample_rows, row, analysis_args.max_plot_points, sample_seen, rng)

                if len(chunk_rows) >= analysis_args.csv_chunk_size:
                    chunk_df = pd.DataFrame(chunk_rows)
                    chunk_df.to_csv(
                        csv_path,
                        mode="a",
                        index=False,
                        encoding="utf-8",
                        header=(not header_written),
                    )
                    header_written = True
                    chunk_rows.clear()

        if (t_idx + 1) % max(1, n_time // 10) == 0:
            print(f"[CSV] 已处理时间步 {t_idx + 1}/{n_time}")

    if chunk_rows:
        chunk_df = pd.DataFrame(chunk_rows)
        chunk_df.to_csv(
            csv_path,
            mode="a",
            index=False,
            encoding="utf-8",
            header=(not header_written),
        )

    overall_mae = float(total_abs_sum / total_count) if total_count > 0 else float("nan")
    overall_rmse = float(np.sqrt(total_sq_sum / total_count)) if total_count > 0 else float("nan")
    overall_r2 = calc_r2_from_stats(total_sq_sum, total_true_sum, total_true_sq_sum, total_count)

    station_rows = []
    for station_id, st in station_stats.items():
        c = st["count"]
        station_rows.append(
            {
                "station_id": station_id,
                "target_station_index": st["target_idx"],
                "original_node_index": st["node_idx"],
                "sample_count": c,
                "MAE": float(st["abs_sum"] / c) if c > 0 else float("nan"),
                "RMSE": float(np.sqrt(st["sq_sum"] / c)) if c > 0 else float("nan"),
                "R2": calc_r2_from_stats(st["sq_sum"], st["true_sum"], st["true_sq_sum"], c),
            }
        )

    station_metrics_df = pd.DataFrame(station_rows)
    if not station_metrics_df.empty:
        station_metrics_df = station_metrics_df.sort_values(["MAE", "RMSE"], ascending=[False, False]).reset_index(drop=True)

    sample_df = pd.DataFrame(sample_rows)

    summary = {
        "overall_MAE": overall_mae,
        "overall_RMSE": overall_rmse,
        "overall_R2": overall_r2,
        "sample_count": int(total_count),
    }
    count_info = {
        "kept": kept_count,
        "dropped": dropped_count,
    }
    return summary, station_metrics_df, sample_df, count_info


def collect_trend_data_from_csv(
    csv_path: str,
    station_ids: Sequence[str],
    max_points_per_station: int,
) -> pd.DataFrame:
    """从 CSV 分块读取指定站点数据，并按步长下采样用于趋势图。"""
    if not os.path.isfile(csv_path):
        return pd.DataFrame()

    station_set = set(station_ids)
    bucket: Dict[str, List[dict]] = defaultdict(list)

    for chunk in pd.read_csv(csv_path, chunksize=200000):
        chunk = chunk[chunk["station_id"].isin(station_set)]
        if chunk.empty:
            continue
        for row in chunk.to_dict(orient="records"):
            bucket[str(row["station_id"])].append(row)

    merged_rows: List[dict] = []
    for sid, rows in bucket.items():
        if not rows:
            continue
        sdf = pd.DataFrame(rows).sort_values("timestamp")
        idx = downsample_by_stride(np.arange(len(sdf)), max_points_per_station)
        sdf = sdf.iloc[idx]
        sdf["station_id"] = sid
        merged_rows.extend(sdf.to_dict(orient="records"))

    return pd.DataFrame(merged_rows)


def generate_trend_plot(trend_df: pd.DataFrame, output_path: str, top_k: int, use_cn_font: bool) -> None:
    if trend_df.empty:
        print("[绘图] 趋势图数据为空，跳过")
        return

    station_order = trend_df["station_id"].value_counts().head(top_k).index.tolist()
    if not station_order:
        print("[绘图] 无可用站点，跳过趋势图")
        return

    n = len(station_order)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.5 * nrows), squeeze=False)

    for i, station_id in enumerate(station_order):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sdf = trend_df[trend_df["station_id"] == station_id].sort_values("timestamp")

        x = np.arange(len(sdf))
        ax.plot(x, sdf["true_value"].values, label=tr("真实值", "Ground Truth", use_cn_font), linewidth=1.4)
        ax.plot(x, sdf["pred_value"].values, label=tr("预测值", "Prediction", use_cn_font), linewidth=1.4, alpha=0.9)
        ax.set_title(tr(f"站点 {station_id}：预测值 vs 真实值", f"Station {station_id}: Prediction vs Ground Truth", use_cn_font))
        ax.set_xlabel(tr("时间索引", "Time Index", use_cn_font))
        ax.set_ylabel(tr("数值", "Value", use_cn_font))
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    # 去掉空子图
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle(tr("测试集预测趋势对比", "Prediction Trend on Test Set", use_cn_font), fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[绘图] 趋势图已保存: {output_path}")


def generate_error_plots(sample_df: pd.DataFrame, station_metrics_df: pd.DataFrame, output_path: str, use_cn_font: bool) -> None:
    if sample_df.empty:
        print("[绘图] 误差分析样本为空，跳过")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) 误差直方图
    ax = axes[0, 0]
    ax.hist(sample_df["abs_error"].values, bins=50, alpha=0.85, color="#1f77b4")
    ax.set_title(tr("绝对误差分布直方图", "Histogram of Absolute Error", use_cn_font))
    ax.set_xlabel(tr("绝对误差", "Absolute Error", use_cn_font))
    ax.set_ylabel(tr("频数", "Frequency", use_cn_font))
    ax.grid(alpha=0.2)

    # 2) 误差箱线图（按站点）
    ax = axes[0, 1]
    top_station_ids = sample_df["station_id"].value_counts().head(10).index.tolist()
    box_data = [sample_df[sample_df["station_id"] == sid]["abs_error"].values for sid in top_station_ids]
    if box_data:
        ax.boxplot(box_data, labels=top_station_ids, showfliers=False)
        ax.set_title(tr("各站点绝对误差箱线图（样本Top10）", "Absolute Error Boxplot by Station (Top10 Samples)", use_cn_font))
        ax.set_xlabel(tr("站点ID", "Station ID", use_cn_font))
        ax.set_ylabel(tr("绝对误差", "Absolute Error", use_cn_font))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "样本不足", ha="center", va="center")
        ax.set_axis_off()

    # 3) 各站点 MAE 对比
    ax = axes[1, 0]
    if station_metrics_df is not None and not station_metrics_df.empty:
        bar_df = station_metrics_df.sort_values("MAE", ascending=False).head(20)
        ax.bar(bar_df["station_id"].astype(str), bar_df["MAE"].values, color="#2ca02c", alpha=0.9)
        ax.set_title(tr("各站点 MAE 对比（Top20）", "Station MAE Comparison (Top20)", use_cn_font))
        ax.set_xlabel(tr("站点ID", "Station ID", use_cn_font))
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2, axis="y")
    else:
        ax.text(0.5, 0.5, "无站点指标数据", ha="center", va="center")
        ax.set_axis_off()

    # 4) 真实值 vs 预测值散点图
    ax = axes[1, 1]
    x = sample_df["true_value"].values
    y = sample_df["pred_value"].values
    ax.scatter(x, y, s=10, alpha=0.35, color="#d62728", edgecolors="none")
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    low = min(x_min, y_min)
    high = max(x_max, y_max)
    ax.plot([low, high], [low, high], "k--", linewidth=1.2, label=tr("理想线 y=x", "Ideal Line y=x", use_cn_font))
    ax.set_title(tr("真实值 vs 预测值散点图", "Scatter: Ground Truth vs Prediction", use_cn_font))
    ax.set_xlabel(tr("真实值", "Ground Truth", use_cn_font))
    ax.set_ylabel(tr("预测值", "Prediction", use_cn_font))
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    fig.suptitle(tr("测试误差分析图", "Test Error Analysis", use_cn_font), fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[绘图] 误差分析图已保存: {output_path}")


def write_report(
    report_path: str,
    station_metrics_path: str,
    opt,
    summary: Dict[str, float],
    count_info: Dict[str, int],
    model_metrics: Dict[str, float],
    station_metrics_df: pd.DataFrame,
    mapping_source: str,
    station_meta_source: str,
    csv_path: str,
    trend_plot_path: str,
    error_plot_path: str,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time

    lines = []
    lines.append("=" * 80)
    lines.append("模型测试与结果分析报告")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"数据集: {opt.dataset_mode}")
    lines.append(f"模型: {opt.model}")
    lines.append(f"预测属性: {opt.pred_attr}")
    lines.append(f"checkpoint目录: {os.path.join(opt.checkpoints_dir, opt.name)}")
    lines.append(f"总耗时: {elapsed:.2f} 秒")
    lines.append("")

    lines.append("[一] 数据映射来源")
    lines.append(f"test_nodes来源: {mapping_source}")
    lines.append(f"站点元数据来源: {station_meta_source}")
    lines.append("")

    lines.append("[二] 样本统计")
    lines.append(f"保留样本数: {count_info.get('kept', 0)}")
    lines.append(f"过滤样本数: {count_info.get('dropped', 0)}")
    lines.append("")

    lines.append("[三] 总体指标")
    lines.append(f"总体 MAE: {summary.get('overall_MAE', float('nan')):.6f}")
    lines.append(f"总体 RMSE: {summary.get('overall_RMSE', float('nan')):.6f}")
    lines.append(f"总体 R2: {summary.get('overall_R2', float('nan')):.6f}")
    lines.append("")

    if model_metrics:
        lines.append("[四] 模型内部 metrics（来自 model.compute_metrics）")
        for k, v in model_metrics.items():
            lines.append(f"{k}: {v:.6f}")
        lines.append("")

    lines.append("[五] 分站点指标（前20条，按 MAE 降序）")
    if station_metrics_df is not None and not station_metrics_df.empty:
        lines.append(station_metrics_df.head(20).to_string(index=False))
    else:
        lines.append("无可用分站点指标。")
    lines.append("")

    lines.append("[六] 产物路径")
    lines.append(f"结构化CSV: {csv_path}")
    lines.append(f"趋势图: {trend_plot_path}")
    lines.append(f"误差图: {error_plot_path}")
    lines.append(f"分站点指标CSV: {station_metrics_path}")
    lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[报告] 分析报告已保存: {report_path}")


def main() -> None:
    start = time.time()
    use_cn_font = set_chinese_font()

    analysis_args, opt, model_config = parse_args()

    print("=" * 80)
    print("开始执行：测试推理 + 结果分析")
    print("=" * 80)

    print("[1/6] 加载测试数据集")
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"测试样本（batch）数量: {dataset_size}")

    print("[2/6] 加载模型并执行推理")
    model = create_model(opt, model_config)
    model.setup(opt)
    model.eval()

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        model.cache_results()
        if (i + 1) % analysis_args.progress_every_batches == 0 or (i + 1) == dataset_size:
            print(f"推理进度: {i + 1}/{dataset_size} batches")

    # 保存原始结果
    model.save_data()
    print(f"[结果] 原始缓存已保存到: {os.path.join(model.save_dir, 'results.pkl')}")

    print("[3/6] 读取结果并计算模型内置指标")
    model.compute_metrics()
    model_metrics = model.get_current_metrics()
    if model_metrics:
        print("模型内置指标:")
        for k, v in model_metrics.items():
            print(f"  {k}: {v:.6f}")

    results = model.results
    required_keys = ["y_pred", "y_target", "time"]
    for k in required_keys:
        if k not in results:
            raise KeyError(f"results 中缺少关键字段: {k}")

    y_pred = np.asarray(results["y_pred"])
    y_target = np.asarray(results["y_target"])
    time_arr = np.asarray(results["time"]).reshape(-1)
    missing_target = np.asarray(results["missing_target"]) if "missing_target" in results else None

    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_target shape: {y_target.shape}")
    print(f"time shape: {time_arr.shape}")
    if missing_target is not None:
        print(f"missing_target shape: {missing_target.shape}")

    print("[4/6] 构建站点映射")
    y_pred_3d = to_numpy_3d(y_pred, "y_pred")
    target_station_count = y_pred_3d.shape[1]

    test_nodes, mapping_source = load_test_nodes(analysis_args, opt, dataset, target_station_count)
    max_node_index = int(max(np.max(test_nodes) if len(test_nodes) > 0 else 0, target_station_count - 1))
    station_ids, station_meta_source = load_station_id_table(analysis_args, opt, dataset, max_node_index)
    station_mapping = build_target_station_mapping(test_nodes, station_ids, target_station_count)

    output_dir = analysis_args.output_dir.strip() if analysis_args.output_dir else os.path.join(model.save_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, analysis_args.csv_name)
    station_metrics_path = os.path.join(output_dir, analysis_args.station_metrics_name)
    trend_plot_path = os.path.join(output_dir, analysis_args.trend_plot_name)
    error_plot_path = os.path.join(output_dir, analysis_args.error_plot_name)
    report_path = os.path.join(output_dir, analysis_args.report_name)

    print("[5/6] 生成结构化 CSV 与统计结果")
    summary, station_metrics_df, sample_df, count_info = generate_csv_and_stats(
        y_pred=y_pred,
        y_target=y_target,
        time_arr=time_arr,
        missing_target=missing_target,
        station_mapping=station_mapping,
        csv_path=csv_path,
        analysis_args=analysis_args,
    )

    station_metrics_df.to_csv(station_metrics_path, index=False, encoding="utf-8")

    print("总体统计:")
    print(f"  MAE : {summary['overall_MAE']:.6f}")
    print(f"  RMSE: {summary['overall_RMSE']:.6f}")
    print(f"  R2  : {summary['overall_R2']:.6f}")
    print(f"  保留样本: {count_info['kept']}")
    print(f"  过滤样本: {count_info['dropped']}")

    print("[6/6] 绘图并输出报告")
    selected_stations = []
    if not station_metrics_df.empty:
        selected_stations = station_metrics_df.sort_values("sample_count", ascending=False)["station_id"].head(analysis_args.trend_top_k_stations).astype(str).tolist()

    trend_df = collect_trend_data_from_csv(
        csv_path=csv_path,
        station_ids=selected_stations,
        max_points_per_station=analysis_args.max_trend_points_per_station,
    )
    generate_trend_plot(trend_df, trend_plot_path, analysis_args.trend_top_k_stations, use_cn_font)
    generate_error_plots(sample_df, station_metrics_df, error_plot_path, use_cn_font)

    write_report(
        report_path=report_path,
        station_metrics_path=station_metrics_path,
        opt=opt,
        summary=summary,
        count_info=count_info,
        model_metrics=model_metrics,
        station_metrics_df=station_metrics_df,
        mapping_source=mapping_source,
        station_meta_source=station_meta_source,
        csv_path=csv_path,
        trend_plot_path=trend_plot_path,
        error_plot_path=error_plot_path,
        start_time=start,
    )

    print("=" * 80)
    print("流程完成")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
