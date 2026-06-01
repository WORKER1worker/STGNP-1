#!/usr/bin/env python3
"""Aggregate sensor strategy experiments and generate 4.1 conclusion draft."""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd


SENSOR_RATIO_MAP = {
    5: "1:6",
    10: "2:5",
    15: "3:4",
    20: "4:3",
    25: "5:2",
    30: "6:1",
}

SENSOR_RATIO_ORDER = ["1:6", "2:5", "3:4", "4:3", "5:2", "6:1"]
STRATEGY_ORDER = ["orig", "mts", "tts", "pmts"]
TARGET_SEEDS = 5


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def find_reports(base_dir: str) -> List[str]:
    results = []
    for root, _, files in os.walk(base_dir):
        if "analysis_report.txt" in files:
            results.append(os.path.join(root, "analysis_report.txt"))
    return results


def parse_from_name(name: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    lower = name.lower()
    strategy_match = re.search(r"(orig|mts|tts|pmts)", lower)
    strategy = strategy_match.group(1) if strategy_match else None

    n_obs = None
    n_obs_patterns = [
        r"(?:^|[_\-])obs(\d+)(?:[_\-]|$)",
        r"(?:^|[_\-])nobs(\d+)(?:[_\-]|$)",
        r"(?:^|[_\-])n_obs(\d+)(?:[_\-]|$)",
    ]
    for pat in n_obs_patterns:
        m = re.search(pat, lower)
        if m:
            n_obs = int(m.group(1))
            break

    seed = None
    seed_match = re.search(r"seed(\d+)", lower)
    if seed_match:
        seed = int(seed_match.group(1))

    return strategy, n_obs, seed


def load_marker_info(report_path: str) -> Dict[str, object]:
    candidates = [
        os.path.join(os.path.dirname(report_path), "strategy_ratio_info.json"),
        os.path.join(os.path.dirname(os.path.dirname(report_path)), "strategy_ratio_info.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                warn(f"Failed to parse marker JSON: {path}")
    return {}


def parse_run_info(report_path: str) -> Tuple[Optional[str], Optional[int], Optional[int], str]:
    parent = os.path.basename(os.path.dirname(report_path))
    strategy, n_obs, seed = parse_from_name(parent)
    if parent.lower() == "analysis":
        run_dir = os.path.dirname(os.path.dirname(report_path))
        parent = os.path.basename(run_dir)
        strategy, n_obs, seed = parse_from_name(parent)

    marker = load_marker_info(report_path)
    if marker:
        strategy = str(marker.get("strategy", strategy) or strategy).lower() if marker else strategy
        n_obs = int(marker.get("observed_count", n_obs)) if marker.get("observed_count") is not None else n_obs
        seed_val = marker.get("seed")
        seed = int(seed_val) if seed_val is not None else seed

    if strategy is None or n_obs is None or seed is None:
        warn(f"Failed to parse strategy/n_obs/seed from {parent} (path={report_path})")
    return strategy, n_obs, seed, os.path.dirname(report_path)


def extract_metrics(text: str) -> Tuple[float, float, float]:
    patterns = {
        "mae": [r"总体 MAE:\s*([0-9.+\-eE]+)", r"Overall MAE:\s*([0-9.+\-eE]+)"],
        "rmse": [r"总体 RMSE:\s*([0-9.+\-eE]+)", r"Overall RMSE:\s*([0-9.+\-eE]+)"],
        "r2": [r"总体 R2:\s*([0-9.+\-eE]+)", r"Overall R2:\s*([0-9.+\-eE]+)"],
    }

    def pick_value(pats: List[str]) -> float:
        for pat in pats:
            m = re.search(pat, text)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    break
        return float("nan")

    mae = pick_value(patterns["mae"])
    rmse = pick_value(patterns["rmse"])
    r2 = pick_value(patterns["r2"])
    return mae, rmse, r2


def ratio_from_nobs(n_obs: Optional[int]) -> str:
    if n_obs is None:
        return "UNKNOWN"
    return SENSOR_RATIO_MAP.get(n_obs, "UNKNOWN")


def safe_format(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NaN"
    return f"{val:.4f}"


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_progress_matrix(summary_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGY_ORDER:
        row = {"strategy": strategy}
        for ratio in SENSOR_RATIO_ORDER:
            mask = (summary_runs["strategy"] == strategy) & (summary_runs["sensor_ratio"] == ratio)
            seeds = summary_runs.loc[mask, "seed"].dropna().unique().tolist()
            count = len(seeds)
            if count < TARGET_SEEDS:
                cell = f"{count} MISSING(need>=5)"
            else:
                cell = str(count)
            row[ratio] = cell
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add totals
    total_row = {"strategy": "TOTAL"}
    for ratio in SENSOR_RATIO_ORDER:
        total = 0
        for strategy in STRATEGY_ORDER:
            mask = (summary_runs["strategy"] == strategy) & (summary_runs["sensor_ratio"] == ratio)
            total += summary_runs.loc[mask, "seed"].dropna().nunique()
        total_row[ratio] = str(total)
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    total_col = []
    for _, row in df.iterrows():
        if row["strategy"] == "TOTAL":
            total_col.append(str(summary_runs["seed"].dropna().nunique()))
            continue
        total = 0
        for ratio in SENSOR_RATIO_ORDER:
            mask = (summary_runs["strategy"] == row["strategy"]) & (summary_runs["sensor_ratio"] == ratio)
            total += summary_runs.loc[mask, "seed"].dropna().nunique()
        total_col.append(str(total))
    df["TOTAL"] = total_col
    return df


def print_progress(summary_by_cell: pd.DataFrame, summary_runs: pd.DataFrame) -> None:
    red = "\033[31m"
    reset = "\033[0m"
    print("\n=== Progress by cell ===")
    for strategy in STRATEGY_ORDER:
        for ratio in SENSOR_RATIO_ORDER:
            mask = (summary_runs["strategy"] == strategy) & (summary_runs["sensor_ratio"] == ratio)
            seeds_done = summary_runs.loc[mask, "seed"].dropna().nunique()

            cell = summary_by_cell[
                (summary_by_cell["strategy"] == strategy) &
                (summary_by_cell["sensor_ratio"] == ratio)
            ]
            if not cell.empty:
                mae_mean = float(cell.iloc[0].get("mae_mean", float("nan")))
                mae_std = float(cell.iloc[0].get("mae_std", float("nan")))
            else:
                mae_mean = float("nan")
                mae_std = float("nan")

            text = (
                f"{strategy} {ratio}: {seeds_done}/{TARGET_SEEDS} | "
                f"MAE {safe_format(mae_mean)} ± {safe_format(mae_std)}"
            )
            if not math.isnan(mae_std) and mae_std > 0.01:
                text = f"{red}{text}{reset}"
            print(text)


def betacf(a: float, b: float, x: float) -> float:
    maxit = 200
    eps = 3.0e-7
    fpmin = 1.0e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delh = d * c
        h *= delh
        if abs(delh - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    if t >= 0:
        return 1.0 - 0.5 * betai(a, b, x)
    return 0.5 * betai(a, b, x)


def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    a = [x for x in a if x is not None and not math.isnan(x)]
    b = [x for x in b if x is not None and not math.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)

    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return float("nan"), float("nan"), float("nan")

    t_stat = (mean_a - mean_b) / se
    df_num = (var_a / len(a) + var_b / len(b)) ** 2
    df_den = (var_a ** 2) / (len(a) ** 2 * (len(a) - 1)) + (var_b ** 2) / (len(b) ** 2 * (len(b) - 1))
    if df_den == 0:
        return float("nan"), float("nan"), float("nan")
    df = df_num / df_den

    cdf = t_cdf(abs(t_stat), df)
    p_value = 2.0 * (1.0 - cdf)
    return t_stat, df, p_value


def build_completeness_table(summary_runs: pd.DataFrame) -> str:
    header = "| Strategy | " + " | ".join(SENSOR_RATIO_ORDER) + " |\n"
    header += "|---" + "|---" * len(SENSOR_RATIO_ORDER) + "|\n"
    lines = [header]
    for strategy in STRATEGY_ORDER:
        row = [strategy]
        for ratio in SENSOR_RATIO_ORDER:
            mask = (summary_runs["strategy"] == strategy) & (summary_runs["sensor_ratio"] == ratio)
            seeds = summary_runs.loc[mask, "seed"].dropna().nunique()
            cell = f"{seeds}"
            if seeds < TARGET_SEEDS:
                cell += " 数据不足"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |\n")
    return "".join(lines)


def conclusion_strategy_compare(summary_runs: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, float]], Dict[str, str]]:
    results = {}
    winner = {}
    sig_vs_tts = 0
    sig_vs_mts_higher = 0
    valid_ratios = 0
    all_small_diff = True

    for ratio in SENSOR_RATIO_ORDER:
        pmts_raw = summary_runs[(summary_runs["strategy"] == "pmts") & (summary_runs["sensor_ratio"] == ratio)]["mae"].tolist()
        tts_raw = summary_runs[(summary_runs["strategy"] == "tts") & (summary_runs["sensor_ratio"] == ratio)]["mae"].tolist()
        mts_raw = summary_runs[(summary_runs["strategy"] == "mts") & (summary_runs["sensor_ratio"] == ratio)]["mae"].tolist()

        pmts = [x for x in pmts_raw if x is not None and not math.isnan(x)]
        tts = [x for x in tts_raw if x is not None and not math.isnan(x)]
        mts = [x for x in mts_raw if x is not None and not math.isnan(x)]

        if len(pmts) < TARGET_SEEDS:
            continue
        if len(tts) < TARGET_SEEDS:
            continue
        if len(mts) < TARGET_SEEDS:
            continue

        valid_ratios += 1

        t_stat_tts, df_tts, p_tts = welch_ttest(pmts, tts)
        delta_tts = (sum(pmts) / len(pmts)) - (sum(tts) / len(tts))
        t_stat_mts, df_mts, p_mts = welch_ttest(pmts, mts)
        delta_mts = (sum(pmts) / len(pmts)) - (sum(mts) / len(mts))

        results[ratio] = {
            "delta_tts": delta_tts,
            "p_tts": p_tts,
            "df_tts": df_tts,
            "delta_mts": delta_mts,
            "p_mts": p_mts,
            "df_mts": df_mts,
        }

        if p_tts < 0.05 and delta_tts < 0:
            sig_vs_tts += 1
        if p_mts < 0.05 and delta_mts > 0:
            sig_vs_mts_higher += 1
        if not (abs(delta_mts) < 0.002 and p_mts > 0.1):
            all_small_diff = False

        means = {
            "pmts": sum(pmts) / len(pmts),
            "tts": sum(tts) / len(tts),
            "mts": sum(mts) / len(mts),
        }
        winner[ratio] = min(means, key=means.get)

    conclusion = "策略效果依赖比例，需个例讨论"
    if valid_ratios >= 4 and sig_vs_tts >= 4 and sig_vs_mts_higher == 0:
        conclusion = "PMTS 显著优于 TTS、不差于 MTS，采纳为 4.2–4.5 主训练策略。"
    elif valid_ratios > 0 and all_small_diff:
        conclusion = "PMTS 与 MTS 差异不显著，正文保留 PMTS，MTS 放补充材料。"

    return conclusion, results, winner


def conclusion_best_ratio(summary_by_cell: pd.DataFrame) -> Tuple[str, str, List[Tuple[str, float, float]]]:
    pmts = summary_by_cell[summary_by_cell["strategy"] == "pmts"].copy()
    pmts = pmts[pmts["sensor_ratio"].isin(SENSOR_RATIO_ORDER)]

    rows = []
    for ratio in SENSOR_RATIO_ORDER:
        row = pmts[pmts["sensor_ratio"] == ratio]
        if row.empty:
            continue
        seeds = int(row.iloc[0].get("n_seeds", 0))
        if seeds < TARGET_SEEDS:
            continue
        rows.append((
            ratio,
            float(row.iloc[0].get("mae_median", float("nan"))),
            float(row.iloc[0].get("mae_std", float("nan"))),
        ))

    rows = [r for r in rows if not math.isnan(r[1])]
    rows.sort(key=lambda x: x[1])

    if not rows:
        return "PMTS 配置数据不足，无法确定最佳比例。", "", []

    best = rows[0]
    second = rows[1] if len(rows) > 1 else None

    conclusion = ""
    if second is not None:
        diff = second[1] - best[1]
        overlap_check = diff > 0.5 * (best[2] + second[2])
        if diff >= 0.002 and overlap_check:
            conclusion = f"采纳 `{best[0]}` 为主传感器配置。"
        else:
            conclusion = (
                f"Top-2 候选：{best[0]} 与 {second[0]}；"
                "二者差异在误差棒以内，4.2 之前需补 seed。"
            )
    else:
        conclusion = f"采纳 `{best[0]}` 为主传感器配置。"

    return conclusion, best[0], rows


def load_framework(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find framework file: {path}")
    bullets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                bullets.append(line)
    if len(bullets) < 3:
        raise ValueError("Framework file must contain at least 3 bullet lines")
    return bullets[:3]


def map_framework(bullets: List[str], strategy_supported: bool, ratio_supported: bool) -> List[str]:
    out = []
    for b in bullets:
        text = b
        lower = b.lower()
        if any(k in lower for k in ["pmts", "tts", "mts", "策略"]):
            if strategy_supported:
                out.append(text)
            else:
                out.append("- 不成立/暂无证据")
        elif any(k in lower for k in ["比例", "传感器", "配置", "ratio", "sensor"]):
            if ratio_supported:
                out.append(text)
            else:
                out.append("- 不成立/暂无证据")
        else:
            out.append("- 不成立/暂无证据")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 4.1 experiments")
    parser.add_argument("--base-dir", type=str, default="checkpoints/SM_trainingStrategy/SM",
                        help="Base directory to scan")
    parser.add_argument("--output-dir", type=str, default="experiments/training_strategy_ratio",
                        help="Output directory for CSV/MD")
    parser.add_argument("--framework-path", type=str, default="experiments/training_strategy_ratio/preset_conclusion_framework_4_1.md",
                        help="Path to preset conclusion framework (3 bullets)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    reports = find_reports(base_dir)
    if not reports:
        warn(f"No analysis_report.txt found under {base_dir}")

    rows = []
    for report in reports:
        try:
            text = open(report, "r", encoding="utf-8").read()
        except Exception:
            text = open(report, "r", errors="ignore").read()

        strategy, n_obs, seed, run_dir = parse_run_info(report)
        sensor_ratio = ratio_from_nobs(n_obs)
        mae, rmse, r2 = extract_metrics(text)

        if math.isnan(mae) or math.isnan(rmse) or math.isnan(r2):
            warn(f"Missing metrics in {report}")

        rows.append({
            "strategy": strategy or "UNKNOWN",
            "sensor_ratio": sensor_ratio,
            "n_obs": n_obs,
            "seed": seed,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "run_path": run_dir,
        })

    summary_runs = pd.DataFrame(rows)
    summary_runs_csv = os.path.join(output_dir, "summary_runs.csv")
    write_csv(summary_runs, summary_runs_csv)

    grouped = summary_runs.groupby(["strategy", "sensor_ratio"], dropna=False)
    summary_by_cell = grouped.agg(
        n_seeds=("seed", lambda x: x.dropna().nunique()),
        mae_mean=("mae", "mean"),
        mae_std=("mae", lambda x: x.std(ddof=0)),
        mae_median=("mae", "median"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", lambda x: x.std(ddof=0)),
        r2_mean=("r2", "mean"),
        r2_std=("r2", lambda x: x.std(ddof=0)),
    ).reset_index()

    summary_by_cell_csv = os.path.join(output_dir, "summary_by_cell.csv")
    write_csv(summary_by_cell, summary_by_cell_csv)

    progress_matrix = build_progress_matrix(summary_runs)
    progress_matrix_csv = os.path.join(output_dir, "progress_matrix.csv")
    write_csv(progress_matrix, progress_matrix_csv)

    print_progress(summary_by_cell, summary_runs)

    # Conclusion draft
    summary_runs = pd.read_csv(summary_runs_csv)
    summary_by_cell = pd.read_csv(summary_by_cell_csv)

    completeness_table = build_completeness_table(summary_runs)
    conclusion_1, ttest_results, winners = conclusion_strategy_compare(summary_runs)
    conclusion_2, best_ratio, ranking_rows = conclusion_best_ratio(summary_by_cell)

    strategy_supported = len(ttest_results) > 0
    ratio_supported = len(ranking_rows) > 0

    framework_path = os.path.abspath(args.framework_path)
    mapped_bullets: List[str] = []
    if os.path.isfile(framework_path):
        bullets = load_framework(framework_path)
        mapped_bullets = map_framework(bullets, strategy_supported, ratio_supported)
    else:
        warn(f"Framework file not found, skip mapping: {framework_path}")

    ttest_lines = []
    for ratio in SENSOR_RATIO_ORDER:
        if ratio not in ttest_results:
            continue
        res = ttest_results[ratio]
        ttest_lines.append(
            f"- {ratio}: ΔMAE(PMTS-TTS)={res['delta_tts']:.4f}, p={res['p_tts']:.4f}; "
            f"ΔMAE(PMTS-MTS)={res['delta_mts']:.4f}, p={res['p_mts']:.4f}"
        )

    ranking_table = "| Ratio | mae_median | mae_std |\n|---|---|---|\n"
    for ratio, median, std in ranking_rows:
        ranking_table += f"| {ratio} | {median:.4f} | {std:.4f} |\n"

    missing_cells = []
    for strategy in STRATEGY_ORDER:
        for ratio in SENSOR_RATIO_ORDER:
            mask = (summary_runs["strategy"] == strategy) & (summary_runs["sensor_ratio"] == ratio)
            seeds = summary_runs.loc[mask, "seed"].dropna().nunique()
            if seeds < TARGET_SEEDS:
                missing_cells.append(f"{strategy} {ratio}")

    high_std_cells = []
    for _, row in summary_by_cell.iterrows():
        std = row.get("mae_std")
        if pd.notna(std) and float(std) > 0.01:
            high_std_cells.append(f"{row.get('strategy')} {row.get('sensor_ratio')}")

    draft_path = os.path.join(output_dir, "4_1_conclusion_draft.md")
    with open(draft_path, "w", encoding="utf-8") as f:
        f.write("## 数据完整性\n")
        f.write(completeness_table + "\n")

        f.write("## 结论 1：策略对比\n")
        f.write(conclusion_1 + "\n\n")
        if ttest_lines:
            f.write("**Welch t-test 结果**\n")
            f.write("\n".join(ttest_lines) + "\n\n")

        f.write("## 结论 2：最佳传感器配置\n")
        f.write(conclusion_2 + "\n\n")
        if ranking_rows:
            f.write("**PMTS 比例排序（mae_median）**\n")
            f.write(ranking_table + "\n")

        f.write("## 对照预设结论框架\n")
        if mapped_bullets:
            for line in mapped_bullets:
                f.write(line + "\n")
        else:
            f.write("- 未提供预设结论框架，跳过对照\n")
        f.write("\n")

        f.write("## 下一步建议\n")
        if missing_cells:
            f.write("- 仍需补跑：" + ", ".join(missing_cells) + "\n")
        else:
            f.write("- 仍需补跑：无\n")
        if high_std_cells:
            f.write("- 潜在风险（std > 0.01）：" + ", ".join(high_std_cells) + "\n")
        else:
            f.write("- 潜在风险（std > 0.01）：无\n")

    print(f"\nWrote summary_runs.csv -> {summary_runs_csv}")
    print(f"Wrote summary_by_cell.csv -> {summary_by_cell_csv}")
    print(f"Wrote progress_matrix.csv -> {progress_matrix_csv}")
    print(f"Wrote 4_1_conclusion_draft.md -> {draft_path}")
    print("\n=== Conclusions (4.1 draft) ===")
    print(conclusion_1)
    print(conclusion_2)


if __name__ == "__main__":
    main()
