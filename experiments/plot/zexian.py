"""
Figure 6: 不同训练策略 (TTS / PMTS K=1 / orig) 在 6 个传感器配置比例下的
未监测站点 MAE 对比（5 个随机种子，误差棒 = ±1 std）

参考 He et al. (2025), Journal of Hydrology 652, 132681, Fig. 6
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

# ============== 1. 数据加载 ==============
CSV_PATH = "../training_strategy_ratio/summary_by_cell.csv"   # ← 改成你的实际路径
df = pd.read_csv(CSV_PATH)

# 只保留 3 种策略；如果你的 csv 里 PMTS K=1 的策略名不是 "pmts"，请相应改写
KEEP = {
    "tts":  "TTS",
    "pmts": "PMTS (K = 1)",
    "orig": "orig",
}
df = df[df["strategy"].isin(KEEP.keys())].copy()
df["strategy_label"] = df["strategy"].map(KEEP)

# 比例顺序：1:6 → 6:1
RATIO_ORDER = ["1:6", "2:5", "3:4", "4:3", "5:2", "6:1"]
df["ratio"] = pd.Categorical(df["sensor_ratio"], categories=RATIO_ORDER, ordered=True)

# summary_by_cell.csv 已经是聚合结果
agg = df[["strategy_label", "ratio", "mae_mean", "mae_std"]].copy()
agg = agg.rename(columns={"mae_mean": "mean", "mae_std": "std"})

# ============== 2. 绘图样式（贴近 He 2025 论文风格） ==============
mpl.rcParams.update({
    "font.family": "DejaVu Sans",      # 中文可换成 "Microsoft YaHei" / "SimHei"
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# He 2025 风格的三色：稳重蓝 / 暖橙 / 深绿（orig 用最显眼色）
STYLE = {
    "TTS":          dict(color="#7f7f7f", marker="s", linestyle="--", label="TTS"),
    "PMTS (K = 1)": dict(color="#f0a500", marker="^", linestyle="-",  label="PMTS (K = 1)"),
    "orig":         dict(color="#1f77b4", marker="o", linestyle="-",  label="orig (selected)"),
}
DRAW_ORDER = ["TTS", "PMTS (K = 1)", "orig"]   # orig 最后画 → 在最上层

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)

x = np.arange(len(RATIO_ORDER))
offset = {"TTS": -0.10, "PMTS (K = 1)": 0.0, "orig": 0.10}  # 微抖避免误差棒重叠

for strat in DRAW_ORDER:
    sub = agg[agg["strategy_label"] == strat].sort_values("ratio")
    s = STYLE[strat]
    ax.errorbar(
        x + offset[strat],
        sub["mean"].values,
        yerr=sub["std"].values,
        marker=s["marker"], markersize=6.5,
        linestyle=s["linestyle"], linewidth=1.6,
        color=s["color"], ecolor=s["color"],
        capsize=3, capthick=1.0, elinewidth=1.0,
        label=s["label"], zorder=3 if strat == "orig" else 2,
    )

# ============== 3. 高亮主设置 (orig + 4:3) ==============
orig_43 = agg[(agg.strategy_label == "orig") & (agg.ratio == "4:3")]
if len(orig_43):
    y_star = orig_43["mean"].values[0]
    ax.scatter([3 + offset["orig"]], [y_star],
               s=180, facecolor="none", edgecolor="#d62728",
               linewidth=2.0, zorder=5, label="downstream main setting")
    ax.annotate(
        "orig + 4:3\n(selected)",
        xy=(3 + offset["orig"], y_star),
        xytext=(3.45, y_star + 0.012),
        fontsize=9, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )

# ============== 4. 轴 / 标题 / 图例 ==============
ax.set_xticks(x)
ax.set_xticklabels(RATIO_ORDER)
ax.set_xlabel("Sensor configuration ratio (monitored : unmonitored)")
ax.set_ylabel("MAE on unmonitored stations")
ax.set_ylim(bottom=0.03)        # 视数据情况微调；He 2025 也手动收紧了 y 轴
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

# 图例放图内右上（He 2025 是这样）
leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5)

# 可选：用 panel letter 标 "(a) MAE"，方便后续做组合图
ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom")

plt.tight_layout()
plt.savefig("fig6_strategy_vs_ratio.pdf", bbox_inches="tight")
plt.savefig("fig6_strategy_vs_ratio.png", bbox_inches="tight", dpi=300)
plt.show()