# Holdout 比例实验总结 (2026-05-07)

## 实验说明
本实验使用 SM_NQ 数据集的 35 个站点，按照不同的 holdout 比例（0.1/0.2/0.3/0.5/0.8）
随机选取对应比例的站点作为未知站点（holdout），其余站点全部用于训练。
训练完成后，使用 holdout 模式对未知站点进行推断，并统计总体 MAE、RMSE、R2 指标。

## 汇总表
| Holdout 比例 | Checkpoint | 总体 MAE | 总体 RMSE | 总体 R2 |
| --- | --- | --- | --- | --- |
| 0.1 | checkpoints/SM/hierarchical_SM_20260507T142853 | 0.029455 | 0.037476 | 0.832191 |
| 0.2 | checkpoints/SM/hierarchical_SM_20260507T144943 | 0.038021 | 0.047019 | 0.836521 |
| 0.3 | checkpoints/SM/hierarchical_SM_20260507T150939 | 0.046065 | 0.055713 | 0.764259 |
| 0.5 | checkpoints/SM/hierarchical_SM_20260507T153005 | 0.059372 | 0.078403 | 0.617025 |
| 0.8 | checkpoints/SM/hierarchical_SM_20260507T154953 | 0.065003 | 0.078552 | 0.596762 |

## 备注
- Holdout 比例表示从 35 个站点中被选为未知站点的比例。
- 指标来自各 checkpoint 的 analysis_report.txt。

---

# Holdout 比例实验补充总结（2026-05-12，4h 时间窗口）

## 实验步骤
1. 固定时间窗口为 4h（t_len=8，30min 数据）。
2. 采用随机比例剔除（holdout ratio=0.1~0.6），每个比例重复多次 seed。
3. 每次实验严格 holdout：未知站点不参与训练/图构建，仅在推断时动态加入。
4. 训练完成后执行 holdout 推断，统计总体 MAE/RMSE/R2。
5. 汇总多次实验的 mean/std，并计算站点重要性（移除导致整体 MAE 增量）。

## 实验设置
- 模型: hierarchical（SM_config1）
- 时间窗口: 4h（t_len=8）
- 比例: 0.1/0.2/0.3/0.4/0.5/0.6
- Seeds: 3,4,5,7,3407 + 11,12,13,17,19（两轮合计 10 seeds）
- 总运行次数: 6 ratios × 10 seeds = 60
- 输出目录:
	- 第一轮: experiments/ratio_holdout_multi/20260511T131138/
	- 第二轮: experiments/ratio_holdout_multi/20260512T094505/

## 总体结果（合并 60 次，mean ± std）
| Holdout 比例 | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| 0.1 | 0.0523 ± 0.0082 | 0.0699 ± 0.0128 | 0.6565 ± 0.1498 |
| 0.2 | 0.0441 ± 0.0086 | 0.0567 ± 0.0097 | 0.7265 ± 0.0863 |
| 0.3 | 0.0512 ± 0.0093 | 0.0678 ± 0.0129 | 0.6708 ± 0.1059 |
| 0.4 | 0.0549 ± 0.0066 | 0.0726 ± 0.0084 | 0.6538 ± 0.0934 |
| 0.5 | 0.0536 ± 0.0067 | 0.0722 ± 0.0094 | 0.6271 ± 0.0686 |
| 0.6 | 0.0535 ± 0.0043 | 0.0717 ± 0.0063 | 0.6640 ± 0.0380 |

## 关键站点（全局排名，按 ΔMAE）
定义: ΔMAE = mean(MAE | 站点被移除) − mean(MAE | 站点未移除)。

Top 10:
1. MS3482 (ΔMAE=0.0075)
2. MS3527 (ΔMAE=0.0054)
3. BC02   (ΔMAE=0.0050)
4. P2     (ΔMAE=0.0050)
5. BC08   (ΔMAE=0.0045)
6. BC03   (ΔMAE=0.0042)
7. MS3488 (ΔMAE=0.0041)
8. MS3501 (ΔMAE=0.0037)
9. CD01   (ΔMAE=0.0034)
10. F3    (ΔMAE=0.0028)

备注:
- 低贡献/可能冗余站点（ΔMAE<0）主要包括: MS3523, MSNQRW, MS3614, BC07, BC05。
- 若需更严格的结论，可对 Top 10 站点做 LOSO 精排验证。

## 站点重要性（RMSE / R2）
定义:
- ΔRMSE = mean(RMSE | 站点被移除) − mean(RMSE | 站点未移除)
- ΔR2 = mean(R2 | 站点未移除) − mean(R2 | 站点被移除)

RMSE Top 10:
1. MS3482 (ΔRMSE=0.0128)
2. BC02   (ΔRMSE=0.0091)
3. MS3527 (ΔRMSE=0.0077)
4. MS3501 (ΔRMSE=0.0062)
5. P2     (ΔRMSE=0.0062)
6. BC08   (ΔRMSE=0.0059)
7. BC03   (ΔRMSE=0.0049)
8. F3     (ΔRMSE=0.0048)
9. MS3488 (ΔRMSE=0.0046)
10. CD01  (ΔRMSE=0.0043)

R2 Top 10:
1. MS3527 (ΔR2=0.0502)
2. MS3576 (ΔR2=0.0492)
3. MS3482 (ΔR2=0.0431)
4. MS3488 (ΔR2=0.0415)
5. MS3620 (ΔR2=0.0410)
6. BC08   (ΔR2=0.0410)
7. BC03   (ΔR2=0.0409)
8. C1     (ΔR2=0.0399)
9. P2     (ΔR2=0.0397)
10. MSBJ  (ΔR2=0.0378)

---

# Top-K Key Stations Only 实验总结（2026-05-13）

## 实验步骤
1. 使用全局站点重要性排序（merged_station_importance_summary.csv），提取 Top-K 站点。
2. 仅保留 Top-K 站点作为观测站点，其余站点全部作为 holdout。
3. 严格 holdout：未知站点不参与训练/图构建，仅在推断阶段动态加入。
4. 对每个 K 重复多 seed 训练与推断，统计 MAE/RMSE/R2 的 mean/std。
5. 输出 Top-K 对比曲线（MAE/RMSE/R2）。

## 实验设置
- K 值: 2 / 4 / 8 / 10 / 14
- Seeds: 3, 4, 5
- 模型: hierarchical（SM_config1）
- 时间窗口: 4h（t_len=8）
- 重要性排序来源:
	- [experiments/ratio_holdout_multi/merged_20260512T193215/merged_station_importance_summary.csv](experiments/ratio_holdout_multi/merged_20260512T193215/merged_station_importance_summary.csv)
- 输出目录:
	- [experiments/topk_key_stations/20260512T202907](experiments/topk_key_stations/20260512T202907)

## 结果汇总（mean ± std）
| Top-K | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| 2  | 0.0924 ± 0.0066 | 0.1177 ± 0.0066 | 0.0252 ± 0.1100 |
| 4  | 0.1053 ± 0.0015 | 0.1368 ± 0.0051 | -0.3913 ± 0.1049 |
| 8  | 0.0682 ± 0.0036 | 0.0914 ± 0.0073 | 0.3700 ± 0.0987 |
| 10 | 0.0704 ± 0.0079 | 0.0955 ± 0.0129 | 0.2185 ± 0.2009 |
| 14 | 0.0588 ± 0.0084 | 0.0714 ± 0.0083 | 0.5654 ± 0.1007 |

## 结论
- Top-2 / Top-4 明显不足，R2 很低甚至为负，无法支撑全局空间推断。
- Top-8 开始具备可用性，误差显著下降，但稳定性仍有波动。
- Top-14 达到当前 Top-K 实验中最优整体表现。

## 参考图表
- [Top-K vs MAE](experiments/topk_key_stations/20260512T202907/plots/topk_vs_mae.png)
- [Top-K vs RMSE](experiments/topk_key_stations/20260512T202907/plots/topk_vs_rmse.png)
- [Top-K vs R2](experiments/topk_key_stations/20260512T202907/plots/topk_vs_r2.png)

---

# Top-K Key Stations Only（排除 self-error 排名）实验总结（2026-05-13）

## 实验步骤
1. 使用“排除 self-error”的贡献度排名作为排序依据（contribution_ranking.csv）。
2. 依次选择 Top-2/4/8/10/14 站点作为训练观测站点，其余全部 holdout。
3. 固定 4h 时间窗口（t_len=8），多 seed 训练与推断。
4. 汇总 MAE/RMSE/R2 的 mean/std，并输出 Top-K 曲线。

## 实验设置
- K 值: 2 / 4 / 8 / 10 / 14
- Seeds: 3, 4, 5
- 模型: hierarchical（SM_config1）
- 时间窗口: 4h（t_len=8）
- 排名来源:
	- [experiments/ratio_holdout_multi/contribution_20260513T152103/contribution_ranking.csv](experiments/ratio_holdout_multi/contribution_20260513T152103/contribution_ranking.csv)
- 输出目录:
	- [experiments/topk_key_stations/20260513T155225](experiments/topk_key_stations/20260513T155225)

## 结果汇总（mean ± std）
| Top-K | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| 2  | 0.0884 ± 0.0023 | 0.1068 ± 0.0028 | 0.2553 ± 0.0391 |
| 4  | 0.0679 ± 0.0006 | 0.0954 ± 0.0014 | 0.4169 ± 0.0167 |
| 8  | 0.0652 ± 0.0004 | 0.0876 ± 0.0011 | 0.5342 ± 0.0112 |
| 10 | 0.0594 ± 0.0002 | 0.0798 ± 0.0005 | 0.6214 ± 0.0051 |
| 14 | 0.0600 ± 0.0006 | 0.0815 ± 0.0005 | 0.6052 ± 0.0047 |

## 结论
- 采用“排除 self-error”的贡献度排名后，Top-K 曲线更平滑。
- Top-10 达到本轮最优整体表现，Top-14 略有回落但差异不大。

## 参考图表
- [Top-K vs MAE](experiments/topk_key_stations/20260513T155225/plots/topk_vs_mae.png)
- [Top-K vs RMSE](experiments/topk_key_stations/20260513T155225/plots/topk_vs_rmse.png)
- [Top-K vs R2](experiments/topk_key_stations/20260513T155225/plots/topk_vs_r2.png)

## 追加实验（K=12/16）
补充目的: 在 Top-10 附近继续采样，验证是否进入平台区间。

实验设置:
- K 值: 12 / 16
- Seeds: 3, 4, 5
- 时间窗口: 4h（t_len=8）
- 输出目录:
	- [experiments/topk_key_stations/20260515T151231](experiments/topk_key_stations/20260515T151231)

结果汇总（mean ± std）:
| Top-K | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| 12 | 0.0607 ± 0.0012 | 0.0817 ± 0.0007 | 0.6094 ± 0.0066 |
| 16 | 0.0602 ± 0.0006 | 0.0824 ± 0.0012 | 0.6046 ± 0.0118 |

结论补充:
- K=12/16 未超过 K=10 的整体表现，说明 Top-10 已接近最优。

---

# 贡献度 Top-10 单次验证（2026-05-13）

## 实验步骤
1. 使用“排除 self-error”的贡献度排名 Top-10 作为训练站点。
2. 其余 25 个站点全部 holdout，严格 zero-participation。
3. 固定 4h 时间窗口（t_len=8），seed=3。
4. 训练 + holdout 推断，输出总体指标与分站点指标。

## 训练站点（Top-10 by contribution）
MS3501, P2, MS3603, MS3620, MS3513, C1, MS3627, F5, BC06, P10

## 总体指标
- MAE: 0.059286
- RMSE: 0.079138
- R2: 0.627211

## 产物路径
- [analysis_report.txt](checkpoints/SM_keySite/SM/hierarchical_SM_20260513T152831/analysis/analysis_report.txt)

---

# Top-10 Key Stations Only 单次验证（2026-05-12）

## 实验步骤
1. 选取 Top-10 关键站点作为训练观测站点，其余 25 个站点全部 holdout。
2. 固定 4h 时间窗口（t_len=8），seed=3。
3. 训练 + holdout 推断，输出总体指标与分站点指标。

## 训练站点（Top-10）
MS3482, MS3527, BC08, P2, BC03, MS3488, BC02, MS3501, MS3576, BC04

## 总体指标
- MAE: 0.057276
- RMSE: 0.075872
- R2: 0.573174

## 产物路径
- [analysis_report.txt](checkpoints/SM_keySite/SM/hierarchical_SM_20260512T194832/analysis/analysis_report.txt)
