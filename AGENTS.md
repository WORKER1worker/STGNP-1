# STGNP 土壤水分重建项目接手说明

本文档用于让后续新对话中的 AI 或协作者快速理解当前项目。请优先阅读本文件，再阅读代码和实验记录。

## 1. 项目位置与基本背景

- 当前本地工作区：`D:\文档\New project`
- 远程 AutoDL 项目路径：`/root/autodl-tmp/over/STGNP`
- 远程登录命令格式：`ssh -p 46476 root@connect.westc.seetacloud.com`
- 项目基础代码来自 KDD 2023 的 STGNP（Spatio-Temporal Graph Neural Processes for Spatio-Temporal Extrapolation）。
- 当前研究方向：将 STGNP 适配到稀疏土壤水分监测网络，进行未监测站点土壤水分重建。
- 当前论文主线建议：面向稀疏土壤水分监测网络的 STGNP 未监测站点重建与传感器配置优化。

注意：早期文档中有“24 个 Pali 站点、10 cm 深度”的适配说明，但当前主实验实际使用的是 NAQU 数据、5 cm 深度、筛选后的 35 个站点。写论文或设计新实验时，优先以当前 NAQU 35 站点设置为准。

## 2. 当前项目核心任务

当前任务不是普通时间序列预测，而是严格的未知站点重建：

1. 一部分站点作为观测站点参与训练。
2. 一部分站点作为未监测或 holdout 站点，在训练阶段完全不参与。
3. 推理阶段再把 holdout 站点动态加入图结构。
4. 利用观测站点的历史土壤水分序列和空间邻接关系，重建未监测站点的土壤水分变化。

这比普通 test-node 插值更严格，更接近真实的传感器未布设场景。

## 3. 当前主要数据设置

活跃实验数据：

- 数据集：SM_NQ / NAQU soil moisture
- 深度：5 cm
- 采样间隔：30 min
- 原始文件：`data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv`
- 站点元数据：`data/dataset/SM_NQ/Stations_information_NAQU.csv`
- 原始站点元数据约 56 行，原始时间记录约 195,793 条。
- 当前主实验使用筛选后的 35 个站点：
  - `dataset/SM_NQ_selected35/SM_NQ-30-minutes_05cm_selected35.csv`
  - `dataset/SM_NQ_selected35/Stations_information_NAQU_selected35.csv`

selected35 站点包括：

`BC02, BC03, BC04, BC05, BC06, BC07, BC08, C1, C2, CD01, CD03, CD07, F3, F4, F5, MS3482, MS3488, MS3494, MS3501, MS3513, MS3518, MS3523, MS3527, MS3533, MS3576, MS3603, MS3614, MS3620, MS3627, MSBJ, MSNQRW, P1, P10, P11, P2`

新增面状产品一致性实验数据：

- 数据目录：`dataset/NASA_STATION/`
- 站点/网格元数据：`NSIDC-0779_station_extent_smap_grid_station_info.csv`
- 土壤水分宽表：`NSIDC-0779_station_extent_20201004_20201010_smap_grid_soil_moisture_wide.csv`
- 网格伪站点数：6586
- 时间范围文件覆盖：2020-10-04 至 2020-10-10
- 当前已完成评价覆盖：2020-10-04 至 2020-10-09 六个有效时间步（`t_len=6`）
- 该实验用于“点尺度模型与 SMAP 面状土壤水分产品的一致性检验”，不要与 NAQU selected35 strict holdout 主实验混为同一类验证。

缺失值：

- 数据中 `-99.0` 表示缺失。
- `SM_dataset.py` 中将 `-99.0` 标记为 missing。
- 模型输入中，缺失值使用站点均值填充；若站点全缺失，使用全局均值兜底。
- 归一化统计量使用非缺失值计算。
- 评估时默认过滤 missing target。

## 4. 关键代码文件

### 数据与划分

- `data/SM_dataset.py`
  - 当前土壤水分数据集实现。
  - 读取站点坐标、土壤水分序列、缺失掩码。
  - 构建基于 Haversine 距离和高斯核的邻接矩阵。
  - 支持 `sm_eval_target_mode=test/holdout/all`。
  - 支持 strict holdout 动态追加。
  - 支持 `--sm_eval_full_time`，非训练阶段可使用完整时间范围。
  - 支持大规模面状/网格数据推理：
    - `--sm_large_graph_threshold`：超过阈值自动启用大图保护。
    - `--sm_adj_top_k`：构建 sparse top-k 空间邻接；默认大图使用 top-64。
    - `--sm_eval_target_chunk_size`：对大量 target 节点分块推理；默认大图自动使用 256。

- `data/base_dataset.py`
  - 通用数据窗口与 context/target 划分逻辑。
  - 训练阶段支持 TTS、MTS、PMTS 的 context/target 划分逻辑。
  - 支持 sparse adjacency 的 1-hop/2-hop 切片，避免大规模全连接矩阵 `A @ A`。
  - 支持 `all` 模式 target-node chunked inference，并在 batch 中返回 `target_node_index`。

- `data/dataset/build_sm_nq_split.py`
  - 生成 strict holdout split 产物。
  - 训练子集中会完全移除 holdout 站点。
  - 推理阶段再从完整数据源动态追加 holdout 站点。

### 训练与测试

- `train.py`
  - 主训练入口。
  - 加载 `TrainOptions`、dataset 和 model。
  - 支持 validation、best checkpoint 保存、自动生成 `run_test.sh`。

- `test.py`
  - 原始测试入口。

- `test_and_analyze.py`
  - 当前更重要的测试与分析入口。
  - 输出 `predictions_structured.csv`、`analysis_report.txt`、`station_metrics.csv`。
  - 计算总体和分站点 MAE、RMSE、R2。
  - 可绘制趋势图和误差图。

- `test_and_analyze_plot.py`
  - 当前 NASA/SMAP 面状数据实验使用的测试与分析入口。
  - 在 `all` + 大规模 target 情况下支持分块结果汇总。
  - 使用 `target_node_index` 将分块输出还原到原始网格/站点 ID。
  - 默认限制趋势图数量，避免 6000+ 网格生成过多图片。

### 实验脚本

- `run_time_window_ablation.py`
  - 时间窗口消融实验。
  - 支持 strict holdout、滑动窗口评估、稳定性指标和极端变化指标。

- `run_holdout_ratio_experiment.py`
  - holdout 比例实验旧版/相关脚本。

- `run_ratio_experiment_ms3603.py`
  - 特定站点/比例相关实验脚本。

- `run_topk_key_stations_experiment.py`
  - Top-K 关键站点实验。
  - 只保留贡献度排名前 K 的站点作为观测站点，其余全部 holdout。

- `analyze_station_contribution.py`
  - 站点贡献度分析。

- `plot_station_metric_map.py`
  - 站点指标空间图。

- `plot_station_metric_map_aggregate.py`
  - 聚合版本空间图。

- `plot_window_vs_mae.py`
  - 时间窗口结果绘图。

### 配置

- `model_configurations/hierarchical_SM_config.yaml`
  - 土壤水分专用配置。

- `model_configurations/hierarchical_config.yaml`
  - hierarchical/STGNP 原始或整合配置。

当前常用配置 `SM_config1`：

- `tcn_channels: [32, 64]`
- `latent_channels: [16, 32]`
- `emd_channel: 16`
- `num_latent_layers: 1`
- `observation_hidden_dim: 128`
- `num_observation_layers: 3`
- `tcn_kernel_size: 3`
- `dropout: 0.1`

## 5. 模型实现要点

当前使用 `hierarchical` 模型，即 STGNP 的层次化 neural process 实现。

关键文件：

- `models/hierarchical/hierarchical_model.py`
- `models/hierarchical/inference_model.py`
- `models/hierarchical/likelihood_model.py`
- `models/hierarchical/st_encoding.py`

模型流程：

1. `SM_dataset.py` 生成 context/target 站点的时间窗口数据。
2. `base_dataset.py` 生成 1-hop 和 2-hop 邻接矩阵：
   - `A_1hop = A[target, context]`
   - `A_2hop = A^2[target, context]`
3. `HierarchicalNP` 使用：
   - deterministic path：`Deterministic`
   - stochastic path：`InferenceModel`
   - observation path：`ObservationModel`
4. 模型输出为预测分布，分析时主要使用预测均值。
5. 训练损失包括：
   - negative log likelihood
   - KL divergence

## 6. 当前训练策略实际是什么

当前代码已经显式支持 TTS、MTS、PMTS。入口参数在 `options/train_options.py`：

```bash
--training_strategy tts|mts|pmts
--pmts_update_freq N
```

三种策略含义：

| 策略 | 当前是否实现 | 说明 |
|---|---|---|
| TTS | 是 | 训练开始时固定一组 target/context，整个训练过程保持不变 |
| MTS | 是 | 每个训练样本随机选 `num_train_target` 个 target，其余训练站点作为 context |
| PMTS | 是 | 周期性固定 context/target 划分；每隔 `pmts_update_freq` 个 epoch 重新采样一次 |

实现位置：

- `options/train_options.py`：定义 `--training_strategy` 和 `--pmts_update_freq`。
- `data/base_dataset.py`：根据策略决定每个样本的 context/target 划分。
- `data/SM_dataset.py`：TTS/PMTS 初始化固定训练划分。
- `train.py`：PMTS 在指定 epoch 周期触发 `reset_train_context_target()`。

论文当前主实验采用 PMTS3，即 `training_strategy=pmts` 且 `pmts_update_freq=3`。MTS 作为随机采样基线保留，TTS 作为固定划分基线。解释时注意：PMTS 的核心不是每个样本都随机，而是在一个短周期内保持站点配置稳定，再周期性刷新，从而在配置稳定性与空间组合多样性之间折中。

## 7. 已完成实验总结

### 7.1 训练策略与传感器配置比例实验

脚本与结果：

- 脚本：`run_training_strategy_experiment.py`
- 结果文件：
  - `experiments/training_strategy_ratio/summary_by_cell.csv`
  - `experiments/training_strategy_ratio_pmts3/results_summary.csv`
- 论文位置：`STGNP 土壤水分论文初稿.md` 第 4.1 节。

设置：

- 数据集：NAQU selected35，5 cm，30 min。
- 模型：hierarchical，`SM_config1`。
- 训练策略：TTS、MTS、PMTS3。
- 观测:未监测比例：1:6、2:5、3:4、4:3、5:2、6:1。
- 每个组合 5 个随机种子。
- 评估：strict holdout，仅在未监测站点非缺失记录上计算 MAE、RMSE、R2。

论文当前采用的核心结果：

- TTS 在稀疏设置下明显不稳定，不作为后续主策略。
- MTS 在 4:3 比例下表现有竞争力：MAE `0.0492 ± 0.0082`，RMSE `0.0643 ± 0.0097`，R2 `0.7023 ± 0.0603`。
- PMTS3 在 6:1 比例下取得最佳总体结果：MAE `0.0420 ± 0.0080`，RMSE `0.0546 ± 0.0103`，R2 `0.7925 ± 0.0219`。
- 后续主实验默认倾向 PMTS3；4:3 可作为成本—精度折中参考，6:1 可作为性能导向配置。

### 7.2 输入时间窗口长度实验

脚本与结果：

- 脚本：`run_time_window_ablation.py`
- 结果文件：`experiments/time_window_ablation/results_summary.csv` 及相关时间戳目录。
- 论文位置：`STGNP 土壤水分论文初稿.md` 第 4.2 节。

论文当前表述：

- 在 PMTS + 6:1 主设置下比较 1、2、3、4、5、6、8、12 h 输入窗口。
- 3 h 输入窗口作为性能导向默认窗口：MAE `0.0442 ± 0.0044`，RMSE `0.0555 ± 0.0036`，R2 `0.8189 ± 0.0245`。
- 6 h 窗口误差略高但跨随机种子更稳定。
- 结论：历史输入长度不是越长越好，适中的近期历史窗口更适合当前任务；过长窗口可能引入冗余动态。

注意：早期固定 holdout 多 seed 实验曾显示 4 h 窗口较稳。写论文时以第 4.2 节当前统一 PMTS 设置为主，不要混用早期窗口结论。

### 7.3 未见站点重建精度空间分布

论文位置：`STGNP 土壤水分论文初稿.md` 第 4.3 节。

当前状态：

- 已有空间分布图脚本：
  - `plot_station_metric_map.py`
  - `plot_station_metric_map_aggregate.py`
- 已有相关图像产物位于 `image/yucejigndu/` 和 `image/zhandian_importance/`。
- 第 4.3 节目前仍偏占位，需要后续将最终站点级 MAE/RMSE/R2 与空间坐标合并后，补充更具体的空间误差解释。

写作重点：

- 解释高误差站点是否位于边缘、观测稀疏区域或局地异质性强区域。
- 不要只描述“颜色深浅”，要把误差空间分布与观测站点覆盖联系起来。

### 7.4 关键站点贡献度与 Top-K 重建实验

脚本与结果：

- 贡献度分析：`analyze_station_contribution.py`
- Top-K 实验：`run_topk_key_stations_experiment.py`
- 当前主要结果文件：`experiments/topk_key_stations/20260528T162352/results_summary_mean.csv`
- 论文位置：`STGNP 土壤水分论文初稿.md` 第 4.4 节。

贡献度定义：

```text
ΔMAE = mean(MAE | station removed, excluding self-error)
       - mean(MAE | station kept, excluding self-error)
```

论文当前采用的 Top-12 关键站点：

`BC02, F5, MS3494, BC05, C1, P10, BC03, F4, MS3518, MS3603, MS3533, BC06`

Top-K 重建结果：

| Top-K | MAE | RMSE | R2 |
|---:|---:|---:|---:|
| 2 | 0.0982 ± 0.0047 | 0.1312 ± 0.0074 | -0.1977 ± 0.1316 |
| 4 | 0.0691 ± 0.0027 | 0.0922 ± 0.0042 | 0.4066 ± 0.0546 |
| 8 | 0.0612 ± 0.0047 | 0.0769 ± 0.0043 | 0.5826 ± 0.0470 |
| 10 | 0.0566 ± 0.0030 | 0.0726 ± 0.0021 | 0.6353 ± 0.0211 |
| 12 | 0.0528 ± 0.0015 | 0.0703 ± 0.0007 | 0.6650 ± 0.0064 |
| 14 | 0.0577 ± 0.0006 | 0.0743 ± 0.0013 | 0.6435 ± 0.0123 |
| 16 | 0.0530 ± 0.0014 | 0.0693 ± 0.0024 | 0.6726 ± 0.0226 |

结论：

- Top-2/Top-4 不足以支撑全局重建。
- Top-8 开始具备可用性。
- Top-12 在误差和部署规模之间最具性价比。
- Top-16 略提升 R2/RMSE，但相对 Top-12 的 MAE 改善不明显。
- 关键站点数量不是越多越好，空间位置和信息贡献比单纯数量更重要。

### 7.5 NASA/SMAP 面状土壤水分产品一致性实验

这是最新完成并已写入论文第 4.6 节的实验。

论文位置：

- `STGNP 土壤水分论文初稿.md` 第 4.6 节：`与面状土壤水分产品的区域一致性检验`
- 原综合讨论已顺延为第 4.7 节。

数据：

- 面状/网格产品：NASA SMAP soil moisture
- 时间范围文件名：2020-10-04 至 2020-10-10
- 当前有效评价：2020-10-04 至 2020-10-09 六个时间步
- 站点/网格数：6586 个 SMAP 网格单元中心作为伪站点
- 站点信息：`dataset/NASA_STATION/NSIDC-0779_station_extent_smap_grid_station_info.csv`
- 土壤水分宽表：`dataset/NASA_STATION/NSIDC-0779_station_extent_20201004_20201010_smap_grid_soil_moisture_wide.csv`
- 空 test nodes：`dataset/NASA_STATION/test_nodes_empty.npy`

运行命令核心参数：

```bash
python test_and_analyze_plot.py \
  --model hierarchical \
  --dataset_mode SM \
  --pred_attr SM \
  --config SM_config1 \
  --phase test \
  --gpu_ids 0 \
  --file_time 20260531T151041 \
  --epoch best \
  --t_len 6 \
  --checkpoints_dir checkpoints/faceData \
  --sm_location_path dataset/NASA_STATION/NSIDC-0779_station_extent_smap_grid_station_info.csv \
  --sm_data_path dataset/NASA_STATION/NSIDC-0779_station_extent_20201004_20201010_smap_grid_soil_moisture_wide.csv \
  --sm_test_nodes_path dataset/NASA_STATION/test_nodes_empty.npy \
  --sm_eval_target_mode all \
  --sm_eval_full_time \
  --output_dir checkpoints/faceData/SM/hierarchical_SM_20260531T151041/analysis_NASA
```

输出目录：

- `checkpoints/faceData/SM/hierarchical_SM_20260531T151041/analysis_NASA`

主要产物：

- `predictions_structured.csv`
- `station_metrics.csv`
- `analysis_report.txt`
- `error_analysis.png`
- `trend_prediction_vs_truth_part01.png`
- `trend_prediction_vs_truth_part02.png`

结果：

| 有效网格数 | 有效样本数 | MAE | RMSE | ubRMSE | R2 |
|---:|---:|---:|---:|---:|---:|
| 6493 | 19165 | 0.0174 | 0.0248 | 0.0232 | 0.4210 |

网格级误差分布：

- MAE 中位数：0.0137
- MAE 平均值：0.0173
- 约 68.0% 有效网格 MAE ≤ 0.02
- 约 93.0% 有效网格 MAE ≤ 0.04
- 约 1.4% 有效网格 MAE > 0.06
- 最高网格 MAE：0.0953

技术改动：

- `data/SM_dataset.py`：大图自动使用 sparse top-k 邻接，6586 网格默认 top-64。
- `data/base_dataset.py`：`all` 模式自动 target 分块，避免一次生成 `2 × 6586 × 6586` 邻接张量。
- `models/hierarchical/hierarchical_model.py`：分块结果缓存 `target_node_index`。
- `test_and_analyze_plot.py`：支持分块结果汇总并生成面状数据分析产物。

解释边界：

- 该实验是“与 SMAP 面状产品的一致性检验”，不是独立外推验证。
- 不要写成 STGNP 优于 SMAP，也不要写成优于传统插值。
- 当前 `t_len=6` 只覆盖 2020-10-04 至 2020-10-09；2020-10-10 尚未纳入最终结果。
- 若要纳入完整七天，需要解决 `t_len=7` 下的显存/内存问题，或实现更低内存的分块图聚合。

## 8. He 2025 风格实验设计的当前状态

用户希望借鉴 He et al. 2025 的稀疏监测水位重建论文，设计对应土壤水分实验。可以借鉴实验结构，但不能机械照搬术语。

当前状态：图 6 对应的训练策略与传感器配置比例实验、图 9 对应的输入窗口实验、关键站点与 Top-K 实验、以及新增的 NASA/SMAP 面状产品一致性实验已经完成并写入论文初稿。空间邻接保留比例实验和典型站点曲线仍需补强。

### 图 6：传感器配置比例 + 训练策略

建议定义比例为：

```text
观测站点 : 未监测站点
```

35 个站点下建议：

| 比例 | 观测站点数 | 未监测站点数 |
|---|---:|---:|
| 1:6 | 5 | 30 |
| 2:5 | 10 | 25 |
| 3:4 | 15 | 20 |
| 4:3 | 20 | 15 |
| 5:2 | 25 | 10 |
| 6:1 | 30 | 5 |

比较：

- TTS
- MTS
- PMTS

指标：

- 未监测站点 MAE，建议加 seed 标准差。

状态：

- 已完成，见第 7.1 节。
- 论文当前结论：PMTS3 + 6:1 为性能导向最佳配置；4:3 可作为成本—精度折中参考。

### 图 7：空间邻接保留比例对训练过程的影响

不要直接叫 He 2025 的 DRR，建议改成：

- Spatial adjacency retention ratio
- Graph sparsification ratio

定义：

- 先构建完整 Haversine-Gaussian 邻接矩阵。
- 保留权重最高的 30%-100% 边。
- 其余边置零。
- 保留自环。

设置：

- 0.3、0.4、0.5、0.6、0.7、0.8、0.9、1.0

图：

- x 轴：epoch
- y 轴：validation loss
- 颜色：空间邻接保留比例

状态：尚未形成最终论文实验。当前 NASA/SMAP 大图推理已实现 sparse top-k 邻接，但这不是系统的空间邻接保留比例消融。

### 图 8：空间邻接保留比例对测试重建精度的影响

设置同图 7。

固定：

- 图 6 最优训练策略，例如 PMTS。
- 图 6 最优比例，例如 4:3。
- 输入窗口 4 h 或图 9 最优窗口。

图：

- x 轴：空间邻接保留比例
- y 轴：未监测站点 median MAE 或 MAE
- 可用箱线图展示站点间分布。

状态：尚未形成最终论文实验。若论文篇幅有限，可暂时不放；如果加入，应与图 7 使用同一套保留比例。

### 图 9：历史输入长度对重建性能的影响

状态：已完成并写入论文第 4.2 节。当前论文采用 1、2、3、4、5、6、8、12 h 设置；3 h 为性能导向默认窗口。

历史计划曾建议在统一最佳策略下重跑：

- 1、2、3、4、5、6、7、8 h
- `t_len = 2, 4, 6, 8, 10, 12, 14, 16`

固定：

- PMTS
- 4:3
- 最优空间邻接保留比例，例如 0.6

图：

- x 轴：输入窗口小时数
- y 轴：未监测站点 MAE

### 图 10：最优设置下空间误差分布

使用图 6-9 确定的最优组合：

- 训练策略：如 PMTS
- 观测:未监测比例：如 4:3
- 空间保留比例：如 0.6
- 输入窗口：如 4 h

输出：

- 每个未监测站点 MAE、RMSE、R2、MAPE。
- 与经纬度合并。
- 绘制空间散点图。

注意：

- 土壤水分 MAPE 在真实值接近 0 时可能不稳定。
- 建议同时展示 MAE 和 MAPE，或至少在文中说明 MAPE 局限。

状态：已有空间图脚本和部分图像产物，但论文第 4.3 节仍需补具体结果解释。

### 图 11：典型站点真实值与重建值对比

从图 10 的误差等级中选：

- 低误差站点
- 中误差站点
- 高误差站点

绘制：

- 实测土壤水分曲线
- 重建土壤水分曲线
- 标注 MAE、RMSE、MAPE

状态：论文第 4.5 节仍为空，需要后续选择典型低/中/高误差站点并补图。

## 9. 建议的后续执行顺序

1. 补第 4.3 节空间误差分布的最终图和文字解释。
2. 补第 4.5 节典型站点重建曲线，建议选择低/中/高误差站点各 1-2 个。
3. 如要完整支持 NASA/SMAP 2020-10-04 至 2020-10-10 七天结果，优化 `t_len=7` 的大图推理内存峰值，或实现更低内存的图聚合。
4. 视论文篇幅决定是否补空间邻接保留比例消融；若补，则同时做训练曲线和测试精度。
5. 更新结论部分，使其包含第 4.6 面状产品一致性实验，同时避免过度声称优于 SMAP 或传统插值。
6. 最后统一图号、表号、图注和 Results/Discussion 编号。

## 10. Notion 记录

已在 Notion 中创建页面：

- 页面标题：`STGNP 土壤水分重建项目计划与实验设计`
- URL：`https://www.notion.so/361a6f0d873b8162a6b4cdca02aaeb35`
- 父页面：`🧪 STGNP 实验记录`

## 11. 写作与论文主线建议

推荐论文结果章节：

1. 训练策略与传感器配置比例的影响
2. 输入时间窗口长度对模型性能的影响
3. 未见站点重建精度的空间分布
4. 关键站点贡献与 Top-K 重建实验
5. 典型站点重建曲线
6. 与面状土壤水分产品的区域一致性检验
7. 综合讨论

核心论点模板：

- PMTS3 在高观测比例下表现最好，MTS 在中等比例下具有竞争力。
- 4:3 可作为成本—精度折中参考，6:1 可作为性能导向配置。
- 历史输入窗口不是越长越好，过长窗口可能引入冗余信息。
- 少量高贡献站点可以保留大部分区域重建能力，Top-12 是当前 Top-K 实验中的较优折中。
- NASA/SMAP 面状产品实验表明模型具备跨尺度区域一致性，但不是独立外推验证。
- 高误差站点可能与空间边缘、缺失率高、局地异质性强或邻近观测不足有关。

## 12. 后续 AI 操作注意事项

- 不要把早期 Pali 24 站点/10 cm 描述误写成当前主实验。
- 当前主实验以 NAQU selected35、5 cm、30 min 为准。
- NASA/SMAP 面状产品实验使用 `dataset/NASA_STATION/` 下的 6586 个网格伪站点，不要与 NAQU selected35 主实验混淆。
- 不要直接照搬 He 2025 的 DRR 名称；应改写为空间邻接保留比例或图稀疏化比例。
- 当前已经实现 TTS/MTS/PMTS；不要再沿用“当前只是 MTS-like 随机 target/context”的旧说法。
- 不要把 holdout 站点放进训练子集；strict holdout 是当前项目的核心设计。
- 写论文时，baseline 不是当前主线的第一优先级；当前更适合围绕“传感器配置、训练策略、时间窗口、关键站点、空间误差分布、面状产品一致性”组织实验。
- 面状产品实验只能写成与 SMAP 的一致性检验；不能写成 STGNP 优于 SMAP 或优于传统插值。
- 当前面状结果覆盖 2020-10-04 至 2020-10-09 六个时间步；若用户要求 2020-10-10，必须先补完整七天推理结果。
- 如果要做方法优势声明，仍需后续补传统插值或时空图模型对比。
- 修改远程代码前先检查 git 状态，避免覆盖用户已有实验结果。
