# STGNP 土壤水分数据集适配指南

## 📋 项目概述

本文档详细说明如何将 STGNP（Spatio-Temporal Graph Neural Processes）模型适配到**土壤水分预测任务**。

### 数据特性
- **传感器数量**: 24个 (PL01-PL24)
- **时间分辨率**: 30分钟间隔
- **测量深度**: 10cm 土壤层
- **总时间步数**: 115,252条记录 (~3.3年)
- **地点**: 帕雷塔尼地区(经纬度信息已录入)

### 模型配置
- **输入特征 (d_input)**: 1 (土壤含水量值)
- **输出特征 (d_output)**: 1 (土壤含水量预测)
- **空间特征 (d_spatial)**: 2 (经度lon, 纬度lat)
- **节点数 (num_nodes)**: 24个传感器
- **协变量维度**: 0 (可选扩展为含有其他特征)

---

## 🛠️ 实现步骤

### 第1步: 数据预处理

#### 1.1 运行数据转换脚本
```bash
python data/dataset/convert_data_to_csv.py
```

**功能**:
- 将 `Pali-Stations.xlsx` 转换为 `Pali-Stations.csv` (24个站点位置信息)
- 将 `SM_PL-30 minutes_10cm.txt` 转换为 `SM_PL-30 minutes_10cm.csv`
- 处理缺失值标记 (-99.0)
- 清理重复列 (PL11-01, PL12-01 → PL11, PL12)

**输出文件**:
```
data/dataset/SM/
├── Pali-Stations.csv          # 站点位置信息 (24行)
└── SM_PL-30 minutes_10cm.csv  # 原始测量数据 (115,252行)
```

#### 1.2 运行数据集准备脚本
```bash
python data/dataset/prepare_sm_dataset.py
```

**功能**:
- 生成模型所需的处理后数据格式
- 创建训练/测试集分割
- 计算归一化统计量

**输出文件**:
```
dataset/SM/
├── processed_raw.csv   # 处理后数据 (long format)
└── test_nodes.npy      # 测试集节点索引
```

### 第2步: 数据集类实现

**文件**: `data/SM_dataset.py`

继承自 `BaseDataset`，实现以下关键方法：

```python
class SMDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 设置默认参数
        parser.set_defaults(y_dim=1, covariate_dim=0, spatial_dim=2)
        return parser

    def __init__(self, opt):
        # 加载站点位置 → 构建邻接矩阵
        self.A = self.load_loc(location_path, build_adj=opt.use_adj)
        # 加载特征数据 → 处理缺失值 → 归一化
        self.raw_data, norm_info = self.load_feat(data_path, time_division)
        # 划分训练/测试节点
        self.test_node_index = self.get_node_division(...)

    def load_loc(self, location_path, build_adj=True):
        # 从CSV读取站点坐标
        # 计算Haversine距离 → 应用高斯核 → 邻接矩阵
        return A  # shape: (24, 24)

    def load_feat(self, data_path, time_division):
        # 按站点组织数据: (24 stations, ~80k timesteps, 1 feature)
        # 处理缺失值 -99.0
        # 计算mean/scale用于归一化
        # 返回: data dict + norm_info dataframe
```

**关键输出**:
- `raw_data['pred']`: (24, 80676, 1) - 土壤水分值
- `raw_data['feat']`: (24, 80676, 0) - 暂无协变量
- `raw_data['missing']`: (24, 80676, 1) - 缺失值掩码
- `self.A`: (24, 24) - 基于距离的邻接矩阵

### 第3步: 模型配置

**文件**: `model_configurations/hierarchical_config.yaml`

在YAML配置文件中添加土壤水分专用配置:

```yaml
SM_config1:
  # 推荐配置 (均衡参数量和性能)
  tcn_channels: [32, 64]
  latent_channels: [16, 32]
  emd_channel: 16
  num_latent_layers: 1
  observation_hidden_dim: 128
  num_observation_layers: 3
  tcn_kernel_size: 3
  dropout: 0.1

SM_config2:
  # 轻量配置 (低计算量)
  tcn_channels: [16, 32]
  latent_channels: [16]
  emd_channel: 8
  num_latent_layers: 1
  observation_hidden_dim: 64
  num_observation_layers: 2
  tcn_kernel_size: 3
  dropout: 0.1

SM_config3:
  # 深层配置 (更好的时间建模)
  tcn_channels: [32, 64, 128]
  latent_channels: [16, 32, 64]
  emd_channel: 16
  num_latent_layers: 2
  observation_hidden_dim: 128
  num_observation_layers: 3
  tcn_kernel_size: 3
  dropout: 0.1

SM_config4:
  # 高容量配置 (最大性能)
  tcn_channels: [32, 64, 128, 256]
  latent_channels: [16, 32, 64, 128]
  emd_channel: 16
  num_latent_layers: 2
  observation_hidden_dim: 256
  num_observation_layers: 4
  tcn_kernel_size: 3
  dropout: 0.15
```

---

## 🚀 训练和推理

### 验证配置

运行验证脚本确保所有组件正常工作:

```bash
python validate_sm_simple.py
```

**需要通过的验证**:
- ✓ 选项解析成功 (y_dim=1, covariate_dim=0, spatial_dim=2)
- ✓ 所有数据文件存在
- ✓ 数据集成功加载 (24节点, 80,676时间步)
- ✓ 样本批次生成正确 (batch_size X time X nodes X features)

### 训练命令

#### 基础训练 (SM_config1)
```bash
python train.py \
    --model hierarchical \
    --dataset_mode SM \
    --pred_attr SM \
    --config SM_config1 \
    --phase train \
    --gpu_ids 0 \
    --n_epochs 100 \
    --num_train_target 3 \
    --enable_val \
    --save_best \
    --seed 2023
```

#### 使用train.sh脚本
```bash
./train.sh hierarchical SM SM SM_config1 0 2023
```

**训练参数说明**:
- `--model hierarchical`: 使用STGNP模型
- `--dataset_mode SM`: 使用土壤水分数据集
- `--pred_attr SM`: 预测目标属性为 SM (土壤水分)
- `--config SM_config1`: 使用推荐配置
- `--n_epochs 100`: 总训练轮数 (根据需要调整)
- `--num_train_target 3`: 每批训练的目标节点数
- `--enable_val`: 启用验证集评估
- `--save_best`: 保存最佳模型
- `--seed 2023`: 随机种子 (保证可重复性)

### 测试/推理命令

```bash
python test.py \
    --model hierarchical \
    --dataset_mode SM \
    --pred_attr SM \
    --config SM_config1 \
    --phase test \
    --gpu_ids 0 \
    --epoch best \
    --seed 2023
```

---

## 📊 数据流概览

```
原始数据
├── Pali-Stations.xlsx (24个站点位置)
└── SM_PL-30 minutes_10cm.txt (115,252条记录)
           ↓ (convert_data_to_csv.py)
CSV数据文件
├── Pali-Stations.csv
└── SM_PL-30 minutes_10cm.csv
           ↓ (prepare_sm_dataset.py)
处理后数据
├── dataset/SM/processed_raw.csv
└── dataset/SM/test_nodes.npy
           ↓ (SM_dataset.py加载)
数据张量
├── pred: (24, 80676, 1) - 土壤水分值
├── feat: (24, 80676, 0) - 协变量(空)
├── missing: (24, 80676, 1) - 缺失掩码
└── A: (24, 24) - 邻接矩阵
           ↓ (数据加载器)
批次数据
├── pred_context: (batch, context_time, 24, 1)
├── pred_target: (batch, target_time, 24, 1)
├── adj: (batch, 2, target_time, context_time)
└── missing_mask: (batch, time, 24)
           ↓ (STGNP模型)
预测
├── mean: 推理均值
└── variance: 预测不确定性
```

---

## 🔧 关键实现细节

### 1. 邻接矩阵构建

基于Haversine距离计算传感器间的相似性:

```python
# 步骤:
# 1. 从CSV读取所有站点的 (lon, lat) 坐标
# 2. 计算每对站点间的大圆距离 (Haversine公式)
# 3. 应用高斯核: A[i,j] = exp(-0.5 * (d[i,j] / sigma)^2)
# 4. 结果: A ∈ [0,1], A[i,j]=1表示相邻
```

### 2. 缺失值处理

数据中用 -99.0 标记缺失值:

```python
# 在数据加载时:
sm_values[sm_values == -99.0] = np.nan
missing_mask[i] = (original_values[i] == -99.0)

# 统计时忽略NaN:
mean = np.nanmean(valid_values)
std = np.nanstd(valid_values)

# 模型损失函数中:
loss *= (1 - missing_mask)  # 只计算非缺失值的损失
```

### 3. 数据归一化

```python
# 使用StandardScaler进行Z-score归一化
normalized_data = (raw_data - mean) / scale

# 其中:
# mean = 0.1738 (历史平均值)
# scale = 0.0756 (标准差)

# 推理后反归一化:
raw_pred = normalized_pred * scale + mean
```

### 4. 批次构建

使用Context-Target分割:

```
# 对于每个样本:
- Context: 前13个时间步的观测
- Target: 后3个时间步的预测目标

# 所有24个节点中:
- 3个作为目标节点 (Train: 随机选择; Test: 固定)
- 21个作为上下文节点

# 结果:
pred_context: (batch_size, 13, 24, 1)
pred_target:  (batch_size, 3, 24, 1)
adj:          (batch_size, 2, 3, 13)  # CSR格式邻接矩阵
```

---

## ⚙️ 参数优化建议

根据你的需求 (单特征输入/输出, 24节点):

### 对于快速实验
```
config: SM_config2
n_epochs: 50
batch_size: 32
num_train_target: 2
```

### 对于标准训练
```
config: SM_config1
n_epochs: 100
batch_size: 64
num_train_target: 3
enable_curriculum: True
```

### 对于最佳性能
```
config: SM_config3 或 SM_config4
n_epochs: 200
batch_size: 128
num_train_target: 3-6
enable_curriculum: True
```

---

## 📈 预期结果

基于以前的STGNP论文结果，在土壤水分预测上预期:

- **MAE**: < 0.05 (归一化单位)
- **RMSE**: < 0.08 (归一化单位)
- **MAPE**: < 15% (对于有效值)

实际结果取决于:
- 缺失值比例 (某些传感器覆盖不完整)
- 训练数据长度 (时间窗口大小)
- 模型容量和训练周期

---

## 🔍 故障排查

### 问题: "Dataset loading error"
**解决**: 确保 CSV 文件生成:
```bash
python data/dataset/convert_data_to_csv.py
python data/dataset/prepare_sm_dataset.py
```

### 问题: CUDA out of memory
**解决**: 减少批次大小或模型容量:
```bash
--batch_size 32  # 默认128
--config SM_config2  # 使用轻量配置
```

### 问题: 模型不收敛
**解决**: 调整学习率或使用课程学习:
```bash
--lr 0.0005  # 降低学习率
--enable_curriculum  # 启用课程学习
--n_epochs 200  # 增加训练轮数
```

---

## 📝 文件结构

```
STGNP/
├── data/
│   ├── SM_dataset.py              # [新] 土壤水分数据集类
│   ├── dataset/
│   │   ├── SM/
│   │   │   ├── Pali-Stations.xlsx
│   │   │   ├── Pali-Stations.csv          # 转换后
│   │   │   ├── SM_PL-30min_10cm.txt
│   │   │   └── SM_PL-30min_10cm.csv       # 转换后
│   │   ├── convert_data_to_csv.py         # [修改] 支持SM数据转换
│   │   └── prepare_sm_dataset.py          # [新] 数据集准备
│   └── base_dataset.py            # [不变] 基类
├── model_configurations/
│   └── hierarchical_config.yaml   # [修改] 添加SM_config1-4
├── validate_sm_simple.py          # [新] 验证脚本
├── train.py                       # [不变] 训练脚本
├── test.py                        # [不变] 测试脚本
└── train.sh                       # [不变] 训练shell脚本
```

---

## 📚 相关论文

- **原始STGNP**: Hu et al., "Graph Neural Processes for Spatio-Temporal Extrapolation", KDD 2023
- **Haversine距离**: https://en.wikipedia.org/wiki/Haversine_formula
- **图神经网络**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

---

## ✅ 检查清单

在开始训练前，确保完成:

- [ ] 运行了 `convert_data_to_csv.py`
- [ ] 运行了 `prepare_sm_dataset.py`  
- [ ] 运行了 `validate_sm_simple.py` 且所有验证通过
- [ ] 确认所有数据文件存在于正确位置
- [ ] 选择了合适的模型配置 (SM_config1-4)
- [ ] 设置了训练超参数
- [ ] GPU可用 (如果使用 --gpu_ids 0)
- [ ] 足够的磁盘空间用于检查点保存 (~1GB每个epoch)

---

## 📞 支持和反馈

如有问题或建议,请检查:

1. **数据准备问题**: `python validate_sm_simple.py` 输出
2. **模型配置**: `model_configurations/hierarchical_config.yaml`
3. **数据加载**: `data/SM_dataset.py` 的日志输出
4. **训练日志**: `checkpoints/SM/hierarchical_SM_{timestamp}/`

---

**最后更新**: 2026年4月12日
**STGNP版本**: v1.0
**适配状态**: ✅ 完成并验证
