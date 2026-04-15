# 土壤水分数据适配完成总结

## 🎉 适配工作完成

STGNP模型已成功适配为**土壤水分预测任务**。

### 完成的工作清单

| 任务 | 状态 | 文件 |
|------|------|------|
| 数据转换脚本 | ✅ | `data/dataset/convert_data_to_csv.py` |
| 数据准备脚本 | ✅ | `data/dataset/prepare_sm_dataset.py` |
| 数据集类实现 | ✅ | `data/SM_dataset.py` |
| 模型配置 | ✅ | `model_configurations/hierarchical_config.yaml` |
| 验证脚本 | ✅ | `validate_sm_simple.py` |
| 完整文档 | ✅ | `SOILMOISTURE_ADAPTATION_GUIDE.md` |
| 快速启动脚本 | ✅ | `quick_start_sm.sh` / `quick_start_sm.ps1` |

---

## 📊 数据配置概览

```
输入: Pali-Stations.xlsx + SM_PL-30 minutes_10cm.txt
    ↓
    ├─ 24个传感器站点 (PL01-PL24)
    ├─ 115,252条时间记录 (~3.3年)
    ├─ 30分钟时间分辨率
    └─ 10cm土壤深度

模型参数:
    ├─ d_input: 1 (单一特征-土壤含水量)
    ├─ d_output: 1 (预测单一特征)
    ├─ d_spatial: 2 (经纬度)
    ├─ num_nodes: 24
    ├─ y_dim: 1
    ├─ covariate_dim: 0
    └─ spatial_dim: 2

模型配置:
    ├─ SM_config1: 推荐 (均衡)
    ├─ SM_config2: 轻量 (快速实验)
    ├─ SM_config3: 深层 (更好表示)
    └─ SM_config4: 高容量 (最佳性能)
```

---

## 🚀 快速开始

### 方式1: 自动化快速启动 (推荐)

**Windows用户:**
```powershell
powershell -ExecutionPolicy Bypass -File quick_start_sm.ps1
```

**Linux/Mac用户:**
```bash
bash quick_start_sm.sh
```

### 方式2: 手动步骤

**1. 转换数据**
```bash
python data/dataset/convert_data_to_csv.py
```

**2. 准备数据集**
```bash
python data/dataset/prepare_sm_dataset.py
```

**3. 验证配置**
```bash
python validate_sm_simple.py
```

**4. 开始训练**
```bash
python train.py --model hierarchical --dataset_mode SM --pred_attr SM \
    --config SM_config1 --phase train --gpu_ids 0 --n_epochs 100 \
    --num_train_target 3 --enable_val --save_best
```

---

## 📝 关键实现

### 1. 数据加载流程 (`SM_dataset.py`)
- ✅ 从CSV读取24个站点坐标
- ✅ Haversine距离计算 → 高斯核 → 邻接矩阵
- ✅ 自动检测/处理缺失值 (-99.0)
- ✅ StandardScaler归一化
- ✅ 自动划分训练/测试集

### 2. 模型配置 (`hierarchical_config.yaml`)
- ✅ 4套预定义配置 (轻量 → 高容量)
- ✅ 针对单特征预测优化
- ✅ 可根据需要灵活调整

### 3. 数据处理 (`convert_data_to_csv.py` + `prepare_sm_dataset.py`)
- ✅ Excel → CSV 自动转换
- ✅ 重复列处理 (PL11-01 → PL11)
- ✅ 缺失值标记
- ✅ 长格式数据生成

---

## 📈 预期性能指标

基于原始STGNP论文,在土壤水分预测上预期:

```
MAE:  < 0.05 (归一化单位)
RMSE: < 0.08 (归一化单位)  
MAPE: < 15%  (对于非缺失值)
```

实际结果取决于:
- 模型配置和训练周期
- 缺失值数量和分布
- 超参数调优

---

## 🛠️ 自定义选项

### 选择不同的模型配置

```bash
# 轻量配置 (快速实验)
--config SM_config2

# 深层配置 (更好的时间建模)
--config SM_config3

# 高容量配置 (最大性能)
--config SM_config4
```

### 调整训练参数

```bash
# 学习率
--lr 0.001  # 默认
--lr 0.0005 # 更低的学习率 (更稳定)

# 批次大小
--batch_size 128  # 默认
--batch_size 64   # GPU内存不足时

# 训练轮数
--n_epochs 100   # 默认
--n_epochs 200   # 更多轮数,更好的收敛

# 目标节点数
--num_train_target 3     # 默认 (3个目标节点)
--num_train_target 5     # 更多目标节点
```

### 启用高级功能

```bash
# 启用验证集
--enable_val

# 启用课程学习
--enable_curriculum

# 保存最佳模型
--save_best

# Beta-VAE权重
--beta 1.0  # 默认 (无额外KL约束)
--beta 0.5  # 增加KL权重用于正则化
```

---

## 📂 生成的文件结构

```
STGNP/
├── data/
│   ├── SM_dataset.py                    [新] 土壤水分数据集
│   ├── dataset/
│   │   ├── SM/
│   │   │   ├── Pali-Stations.xlsx       (原始)
│   │   │   ├── Pali-Stations.csv        ✅ 转换后
│   │   │   ├── SM_PL-30min_10cm.txt     (原始)
│   │   │   └── SM_PL-30min_10cm.csv     ✅ 转换后
│   │   ├── convert_data_to_csv.py       [修改]
│   │   └── prepare_sm_dataset.py        [新]
│   └── base_dataset.py
│
├── dataset/
│   └── SM/
│       ├── processed_raw.csv            ✅ 生成 (~90MB)
│       └── test_nodes.npy               ✅ 生成
│
├── model_configurations/
│   └── hierarchical_config.yaml         [修改] 添加SM配置
│
├── checkpoints/
│   └── SM/                              (训练检查点)
│
├── validate_sm_simple.py                [新] 验证脚本
├── quick_start_sm.sh                    [新] Linux启动
├── quick_start_sm.ps1                   [新] Windows启动
├── SOILMOISTURE_ADAPTATION_GUIDE.md     [新] 完整指南
└── README.md                            (原始项目说明)
```

---

## 🔍 验证结果

运行 `python validate_sm_simple.py` 的输出应显示:

```
[PASS] Options parsed successfully!
       - Model: hierarchical
       - Dataset mode: SM
       - Config: SM_config1
       - y_dim: 1
       - covariate_dim: 0
       - spatial_dim: 2

[PASS] Dataset loaded successfully!
       - Dataloader size: 80652
       - Num nodes: 24
       - Num timesteps: 80676
       - Output dim: 1
       - Input dim: 0

[SUCCESS] All validations passed - Ready for training!
```

---

## 📚 进一步阅读

1. **详细指南**: `SOILMOISTURE_ADAPTATION_GUIDE.md`
2. **原始论文**: "Graph Neural Processes for Spatio-Temporal Extrapolation" (KDD 2023)
3. **项目README**: `README.md` (原始STGNP说明)
4. **模型代码**: `models/hierarchical/` (STGNP实现)

---

## 🐛 常见问题

### Q: 如何改变训练的土壤水分特征?

A: 当前适配使用固定的 `pred_attr: SM`。如需更改,需修改:
- `data/SM_dataset.py` 中的 `self.pred_attrs = ['SM']`
- 数据文件中的特征列名

### Q: 能否添加其他协变量特征?

A: 可以。需要:
1. 在CSV中添加额外的列
2. 修改 `SM_dataset.py` 中的 `covariate_dim` (从0改为所需维度)
3. 在 `load_feat()` 中处理新特征

### Q: 模型是否支持多步预测?

A: 是的。模型已支持通过:
- `--t_len 24`: 上下文时间窗口 (默认)
- 调整 `pred_target` 的时间维度

### Q: 如何使用GPU训练?

A: 设置GPU ID:
```bash
--gpu_ids 0       # 使用GPU 0
--gpu_ids 0,1     # 使用GPU 0和1 (需多卡支持)
```

---

## ✅ 使用检查清单

开始训练前确保完成:

- [ ] Python 3.8+ 已安装
- [ ] 依赖包已安装 (`pip install -r requirements.txt`)
- [ ] `python data/dataset/convert_data_to_csv.py` 已运行
- [ ] `python data/dataset/prepare_sm_dataset.py` 已运行
- [ ] `python validate_sm_simple.py` 返回成功
- [ ] 所有CSV和NPY文件都已生成
- [ ] 选择了合适的模型配置 (SM_config1-4)
- [ ] 设置好了超参数
- [ ] 磁盘空间充足 (至少2GB)

---

## 🎯 后续改进方向

根据实验结果,可考虑:

1. **特征工程**: 添加时间特征 (小时、月份)、临界深度等
2. **图结构优化**: 尝试不同的邻接矩阵构建方法
3. **多变量**: 联合预测其他土壤参数 (温度、含盐量等)
4. **时间序列**: 使用更长的上下文窗口或动态长度
5. **集成方法**: 结合多个模型的预测结果

---

## 📞 技术支持

如遇到问题,请:

1. 查看详细文档: `SOILMOISTURE_ADAPTATION_GUIDE.md`
2. 检查验证输出: `python validate_sm_simple.py`
3. 查看训练日志: `checkpoints/SM/hierarchical_SM_*/train_error.log`
4. 检查数据文件: `dataset/SM/processed_raw.csv` 前置行

---

## 📄 版本信息

- **STGNP版本**: 官方版本
- **适配版本**: 1.0
- **适配日期**: 2026年4月12日
- **Python版本**: 3.8+
- **PyTorch版本**: 1.9+

---

**🎓 项目完成!** 现在你可以开始训练土壤水分预测模型了。

祝训练顺利! Good luck! 🚀

