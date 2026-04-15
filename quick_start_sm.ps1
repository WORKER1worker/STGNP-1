# STGNP Soil Moisture Quick Start for Windows
# 土壤水分数据集快速开始指南 (Windows)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "STGNP 土壤水分预测模型 - 快速启动 (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 1: 转换原始数据
Write-Host "[1/5] 转换原始数据为CSV格式..." -ForegroundColor Yellow
python data/dataset/convert_data_to_csv.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 数据转换失败" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: 准备处理后的数据集
Write-Host "[2/5] 准备处理后的数据集..." -ForegroundColor Yellow
python data/dataset/prepare_sm_dataset.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 数据准备失败" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: 验证配置
Write-Host "[3/5] 验证配置..." -ForegroundColor Yellow
python validate_sm_simple.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 配置验证失败" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: 显示训练命令
Write-Host "[4/5] 准备开始训练..." -ForegroundColor Yellow
Write-Host ""

Write-Host "========== 推荐的训练命令 ==========" -ForegroundColor Cyan
Write-Host ""

Write-Host "基础训练 (SM_config1):" -ForegroundColor Green
Write-Host "  python train.py --model hierarchical --dataset_mode SM "--pred_attr SM" -ForegroundColor Gray
Write-Host "    --config SM_config1 --phase train --gpu_ids 0 --n_epochs 100" -ForegroundColor Gray
Write-Host "    --num_train_target 3 --enable_val --save_best" -ForegroundColor Gray
Write-Host ""

Write-Host "Light 实验配置 (SM_config2):" -ForegroundColor Green
Write-Host "  python train.py --model hierarchical --dataset_mode SM "--pred_attr SM" -ForegroundColor Gray
Write-Host "    --config SM_config2 --phase train --gpu_ids 0 --n_epochs 50" -ForegroundColor Gray
Write-Host "    --batch_size 32 --num_train_target 2" -ForegroundColor Gray
Write-Host ""

Write-Host "完整训练 (SM_config3):" -ForegroundColor Green
Write-Host "  python train.py --model hierarchical --dataset_mode SM "--pred_attr SM" -ForegroundColor Gray
Write-Host "    --config SM_config3 --phase train --gpu_ids 0 --n_epochs 200" -ForegroundColor Gray
Write-Host "    --enable_curriculum --enable_val --save_best" -ForegroundColor Gray
Write-Host ""

Write-Host "高容量配置 (SM_config4):" -ForegroundColor Green
Write-Host "  python train.py --model hierarchical --dataset_mode SM "--pred_attr SM" -ForegroundColor Gray
Write-Host "    --config SM_config4 --phase train --gpu_ids 0 --n_epochs 200" -ForegroundColor Gray
Write-Host "    --enable_curriculum --enable_val --save_best" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] 所有准备完成!" -ForegroundColor Green
Write-Host "详细指南查看: SOILMOISTURE_ADAPTATION_GUIDE.md" -ForegroundColor Green
Write-Host ""
Write-Host "数据统计:" -ForegroundColor Cyan
Write-Host "  - 24个传感器站点" -ForegroundColor Gray
Write-Host "  - 115,252条时间序列记录" -ForegroundColor Gray
Write-Host "  - 单一输入/输出特征" -ForegroundColor Gray
Write-Host "  - 已生成邻接矩阵和测试集划分" -ForegroundColor Gray
Write-Host ""
