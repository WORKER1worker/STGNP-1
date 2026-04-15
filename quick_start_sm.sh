#!/bin/bash
# STGNP Soil Moisture Quick Start
# 土壤水分数据集快速开始指南

echo "=========================================="
echo "STGNP 土壤水分预测模型 - 快速启动"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi

echo "[1/5] 准备环境..."
python --version
echo ""

echo "[2/5] 转换原始数据为CSV格式..."
python data/dataset/convert_data_to_csv.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 数据转换失败"
    exit 1
fi
echo ""

echo "[3/5] 准备处理后的数据集..."
python data/dataset/prepare_sm_dataset.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 数据准备失败"
    exit 1
fi
echo ""

echo "[4/5] 验证配置..."
python validate_sm_simple.py
if [ $? -ne 0 ]; then
    echo "[ERROR] 配置验证失败"
    exit 1
fi
echo ""

echo "[5/5] 准备开始训练..."
echo ""
echo "========== 推荐的训练命令 =========="
echo ""
echo "基础训练 (SM_config1):"
echo "  python train.py --model hierarchical --dataset_mode SM --pred_attr SM \\"
echo "    --config SM_config1 --phase train --gpu_ids 0 --n_epochs 100 \\"
echo "    --num_train_target 3 --enable_val --save_best"
echo ""
echo "或使用train.sh脚本:"
echo "  ./train.sh hierarchical SM SM SM_config1 0 2023"
echo ""
echo "Light实验 (SM_config2):"
echo "  python train.py --model hierarchical --dataset_mode SM --pred_attr SM \\"
echo "    --config SM_config2 --phase train --gpu_ids 0 --n_epochs 50 \\"
echo "    --batch_size 32 --num_train_target 2"
echo ""
echo "完整训练 (SM_config3):"
echo "  python train.py --model hierarchical --dataset_mode SM --pred_attr SM \\"
echo "    --config SM_config3 --phase train --gpu_ids 0 --n_epochs 200 \\"
echo "    --enable_curriculum --enable_val --save_best"
echo ""
echo "========================================"
echo ""
echo "[SUCCESS] 所有准备完成!"
echo "详细指南请查看: SOILMOISTURE_ADAPTATION_GUIDE.md"
echo ""
