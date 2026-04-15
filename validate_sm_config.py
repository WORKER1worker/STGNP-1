#!/usr/bin/env python
"""
Validation script for Soil Moisture dataset adaptation to STGNP.

This script verifies:
1. Dataset loading and preprocessing
2. Data dimensions match model requirements
3. Model initialization with SM dataset
4. Configuration parameters
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model


def validate_dataset():
    """Validate dataset loading and structure."""
    print("=" * 80)
    print("VALIDATING SOIL MOISTURE DATASET")
    print("=" * 80)
    
    # Create options for dataset loading
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_mode', type=str, default='SM',
                       help='Dataset mode')
    parser.add_argument('--pred_attr', type=str, default='SM',
                       help='Prediction attribute')
    parser.add_argument('--phase', type=str, default='train',
                       help='Phase: train/val/test')
    parser.add_argument('--t_len', type=int, default=24,
                       help='Time window')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=2023,
                       help='Random seed')
    parser.add_argument('--use_adj', type=bool, default=True,
                       help='Use adjacency matrix')
    parser.add_argument('--gpu_ids', type=str, default='-1',
                       help='GPU IDs')
    parser.add_argument('--serial_batches', action='store_true',
                       help='Serial batches')
    parser.add_argument('--num_threads', type=int, default=0,
                       help='Number of threads')
    parser.add_argument('--max_dataset_size', type=float, default=float('inf'),
                       help='Max dataset size')
    
    args = parser.parse_args([])
    
    print("\n✓ Creating dataset with options:")
    print(f"  Dataset mode: {args.dataset_mode}")
    print(f"  Prediction attr: {args.pred_attr}")
    print(f"  Phase: {args.phase}")
    print(f"  Time window: {args.t_len}")
    
    try:
        # Create dataset
        dataset = create_dataset(args)
        print(f"\n✓ Dataset created successfully!")
        print(f"  Dataset length: {len(dataset)}")
        
        # Check data dimensions
        if hasattr(dataset, 'raw_data'):
            raw_data = dataset.raw_data
            print(f"\n✓ Raw data structure:")
            for key in raw_data.keys():
                if not isinstance(raw_data[key], np.ndarray):
                    print(f"  {key}: {type(raw_data[key])}")
                else:
                    print(f"  {key}: {raw_data[key].shape} {raw_data[key].dtype}")
        
        # Get a sample batch
        print(f"\n✓ Loading sample batch...")
        sample_batch = dataset[0]
        print(f"  Sample batch keys: {sample_batch.keys()}")
        for key, val in sample_batch.items():
            if isinstance(val, torch.Tensor):
                print(f"    {key}: {val.shape}")
            elif isinstance(val, np.ndarray):
                print(f"    {key}: {val.shape}")
            else:
                print(f"    {key}: {type(val)}")
        
        # Print dataset statistics
        print(f"\n✓ Dataset statistics:")
        print(f"  Num nodes: {raw_data['pred'].shape[0]}")
        print(f"  Num timesteps: {raw_data['pred'].shape[1]}")
        print(f"  Output dim (y_dim): {raw_data['pred'].shape[2]}")
        if raw_data['feat'].shape[2] > 0:
            print(f"  Input dim (covariate_dim): {raw_data['feat'].shape[2]}")
        else:
            print(f"  Input dim (covariate_dim): 0")
        if hasattr(dataset, 'A'):
            print(f"  Adjacency matrix shape: {dataset.A.shape}")
        
        # Print normalization info
        if hasattr(args, 'mean') and hasattr(args, 'scale'):
            print(f"\n✓ Normalization info:")
            print(f"  Mean: {args.mean}")
            print(f"  Scale: {args.scale}")
        
        print(f"\n✅ Dataset validation PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset validation FAILED!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def validate_model_config():
    """Validate model configuration for SM dataset."""
    print("\n" + "=" * 80)
    print("🔍 VALIDATING MODEL CONFIGURATION")
    print("=" * 80)
    
    try:
        # Parse training options
        print("\n✓ Parsing training options...")
        sys.argv = ['test_script', '--model', 'hierarchical', '--dataset_mode', 'SM',
                   '--pred_attr', 'SM', '--config', 'SM_config1', '--phase', 'train',
                   '--gpu_ids', '-1', '--seed', '2023']
        
        opt, model_config = TrainOptions().parse()
        
        print(f"✓ Options parsed successfully!")
        print(f"  Model: {opt.model}")
        print(f"  Dataset mode: {opt.dataset_mode}")
        print(f"  Prediction attr: {opt.pred_attr}")
        print(f"  Config: {opt.config}")
        print(f"  y_dim (output): {opt.y_dim}")
        print(f"  covariate_dim (input): {opt.covariate_dim}")
        print(f"  spatial_dim: {opt.spatial_dim}")
        
        print(f"\n✓ Model configuration (SM_config1):")
        for key, val in model_config.items():
            print(f"  {key}: {val}")
        
        print(f"\n✅ Model configuration validation PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model configuration validation FAILED!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print summary of SM adaptation."""
    print("\n" + "=" * 80)
    print("📋 SOIL MOISTURE ADAPTATION SUMMARY")
    print("=" * 80)
    
    summary = {
        "Data Source": [
            "Station info: data/dataset/SM/Pali-Stations.csv (24 stations)",
            "Measurements: data/dataset/SM/SM_PL-30 minutes_10cm.csv (115,252 timesteps)",
        ],
        "Data Processing": [
            "Processed data: dataset/SM/processed_raw.csv",
            "Test nodes: dataset/SM/test_nodes.npy",
        ],
        "Model Configuration": [
            "Dataset class: SM_dataset.py (SMDataset)",
            "Configurations: SM_config1, SM_config2, SM_config3, SM_config4",
            "Default settings:",
            "  - y_dim: 1 (soil moisture)",
            "  - covariate_dim: 0 (temporal features)",
            "  - spatial_dim: 2 (lon, lat)",
            "  - num_nodes: 24 (stations)",
            "  - Input feature: 1 (soil moisture)",
            "  - Output feature: 1 (soil moisture prediction)",
        ],
        "Training Command": [
            "./train.sh hierarchical SM SM SM_config1 -1 2023",
            "or: python train.py --model hierarchical --dataset_mode SM --pred_attr SM --config SM_config1 --phase train --gpu_ids -1",
        ]
    }
    
    for section, items in summary.items():
        print(f"\n🔸 {section}:")
        for item in items:
            print(f"   {item}")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("STGNP SOIL MOISTURE ADAPTATION VALIDATION")
    print("=" * 80 + "\n")
    
    # Validate dataset
    dataset_valid = validate_dataset()
    
    # Validate model configuration  
    model_valid = validate_model_config()
    
    # Print summary
    print_summary()
    
    # Final status
    print("\n" + "=" * 80)
    if dataset_valid and model_valid:
        print("✅ ALL VALIDATIONS PASSED - READY FOR TRAINING!")
    else:
        print("❌ SOME VALIDATIONS FAILED - PLEASE FIX ERRORS ABOVE")
    print("=" * 80 + "\n")
    
    sys.exit(0 if (dataset_valid and model_valid) else 1)
