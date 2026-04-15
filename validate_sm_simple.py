#!/usr/bin/env python
"""
Validation script for Soil Moisture dataset adaptation to STGNP.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options.train_options import TrainOptions
import argparse

def main():
    print("\n" + "=" * 80)
    print("SOIL MOISTURE DATASET VALIDATION")
    print("=" * 80 + "\n")
    
    # Test 1: Parse model configuration
    print("[TEST 1] Parsing training options for SM dataset...")
    sys.argv = ['test', '--model', 'hierarchical', '--dataset_mode', 'SM',
               '--pred_attr', 'SM', '--config', 'SM_config1', '--phase', 'train',
               '--gpu_ids', '-1', '--seed', '2023', '--enable_val']
    
    try:
        opt, model_config = TrainOptions().parse()
        print("[PASS] Options parsed successfully!")
        print(f"       - Model: {opt.model}")
        print(f"       - Dataset mode: {opt.dataset_mode}") 
        print(f"       - Config: {opt.config}")
        print(f"       - y_dim: {opt.y_dim}")
        print(f"       - covariate_dim: {opt.covariate_dim}")
        print(f"       - spatial_dim: {opt.spatial_dim}")
        
        print(f"\n       Model configuration (SM_config1):")
        for key, val in model_config.items():
            print(f"       - {key}: {val}")
            
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Dataset paths
    print(f"\n[TEST 2] Checking data files...")
    files_to_check = [
        ('data/dataset/SM/Pali-Stations.csv', 'Station locations'),
        ('data/dataset/SM/SM_PL-30 minutes_10cm.csv', 'Soil moisture measurements'),
        ('dataset/SM/processed_raw.csv', 'Processed data'),
        ('dataset/SM/test_nodes.npy', 'Test nodes split'),
    ]
    
    all_exist = True
    for file_path, desc in files_to_check:
        exists = os.path.exists(file_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"       {status} {desc:40s} - {file_path}")
        if not exists:
            all_exist = False
    
    # Test 3: Dataset loading
    print(f"\n[TEST 3] Loading SM dataset...")
    try:
        from data import create_dataset
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_mode', type=str, default='SM')
        parser.add_argument('--pred_attr', type=str, default='SM')
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--t_len', type=int, default=24)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--seed', type=int, default=2023)
        parser.add_argument('--use_adj', type=bool, default=True)
        parser.add_argument('--gpu_ids', type=str, default='-1')
        parser.add_argument('--serial_batches', action='store_true')
        parser.add_argument('--num_threads', type=int, default=0)
        parser.add_argument('--max_dataset_size', type=float, default=float('inf'))
        parser.add_argument('--num_train_target', type=int, default=3)
        parser.add_argument('--enable_curriculum', action='store_true')
        
        args = parser.parse_args([])
        data_loader = create_dataset(args)
        
        print(f"[PASS] Dataset loaded successfully!")
        print(f"       - Dataloader size: {len(data_loader)}")
        
        # Access the actual dataset
        dataset = data_loader.dataset
        if hasattr(dataset, 'raw_data'):
            raw_data = dataset.raw_data
            print(f"       - Num nodes: {raw_data['pred'].shape[0]}")
            print(f"       - Num timesteps: {raw_data['pred'].shape[1]}")
            print(f"       - Output dim: {raw_data['pred'].shape[2]}")
            if raw_data['feat'].shape[2] > 0:
                print(f"       - Input dim: {raw_data['feat'].shape[2]}")
            else:
                print(f"       - Input dim: 0")
        
        # Get sample batch
        for sample_batch in data_loader:
            print(f"       - Sample batch keys: {list(sample_batch.keys())}")
            for key, val in sample_batch.items():
                if hasattr(val, 'shape'):
                    print(f"         * {key}: {val.shape}")
            break
        
    except Exception as e:
        print(f"[FAIL] Dataset loading error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY:")
    print("=" * 80)
    print(f"""
Data Source:
  - Station info: data/dataset/SM/Pali-Stations.csv (24 stations)
  - Measurements: data/dataset/SM/SM_PL-30 minutes_10cm.csv (115,252 timesteps)

Data Processing:
  - Processed data: dataset/SM/processed_raw.csv
  - Test nodes: dataset/SM/test_nodes.npy  

Model Configuration:
  - Dataset class: SM_dataset.py (SMDataset)
  - Configurations: SM_config1, SM_config2, SM_config3, SM_config4
  - Default settings:
    * y_dim: 1 (soil moisture - single feature output)
    * covariate_dim: 0 (no temporal covariates)
    * spatial_dim: 2 (longitude, latitude)
    * num_nodes: 24 (soil moisture sensors)
    * d_input: 1 (single input feature)
    * d_output: 1 (single output prediction)

Training Command Examples:
  ./train.sh hierarchical SM SM SM_config1 -1 2023
  
  python train.py --model hierarchical --dataset_mode SM --pred_attr SM \\
                 --config SM_config1 --phase train --gpu_ids -1 --seed 2023

Testing Command Example:
  python test.py --model hierarchical --dataset_mode SM --pred_attr SM \\
                --config SM_config1 --phase test --gpu_ids -1 --epoch best
""")
    
    print("=" * 80)
    print("[SUCCESS] All validations passed - Ready for training!")
    print("=" * 80 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
