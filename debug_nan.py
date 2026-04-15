#!/usr/bin/env python
"""Debug script to trace NaN appearance in model"""

import sys
import torch
import numpy as np

# Set command line arguments
sys.argv = ['debug_nan.py', '--model', 'hierarchical', '--dataset_mode', 'SM', '--pred_attr', 'SM',
            '--config', 'SM_config1', '--phase', 'train', '--gpu_ids', '-1',
            '--n_epochs', '1', '--seed', '2023']

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# Parse arguments
opt, model_config = TrainOptions().parse()
opt.batch_size = 8  # Smaller batch for debugging
opt.num_workers = 0

print("\n" + "="*60)
print("STEP 1: Load Dataset")
print("="*60)

dataset = create_dataset(opt)
print(f"Dataset created: {type(dataset)}")

# Get first batch - dataset is already a dataloader
try:
    batch_data = next(iter(dataset))
except StopIteration:
    print("ERROR: Could not get batch from dataset")
    sys.exit(1)

print("\n--- Batch Data Shapes ---")
for key in batch_data.keys():
    if torch.is_tensor(batch_data[key]):
        print(f"  {key}: {batch_data[key].shape}, dtype={batch_data[key].dtype}")
        # Check for NaN in raw data
        if torch.isnan(batch_data[key]).any():
            print(f"    [WARNING] Contains NaN in RAW DATA!")
            print(f"    NaN count: {torch.isnan(batch_data[key]).sum().item()}")
        # Check value ranges
        valid_mask = ~torch.isnan(batch_data[key])
        if valid_mask.any():
            valid_vals = batch_data[key][valid_mask]
            print(f"    Value range (excluding NaN): [{valid_vals.min():.4f}, {valid_vals.max():.4f}]")

print("\n" + "="*60)
print("INVESTIGATING: Why are there NaN in the input data?")
print("="*60)

# Check if NaN correlate with missing_mask_target
nan_mask_target = torch.isnan(batch_data['pred_target'])
missing_mask = batch_data['missing_mask_target'].bool()
print(f"\nNaNs in pred_target:")
print(f"  Total NaN positions: {nan_mask_target.sum().item()}")
print(f"  Total missing positions: {missing_mask.sum().item()}")

# Check correlation
print(f"\nChecking correlation between NaN and missing_mask:")
for b in range(batch_data['pred_target'].shape[0]):
    nan_count = nan_mask_target[b].sum().item()
    if nan_count > 0:
        # For each position with NaN, check if it's marked as missing
        matching = 0
        for t in range(batch_data['pred_target'].shape[1]):
            for n in range(batch_data['pred_target'].shape[2]):
                if nan_mask_target[b, t, n, 0]:
                    # This position is NaN, check if it's marked as missing
                    # Note: missing_mask has shape [batch, time, num_nodes]
                    if missing_mask[b, t, n] == 1.0:
                        matching += 1
        print(f"  Batch {b}: {nan_count} NaN values, {matching} marked as missing")

# Let's look at raw values before normalization
print(f"\n*** This suggests: NaN values might be created during data processing ***")
print(f"*** Need to check where missing values (-99.0) are being handled ***")

print("\n" + "="*60)
print("STEP 2: Create Model")
print("="*60)

model = create_model(opt, model_config)
model.setup(opt)

print(f"Model created and set up")
print(f"Model device: {next(model.netHierarchicalNP.parameters()).device}")

print("\n" + "="*60)
print("STEP 3: Forward Pass with First Batch")
print("="*60)

# Move batch to model device
device = next(model.netHierarchicalNP.parameters()).device
for key in batch_data.keys():
    if torch.is_tensor(batch_data[key]):
        batch_data[key] = batch_data[key].to(device)

# Set data in model
model.set_input(batch_data)

print(f"\n--- After set_input ---")
print(f"  pred_context shape: {model.pred_context.shape}")
print(f"  pred_target shape: {model.pred_target.shape}")
if torch.isnan(model.pred_target).any():
        print(f"    [WARNING] pred_target contains NaN!")
# Try forward pass
print(f"\n--- Attempting forward pass ---")
try:
    model.forward(training=True)
    print(f"Forward pass completed successfully")
    
    if torch.isnan(model.p_y_pred.mean).any():
        print(f"    [WARNING] p_y_pred.mean contains NaN!")
        print(f"    NaN count: {torch.isnan(model.p_y_pred.mean).sum().item()}")
    if torch.isnan(model.p_y_pred.variance).any():
        print(f"    [WARNING] p_y_pred.variance contains NaN!")
        print(f"    NaN count: {torch.isnan(model.p_y_pred.variance).sum().item()}")
    
    # Check variance validity for Normal distribution
    if (model.p_y_pred.variance < 0).any():
        print(f"    [WARNING] p_y_pred.variance has negative values!")
        print(f"    Min variance: {model.p_y_pred.variance.min().item()}")
        
except Exception as e:
    print(f"Forward pass failed with error:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)[:200]}...")

print("\n" + "="*60)
print("STEP 4: Backward Pass / Loss Calculation")
print("="*60)

try:
    # Manually compute loss like in backward()
    print(f"\nAttempting loss computation...")
    model.loss_nll, model.loss_kl = model.criterion(
        model.p_y_pred, 
        model.pred_target, 
        model.q_dists, 
        model.p_dists, 
        model.missing_mask_target
    )
    print(f"Loss computed successfully")
    print(f"  NLL: {model.loss_nll}")
    print(f"  KL: {model.loss_kl}")
    
except Exception as e:
    print(f"Loss computation failed:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    
    # Try to isolate the exact failing line
    print(f"\nDebugging loss computation:")
    print(f"  model.pred_target shape: {model.pred_target.shape}")
    print(f"  model.pred_target has NaN: {torch.isnan(model.pred_target).any()}")
    print(f"  model.p_y_pred.mean shape: {model.p_y_pred.mean.shape}")
    print(f"  model.p_y_pred.mean has NaN: {torch.isnan(model.p_y_pred.mean).any()}")


print("\n" + "="*60)
print("END DEBUG")
print("="*60)
