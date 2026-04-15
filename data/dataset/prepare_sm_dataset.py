# prepare_sm_dataset.py
"""
Prepare Soil Moisture dataset for STGNP model training.
This script creates the processed data needed by SM_dataset.py
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def prepare_sm_dataset():
    """
    Prepare soil moisture dataset:
    1. Load station locations
    2. Load raw measurements
    3. Organize into (num_nodes, num_timesteps, num_features) format
    4. Create test_nodes.npy
    """
    # Define paths
    sm_dir = 'dataset/SM'
    os.makedirs(sm_dir, exist_ok=True)
    
    stations_csv = 'data/dataset/SM/Pali-Stations.csv'
    soil_csv = 'data/dataset/SM/SM_PL-30 minutes_10cm.csv'
    
    if not os.path.exists(stations_csv) or not os.path.exists(soil_csv):
        print(f"❌ CSV files not found. Running convert_data_to_csv.py first...")
        os.system('python data/dataset/convert_data_to_csv.py')
    
    print("="*70)
    print("📊 Preparing Soil Moisture Dataset")
    print("="*70 + "\n")
    
    # Load data
    print("1️⃣  Loading station information...")
    stations = pd.read_csv(stations_csv)
    print(f"   {len(stations)} stations loaded")
    
    print("2️⃣  Loading soil moisture measurements...")
    sm_data = pd.read_csv(soil_csv)
    print(f"   {len(sm_data)} time steps loaded")
    
    # Extract sensor columns
    sensor_cols = [col for col in sm_data.columns 
                   if col.startswith('PL') and len(col) == 4]
    sensor_cols = sorted(sensor_cols, key=lambda x: int(x[2:]))
    
    print(f"   {len(sensor_cols)} sensors: {sensor_cols}")
    
    # Create processed data
    print("3️⃣  Creating processed dataset...")
    num_nodes = len(sensor_cols)
    num_timesteps = len(sm_data)
    
    # Extract soil moisture values (num_nodes, num_timesteps)
    sm_values = sm_data[sensor_cols].values.T  # Transpose to (nodes, time)
    
    print(f"   Data shape: ({num_nodes}, {num_timesteps})")
    
    # Create missing value mask
    missing_mask = (sm_values == -99.0).astype(int)
    
    # Replace missing values with NaN for statistics calculation
    sm_values_clean = sm_values.astype(float).copy()
    sm_values_clean[missing_mask == 1] = np.nan
    
    # Compute statistics for normalization
    mean_val = np.nanmean(sm_values_clean)
    std_val = np.nanstd(sm_values_clean)
    scale_val = std_val if std_val > 0 else 1.0
    
    print(f"   Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    
    # Create processed raw data
    processed_raw = pd.DataFrame()
    processed_raw['time'] = pd.to_datetime(
        sm_data[['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']]
        .rename(columns={
            'yyyy': 'year', 'mm': 'month', 'dd': 'day',
            'HH': 'hour', 'MM': 'minute', 'SS': 'second'
        })
    )
    
    # Add soil moisture data
    for i, col in enumerate(sensor_cols):
        processed_raw[col] = sm_data[col].values
        # Add missing mask column
        processed_raw[f'{col}_Missing'] = missing_mask[i]
    
    # Add station_id column (repeat for each timestep)
    processed_raw['station_id'] = None  # Will be handled differently
    
    # Save as processed_raw.csv
    print("4️⃣  Saving processed data...")
    processed_raw_path = os.path.join(sm_dir, 'processed_raw.csv')
    
    # Reshape for stacking - pivot to long format
    pivot_data = []
    for idx, station in enumerate(sensor_cols):
        station_data = processed_raw.copy()
        station_data['station_id'] = station
        station_data['SM'] = processed_raw[station].values
        station_data['SM_Missing'] = processed_raw[f'{station}_Missing'].values
        # Keep only necessary columns
        station_data = station_data[['time', 'station_id', 'SM', 'SM_Missing']]
        pivot_data.append(station_data)
    
    processed_raw_final = pd.concat(pivot_data, ignore_index=True)
    processed_raw_final.to_csv(processed_raw_path, index=False)
    print(f"   Saved: {processed_raw_path}")
    
    # Create test_nodes.npy
    # Use last 1/3 of stations as test nodes
    test_nodes = np.arange(num_nodes)
    test_count = max(1, num_nodes // 3)
    test_node_index = np.array([num_nodes - i - 1 for i in range(test_count)])
    
    test_nodes_path = os.path.join(sm_dir, 'test_nodes.npy')
    np.save(test_nodes_path, test_node_index)
    print(f"   Saved: {test_nodes_path}")
    print(f"   Test nodes: {test_node_index} (count: {len(test_node_index)})")
    
    print("\n" + "="*70)
    print("✅ Dataset preparation complete!")
    print("="*70)
    print(f"\nDataset statistics:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Time steps: {num_timesteps}")
    print(f"  - Features: 1 (Soil Moisture)")
    print(f"  - Mean: {mean_val:.6f}")
    print(f"  - Scale: {scale_val:.6f}")
    print(f"  - Missing values: {np.sum(missing_mask)} / {missing_mask.size}")


if __name__ == '__main__':
    prepare_sm_dataset()
