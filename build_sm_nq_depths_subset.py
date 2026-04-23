
#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# User-selected stations (intended pool)
selected_stations = [
    'MS3627','MS3518','MS3614','MS3488','MS3533','MS3523','BC04','MS3603','BC05','MSBJ',
    'C1','BC08','MS3482','MS3494','BC02','BC06','MS3576','MS3501','CD02','P2',
    'P11','F4','C4','CD01','MS3475',
]

# User-fixed test stations
fixed_test_stations = ['MS3627','MS3533','CD07','MS3518','MS3603','MS3513']

base_dir = 'data/dataset/SM_NQ_depths'
out_dir = 'dataset/SM_NQ_selected25_depths'
os.makedirs(out_dir, exist_ok=True)

loc_path = os.path.join(base_dir, 'Stations_information_NAQU.csv')
depth_files = [
    'SM_NQ-30-minutes_05cm.csv',
    'SM_NQ-30-minutes_10cm.csv',
    'SM_NQ-30-minutes_20cm.csv',
    'SM_NQ-30-minutes_40cm.csv',
]

loc = pd.read_csv(loc_path)
if 'station_id' not in loc.columns:
    raise ValueError('Stations_information_NAQU.csv missing station_id column')

meta_cols = ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']

# Ensure test stations are included in the pool; current split method requires this.
extra_test_stations = [s for s in fixed_test_stations if s not in selected_stations]
if extra_test_stations:
    print('[WARN] Following test stations are not in selected_stations, auto-adding them:')
    print('       ', extra_test_stations)
    selected_stations = selected_stations + extra_test_stations

loc_ids = set(loc['station_id'].astype(str).tolist())
missing_in_loc = [s for s in selected_stations if s not in loc_ids]
if missing_in_loc:
    raise ValueError(f'Stations missing in location file: {missing_in_loc}')

# Load all depth data once and validate station columns exist.
depth_dfs = {}
for fname in depth_files:
    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)
    for c in meta_cols:
        if c not in df.columns:
            raise ValueError(f'{fname} missing time column: {c}')
    missing_in_depth = [s for s in selected_stations if s not in df.columns]
    if missing_in_depth:
        raise ValueError(f'Stations missing in {fname}: {missing_in_depth}')
    depth_dfs[fname] = df

loc_sub = loc[loc['station_id'].astype(str).isin(selected_stations)].copy()
loc_sub = loc_sub.sort_values(by='station_id').reset_index(drop=True)
station_sorted = loc_sub['station_id'].astype(str).tolist()

# Build subset files for each depth using the same sorted station order.
written_depth_paths = []
for fname in depth_files:
    df = depth_dfs[fname]
    sub = df[meta_cols + station_sorted].copy()
    out_path = os.path.join(out_dir, fname)
    sub.to_csv(out_path, index=False)
    written_depth_paths.append(out_path)

# Convert fixed test station IDs to sorted-index positions used by dataset.
fixed_test_indices = np.array([station_sorted.index(s) for s in fixed_test_stations], dtype=np.int64)

loc_out = os.path.join(out_dir, 'Stations_information_NAQU.csv')
test_out = os.path.join(out_dir, 'test_nodes.npy')
sel_out = os.path.join(out_dir, 'selected_stations.txt')
test_station_out = os.path.join(out_dir, 'test_stations.txt')

loc_sub.to_csv(loc_out, index=False)
np.save(test_out, fixed_test_indices)

with open(sel_out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(station_sorted) + '\n')
with open(test_station_out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_test_stations) + '\n')

print('subset_count:', len(station_sorted))
print('fixed_test_stations:', fixed_test_stations)
print('fixed_test_indices:', fixed_test_indices.tolist())
print('sorted_station_order:', station_sorted)
print('written:', loc_out)
for p in written_depth_paths:
    print('written:', p)
print('written:', test_out)
print('written:', sel_out)
print('written:', test_station_out)
