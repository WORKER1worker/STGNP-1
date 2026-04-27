from data.base_dataset import BaseDataset
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class SMDataset(BaseDataset):
    """
    Soil Moisture Dataset
    Data structure:
    - 24 soil moisture sensor stations (PL01-PL24)
    - Each station has soil moisture measurements at 30-minute intervals (10cm depth)
    - Location: Pali region (longitude, latitude)
    - Missing values are marked as -99.0
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add dataset-specific options for soil moisture dataset.
        
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase
        Returns:
            the modified parser
        """
        # y_dim: number of predicted features (soil moisture: 1)
        # covariate_dim: number of covariates (soil moisture only: 0, can be expanded to include temporal features)
        # spatial_dim: dimension of spatial features used for adjacency (lon, lat: 2)
        parser.set_defaults(y_dim=1, covariate_dim=0, spatial_dim=2)
        parser.add_argument('--sm_location_path', type=str,
                    default='data/dataset/SM_NQ/Stations_information_NAQU.csv',
                    help='path to station metadata csv')
        parser.add_argument('--sm_data_path', type=str,
                    default='data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv',
                    help='path to soil moisture csv')
        parser.add_argument('--sm_test_nodes_path', type=str,
                    default='dataset/SM_NQ/test_nodes.npy',
                    help='path to test node split file')
        parser.add_argument('--sm_holdout_nodes_path', type=str,
                default='',
                help='optional path to holdout node split file (.npy)')
        parser.add_argument('--sm_holdout_station_id', type=str,
                default='',
                help='optional holdout station id for unseen-site inference')
        parser.add_argument('--sm_holdout_lon', type=float,
                default=None,
                help='optional holdout station longitude for consistency check')
        parser.add_argument('--sm_holdout_lat', type=float,
                default=None,
                help='optional holdout station latitude for consistency check')
        parser.add_argument('--sm_eval_target_mode', type=str,
                default='test',
                choices=['test', 'holdout'],
                help='target node group for non-train phases')
        parser.add_argument('--sm_holdout_location_path', type=str,
            default='data/dataset/SM_NQ/Stations_information_NAQU.csv',
            help='full location csv path used for strict holdout inference')
        parser.add_argument('--sm_holdout_data_path', type=str,
            default='data/dataset/SM_NQ/SM_NQ-30-minutes_05cm.csv',
            help='full data csv path used for strict holdout inference')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        Initialize the soil moisture dataset.
        """
        self.opt = opt
        
        self.pred_attrs = [opt.pred_attr]  # ['SM'] - soil moisture
        self.drop_attrs = ['SM_Missing']  # columns to drop after loading
        self.cont_cols = ['SM']  # continuous columns to normalize
        
        # Define data paths
        location_path = opt.sm_location_path
        data_path = opt.sm_data_path
        test_nodes_path = opt.sm_test_nodes_path
        eval_mode = getattr(self.opt, 'sm_eval_target_mode', 'test')
        
        # Load station locations and build adjacency matrix
        self.A = self.load_loc(location_path, build_adj=opt.use_adj)
        
        # Load features and normalization info
        self.raw_data, norm_info = self.load_feat(
            data_path, 
            self.time_division[opt.phase],
            self.station_ids,
        )

        mean_val = float(norm_info.at['mean', opt.pred_attr])
        scale_val = float(norm_info.at['scale', opt.pred_attr])
        
        # Get data division index (train/test split)
        self.test_node_index = self.get_node_division(
            test_nodes_path, 
            num_nodes=self.raw_data['pred'].shape[0]
        )
        self.test_node_index = self._normalize_node_index(
            self.test_node_index,
            num_nodes=self.raw_data['pred'].shape[0],
            name='test_node_index',
        )

        self.holdout_node_index = self.get_holdout_division(
            num_nodes=self.raw_data['pred'].shape[0]
        )

        # Strict zero-participation fallback: append holdout only for holdout inference.
        if self.opt.phase != 'train' and eval_mode == 'holdout' and self.holdout_node_index.size == 0:
            self.holdout_node_index = self.append_holdout_node_for_eval(
                holdout_station_id=str(getattr(self.opt, 'sm_holdout_station_id', '')).strip(),
                holdout_location_path=str(getattr(self.opt, 'sm_holdout_location_path', '')).strip(),
                holdout_data_path=str(getattr(self.opt, 'sm_holdout_data_path', '')).strip(),
                time_division=self.time_division[opt.phase],
                mean_val=mean_val,
                scale_val=scale_val,
            )

        overlap = np.intersect1d(self.test_node_index, self.holdout_node_index)
        if overlap.size > 0:
            raise ValueError(
                f'test_node_index and holdout_node_index overlap: {overlap.tolist()}'
            )

        excluded_nodes = np.union1d(self.test_node_index, self.holdout_node_index)
        self.train_node_index = np.setdiff1d(
            np.arange(self.raw_data['pred'].shape[0]), 
            excluded_nodes
        )

        if self.train_node_index.size == 0:
            raise ValueError('No training nodes left after excluding test/holdout nodes.')

        if self.opt.phase == 'train':
            self.eval_target_node_index = None
        else:
            if eval_mode == 'holdout':
                if self.holdout_node_index.size == 0:
                    raise ValueError('sm_eval_target_mode=holdout but no holdout nodes were provided.')
                self.eval_target_node_index = self.holdout_node_index
            else:
                if self.test_node_index.size == 0:
                    raise ValueError('sm_eval_target_mode=test but test_node_index is empty.')
                self.eval_target_node_index = self.test_node_index

        print(
            '  Node split summary: '
            f'train={self.train_node_index.size}, '
            f'test={self.test_node_index.size}, '
            f'holdout={self.holdout_node_index.size}, '
            f'eval_mode={getattr(self.opt, "sm_eval_target_mode", "test") if self.opt.phase != "train" else "train"}'
        )
        
        # Add normalization info to options
        self.add_norm_info(mean_val, scale_val)
        
        # Validate data format
        self._data_format_check()
    
    def load_loc(self, location_path, build_adj=True):
        """
        Load station locations from CSV and build adjacency matrix.
        
        Args:
            location_path: path to Pali-Stations.csv
            build_adj: if True, build adjacency matrix based on distances;
                      if False, return relative coordinates
        
        Returns:
            A: adjacency matrix (num_nodes, num_nodes) or 
               distance matrix (num_nodes, num_nodes, 2)
        """
        print('Loading station locations...')
        
        # Load location data
        location = pd.read_csv(location_path)
        location = location[['station_id', 'lon', 'lat']]
        location = location.sort_values(by=['station_id'])
        self.location_df = location.reset_index(drop=True)
        self.station_ids = location['station_id'].astype(str).tolist()
        
        num_station = len(location)
        print(f'  Found {num_station} stations')
        
        if build_adj:
            print('  Building adjacency matrix from distances...')
            A = self._build_adjacency_from_location(location)
            print(f'  Adjacency matrix shape: {A.shape}')
            
        else:
            # Return relative coordinates
            print('  Using relative coordinates as spatial features...')
            A = np.zeros((num_station, num_station, 2))
            
            for i in range(num_station):
                for j in range(num_station):
                    A[i, j, 0] = location.iloc[i]['lon'] - location.iloc[j]['lon']
                    A[i, j, 1] = location.iloc[i]['lat'] - location.iloc[j]['lat']
        
        return A

    def _build_adjacency_from_location(self, location_df):
        num_station = len(location_df)
        dist = np.zeros((num_station, num_station), dtype=float)
        for i in range(num_station):
            for j in range(num_station):
                dist[i, j] = self.haversine(
                    location_df.iloc[i]['lon'],
                    location_df.iloc[i]['lat'],
                    location_df.iloc[j]['lon'],
                    location_df.iloc[j]['lat'],
                )

        sigma = np.std(dist)
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        return np.exp(-0.5 * (dist / sigma) ** 2)

    @staticmethod
    def _read_sm_data_frame(data_path):
        sm_data = pd.read_csv(data_path)
        sm_data['datetime'] = pd.to_datetime(
            sm_data[['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']]
            .rename(columns={
                'yyyy': 'year',
                'mm': 'month',
                'dd': 'day',
                'HH': 'hour',
                'MM': 'minute',
                'SS': 'second'
            })
        )
        return sm_data
    
    def load_feat(self, data_path, time_division, station_ids):
        """
        Load soil moisture features and prepare data.
        
        Args:
            data_path: path to soil moisture csv
            time_division: [start_ratio, end_ratio] for train/val/test split
            station_ids: station ids from location csv (sorted)
        
        Returns:
            data: dict with keys 'feat', 'pred', 'missing', 'time'
            norm_info: normalization statistics (mean, scale, var)
        """
        print('Loading soil moisture features...')
        
        # Read raw data
        sm_data = self._read_sm_data_frame(data_path)
        
        # Keep station order consistent with location file.
        sensor_cols = [station for station in station_ids if station in sm_data.columns]
        if len(sensor_cols) == 0:
            raise RuntimeError(f'No station columns found in {data_path}.')

        extra_cols = [
            col for col in sm_data.columns
            if col not in ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS', 'datetime'] and col not in sensor_cols
        ]
        if len(extra_cols) > 0:
            print(f'  Ignoring {len(extra_cols)} non-station columns: {extra_cols[:10]}')

        # Global fallback for stations that are fully missing in a split.
        all_sensor_values = sm_data[sensor_cols].replace(-99.0, np.nan).to_numpy(dtype=float)
        global_mean = np.nanmean(all_sensor_values)
        if not np.isfinite(global_mean):
            global_mean = 0.0
        
        print(f'  Found {len(sensor_cols)} sensors')
        print(f'  Sensors: {sensor_cols}')
        print(f'  Total records: {len(sm_data)}')
        
        # Prepare data containers
        data = {
            'feat': [],
            'pred': [],
            'missing': [],
            'time': []
        }
        
        # Process each station
        station_order = sensor_cols
        for idx, station in enumerate(station_order):
            # Extract station data
            station_values = sm_data[station].values.copy().astype(float)
            
            # Create SM column for this station
            station_sm = pd.DataFrame({
                'SM': station_values,
                'time': sm_data['datetime']
            })
            
            # Mark missing values (-99.0)
            sm_missing = (station_sm['SM'] == -99.0).astype(int)
            
            # Replace missing values with NaN for processing
            station_sm.loc[sm_missing.astype(bool), 'SM'] = np.nan
            
            # Fill missing values with station mean (before normalization)
            # This is better than using 0 which could be ambiguous with real data
            station_mean = station_sm['SM'].mean()
            if not np.isfinite(station_mean):
                station_mean = global_mean
            station_sm['SM'] = station_sm['SM'].fillna(station_mean)
            
            # Create missing mask
            station_sm['SM_Missing'] = sm_missing
            
            # For now, covariate_dim=0, so no extra features
            # (can be extended to include temporal features)
            station_feat = station_sm[[]].copy()  # Empty features
            
            # Split data according to time_division
            data_length = len(station_sm)
            start_index = int(time_division[0] * data_length)
            end_index = int(time_division[1] * data_length)
            
            # Extract time series data
            feat_data = station_feat.iloc[start_index:end_index].to_numpy()
            missing_data = station_sm['SM_Missing'].iloc[start_index:end_index].to_numpy()
            pred_data = station_sm['SM'].iloc[start_index:end_index].to_numpy()
            
            # Store with shape (1, time_steps, num_features)
            data['feat'].append(feat_data[np.newaxis, :, :] if feat_data.shape[1] > 0 else np.zeros((1, len(feat_data), 0)))
            data['missing'].append(missing_data[np.newaxis, :, np.newaxis])
            data['pred'].append(pred_data[np.newaxis, :, np.newaxis])
            
            if idx == 0:
                data['time'] = station_sm['time'].iloc[start_index:end_index].values
        
        # Concatenate all stations
        data['feat'] = np.concatenate(data['feat'], axis=0)
        data['missing'] = np.concatenate(data['missing'], axis=0)
        data['pred'] = np.concatenate(data['pred'], axis=0)
        
        print(f'  Pred data shape: {data["pred"].shape}')
        print(f'  Feat data shape: {data["feat"].shape}')
        print(f'  Missing data shape: {data["missing"].shape}')
        
        # Compute normalization statistics
        print('Computing normalization info...')
        
        # For normalization, we use non-missing values only
        pred_data_valid = data['pred'].copy()
        pred_data_valid[data['missing'].squeeze() == 1] = np.nan
        
        # Compute statistics
        mean_val = np.nanmean(pred_data_valid)
        std_val = np.nanstd(pred_data_valid)
        if not np.isfinite(mean_val):
            mean_val = global_mean
        if not np.isfinite(std_val):
            std_val = 1.0
        scale_val = std_val if std_val > 0 else 1.0
        
        # Create norm info dataframe
        norm_info = pd.DataFrame(
            [[mean_val], [scale_val], [std_val ** 2]],
            columns=['SM'],
            index=['mean', 'scale', 'var']
        )
        
        print(f'  Mean: {mean_val:.4f}, Scale: {scale_val:.4f}')
        
        # Normalize prediction data
        data['pred'] = (data['pred'] - mean_val) / scale_val
        
        # Convert time to seconds since epoch
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00')) 
                       / np.timedelta64(1, 's'))
        
        return data, norm_info
    
    def get_node_division(self, test_nodes_path, num_nodes):
        """
        Get train/test node split.
        
        Args:
            test_nodes_path: path to test_nodes.npy file
            num_nodes: total number of nodes
        
        Returns:
            test_node_index: array of test node indices
        """
        if os.path.exists(test_nodes_path):
            print(f'Loading test nodes from {test_nodes_path}...')
            test_node_index = np.load(test_nodes_path)
        else:
            # If test_nodes.npy doesn't exist, randomly select test nodes
            # For soil moisture: typically use bottom 1/3 stations as test
            print(f'Test nodes file not found, randomly selecting {num_nodes // 3} nodes as test...')
            test_node_index = np.random.choice(
                np.arange(num_nodes), 
                size=max(1, num_nodes // 3),
                replace=False
            )
            # Save it for reproducibility
            os.makedirs(os.path.dirname(test_nodes_path) or '.', exist_ok=True)
            np.save(test_nodes_path, test_node_index)

        return test_node_index

    def _normalize_node_index(self, node_index, num_nodes, name):
        arr = np.asarray(node_index).reshape(-1)
        if arr.size == 0:
            return np.array([], dtype=np.int64)
        arr = np.unique(arr.astype(np.int64))
        if arr.min() < 0 or arr.max() >= num_nodes:
            raise ValueError(
                f'{name} has out-of-range index values. '
                f'valid range=[0,{num_nodes - 1}], got min={int(arr.min())}, max={int(arr.max())}'
            )
        return np.sort(arr)

    def get_holdout_division(self, num_nodes):
        holdout_nodes_path = str(getattr(self.opt, 'sm_holdout_nodes_path', '')).strip()
        holdout_station_id = str(getattr(self.opt, 'sm_holdout_station_id', '')).strip()
        eval_mode = getattr(self.opt, 'sm_eval_target_mode', 'test')

        holdout_index = np.array([], dtype=np.int64)
        if holdout_nodes_path:
            if not os.path.exists(holdout_nodes_path):
                raise FileNotFoundError(f'Cannot find holdout nodes file: {holdout_nodes_path}')
            print(f'Loading holdout nodes from {holdout_nodes_path}...')
            holdout_index = np.load(holdout_nodes_path)
        elif holdout_station_id:
            if holdout_station_id not in self.station_ids:
                if self.opt.phase != 'train' and eval_mode == 'holdout':
                    # Strict mode: holdout may be intentionally excluded from training subset.
                    holdout_index = np.array([], dtype=np.int64)
                else:
                    raise ValueError(
                        f'sm_holdout_station_id={holdout_station_id} not found in station ids.'
                    )
            else:
                holdout_idx = self.station_ids.index(holdout_station_id)
                holdout_index = np.array([holdout_idx], dtype=np.int64)

        if holdout_index.size > 0:
            holdout_lon = getattr(self.opt, 'sm_holdout_lon', None)
            holdout_lat = getattr(self.opt, 'sm_holdout_lat', None)
            if holdout_lon is not None and holdout_lat is not None and hasattr(self, 'location_df'):
                station_row = self.location_df[self.location_df['station_id'].astype(str) == holdout_station_id]
                if len(station_row) == 1:
                    lon_ref = float(station_row.iloc[0]['lon'])
                    lat_ref = float(station_row.iloc[0]['lat'])
                    if abs(lon_ref - float(holdout_lon)) > 1e-6 or abs(lat_ref - float(holdout_lat)) > 1e-6:
                        raise ValueError(
                            'Provided holdout coordinates do not match station metadata: '
                            f'station={holdout_station_id}, expected=({lon_ref}, {lat_ref}), '
                            f'got=({holdout_lon}, {holdout_lat})'
                        )

        return self._normalize_node_index(holdout_index, num_nodes, 'holdout_node_index')

    def append_holdout_node_for_eval(self, holdout_station_id, holdout_location_path, holdout_data_path,
                                     time_division, mean_val, scale_val):
        if holdout_station_id == '':
            raise ValueError('sm_eval_target_mode=holdout requires sm_holdout_station_id when holdout index file is empty.')
        if holdout_station_id in self.station_ids:
            holdout_idx = self.station_ids.index(holdout_station_id)
            return np.array([holdout_idx], dtype=np.int64)

        if not os.path.isfile(holdout_location_path):
            raise FileNotFoundError(f'Cannot find holdout location file: {holdout_location_path}')
        if not os.path.isfile(holdout_data_path):
            raise FileNotFoundError(f'Cannot find holdout data file: {holdout_data_path}')

        location_df = pd.read_csv(holdout_location_path)
        if not {'station_id', 'lon', 'lat'}.issubset(set(location_df.columns)):
            raise ValueError('Holdout location csv must contain station_id/lon/lat columns.')

        station_row = location_df[location_df['station_id'].astype(str) == holdout_station_id]
        if len(station_row) != 1:
            raise ValueError(f'Cannot locate unique holdout station in location file: {holdout_station_id}')

        holdout_lon = getattr(self.opt, 'sm_holdout_lon', None)
        holdout_lat = getattr(self.opt, 'sm_holdout_lat', None)
        lon_ref = float(station_row.iloc[0]['lon'])
        lat_ref = float(station_row.iloc[0]['lat'])
        if holdout_lon is not None and holdout_lat is not None:
            if abs(lon_ref - float(holdout_lon)) > 1e-6 or abs(lat_ref - float(holdout_lat)) > 1e-6:
                raise ValueError(
                    'Provided holdout coordinates do not match holdout location file: '
                    f'station={holdout_station_id}, expected=({lon_ref}, {lat_ref}), '
                    f'got=({holdout_lon}, {holdout_lat})'
                )

        sm_data = self._read_sm_data_frame(holdout_data_path)
        if holdout_station_id not in sm_data.columns:
            raise ValueError(f'Holdout station column not found in holdout data csv: {holdout_station_id}')

        values = sm_data[holdout_station_id].values.copy().astype(float)
        missing = (values == -99.0).astype(int)
        values[missing.astype(bool)] = np.nan

        station_mean = np.nanmean(values)
        if not np.isfinite(station_mean):
            station_mean = mean_val
        values = np.nan_to_num(values, nan=station_mean)

        data_length = len(values)
        start_index = int(time_division[0] * data_length)
        end_index = int(time_division[1] * data_length)

        pred_data = values[start_index:end_index]
        missing_data = missing[start_index:end_index]
        holdout_time = sm_data['datetime'].iloc[start_index:end_index].values

        if len(pred_data) != self.raw_data['pred'].shape[1]:
            raise ValueError(
                f'Holdout sequence length mismatch: holdout={len(pred_data)} '
                f'vs context={self.raw_data["pred"].shape[1]}'
            )

        holdout_time_sec = ((holdout_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        if holdout_time_sec.shape[0] != self.raw_data['time'].shape[0] or not np.allclose(holdout_time_sec, self.raw_data['time']):
            raise ValueError('Holdout timestamps do not align with context dataset timestamps.')

        holdout_feat = np.zeros((1, len(pred_data), 0), dtype=float)
        holdout_missing = missing_data[np.newaxis, :, np.newaxis]
        holdout_pred = ((pred_data - mean_val) / scale_val)[np.newaxis, :, np.newaxis]

        self.raw_data['feat'] = np.concatenate([self.raw_data['feat'], holdout_feat], axis=0)
        self.raw_data['missing'] = np.concatenate([self.raw_data['missing'], holdout_missing], axis=0)
        self.raw_data['pred'] = np.concatenate([self.raw_data['pred'], holdout_pred], axis=0)

        self.location_df = pd.concat([
            self.location_df,
            pd.DataFrame([{'station_id': holdout_station_id, 'lon': lon_ref, 'lat': lat_ref}]),
        ], ignore_index=True)
        self.station_ids.append(holdout_station_id)

        if getattr(self.opt, 'use_adj', True):
            self.A = self._build_adjacency_from_location(self.location_df)

        holdout_idx = len(self.station_ids) - 1
        print(f'  Strict holdout appended for evaluation: station={holdout_station_id}, index={holdout_idx}')
        return np.array([holdout_idx], dtype=np.int64)


# Register this dataset
