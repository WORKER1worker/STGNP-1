from data.base_dataset import BaseDataset
import os
import re
import pandas as pd
import numpy as np


class SMDataset(BaseDataset):
    """
    Soil Moisture Dataset
    Data structure:
    - Station-wise soil moisture time series at 30-minute intervals
    - Supports single-depth input and auto-discovered multi-depth input
    - Multi-depth mode stacks channels as [N, T, D] (e.g., D=4)
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
        # y_dim: number of predicted features (single-depth: 1, multi-depth: dynamic override in __init__)
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
        parser.add_argument('--sm_graph_mode', type=str, default='channel', choices=['channel', 'joint_4n'],
                help='channel: [N,T,D] multi-depth channels; joint_4n: [4N,T,1] depth-aware graph nodes')
        parser.add_argument('--sm_vertical_init', type=float, default=1.0,
                help='initial vertical edge weight for joint_4n adjacency construction')
        parser.add_argument('--sm_learn_vertical_weight', type=int, default=1,
                help='whether to learn vertical edge weight in joint_4n mode (1 enable, 0 disable)')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        Initialize the soil moisture dataset.
        """
        self.opt = opt
        self.graph_mode = getattr(opt, 'sm_graph_mode', 'channel')
        self.sm_vertical_init = float(getattr(opt, 'sm_vertical_init', 1.0))
        
        self.pred_attrs = [opt.pred_attr]  # ['SM'] - soil moisture
        self.drop_attrs = ['SM_Missing']  # columns to drop after loading
        self.cont_cols = ['SM']  # continuous columns to normalize
        
        # Define data paths
        location_path = opt.sm_location_path
        data_path = opt.sm_data_path
        test_nodes_path = opt.sm_test_nodes_path

        # Discover depth files once and keep a stable depth order for graph/data alignment.
        self.depth_files = sorted(self._discover_depth_files(data_path), key=lambda p: self._extract_depth_info(p)[0])
        self.depth_labels = [self._extract_depth_info(path)[1] for path in self.depth_files]
        self.num_depths = len(self.depth_files)
        if self.graph_mode == 'joint_4n' and self.num_depths <= 1:
            raise RuntimeError('sm_graph_mode=joint_4n requires at least 2 depth files.')
        
        # Load station locations and build adjacency matrix
        self.A = self.load_loc(location_path, build_adj=opt.use_adj)
        
        # Load features and normalization info.
        self.raw_data, norm_info, used_station_ids = self.load_feat(
            data_path, 
            self.time_division[opt.phase],
            self.base_station_ids if self.graph_mode == 'joint_4n' else self.station_ids,
        )

        # Keep adjacency and station ids consistent with actual loaded station set.
        if self.graph_mode == 'joint_4n':
            if len(used_station_ids) != len(self.base_station_ids):
                station_to_index = {station: idx for idx, station in enumerate(self.base_station_ids)}
                keep_base_index = [station_to_index[station] for station in used_station_ids]
                expanded_keep_index = []
                num_base = len(self.base_station_ids)
                for depth_idx in range(self.num_depths):
                    expanded_keep_index.extend([depth_idx * num_base + idx for idx in keep_base_index])

                self.A = self.A[np.ix_(expanded_keep_index, expanded_keep_index)]
                if hasattr(self, 'A_horizontal') and hasattr(self, 'A_vertical'):
                    self.A_horizontal = self.A_horizontal[np.ix_(expanded_keep_index, expanded_keep_index)]
                    self.A_vertical = self.A_vertical[np.ix_(expanded_keep_index, expanded_keep_index)]

                self.base_station_ids = used_station_ids
                self.station_ids = self._expand_station_ids(self.base_station_ids, self.depth_labels)
                print(f'  Using {len(self.base_station_ids)} aligned base stations after depth intersection')

            # joint_4n encodes depth in node axis, so feature channel is 1.
            self.opt.y_dim = 1
        else:
            if len(used_station_ids) != len(self.station_ids):
                station_to_index = {station: idx for idx, station in enumerate(self.station_ids)}
                keep_index = [station_to_index[station] for station in used_station_ids]
                self.A = self.A[np.ix_(keep_index, keep_index)]
                self.station_ids = used_station_ids
                print(f'  Using {len(used_station_ids)} aligned stations after depth intersection')

            # Dynamic y_dim from loaded channel dimension (1 for single-depth, 4 for multi-depth).
            self.opt.y_dim = int(self.raw_data['pred'].shape[-1])

        print(f'  Dataset output channels (y_dim): {self.opt.y_dim}')
        
        # Get data division index (train/test split)
        self.test_node_index = self.get_node_division(
            test_nodes_path, 
            num_nodes=self.raw_data['pred'].shape[0]
        )
        self.train_node_index = np.setdiff1d(
            np.arange(self.raw_data['pred'].shape[0]), 
            self.test_node_index
        )
        
        # Add normalization info to options.
        mean_values = norm_info.loc['mean'].to_numpy(dtype=float)
        scale_values = norm_info.loc['scale'].to_numpy(dtype=float)
        if mean_values.size == 1:
            mean_values = float(mean_values[0])
            scale_values = float(scale_values[0])
        self.add_norm_info(mean_values, scale_values)
        
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
        self.base_station_ids = location['station_id'].astype(str).tolist()
        self.station_ids = self.base_station_ids.copy()
        
        num_station = len(location)
        print(f'  Found {num_station} stations')

        # Build base spatial similarity matrix first.
        A_spatial = np.zeros((num_station, num_station), dtype=np.float32)
        for i in range(num_station):
            for j in range(num_station):
                dis = self.haversine(
                    location.iloc[i]['lon'],
                    location.iloc[i]['lat'],
                    location.iloc[j]['lon'],
                    location.iloc[j]['lat']
                )
                A_spatial[i, j] = dis

        A_spatial = np.exp(-0.5 * (A_spatial / np.std(A_spatial)) ** 2)
        
        if build_adj:
            print('  Building adjacency matrix from distances...')
            if self.graph_mode == 'joint_4n' and self.num_depths > 1:
                depth_count = self.num_depths
                total_nodes = num_station * depth_count

                # Horizontal graph: same depth, cross-station spatial edges.
                self.A_horizontal = np.zeros((total_nodes, total_nodes), dtype=np.float32)
                for depth_idx in range(depth_count):
                    start = depth_idx * num_station
                    end = start + num_station
                    self.A_horizontal[start:end, start:end] = A_spatial

                # Vertical graph: same station, adjacent-depth edges.
                self.A_vertical = np.zeros((total_nodes, total_nodes), dtype=np.float32)
                for depth_idx in range(depth_count - 1):
                    for station_idx in range(num_station):
                        src = depth_idx * num_station + station_idx
                        dst = (depth_idx + 1) * num_station + station_idx
                        self.A_vertical[src, dst] = 1.0
                        self.A_vertical[dst, src] = 1.0

                A = self.A_horizontal + self.sm_vertical_init * self.A_vertical
                self.station_ids = self._expand_station_ids(self.base_station_ids, self.depth_labels)
                print(f'  Joint graph mode enabled: horizontal+vertical, nodes={total_nodes}')
                print(f'  Adjacency matrix shape: {A.shape}')
            else:
                A = A_spatial
                print(f'  Adjacency matrix shape: {A.shape}')
            
        else:
            if self.graph_mode == 'joint_4n':
                raise RuntimeError('sm_graph_mode=joint_4n requires --use_adj True.')
            # Return relative coordinates
            print('  Using relative coordinates as spatial features...')
            A = np.zeros((num_station, num_station, 2))
            
            for i in range(num_station):
                for j in range(num_station):
                    A[i, j, 0] = location.iloc[i]['lon'] - location.iloc[j]['lon']
                    A[i, j, 1] = location.iloc[i]['lat'] - location.iloc[j]['lat']
        
        return A

    @staticmethod
    def _expand_station_ids(base_station_ids, depth_labels):
        """Expand base station ids to depth-aware node ids in depth-major order."""
        expanded = []
        for depth_label in depth_labels:
            for station_id in base_station_ids:
                expanded.append(f'{station_id}@{depth_label}')
        return expanded

    @staticmethod
    def _extract_depth_info(file_path):
        """Return sortable depth number and display label from CSV file name."""
        name = os.path.basename(file_path)
        match = re.search(r'_(\d+)cm\.csv$', name)
        if match:
            depth_num = int(match.group(1))
            return depth_num, f'{depth_num}cm'
        return 9999, 'single'

    @staticmethod
    def _build_datetime(df, file_path):
        """Build datetime column from standard time columns and validate it."""
        time_cols = ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']
        missing_cols = [col for col in time_cols if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f'Missing time columns in {file_path}: {missing_cols}')

        datetime_series = pd.to_datetime(
            df[time_cols].rename(columns={
                'yyyy': 'year',
                'mm': 'month',
                'dd': 'day',
                'HH': 'hour',
                'MM': 'minute',
                'SS': 'second',
            }),
            errors='coerce'
        )
        if datetime_series.isna().any():
            raise RuntimeError(f'Invalid timestamp rows found in {file_path}.')
        if datetime_series.duplicated().any():
            raise RuntimeError(f'Duplicate timestamps found in {file_path}. Cannot align safely.')

        df = df.copy()
        df['datetime'] = datetime_series
        return df

    def _discover_depth_files(self, data_path):
        """Auto-discover depth CSVs in the same directory when filename matches *_XXcm.csv."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Soil moisture file not found: {data_path}')

        data_path = os.path.abspath(data_path)
        data_dir = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        match = re.match(r'^(.*)_(\d+)cm\.csv$', base_name)
        if not match:
            return [data_path]

        prefix = match.group(1)
        pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)cm\.csv$')
        candidates = []
        for name in os.listdir(data_dir):
            depth_match = pattern.match(name)
            if depth_match:
                candidates.append((int(depth_match.group(1)), os.path.join(data_dir, name)))

        candidates = sorted(candidates, key=lambda x: x[0])
        if len(candidates) <= 1:
            return [data_path]
        return [path for _, path in candidates]

    def _load_depth_frame(self, file_path, station_ids):
        """Load one depth CSV and return frame indexed by datetime with aligned station columns."""
        sm_data = pd.read_csv(file_path)
        sm_data = self._build_datetime(sm_data, file_path)

        sensor_cols = [station for station in station_ids if station in sm_data.columns]
        if len(sensor_cols) == 0:
            raise RuntimeError(f'No station columns from location file found in {file_path}.')

        extra_cols = [
            col for col in sm_data.columns
            if col not in ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS', 'datetime'] and col not in sensor_cols
        ]
        if len(extra_cols) > 0:
            print(f'  Ignoring {len(extra_cols)} non-station columns in {os.path.basename(file_path)}: {extra_cols[:10]}')

        depth_df = sm_data[['datetime'] + sensor_cols].copy()
        for col in sensor_cols:
            depth_df[col] = pd.to_numeric(depth_df[col], errors='coerce')

        depth_df = depth_df.set_index('datetime').sort_index()
        return depth_df, sensor_cols
    
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

        depth_files = self.depth_files
        depth_labels = self.depth_labels

        if len(depth_files) > 1:
            print(f'  Multi-depth mode: discovered {len(depth_files)} files')
            for path in depth_files:
                print(f'    - {path}')
        else:
            print(f'  Single-depth mode: {depth_files[0]}')

        depth_frames = []
        depth_station_sets = []
        for depth_path in depth_files:
            depth_df, sensor_cols = self._load_depth_frame(depth_path, station_ids)
            depth_frames.append(depth_df)
            depth_station_sets.append(set(sensor_cols))

        # Strict station alignment: station must exist in all depth files.
        station_order = [
            station for station in station_ids
            if all(station in station_set for station_set in depth_station_sets)
        ]
        if len(station_order) == 0:
            raise RuntimeError('No common stations found across all depth files.')

        dropped_stations = [station for station in station_ids if station not in station_order]
        if dropped_stations:
            print(f'  WARNING: dropping {len(dropped_stations)} stations missing in some depth files.')
            print(f'  Dropped station ids: {dropped_stations[:20]}')

        depth_frames = [frame[station_order] for frame in depth_frames]

        # Strict time alignment by timestamp intersection.
        common_time_index = depth_frames[0].index
        for frame in depth_frames[1:]:
            common_time_index = common_time_index.intersection(frame.index)
        common_time_index = common_time_index.sort_values()

        if len(common_time_index) == 0:
            raise RuntimeError('No common timestamps found across all depth files.')

        depth_frames = [frame.loc[common_time_index] for frame in depth_frames]

        print(f'  Found {len(station_order)} aligned stations')
        print(f'  Aligned timestamps: {len(common_time_index)}')
        print(f'  Depth labels: {depth_labels}')

        # Per-depth global fallback mean for stations/channels that are fully missing in split.
        global_mean_by_depth = []
        for frame in depth_frames:
            values = frame.to_numpy(dtype=float)
            values[values == -99.0] = np.nan
            depth_mean = np.nanmean(values)
            if not np.isfinite(depth_mean):
                depth_mean = 0.0
            global_mean_by_depth.append(depth_mean)
        global_mean_by_depth = np.asarray(global_mean_by_depth, dtype=float)
        
        # Prepare data containers
        data = {
            'feat': [],
            'pred': [],
            'missing': [],
            'time': []
        }

        # Split time once for all nodes.
        data_length = len(common_time_index)
        start_index = int(time_division[0] * data_length)
        end_index = int(time_division[1] * data_length)
        sliced_time = common_time_index[start_index:end_index].to_numpy()

        if self.graph_mode == 'joint_4n' and len(depth_files) > 1:
            # Depth-aware node expansion: [N, T, D] -> [D*N, T, 1] in depth-major order.
            for depth_idx, frame in enumerate(depth_frames):
                for station in station_order:
                    node_values = frame[station].to_numpy(dtype=float)
                    node_missing = np.isnan(node_values) | (node_values == -99.0)

                    node_values[node_missing] = np.nan
                    if np.all(node_missing):
                        station_mean = global_mean_by_depth[depth_idx]
                    else:
                        station_mean = np.nanmean(node_values)
                        if not np.isfinite(station_mean):
                            station_mean = global_mean_by_depth[depth_idx]
                    node_values = np.where(np.isnan(node_values), station_mean, node_values)

                    pred_data = node_values[start_index:end_index][:, np.newaxis]
                    missing_data = node_missing[start_index:end_index][:, np.newaxis].astype(np.float32)
                    feat_data = np.zeros((len(pred_data), 0), dtype=np.float32)

                    data['feat'].append(feat_data[np.newaxis, :, :])
                    data['missing'].append(missing_data[np.newaxis, :, :])
                    data['pred'].append(pred_data[np.newaxis, :, :])

            data['time'] = sliced_time
        else:
            # Channel stacking mode: keep node axis as station axis, channels as depth axis.
            for station in station_order:
                station_depth_values = []
                station_depth_missing = []

                for depth_idx, frame in enumerate(depth_frames):
                    channel_values = frame[station].to_numpy(dtype=float)
                    channel_missing = np.isnan(channel_values) | (channel_values == -99.0)

                    channel_values[channel_missing] = np.nan
                    if np.all(channel_missing):
                        station_mean = global_mean_by_depth[depth_idx]
                    else:
                        station_mean = np.nanmean(channel_values)
                        if not np.isfinite(station_mean):
                            station_mean = global_mean_by_depth[depth_idx]
                    channel_values = np.where(np.isnan(channel_values), station_mean, channel_values)

                    station_depth_values.append(channel_values)
                    station_depth_missing.append(channel_missing.astype(np.float32))

                station_values = np.stack(station_depth_values, axis=-1)  # [time, depth]
                station_missing = np.stack(station_depth_missing, axis=-1)  # [time, depth]

                feat_data = np.zeros((len(common_time_index), 0), dtype=np.float32)[start_index:end_index]
                missing_data = station_missing[start_index:end_index]
                pred_data = station_values[start_index:end_index]

                data['feat'].append(feat_data[np.newaxis, :, :])
                data['missing'].append(missing_data[np.newaxis, :, :])
                data['pred'].append(pred_data[np.newaxis, :, :])

            data['time'] = sliced_time

        # Concatenate all stations
        data['feat'] = np.concatenate(data['feat'], axis=0).astype(np.float32)
        data['missing'] = np.concatenate(data['missing'], axis=0).astype(np.float32)
        data['pred'] = np.concatenate(data['pred'], axis=0).astype(np.float32)

        print(f'  Pred data shape: {data["pred"].shape}')
        print(f'  Feat data shape: {data["feat"].shape}')
        print(f'  Missing data shape: {data["missing"].shape}')

        # Compute normalization statistics
        print('Computing normalization info...')

        # For normalization, use non-missing values only.
        pred_data_valid = data['pred'].copy()
        pred_data_valid[data['missing'] == 1] = np.nan

        # Compute per-channel statistics.
        mean_val = np.nanmean(pred_data_valid, axis=(0, 1))
        std_val = np.nanstd(pred_data_valid, axis=(0, 1))
        mean_val = np.atleast_1d(mean_val).astype(float)
        std_val = np.atleast_1d(std_val).astype(float)

        for idx in range(len(mean_val)):
            if not np.isfinite(mean_val[idx]):
                mean_val[idx] = global_mean_by_depth[idx]
            if not np.isfinite(std_val[idx]) or std_val[idx] <= 0:
                std_val[idx] = 1.0
        scale_val = std_val.copy()

        if self.graph_mode == 'joint_4n':
            mean_val = np.array([float(np.nanmean(pred_data_valid))], dtype=float)
            std_scalar = float(np.nanstd(pred_data_valid))
            if not np.isfinite(mean_val[0]):
                mean_val[0] = float(np.nanmean(global_mean_by_depth))
            if not np.isfinite(std_scalar) or std_scalar <= 0:
                std_scalar = 1.0
            scale_val = np.array([std_scalar], dtype=float)
            channel_names = ['SM']
        elif len(depth_files) == 1:
            channel_names = ['SM']
        else:
            channel_names = [f'SM_{depth}' for depth in depth_labels]

        # Create norm info dataframe
        norm_info = pd.DataFrame(
            [mean_val, scale_val, std_val ** 2],
            columns=channel_names,
            index=['mean', 'scale', 'var']
        )

        print(f'  Mean: {mean_val}')
        print(f'  Scale: {scale_val}')

        # Normalize prediction data
        data['pred'] = (data['pred'] - mean_val.reshape(1, 1, -1)) / scale_val.reshape(1, 1, -1)

        # Convert time to seconds since epoch
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00'))
                        / np.timedelta64(1, 's')).astype(np.float64)

        return data, norm_info, station_order
    
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
            test_node_index = np.asarray(test_node_index).astype(int).reshape(-1)

            # For joint_4n, support legacy base-station split by expanding to all depths.
            if self.graph_mode == 'joint_4n' and hasattr(self, 'base_station_ids'):
                base_num = len(self.base_station_ids)
                if base_num > 0 and test_node_index.size > 0 and np.max(test_node_index) < base_num:
                    expanded = []
                    for depth_idx in range(self.num_depths):
                        expanded.extend((test_node_index + depth_idx * base_num).tolist())
                    test_node_index = np.asarray(expanded, dtype=int)
                    print(f'  Expanded base test nodes to joint graph nodes: {len(test_node_index)}')
        else:
            # If test_nodes.npy doesn't exist, randomly select test nodes
            # For soil moisture: typically use bottom 1/3 stations as test
            if self.graph_mode == 'joint_4n' and hasattr(self, 'base_station_ids'):
                base_num = len(self.base_station_ids)
                base_test_num = max(1, base_num // 3)
                print(f'Test nodes file not found, randomly selecting {base_test_num} base stations and expanding to all depths...')
                base_test_nodes = np.random.choice(np.arange(base_num), size=base_test_num, replace=False)
                expanded = []
                for depth_idx in range(self.num_depths):
                    expanded.extend((base_test_nodes + depth_idx * base_num).tolist())
                test_node_index = np.asarray(expanded, dtype=int)
            else:
                print(f'Test nodes file not found, randomly selecting {num_nodes // 3} nodes as test...')
                test_node_index = np.random.choice(
                    np.arange(num_nodes), 
                    size=max(1, num_nodes // 3),
                    replace=False
                )
            # Save it for reproducibility
            os.makedirs(os.path.dirname(test_nodes_path) or '.', exist_ok=True)
            np.save(test_nodes_path, test_node_index)
        
        return np.unique(test_node_index)


# Register this dataset
