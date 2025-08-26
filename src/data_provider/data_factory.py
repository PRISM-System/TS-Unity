import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path

from config.base_config import BaseConfig
from utils.tools import StandardScaler


class _SyntheticTimeSeriesDataset(Dataset):
    """Minimal dataset yielding tuples expected by Exp classes.

    Each item: (batch_x, batch_y, batch_x_mark, batch_y_mark)
    Shapes match usages in exp modules.
    """

    def __init__(self, args: BaseConfig, length: int = 500):
        self.args = args
        self.length = length

        seq_len = args.seq_len
        label_len = getattr(args, 'label_len', max(1, seq_len // 2))
        pred_len = getattr(args, 'pred_len', 1)
        enc_in = args.enc_in
        c_out = args.c_out

        # Pre-generate tensors to be indexed
        self.batch_x = torch.randn(length, seq_len, enc_in)
        # batch_y length must cover label_len + pred_len for forecasting code paths
        self.batch_y = torch.randn(length, label_len + pred_len, c_out)
        # time feature markers (not used heavily in our minimal path but required by signature)
        self.batch_x_mark = torch.zeros(length, seq_len, enc_in)
        self.batch_y_mark = torch.zeros(length, label_len + pred_len, c_out)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return (
            self.batch_x[idx],
            self.batch_y[idx],
            self.batch_x_mark[idx],
            self.batch_y_mark[idx],
        )


class DataFactory:
    """Provide datasets and dataloaders for different flags.

    This is a lightweight fallback to unblock pipeline execution.
    """

    def __init__(self, args: BaseConfig):
        self.args = args

    def get_dataset(self, flag: str) -> Tuple[Dataset, DataLoader]:
        flag = flag.lower()
        if flag not in { 'train', 'val', 'test' }:
            raise ValueError(f"Unsupported flag: {flag}")
        
        # If using ETT datasets, load real CSVs
        if str(self.args.data).lower() in { 'etth1', 'etth2', 'ettm1', 'ettm2' }:
            dataset = _ETTDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'electricity', 'ecl' }:
            dataset = _ElectricityDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'exchange_rate', 'exchange' }:
            dataset = _ExchangeRateDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'illness', 'national_illness' }:
            dataset = _IllnessDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'traffic' }:
            dataset = _TrafficDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'weather' }:
            dataset = _WeatherDataset(self.args, flag=flag)
        elif str(self.args.data).lower() in { 'swat' }:
            dataset = _SWaTDataset(self.args, flag=flag)
        else:
            # Fallback synthetic data
            length = 800 if flag == 'train' else 200
            dataset = _SyntheticTimeSeriesDataset(self.args, length=length)

        shuffle = flag == 'train'
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.args, 'num_workers', 0),
            drop_last=False,
        )
        return dataset, loader


def data_provider(args: BaseConfig, flag: str):
    factory = DataFactory(args)
    return factory.get_dataset(flag)


class _ETTDataset(Dataset):
    """ETT dataset loader for forecasting.

    Expects files under /TS-Unity/datasets/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
    Returns tuples: (batch_x, batch_y, batch_x_mark, batch_y_mark)
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        name_map = {
            'etth1': 'ETTh1.csv',
            'etth2': 'ETTh2.csv',
            'ettm1': 'ETTm1.csv',
            'ettm2': 'ETTm2.csv',
        }
        data_name = name_map[str(args.data).lower()]
        csv_path = Path('/TS-Unity/datasets/ETT-small') / data_name
        if not csv_path.exists():
            raise FileNotFoundError(f"ETT file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # First column is timestamp
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        # Train/Val/Test split as common in ETT: 12/4/4 months equivalent
        n = len(values)
        # Rough split indices
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Fit scaler on train split and transform current split
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        # Set enc/dec dims if not aligned
        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        # Ensure args dims match data
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        # Build time feature markers: [month, day, weekday, hour]
        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _SWaTDataset(Dataset):
    """SWaT dataset loader using PKL files.

    Expects files under /TS-Unity/datasets/SWaT/:
      - train.pkl (or train*.pkl)
      - test.pkl  (or test*.pkl)

    Each PKL should load to a pandas DataFrame or dict-like containing time index/column
    and multivariate features. Optional 'label' column will be ignored for inputs.
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        base = Path('/TS-Unity/datasets/SWaT')
        if not base.exists():
            raise FileNotFoundError(f"SWaT directory not found: {base}")

        # Find pkl files (support common SWaT names)
        import glob
        normal_candidates = (
            sorted(glob.glob(str(base / 'SWaT_Dataset_Normal*.pkl'))) or
            sorted(glob.glob(str(base / 'train*.pkl'))) or
            [str(base / 'train.pkl')]
        )
        attack_candidates = (
            sorted(glob.glob(str(base / 'SWaT_Dataset_Attack*.pkl'))) or
            sorted(glob.glob(str(base / 'test*.pkl'))) or
            [str(base / 'test.pkl')]
        )

        # Load appropriate file
        import pickle
        if flag in ('train', 'val'):
            pkl_path = Path(normal_candidates[0])
        else:
            pkl_path = Path(attack_candidates[0])
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)

        # Convert to DataFrame if needed
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, dict) and 'data' in obj:
            df = pd.DataFrame(obj['data'])
        else:
            df = pd.DataFrame(obj)

        # SWaT conventions per reference snippet
        time_col_candidates = [' Timestamp', 'timestamp', 'time']
        label_col_candidates = ['Normal/Attack', 'label', 'anomaly', 'attack', 'y']
        time_col = next((c for c in time_col_candidates if c in df.columns), None)
        label_col = next((c for c in label_col_candidates if c in df.columns), None)

        if time_col is not None and time_col in df:
            tseries = pd.to_datetime(df[time_col], errors='coerce')
        else:
            # Fallback: generate monotonically increasing times (hourly)
            tseries = pd.date_range('2000-01-01', periods=len(df), freq='H')

        # Feature matrix: drop time/label columns, keep numeric only
        feature_df = df.drop(columns=[c for c in [time_col, label_col] if c is not None])
        feature_df = feature_df.select_dtypes(include=[np.number])
        # If user constrained feature count (via --nvars/enc_in), subselect columns
        try:
            requested_vars = int(getattr(self.args, 'enc_in', 0))
            if requested_vars and requested_vars > 0 and requested_vars <= feature_df.shape[1]:
                feature_df = feature_df.iloc[:, :requested_vars]
        except Exception:
            pass
        values = feature_df.values.astype(np.float32)

        # Optional labels from CSV if available (for test)
        labels_series = None
        label_csv_candidates = sorted(glob.glob(str(base / '*label*.csv')))
        if label_csv_candidates:
            label_df = pd.read_csv(label_csv_candidates[0])
            # Try to find a column with label info
            for c in label_df.columns:
                if str(c).lower() in {'label', 'attack', 'anomaly', 'y'}:
                    labels_series = label_df[c]
                    break
            if labels_series is None and label_df.shape[1] >= 1:
                labels_series = label_df.iloc[:, -1]

        # Split train/val/test per provided reference
        valid_split_rate = 0.9
        if flag == 'train':
            split_idx = int(len(values) * valid_split_rate)
            self.data = values[:split_idx]
            self.time = pd.Series(tseries[:split_idx]).reset_index(drop=True)
            self.has_labels = False
        elif flag == 'val':
            split_idx = int(len(values) * valid_split_rate)
            self.data = values[split_idx:]
            self.time = pd.Series(tseries[split_idx:]).reset_index(drop=True)
            self.has_labels = False
        else:
            self.data = values
            self.time = pd.Series(tseries).reset_index(drop=True)
            # Attach labels from CSV or attack PKL
            if labels_series is not None:
                lab = np.asarray(labels_series).astype(np.float32)
            elif label_col is not None and label_col in df.columns:
                lab_raw = df[label_col].copy()
                lab = np.where(lab_raw.astype(str) == 'Normal', 0.0, 1.0).astype(np.float32)
            else:
                lab = None

            if lab is not None:
                if len(lab) != len(self.data):
                    m = min(len(lab), len(self.data))
                    lab = lab[:m]
                    self.data = self.data[:m]
                    self.time = self.time.iloc[:m].reset_index(drop=True)
                self.labels = lab
                self.has_labels = True
            else:
                self.has_labels = False

        # Standardize using train stats (load train pkl for stats)
        train_candidates = (
            sorted(glob.glob(str(base / 'SWaT_Dataset_Normal*.pkl'))) or
            sorted(glob.glob(str(base / 'train*.pkl'))) or
            [str(base / 'train.pkl')]
        )
        with open(train_candidates[0], 'rb') as f:
            train_obj = pickle.load(f)
        if isinstance(train_obj, pd.DataFrame):
            train_df = train_obj.copy()
        elif isinstance(train_obj, dict) and 'data' in train_obj:
            train_df = pd.DataFrame(train_obj['data'])
        else:
            train_df = pd.DataFrame(train_obj)
        if time_col is not None and time_col in train_df:
            train_df = train_df.drop(columns=[time_col])
        if label_col is not None and label_col in train_df:
            train_df = train_df.drop(columns=[label_col])
        # Ensure numeric-only and match requested feature count
        train_df = train_df.select_dtypes(include=[np.number])
        try:
            requested_vars = int(getattr(self.args, 'enc_in', 0))
            if requested_vars and requested_vars > 0 and requested_vars <= train_df.shape[1]:
                train_df = train_df.iloc[:, :requested_vars]
        except Exception:
            pass
        train_values = train_df.values.astype(np.float32)

        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = self.data.shape[1]
        self.c_out = self.data.shape[1]
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        # For anomaly detection, we still generate windows for batching
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        # For anomaly detection, provide label window if available; else zeros
        if getattr(self, 'has_labels', False):
            lab_slice = self.labels[r_begin:r_end].reshape(-1, 1)
            seq_y = lab_slice.astype(np.float32)
        else:
            seq_y = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)

        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _WeatherDataset(Dataset):
    """Weather dataset loader.

    Assumes CSV at /TS-Unity/datasets/weather/weather.csv with first column timestamp
    and remaining columns as weather variables.
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        csv_path = Path('/TS-Unity/datasets/weather/weather.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Weather file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Standardize using train stats
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _TrafficDataset(Dataset):
    """Traffic dataset loader.

    Assumes CSV at /TS-Unity/datasets/traffic/traffic.csv with first column timestamp
    and remaining columns as traffic flow series.
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        csv_path = Path('/TS-Unity/datasets/traffic/traffic.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Traffic file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Standardize using train stats
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _IllnessDataset(Dataset):
    """National Illness dataset loader.

    Assumes CSV at /TS-Unity/datasets/illness/national_illness.csv with first column timestamp
    and remaining columns as series.
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        csv_path = Path('/TS-Unity/datasets/illness/national_illness.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Illness file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Standardize using train stats
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        def _tf(dt_series: pd.Series) -> np.ndarray:
            # Daily/weekly; hour set to 0
            hours = np.zeros_like(dt_series.dt.day.values, dtype=np.int64)
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                hours,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _ExchangeRateDataset(Dataset):
    """Exchange rate dataset loader.

    Assumes CSV at /TS-Unity/datasets/exchange_rate/exchange_rate.csv with first column timestamp
    and remaining columns as exchange rate series.
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        csv_path = Path('/TS-Unity/datasets/exchange_rate/exchange_rate.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Exchange rate file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Standardize by train split
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


class _ElectricityDataset(Dataset):
    """Electricity dataset loader (UCI Electricity Load Diagrams 2011-2014 variant).

    Assumes CSV at /TS-Unity/datasets/electricity/electricity.csv with first column timestamp
    and remaining columns as features (household loads).
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = getattr(args, 'pred_len', 1)

        csv_path = Path('/TS-Unity/datasets/electricity/electricity.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Electricity file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        time_col = pd.to_datetime(df.iloc[:, 0])
        values = df.iloc[:, 1:].values.astype(np.float32)

        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        if flag == 'train':
            self.data = values[:train_end]
            self.time = time_col.iloc[:train_end].reset_index(drop=True)
        elif flag == 'val':
            self.data = values[train_end:val_end]
            self.time = time_col.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = values[val_end:]
            self.time = time_col.iloc[val_end:].reset_index(drop=True)

        # Fit scaler on train split and transform current split
        train_values = values[:train_end]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0) + 1e-8
        self.scaler = StandardScaler(mean, std)
        self.data_scaled = self.scaler.transform(self.data)

        self.enc_in = values.shape[1]
        self.c_out = values.shape[1]
        # Align args with actual data dims
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_scaled[s_begin:s_end]
        seq_y = self.data_scaled[r_begin:r_end]

        def _tf(dt_series: pd.Series) -> np.ndarray:
            return np.stack([
                dt_series.dt.month.values,
                dt_series.dt.day.values,
                dt_series.dt.weekday.values,
                dt_series.dt.hour.values,
            ], axis=1).astype(np.float32)

        x_mark = _tf(self.time.iloc[s_begin:s_end])
        y_mark = _tf(self.time.iloc[r_begin:r_end])

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )