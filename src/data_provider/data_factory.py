import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config.base_config import BaseConfig
# from utils.tools import StandardScaler

class DataScaler:
    """데이터 스케일링을 담당하는 클래스"""
    
    def __init__(self, scale_method: str):
        """
        Args:
            scale_method: 스케일링 방법 ('minmax', 'minmax_square', 'minmax_m1p1', 'standard')
        """
        self.scale_method = scale_method
        self.scaler = None
        
    def fit_transform(self, train_data: np.ndarray, val_data: np.ndarray, 
                     test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터를 스케일링합니다.
        
        Args:
            train_data: 학습 데이터
            val_data: 검증 데이터
            test_data: 테스트 데이터
            
        Returns:
            스케일링된 데이터 튜플 (train, val, test)
        """
        if self.scale_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            train_scaled = self.scaler.fit_transform(train_data)
            val_scaled = self.scaler.transform(val_data)
            test_scaled = self.scaler.transform(test_data)
            
        elif self.scale_method == 'minmax_square':
            self.scaler = MinMaxScaler()
            train_scaled = self.scaler.fit_transform(train_data) ** 2
            val_scaled = self.scaler.transform(val_data) ** 2
            test_scaled = self.scaler.transform(test_data) ** 2
            
        elif self.scale_method == 'minmax_m1p1':
            train_scaled = 2 * (train_data / train_data.max(axis=0)) - 1
            val_scaled = 2 * (val_data / val_data.max(axis=0)) - 1
            test_scaled = 2 * (test_data / test_data.max(axis=0)) - 1
            
        elif self.scale_method == 'standard':
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(train_data)
            val_scaled = self.scaler.transform(val_data)
            test_scaled = self.scaler.transform(test_data)
            
        else:
            raise ValueError(f"지원하지 않는 스케일링 방법: {self.scale_method}")
            
        print(f'{self.scale_method} 정규화 완료')
        return train_scaled, val_scaled, test_scaled


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
        elif str(self.args.data).lower() in { 'psm' }:
            dataset = _PSMDataset(self.args, flag=flag)
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
    """ETT dataset loader for forecasting based on the reference implementation.

    Expects files under /TS-Unity/datasets/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
    Returns tuples: (batch_x, batch_y, batch_x_mark, batch_y_mark)
    """

    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.label_len = getattr(args, 'label_len', max(1, self.seq_len // 2))
        self.pred_len = args.pred_len
        
        # Features handling
        self.features = getattr(args, 'features', 'M')
        self.target = getattr(args, 'target', 'OT')
        self.scale = getattr(args, 'scale', True)
        self.timeenc = getattr(args, 'timeenc', 0)
        self.freq = getattr(args, 'freq', 'h')

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

        self.__read_data__(csv_path)

    def __read_data__(self, csv_path):
        """Read and preprocess the ETT data following the reference implementation."""
        df_raw = pd.read_csv(csv_path)
        
        # Define borders for train/val/test split (12/4/4 months equivalent)
        # 12 months = 12 * 30 * 24 = 8640 hours
        # 4 months = 4 * 30 * 24 = 2880 hours
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        
        # Map flag to set_type
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Handle features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # Skip timestamp column
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            # Default to multivariate
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        # Scaling via DataScaler (fit on train, apply to val/test)
        data_full = df_data.values.astype(np.float32)
        if self.scale:
            train_vals = data_full[border1s[0]:border2s[0]]
            val_vals = data_full[border1s[1]:border2s[1]]
            test_vals = data_full[border1s[2]:border2s[2]]
            scaler = DataScaler(getattr(self.args, 'scale_method', 'standard'))
            train_scaled, val_scaled, test_scaled = scaler.fit_transform(train_vals, val_vals, test_vals)
            data = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)
            self.scaler = None
        else:
            data = data_full
            self.scaler = None

        # Time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 0:
            # Basic time features: month, day, weekday, hour
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Advanced time features (if time_features function exists)
            try:
                from utils.tools import time_features
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            except ImportError:
                # Fallback to basic features
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            # Default to basic features
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Set enc/dec dims if not aligned
        self.enc_in = self.data_x.shape[1]
        self.c_out = self.data_x.shape[1]
        # Ensure args dims match data
        self.args.enc_in = self.enc_in
        self.args.dec_in = self.enc_in
        self.args.c_out = self.c_out

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
        )

    def inverse_transform(self, data):
        """Inverse transform scaled data back to original scale."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

class _BaseADDataset(Dataset):
    """Base dataset class for anomaly detection."""
    
    def __init__(self, args: BaseConfig, flag: str):
        self.args = args
        self.flag = flag        
        
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        raise NotImplementedError
    
class _PSMDataset(_BaseADDataset):
    """PSM dataset loader for anomaly detection."""
    
    def __init__(self, args: BaseConfig, flag: str):
        super().__init__(args, flag)
        
        train_data = pd.read_csv(Path(args.root_path) / 'train.csv')
        train_label = pd.read_csv(Path(args.root_path) / 'train_label.csv')
        test_data = pd.read_csv(Path(args.root_path) / 'test.csv')
        test_label = pd.read_csv(Path(args.root_path) / 'test_label.csv')
        
        train_data = train_data.dropna()
        train_label = train_label.dropna()
        test_data = test_data.dropna()
        test_label = test_label.dropna()
        
        train_total_len = len(train_data)
        train_len = int(train_total_len * self.args.valid_split_rate)
        
        valid_data = train_data[train_len:]
        valid_label = train_label[train_len:]
        train_data = train_data[:train_len]
        train_label = train_label[:train_len]
        
        feature_cols = [col for col in train_data.columns if col != 'timestamp_(min)']
        train_data = train_data[feature_cols]
        valid_data = valid_data[feature_cols]
        test_data = test_data[feature_cols]
        
        ds = DataScaler(getattr(self.args, 'scale_method', 'standard'))
        train_data, valid_data, test_data = ds.fit_transform(train_data, valid_data, test_data)

        if self.flag == 'train':
            self.data = train_data
            self.label_data = train_label
        elif self.flag == 'val':
            self.data = valid_data
            self.label_data = valid_label
        else:
            self.data = test_data
            self.label_data = test_label
        
        self.features = self.data
        self.label_data = self.label_data
        # 라벨 처리 - 연속값을 이진값으로 변환
        if 'attack' in self.label_data.columns:
            # attack 컬럼이 있는 경우
            raw_labels = self.label_data['attack']
        else:
            # attack 컬럼이 없는 경우, 첫 번째 컬럼 사용
            raw_labels = self.label_data.iloc[:, 1]
            
        self.binary_labels = (raw_labels > 0).astype(int)
        
        # 시퀀스 길이 설정
        self.seq_len = args.seq_len
        self.feature_num = len(self.features[0])
        
        # USAD 모델을 위한 차원 설정
        self.win_size = self.seq_len
        self.enc_in = self.feature_num
        self.window_size = self.seq_len  # USAD 모델 호환성
        
        # BuildDataset과 유사한 방식으로 유효한 윈도우 인덱스 생성
        self.valid_idxs = self._generate_valid_indices()
        
        print(f"PSM {flag} dataset - 유효한 윈도우 수: {len(self.valid_idxs)}")
        print(f"특성 수: {self.feature_num}, 시퀀스 길이: {self.seq_len}")
    
    def _generate_valid_indices(self) -> List[int]:
        """유효한 윈도우 인덱스를 생성합니다."""
        valid_idxs = []
        
        # argument의 seq_len을 기반으로 stride 계산
        # train/validation은 stride=seq_len//2, test는 stride=seq_len으로 설정
        if self.flag in ['train', 'val']:
            slide_size = self.seq_len // 2  # 오버랩 (100//2 = 50)
        else:  # test
            slide_size = self.seq_len  # 오버랩 없음 (100)
        
        for i in range(0, len(self.features) - self.seq_len + 1, slide_size):
            valid_idxs.append(i)
            
        return valid_idxs
    
    def __len__(self):
        return len(self.valid_idxs)
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_idxs):
            raise IndexError(f"인덱스 {idx}가 범위를 벗어났습니다.")
        
        start_idx = self.valid_idxs[idx]
        end_idx = start_idx + self.seq_len
        
        # 시퀀스 데이터 추출
        seq = self.features[start_idx:end_idx]
        seq_labels = self.binary_labels[start_idx:end_idx]
        
        # batch_x: (seq_len, feature_num)
        batch_x = torch.FloatTensor(seq)
        
        # batch_y: (seq_len, feature_num) - reconstruction을 위해 입력과 동일
        batch_y = torch.FloatTensor(seq)
        
        # batch_x_mark: (seq_len, feature_num) - 시간 특성 (간단하게 0으로 설정)
        batch_x_mark = torch.zeros(self.seq_len, self.feature_num)
        
        # batch_y_mark: (seq_len, feature_num) - 시간 특성 (간단하게 0으로 설정)
        batch_y_mark = torch.zeros(self.seq_len, self.feature_num)
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark
    
    @property
    def labels(self):
        """시퀀스 라벨을 반환합니다."""
        # 모든 시퀀스의 라벨을 하나의 배열로 변환
        all_labels = []
        for idx in self.valid_idxs:
            start_idx = idx
            end_idx = start_idx + self.seq_len
            seq_labels = self.binary_labels[start_idx:end_idx]
            all_labels.extend(seq_labels)
        return np.array(all_labels)


class _SWaTDataset(_BaseADDataset):
    """SWaT dataset loader using PKL files.

    Expects files under /TS-Unity/datasets/SWaT/:
      - train.pkl (or train*.pkl)
      - test.pkl  (or test*.pkl)

    Each PKL should load to a pandas DataFrame or dict-like containing time index/column
    and multivariate features. Optional 'label' column will be ignored for inputs.
    """

    def __init__(self, args: BaseConfig, flag: str):
        super().__init__(args, flag)
        self.seq_len = args.seq_len
        
        # SWaT (Secure Water Treatment) 데이터셋 로드
        # pickle 파일에서 데이터를 읽고 불필요한 컬럼 제거
        trainset = pd.read_pickle(Path(args.root_path) / 'SWaT_Dataset_Normal_v1.pkl').drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        
        # ===== 학습/검증 데이터 분할 =====
        # 비율 기반으로 학습 데이터를 학습용과 검증용으로 분할
        valid_split_index = int(len(trainset) * self.args.valid_split_rate)  # 분할 지점 계산
        validset = trainset[valid_split_index:]  # 검증 데이터 (뒤쪽 20%)
        trainset = trainset[:valid_split_index]  # 학습 데이터 (앞쪽 80%)
        train_label = np.zeros(len(trainset))
        valid_label = np.zeros(len(validset))
        
        # ===== 테스트 데이터 로드 =====
        # 테스트 데이터는 별도 파일에서 로드
        testset = pd.read_pickle(Path(args.root_path) / 'SWaT_Dataset_Attack_v0.pkl')
        
        # ===== 타임스탬프 생성 =====
        # SWaT 데이터는 연속적인 인덱스를 타임스탬프로 사용
        train_timestamp = np.arange(len(trainset))    # 학습 데이터 타임스탬프
        valid_timestamp = np.arange(len(validset))    # 검증 데이터 타임스탬프
        test_timestamp = np.arange(len(testset))      # 테스트 데이터 타임스탬프
        
        # ===== 이상 라벨 처리 =====
        # 테스트 데이터의 이상 라벨 추출 및 변환
        test_label = testset['Normal/Attack']  # 원본 라벨 추출
        test_label[test_label == 'Normal'] = 0  # 'Normal'을 0으로 변환
        test_label[test_label != 0] = 1         # 나머지('Attack')를 1로 변환
        
        # ===== 테스트 데이터 정리 =====
        # 이상 라벨과 타임스탬프 컬럼을 제거하고 numpy 배열로 변환
        testset = testset.drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        
        ds = DataScaler(getattr(self.args, 'scale_method', 'standard'))
        trainset, validset, testset = ds.fit_transform(trainset, validset, testset)
        
        if self.flag == 'train':
            self.data = trainset
            self.label_data = train_label
        elif self.flag == 'val':
            self.data = validset
            self.label_data = valid_label
        else:
            self.data = testset
            self.label_data = test_label
        
        self.feature_num = self.data.shape[1]
        self.valid_idxs = self._generate_valid_indices()
        
    def _generate_valid_indices(self) -> List[int]:
        """유효한 윈도우 인덱스를 생성합니다."""
        valid_idxs = []
        
        # argument의 seq_len을 기반으로 stride 계산
        # train/validation은 stride=seq_len//2, test는 stride=seq_len으로 설정
        if self.flag in ['train', 'val']:
            slide_size = self.seq_len // 2  # 오버랩 (100//2 = 50)
        else:  # test
            slide_size = self.seq_len  # 오버랩 없음 (100)
            
        for i in range(0, len(self.data) - self.seq_len + 1, slide_size):
            valid_idxs.append(i)
            
        return valid_idxs

    def __len__(self) -> int:
        # For anomaly detection, we still generate windows for batching
        return len(self.valid_idxs)

    def __getitem__(self, index: int):
        if index >= len(self.valid_idxs):
            raise IndexError(f"인덱스 {index}가 범위를 벗어났습니다.")
        
        start_idx = self.valid_idxs[index]
        end_idx = start_idx + self.seq_len
        
        # 시퀀스 데이터 추출
        seq = self.data[start_idx:end_idx]
        seq_labels = self.label_data[start_idx:end_idx]
        
        # batch_x: (seq_len, feature_num)
        batch_x = torch.FloatTensor(seq)
        
        # batch_y: (seq_len, feature_num) - reconstruction을 위해 입력과 동일
        batch_y = torch.FloatTensor(seq)
        
        # batch_x_mark: (seq_len, feature_num) - 시간 특성 (간단하게 0으로 설정)
        batch_x_mark = torch.zeros(self.seq_len, self.feature_num)
        
        # batch_y_mark: (seq_len, feature_num) - 시간 특성 (간단하게 0으로 설정)
        batch_y_mark = torch.zeros(self.seq_len, self.feature_num)
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark


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