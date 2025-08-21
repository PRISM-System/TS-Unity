"""
Dataset Module

이 모듈은 이상 탐지 모델을 위한 데이터셋 클래스와 데이터 로딩 함수를 제공합니다.
"""

from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import dateutil
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple, Optional, Union


def dataframe_from_csv(target: str) -> pd.DataFrame:
    """
    CSV 파일에서 데이터프레임을 로드합니다.
    
    Args:
        target: CSV 파일 경로
        
    Returns:
        로드된 데이터프레임
    """
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets: List[str]) -> pd.DataFrame:
    """
    여러 CSV 파일에서 데이터프레임을 로드하고 결합합니다.
    
    Args:
        targets: CSV 파일 경로 리스트
        
    Returns:
        결합된 데이터프레임
    """
    return pd.concat([dataframe_from_csv(x) for x in targets])


class BuildDataset(Dataset):
    """
    이상 탐지용 데이터셋 클래스
    
    시계열 데이터를 윈도우 단위로 분할하여 배치 학습에 적합한 형태로 변환합니다.
    
    Attributes:
        ts: 시계열 데이터의 타임스탬프
        tag_values: 시계열 데이터 값
        window_size: 윈도우 크기
        model_type: 모델 타입 ('reconstruction' 또는 'prediction')
        time_unit: 시간 단위
        valid_idxs: 유효한 윈도우의 시작 인덱스 리스트
        attacks: 이상 라벨 (선택사항)
    """

    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 slide_size: int = 1,
                 attacks: Optional[np.ndarray] = None,
                 model_type: str = 'reconstruction',
                 time_unit: str = 'seconds'):
        """
        Args:
            data: 시계열 데이터 (시간, 특성 수)
            timestamps: 시계열 데이터의 타임스탬프
            window_size: 윈도우 크기
            slide_size: 이동 윈도우 크기
            attacks: 이상 라벨 (선택사항)
            model_type: 모델 타입 ('reconstruction' 또는 'prediction')
            time_unit: 시간 단위 ('seconds', 'minutes', 'hours', 'days')
        """
        self.ts = np.array(timestamps)
        self.tag_values = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.model_type = model_type
        self.time_unit = time_unit

        # 시간 간격 설정
        timedelta_kwargs = {
            self.time_unit: window_size - (1 if self.model_type == 'reconstruction' else 0)
        }
                
        # 유효한 윈도우 인덱스 생성
        self.valid_idxs = self._generate_valid_indices(timedelta_kwargs, slide_size)

        print(f"유효한 윈도우 수: {len(self.valid_idxs)}")

        # 이상 라벨 설정
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
        else:
            self.attacks = None

    def _generate_valid_indices(self, timedelta_kwargs: Dict[str, int], slide_size: int) -> List[int]:
        """
        유효한 윈도우 인덱스를 생성합니다.
        
        Args:
            timedelta_kwargs: 시간 간격 설정
            slide_size: 이동 윈도우 크기
            
        Returns:
            유효한 윈도우의 시작 인덱스 리스트
        """
        valid_idxs = []
        
        if self.model_type == 'reconstruction':
            for L in range(0, len(self.ts) - self.window_size + 1, slide_size):
                R = L + self.window_size - 1
                if self._is_valid_window(L, R, timedelta_kwargs):
                    valid_idxs.append(L)
        elif self.model_type == 'prediction':
            for L in range(0, len(self.ts) - self.window_size, slide_size):
                R = L + self.window_size
                if self._is_valid_window(L, R, timedelta_kwargs):
                    valid_idxs.append(L)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
            
        return valid_idxs

    def _is_valid_window(self, start_idx: int, end_idx: int, 
                        timedelta_kwargs: Dict[str, int]) -> bool:
        """
        윈도우가 유효한지 확인합니다.
        
        Args:
            start_idx: 시작 인덱스
            end_idx: 끝 인덱스
            timedelta_kwargs: 시간 간격 설정
            
        Returns:
            윈도우 유효성 여부
        """
        try:
            # 날짜 파싱 가능한 경우
            start_time = dateutil.parser.parse(self.ts[start_idx])
            end_time = dateutil.parser.parse(self.ts[end_idx])
            expected_delta = timedelta(**timedelta_kwargs)
            return (end_time - start_time) == expected_delta
        except (ValueError, TypeError):
            # 숫자 타임스탬프인 경우
            if self.model_type == 'reconstruction':
                return self.ts[end_idx] - self.ts[start_idx] == self.window_size - 1
            else:
                return self.ts[end_idx] - self.ts[start_idx] == self.window_size

    def __len__(self) -> int:
        """데이터셋의 길이를 반환합니다."""
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 데이터를 반환합니다.
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            데이터 딕셔너리 (given, ts, answer, attack)
        """
        if idx >= len(self.valid_idxs):
            raise IndexError(f"인덱스 {idx}가 범위를 벗어났습니다.")
            
        start_idx = self.valid_idxs[idx]
        
        if self.model_type == 'reconstruction':
            end_idx = start_idx + self.window_size
            given = self.tag_values[start_idx:end_idx]
            answer = self.tag_values[start_idx:end_idx]
            ts = self.ts[start_idx:end_idx]
        else:  # prediction
            end_idx = start_idx + self.window_size
            given = self.tag_values[start_idx:end_idx-1]
            answer = self.tag_values[end_idx-1:end_idx]
            ts = self.ts[start_idx:end_idx]

        # 타임스탬프를 숫자 인덱스로 변환하여 직렬화 가능하게 만듦
        if isinstance(ts[0], (np.datetime64, np.timedelta64)):
            ts = np.arange(len(ts), dtype=np.float32)
        elif isinstance(ts[0], (int, float, np.integer, np.floating)):
            ts = ts.astype(np.float32)
        else:
            # 문자열이나 다른 타입인 경우 인덱스로 변환
            ts = np.arange(len(ts), dtype=np.float32)

        result = {
            'given': torch.FloatTensor(given),
            'ts': torch.FloatTensor(ts),
            'answer': torch.FloatTensor(answer)
        }
        
        if self.attacks is not None:
            if self.model_type == 'reconstruction':
                attack_data = self.attacks[start_idx:end_idx]
            else:
                attack_data = self.attacks[end_idx-1:end_idx]
            
            # attack 데이터가 1차원이 아닌 경우 처리
            if len(attack_data.shape) > 1:
                attack_data = attack_data.flatten()
            
            result['attack'] = torch.FloatTensor(attack_data)
        else:
            # attack이 없는 경우 answer와 같은 형태로 0으로 채움
            if self.model_type == 'reconstruction':
                result['attack'] = torch.zeros_like(result['answer'])
            else:
                result['attack'] = torch.zeros_like(result['answer'])

        return result


def load_dataset(dataname: str, datainfo: Dict[str, Any], 
                subdataname: Optional[str] = None, 
                valid_split_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터셋을 로드하고 학습/검증/테스트 세트로 분할합니다.
    
    Args:
        dataname: 데이터셋 이름 ('SWaT', 'SKAB', 'KOGAS', 'KOGAS2', 'KOGAS3')
        datainfo: 데이터 정보 딕셔너리
        subdataname: 서브 데이터셋 이름 (선택사항)
        valid_split_rate: 검증 세트 분할 비율
        
    Returns:
        (train_data, train_ts, val_data, val_ts, test_data, test_ts, labels) 튜플
        
    Raises:
        AssertionError: 지원하지 않는 데이터셋인 경우
    """
    try:
        assert dataname in ['SWaT', 'SKAB', 'KOGAS', 'KOGAS2', 'KOGAS3']
    except AssertionError as e:
        raise AssertionError(f"지원하지 않는 데이터셋: {dataname}") from e

    if dataname == 'SWaT':
        trainset = pd.read_pickle(datainfo.train_path).drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        testset = pd.read_pickle(datainfo.test_path)
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label = testset['Normal/Attack']
        test_label[test_label == 'Normal'] = 0
        test_label[test_label != 0] = 1
        testset = testset.drop(['Normal/Attack', ' Timestamp'],
                               axis=1).to_numpy()

    elif dataname == 'SKAB':
        dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{subdataname}.csv'), 
                             sep=";", index_col="datetime", parse_dates=True).drop(['changepoint'], axis=1)
        trainset = dataset.copy()[:400].drop(['anomaly'], axis=1).to_numpy()
        validset = dataset.copy()[400:550].drop(['anomaly'], axis=1).to_numpy()
        testset = dataset.copy()[550:]
        test_label = testset.copy()['anomaly'].to_numpy()
        testset = testset.drop(['anomaly'], axis=1).to_numpy()

        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        
    elif dataname == 'KOGAS':
        if subdataname is None:
            dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{dataname}.csv')).dropna()
        else:
            dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{subdataname}.csv')).dropna()
        dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y.%m.%d %H:%M:%S')
        dataset.set_index('datetime', inplace=True)

        trainset = dataset.loc['2021-01-01':'2022-06-30']
        validset = dataset.loc['2022-07-01':'2022-12-31']
        testset = dataset.loc['2023-01-01':'2023-12-31']
        # test_label = np.zeros([len(testset)])
        if subdataname is None:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{dataname}_labels_manual.npy'))
        else:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{subdataname}_labels_manual.npy'))
        
        train_timestamp = np.datetime_as_string(trainset.index)
        valid_timestamp = np.datetime_as_string(validset.index)
        test_timestamp = np.datetime_as_string(testset.index)
        
    elif dataname == 'KOGAS2':
        dataset = pd.read_csv(datainfo.train_path).dropna()
        dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y.%m.%d %H:%M:%S')
        dataset.set_index('datetime', inplace=True)

        trainset = dataset.loc['2021-01-01':'2022-12-31']
        validset = dataset.loc['2023-01-01':'2023-12-31']
        testset = pd.read_csv(datainfo.test_path).dropna()
        testset['datetime'] = pd.to_datetime(testset['datetime'], format='%Y.%m.%d %H:%M:%S')
        testset.set_index('datetime', inplace=True)
        testset = testset.loc['2023-01-01':'2023-12-31']
        
        if subdataname is None:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{dataname}_labels_manual.npy'))
        else:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{subdataname}_labels_manual.npy'))
        
        train_timestamp = np.datetime_as_string(trainset.index)
        valid_timestamp = np.datetime_as_string(validset.index)
        test_timestamp = np.datetime_as_string(testset.index)
        
    elif dataname == 'KOGAS3':
        if subdataname is None:
            dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{dataname}.csv')).dropna()
        else:
            dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{subdataname}.csv')).dropna()
        dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y.%m.%d %H:%M:%S')
        dataset.set_index('datetime', inplace=True)

        trainset = dataset.loc['2021-01-01':'2021-06-30']
        validset = dataset.loc['2021-07-01':'2021-12-31']
        testset = dataset.loc['2023-01-01':'2023-12-31']
        # test_label = np.zeros([len(testset)])
        
        if subdataname is None:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{dataname}_labels_manual.npy'))
        else:
            test_label = np.load(os.path.join(datainfo.data_dir, f'{subdataname}_labels_manual.npy'))

        train_timestamp = np.datetime_as_string(trainset.index)
        valid_timestamp = np.datetime_as_string(validset.index)
        test_timestamp = np.datetime_as_string(testset.index)
    
    else:
        trainset = np.load(os.path.join(datainfo.train_dir, f'{subdataname}.npy'))
        testset = np.load(os.path.join(datainfo.test_dir, f'{subdataname}.npy'))
        valid_split_index = int(len(trainset) * valid_split_rate)
        validset = trainset[valid_split_index:]
        trainset = trainset[:valid_split_index]
        train_timestamp = np.arange(len(trainset))
        valid_timestamp = np.arange(len(validset))
        test_timestamp = np.arange(len(testset))
        test_label_info = pd.read_csv(datainfo.test_label_path, index_col=0).loc[subdataname]
        test_label = np.zeros([int(test_label_info.num_values)], dtype=np.int)

        for i in eval(test_label_info.anomaly_sequences):
            if type(i) == list:
                test_label[i[0]:i[1] + 1] = 1
            else:
                test_label[i] = 1

    return trainset, train_timestamp, validset, valid_timestamp, testset, test_timestamp, test_label