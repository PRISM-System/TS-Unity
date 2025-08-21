"""
Data Loader Module

이 모듈은 이상 탐지 모델을 위한 데이터 로더를 제공합니다.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from data_provider.dataset import BuildDataset, load_dataset


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


def get_dataloader(
    data_name: str,
    sub_data_name: Optional[str],
    data_info: Dict[str, Any],
    loader_params: Dict[str, Any],
    scale: Optional[str] = None,
    window_size: int = 60,
    slide_size: int = 30,
    model_type: str = 'reconstruction'
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    데이터 로더를 반환합니다.

    Args:
        data_name: 데이터셋 이름
        sub_data_name: 서브 데이터셋 이름 (SMD, SMAP, MSL 데이터에서만 사용)
        data_info: config.yaml의 데이터 정보
        loader_params: 데이터 로더 파라미터
        scale: 스케일링 방법 (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')
        window_size: 시계열 조건을 위한 윈도우 크기
        slide_size: 이동 윈도우 크기
        model_type: 모델 타입 ('reconstruction', 'prediction')

    Returns:
        학습/검증/테스트/인터벌 데이터 로더 튜플
        
        각 데이터 로더는 다음을 반환:
        - given: 입력 시계열 데이터 (batch_size, window_size, num_features)
        - ts: 입력 타임스탬프 (batch_size, window_size)
        - answer: 타겟 시계열 데이터 (batch_size, window_size, num_features)
                 (model_type이 prediction인 경우 window_size는 1)
        - attack: 이상 라벨

    Raises:
        ValueError: 지원하지 않는 스케일링 방법인 경우
    """
    # 스케일링 방법 검증
    valid_scales = (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')
    if scale not in valid_scales:
        raise ValueError(f"지원하지 않는 스케일링 방법: {scale}. "
                        f"지원되는 방법: {valid_scales}")

    # 데이터셋 로드 (데이터, 타임스탬프, 라벨)
    train_data, train_ts, val_data, val_ts, test_data, test_ts, labels = load_dataset(
        dataname=data_name,
        datainfo=data_info,
        subdataname=sub_data_name
    )

    # 스케일링 적용
    if scale is not None:
        scaler = DataScaler(scale)
        train_data, val_data, test_data = scaler.fit_transform(
            train_data, val_data, test_data
        )

    # 데이터셋 빌드
    print("학습 데이터셋 빌드 시작...")
    train_dataset = BuildDataset(
        train_data, train_ts, window_size, slide_size,
        attacks=None, model_type=model_type, time_unit=data_info.time_unit
    )
    print(f"학습 데이터셋 빌드 완료 - 크기: {len(train_dataset)}")

    print("검증 데이터셋 빌드 시작...")
    val_dataset = BuildDataset(
        val_data, val_ts, window_size, slide_size,
        attacks=None, model_type=model_type, time_unit=data_info.time_unit
    )
    print(f"검증 데이터셋 빌드 완료 - 크기: {len(val_dataset)}")

    print("테스트 데이터셋 빌드 시작...")
    test_dataset = BuildDataset(
        test_data, test_ts, window_size, window_size,
        attacks=labels, model_type=model_type, time_unit=data_info.time_unit
    )
    print(f"테스트 데이터셋 빌드 완료 - 크기: {len(test_dataset)}")

    print("추론 데이터셋 빌드 시작...")
    inter_dataset = BuildDataset(
        test_data, test_ts, window_size, window_size,
        attacks=labels, model_type=model_type, time_unit=data_info.time_unit
    )
    print(f"추론 데이터셋 빌드 완료 - 크기: {len(inter_dataset)}")

    # PyTorch 데이터 로더 생성
    print("학습 데이터로더 생성 시작...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=loader_params['batch_size'],
        shuffle=loader_params['shuffle'],
        num_workers=loader_params['num_workers'],
        pin_memory=loader_params['pin_memory'],
        drop_last=loader_params.get('drop_last', False)
    )
    print(f"학습 데이터로더 생성 완료 - 배치 수: {len(train_dataloader)}")

    print("검증 데이터로더 생성 시작...")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=loader_params['batch_size'],
        shuffle=False,
        num_workers=loader_params['num_workers'],
        pin_memory=loader_params['pin_memory'],
        drop_last=False
    )
    print(f"검증 데이터로더 생성 완료 - 배치 수: {len(val_dataloader)}")

    print("테스트 데이터로더 생성 시작...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=loader_params['batch_size'],
        shuffle=False,
        num_workers=loader_params['num_workers'],
        pin_memory=loader_params['pin_memory'],
        drop_last=False
    )
    print(f"테스트 데이터로더 생성 완료 - 배치 수: {len(test_dataloader)}")

    print("추론 데이터로더 생성 시작...")
    inter_dataloader = DataLoader(
        inter_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=loader_params['num_workers'],
        pin_memory=loader_params['pin_memory'],
        drop_last=False
    )
    print(f"추론 데이터로더 생성 완료 - 배치 수: {len(inter_dataloader)}")
    
    print("모든 데이터로더 생성이 완료되었습니다!")

    return train_dataloader, val_dataloader, test_dataloader, inter_dataloader