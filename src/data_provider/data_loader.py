"""
Data Loader Module
"""

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from data_provider.dataset import BuildDataset, load_dataset


class DataScaler:
    """
    데이터 스케일링을 담당하는 클래스
    
    시계열 데이터의 정규화를 수행하여 모델 학습의 안정성과 성능을 향상시킵니다.
    다양한 스케일링 방법을 지원하며, 학습 데이터를 기준으로 스케일러를 학습하고
    검증/테스트 데이터에 동일한 변환을 적용합니다.
    
    지원하는 스케일링 방법:
    - minmax: Min-Max 정규화 (-1 ~ 1 범위)
    - minmax_square: Min-Max 정규화 후 제곱 적용
    - minmax_m1p1: 수동 Min-Max 정규화 (-1 ~ 1 범위)
    - standard: Z-score 정규화 (평균 0, 표준편차 1)
    """
    
    def __init__(self, scale_method: str):
        """
        DataScaler 인스턴스를 초기화합니다.
        
        Args:
            scale_method (str): 사용할 스케일링 방법
                - 'minmax': MinMaxScaler를 사용하여 -1~1 범위로 정규화
                - 'minmax_square': MinMaxScaler 적용 후 제곱 연산 수행
                - 'minmax_m1p1': 수동으로 -1~1 범위로 정규화
                - 'standard': StandardScaler를 사용하여 Z-score 정규화
        """
        self.scale_method = scale_method  # 스케일링 방법 저장
        self.scaler = None  # 실제 스케일러 객체 (fit 후 생성됨)
        
    def fit_transform(self, train_data: np.ndarray, val_data: np.ndarray, 
                     test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터를 스케일링합니다.
        
        학습 데이터를 기준으로 스케일러를 학습(fit)하고, 모든 데이터에 동일한 변환을 적용합니다.
        이는 데이터 누수(data leakage)를 방지하기 위해 학습 데이터의 통계량만을 사용합니다.
        
        Args:
            train_data (np.ndarray): 학습 데이터
                - shape: (samples, features)
                - 스케일러 학습에 사용되는 기준 데이터
            val_data (np.ndarray): 검증 데이터
                - shape: (samples, features)
                - 학습된 스케일러로 변환됨
            test_data (np.ndarray): 테스트 데이터
                - shape: (samples, features)
                - 학습된 스케일러로 변환됨
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 스케일링된 데이터 튜플
                - (train_scaled, val_scaled, test_scaled)
                - 모든 데이터의 shape는 원본과 동일
                
        Raises:
            ValueError: 지원하지 않는 스케일링 방법인 경우
            
        Note:
            - 학습 데이터의 통계량만을 사용하여 스케일러를 학습
            - 검증/테스트 데이터는 학습된 스케일러로만 변환
            - 이는 모델 평가의 공정성을 보장하기 위함
        """
        # ===== Min-Max 정규화 (-1 ~ 1 범위) =====
        if self.scale_method == 'minmax':
            # sklearn의 MinMaxScaler를 -1~1 범위로 설정
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            # 학습 데이터로 스케일러 학습 및 변환
            train_scaled = self.scaler.fit_transform(train_data)
            # 학습된 스케일러로 검증/테스트 데이터 변환
            val_scaled = self.scaler.transform(val_data)
            test_scaled = self.scaler.transform(test_data)
            
        # ===== Min-Max 정규화 후 제곱 적용 =====
        elif self.scale_method == 'minmax_square':
            # 기본 MinMaxScaler (0~1 범위) 사용
            self.scaler = MinMaxScaler()
            # 정규화 후 제곱 연산으로 비선형 변환
            train_scaled = self.scaler.fit_transform(train_data) ** 2
            val_scaled = self.scaler.transform(val_data) ** 2
            test_scaled = self.scaler.transform(test_data) ** 2
            
        # ===== 수동 Min-Max 정규화 (-1 ~ 1 범위) =====
        elif self.scale_method == 'minmax_m1p1':
            # 수동으로 -1~1 범위로 정규화
            # 공식: 2 * (x / max(x)) - 1
            train_scaled = 2 * (train_data / train_data.max(axis=0)) - 1
            val_scaled = 2 * (val_data / val_data.max(axis=0)) - 1
            test_scaled = 2 * (test_data / test_data.max(axis=0)) - 1
            
        # ===== Z-score 정규화 (평균 0, 표준편차 1) =====
        elif self.scale_method == 'standard':
            # sklearn의 StandardScaler 사용
            self.scaler = StandardScaler()
            # 학습 데이터로 스케일러 학습 및 변환
            train_scaled = self.scaler.fit_transform(train_data)
            # 학습된 스케일러로 검증/테스트 데이터 변환
            val_scaled = self.scaler.transform(val_data)
            test_scaled = self.scaler.transform(test_data)
            
        else:
            # 지원하지 않는 스케일링 방법인 경우 오류 발생
            raise ValueError(f"지원하지 않는 스케일링 방법: {self.scale_method}")
            
        # 스케일링 완료 메시지 출력
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
    이상 탐지 모델을 위한 데이터 로더를 생성하고 반환합니다.
    
    시계열 데이터를 윈도우 단위로 분할하여 학습/검증/테스트/추론용 데이터 로더를 생성합니다.
    데이터 스케일링, 윈도우 생성, 배치 처리 등의 전처리 과정을 포함합니다.
    
    Args:
        data_name (str): 사용할 데이터셋 이름
            - 지원 데이터셋: SWaT, SKAB, KOGAS
        sub_data_name (Optional[str]): 서브 데이터셋 이름
            - SKAB, KOGAS 데이터에서만 사용
            - None인 경우 전체 데이터셋 사용
        data_info (Dict[str, Any]): config.yaml의 데이터 정보
            - 데이터셋 경로, 시간 단위, 특성 정보 등 포함
        loader_params (Dict[str, Any]): 데이터 로더 파라미터
            - batch_size: 배치 크기
            - shuffle: 학습 데이터 셔플 여부
            - num_workers: 데이터 로딩 워커 수
            - pin_memory: GPU 메모리 고정 여부
            - drop_last: 마지막 불완전한 배치 제거 여부
        scale (Optional[str]): 데이터 스케일링 방법
            - None: 스케일링 적용 안함
            - 'minmax': Min-Max 정규화 (-1~1)
            - 'minmax_square': Min-Max 정규화 후 제곱
            - 'minmax_m1p1': 수동 Min-Max 정규화 (-1~1)
            - 'standard': Z-score 정규화
        window_size (int): 시계열 윈도우 크기
            - 기본값: 60 (60개 시점을 하나의 시퀀스로 사용)
            - 모델의 입력 시퀀스 길이를 결정
        slide_size (int): 윈도우 이동 크기
            - 기본값: 30 (30개 시점씩 이동)
            - 학습 데이터 생성 시 사용 (중복 데이터 생성)
        model_type (str): 모델 타입
            - 'reconstruction': 재구성 기반 모델 (기본값)
            - 'prediction': 예측 기반 모델

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]: 데이터 로더 튜플
            - train_dataloader: 학습용 데이터 로더
            - val_dataloader: 검증용 데이터 로더  
            - test_dataloader: 테스트용 데이터 로더
            - inter_dataloader: 추론용 데이터 로더 (배치 크기 1)
        
        각 데이터 로더는 다음을 반환하는 배치:
        - given (torch.Tensor): 입력 시계열 데이터
            - shape: (batch_size, window_size, num_features)
        - ts (torch.Tensor): 입력 타임스탬프
            - shape: (batch_size, window_size)
        - answer (torch.Tensor): 타겟 시계열 데이터
            - reconstruction: (batch_size, window_size, num_features)
            - prediction: (batch_size, 1, num_features)
        - attack (torch.Tensor): 이상 라벨
            - shape: (batch_size, window_size) 또는 (batch_size, 1)

    Raises:
        ValueError: 지원하지 않는 스케일링 방법인 경우
        
    Note:
        - 학습 데이터는 slide_size로 중복 생성하여 데이터 증강 효과
        - 검증/테스트 데이터는 window_size로만 생성하여 중복 없음
        - 추론 데이터는 배치 크기 1로 설정하여 실시간 처리 가능
        - 데이터 누수 방지를 위해 학습 데이터로만 스케일러 학습
    """
    # ===== 스케일링 방법 검증 =====
    # 지원되는 스케일링 방법들을 튜플로 정의
    valid_scales = (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')
    if scale not in valid_scales:
        # 지원하지 않는 스케일링 방법인 경우 오류 발생
        raise ValueError(f"지원하지 않는 스케일링 방법: {scale}. "
                        f"지원되는 방법: {valid_scales}")

    # ===== 데이터셋 로드 =====
    # load_dataset 함수를 통해 원본 데이터, 타임스탬프, 라벨을 로드
    # 반환값: (학습데이터, 학습타임스탬프, 검증데이터, 검증타임스탬프, 테스트데이터, 테스트타임스탬프, 라벨)
    train_data, train_ts, val_data, val_ts, test_data, test_ts, labels = load_dataset(
        dataname=data_name,      # 데이터셋 이름
        datainfo=data_info,      # 데이터 정보 (config.yaml)
        subdataname=sub_data_name # 서브 데이터셋 이름 (선택사항)
    )

    # ===== 데이터 스케일링 적용 =====
    # 스케일링이 요청된 경우 DataScaler를 사용하여 정규화 수행
    if scale is not None:
        # 선택된 스케일링 방법으로 DataScaler 인스턴스 생성
        scaler = DataScaler(scale)
        # 학습 데이터를 기준으로 스케일러 학습 후 모든 데이터에 적용
        # 데이터 누수 방지를 위해 학습 데이터의 통계량만 사용
        train_data, val_data, test_data = scaler.fit_transform(
            train_data, val_data, test_data
        )

    # ===== 데이터셋 빌드 =====
    # BuildDataset 클래스를 사용하여 시계열 윈도우 데이터셋 생성
    
    # ===== 학습 데이터셋 빌드 =====
    print("학습 데이터셋 빌드 시작...")
    train_dataset = BuildDataset(
        train_data, train_ts, window_size, slide_size,  # 데이터, 타임스탬프, 윈도우크기, 슬라이드크기
        attacks=None,  # 학습 데이터에는 라벨이 없음 (비지도 학습)
        model_type=model_type,  # 모델 타입 (reconstruction/prediction)
        time_unit=data_info.time_unit  # 시간 단위 (초, 분, 시간 등)
    )
    print(f"학습 데이터셋 빌드 완료 - 크기: {len(train_dataset)}")

    # ===== 검증 데이터셋 빌드 =====
    print("검증 데이터셋 빌드 시작...")
    val_dataset = BuildDataset(
        val_data, val_ts, window_size, slide_size,  # 검증 데이터도 슬라이드 사용
        attacks=None,  # 검증 데이터에도 라벨이 없음 (모델 선택용)
        model_type=model_type,  # 모델 타입
        time_unit=data_info.time_unit  # 시간 단위
    )
    print(f"검증 데이터셋 빌드 완료 - 크기: {len(val_dataset)}")

    # ===== 테스트 데이터셋 빌드 =====
    print("테스트 데이터셋 빌드 시작...")
    test_dataset = BuildDataset(
        test_data, test_ts, window_size, window_size,  # 테스트는 슬라이드 없이 윈도우 크기만 사용
        attacks=labels,  # 테스트 데이터에는 라벨이 있음 (성능 평가용)
        model_type=model_type,  # 모델 타입
        time_unit=data_info.time_unit  # 시간 단위
    )
    print(f"테스트 데이터셋 빌드 완료 - 크기: {len(test_dataset)}")

    # ===== 추론 데이터셋 빌드 =====
    print("추론 데이터셋 빌드 시작...")
    inter_dataset = BuildDataset(
        test_data, test_ts, window_size, window_size,  # 추론도 테스트와 동일한 설정
        attacks=labels,  # 추론 데이터에도 라벨이 있음 (결과 분석용)
        model_type=model_type,  # 모델 타입
        time_unit=data_info.time_unit  # 시간 단위
    )
    print(f"추론 데이터셋 빌드 완료 - 크기: {len(inter_dataset)}")

    # ===== PyTorch 데이터 로더 생성 =====
    # 각 데이터셋에 대해 PyTorch DataLoader를 생성하여 배치 처리 지원
    
    # ===== 학습 데이터로더 생성 =====
    print("학습 데이터로더 생성 시작...")
    train_dataloader = DataLoader(
        train_dataset,  # 학습 데이터셋
        batch_size=loader_params['batch_size'],  # 배치 크기 (예: 32, 64, 128)
        shuffle=loader_params['shuffle'],  # 데이터 셔플 여부 (학습 시 True 권장)
        num_workers=loader_params['num_workers'],  # 데이터 로딩 워커 수 (병렬 처리)
        pin_memory=loader_params['pin_memory'],  # GPU 메모리 고정 (GPU 사용 시 True 권장)
        drop_last=loader_params.get('drop_last', False)  # 마지막 불완전한 배치 제거 여부
    )
    print(f"학습 데이터로더 생성 완료 - 배치 수: {len(train_dataloader)}")

    # ===== 검증 데이터로더 생성 =====
    print("검증 데이터로더 생성 시작...")
    val_dataloader = DataLoader(
        val_dataset,  # 검증 데이터셋
        batch_size=loader_params['batch_size'],  # 학습과 동일한 배치 크기
        shuffle=False,  # 검증 시에는 셔플하지 않음 (일관된 평가를 위해)
        num_workers=loader_params['num_workers'],  # 워커 수
        pin_memory=loader_params['pin_memory'],  # 메모리 고정
        drop_last=False  # 검증에서는 모든 데이터 사용 (불완전한 배치도 포함)
    )
    print(f"검증 데이터로더 생성 완료 - 배치 수: {len(val_dataloader)}")

    # ===== 테스트 데이터로더 생성 =====
    print("테스트 데이터로더 생성 시작...")
    test_dataloader = DataLoader(
        test_dataset,  # 테스트 데이터셋
        batch_size=loader_params['batch_size'],  # 학습과 동일한 배치 크기
        shuffle=False,  # 테스트 시에는 셔플하지 않음 (일관된 평가를 위해)
        num_workers=loader_params['num_workers'],  # 워커 수
        pin_memory=loader_params['pin_memory'],  # 메모리 고정
        drop_last=False  # 테스트에서는 모든 데이터 사용
    )
    print(f"테스트 데이터로더 생성 완료 - 배치 수: {len(test_dataloader)}")

    # ===== 추론 데이터로더 생성 =====
    print("추론 데이터로더 생성 시작...")
    inter_dataloader = DataLoader(
        inter_dataset,  # 추론 데이터셋 (테스트 데이터와 동일)
        batch_size=1,  # 추론 시에는 배치 크기 1 (실시간 처리)
        shuffle=False,  # 추론 시에는 셔플하지 않음
        num_workers=loader_params['num_workers'],  # 워커 수
        pin_memory=loader_params['pin_memory'],  # 메모리 고정
        drop_last=False  # 모든 데이터 사용
    )
    print(f"추론 데이터로더 생성 완료 - 배치 수: {len(inter_dataloader)}")
    
    # ===== 완료 메시지 =====
    print("모든 데이터로더 생성이 완료되었습니다!")

    # ===== 데이터로더 반환 =====
    # (학습, 검증, 테스트, 추론) 순서로 튜플 반환
    return train_dataloader, val_dataloader, test_dataloader, inter_dataloader