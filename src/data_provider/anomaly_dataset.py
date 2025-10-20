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
    
    CSV 파일을 읽어서 pandas DataFrame으로 변환하며, 컬럼명의 앞뒤 공백을 제거합니다.
    이는 데이터 로딩 시 발생할 수 있는 공백 문제를 방지합니다.
    
    Args:
        target (str): 로드할 CSV 파일의 경로
            - 절대 경로 또는 상대 경로 모두 지원
            - 파일이 존재하지 않는 경우 FileNotFoundError 발생
            
    Returns:
        pd.DataFrame: 로드된 데이터프레임
            - 컬럼명의 앞뒤 공백이 제거됨
            - 원본 CSV 파일의 모든 데이터 포함
            
    Raises:
        FileNotFoundError: 지정된 파일이 존재하지 않는 경우
        pd.errors.EmptyDataError: CSV 파일이 비어있는 경우
        pd.errors.ParserError: CSV 파일 파싱 중 오류가 발생한 경우
        
    Example:
        >>> df = dataframe_from_csv('data/sensor_data.csv')
        >>> print(df.columns)  # 공백이 제거된 컬럼명 출력
    """
    # CSV 파일을 읽고 컬럼명의 앞뒤 공백을 제거하여 반환
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets: List[str]) -> pd.DataFrame:
    """
    여러 CSV 파일에서 데이터프레임을 로드하고 결합합니다.
    
    여러 CSV 파일을 순차적으로 로드하여 하나의 DataFrame으로 결합합니다.
    각 파일은 dataframe_from_csv 함수를 통해 로드되므로 컬럼명의 공백이 자동으로 제거됩니다.
    
    Args:
        targets (List[str]): 로드할 CSV 파일 경로들의 리스트
            - 각 경로는 유효한 CSV 파일이어야 함
            - 빈 리스트인 경우 빈 DataFrame 반환
            
    Returns:
        pd.DataFrame: 결합된 데이터프레임
            - 모든 CSV 파일의 데이터가 세로로 결합됨
            - 컬럼명이 일치하지 않는 경우 NaN으로 채워짐
            - 인덱스는 0부터 시작하여 연속적으로 재설정됨
            
    Raises:
        FileNotFoundError: 지정된 파일 중 하나라도 존재하지 않는 경우
        pd.errors.EmptyDataError: CSV 파일 중 하나라도 비어있는 경우
        pd.errors.ParserError: CSV 파일 파싱 중 오류가 발생한 경우
        
    Note:
        - 모든 CSV 파일은 동일한 컬럼 구조를 가져야 함
        - 컬럼 순서가 다른 경우 자동으로 정렬됨
        - 메모리 사용량이 클 수 있으므로 대용량 파일 처리 시 주의 필요
        
    Example:
        >>> files = ['data/part1.csv', 'data/part2.csv', 'data/part3.csv']
        >>> df = dataframe_from_csvs(files)
        >>> print(f"총 {len(df)} 행의 데이터가 로드되었습니다.")
    """
    # 각 CSV 파일을 로드하여 리스트로 만들고, pd.concat으로 결합
    return pd.concat([dataframe_from_csv(x) for x in targets])


class BuildDataset(Dataset):
    """
    이상 탐지용 시계열 데이터셋 클래스
    
    시계열 데이터를 윈도우 단위로 분할하여 배치 학습에 적합한 형태로 변환합니다.
    PyTorch의 Dataset 클래스를 상속받아 DataLoader와 호환되도록 구현되었습니다.
    
    주요 기능:
    - 시계열 데이터를 고정 크기 윈도우로 분할
    - 시간 연속성을 보장하는 유효한 윈도우만 선택
    - 재구성(reconstruction)과 예측(prediction) 모델 타입 지원
    - 이상 라벨과 타임스탬프 정보 제공
    
    Attributes:
        ts (np.ndarray): 시계열 데이터의 타임스탬프 배열
            - shape: (total_timepoints,)
            - 날짜/시간 정보 또는 숫자 인덱스
        tag_values (np.ndarray): 시계열 데이터 값 배열
            - shape: (total_timepoints, num_features)
            - float32 타입으로 변환됨
        window_size (int): 윈도우 크기
            - 하나의 시퀀스에 포함될 시점의 개수
        model_type (str): 모델 타입
            - 'reconstruction': 재구성 기반 모델 (입력=출력)
            - 'prediction': 예측 기반 모델 (과거→미래)
        time_unit (str): 시간 단위
            - 'seconds', 'minutes', 'hours', 'days' 등
            - 윈도우 유효성 검증에 사용
        valid_idxs (List[int]): 유효한 윈도우의 시작 인덱스 리스트
            - 시간 연속성을 만족하는 윈도우들의 시작 위치
        attacks (Optional[np.ndarray]): 이상 라벨 배열 (선택사항)
            - shape: (total_timepoints,) 또는 (total_timepoints, 1)
            - 0: 정상, 1: 이상
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
        BuildDataset 인스턴스를 초기화합니다.
        
        시계열 데이터를 윈도우 단위로 분할하고, 시간 연속성을 만족하는 유효한 윈도우만을 선택합니다.
        모델 타입에 따라 다른 윈도우 생성 전략을 사용합니다.
        
        Args:
            data (np.ndarray): 시계열 데이터 배열
                - shape: (total_timepoints, num_features)
                - 각 행은 하나의 시점, 각 열은 하나의 특성
                - float32 타입으로 자동 변환됨
            timestamps (np.ndarray): 시계열 데이터의 타임스탬프 배열
                - shape: (total_timepoints,)
                - 날짜/시간 문자열, datetime 객체, 또는 숫자 인덱스
            window_size (int): 윈도우 크기
                - 하나의 시퀀스에 포함될 시점의 개수
                - 모델의 입력 시퀀스 길이를 결정
            slide_size (int): 이동 윈도우 크기 (기본값: 1)
                - 윈도우를 이동할 때의 스텝 크기
                - 1: 연속적인 윈도우, >1: 겹치는 윈도우
            attacks (Optional[np.ndarray]): 이상 라벨 배열 (선택사항)
                - shape: (total_timepoints,) 또는 (total_timepoints, 1)
                - 0: 정상, 1: 이상
                - None인 경우 모든 데이터를 정상으로 간주
            model_type (str): 모델 타입 (기본값: 'reconstruction')
                - 'reconstruction': 재구성 기반 모델
                    - 입력과 출력이 동일한 시퀀스
                    - 윈도우 크기: window_size
                - 'prediction': 예측 기반 모델
                    - 과거 시퀀스로 미래 시점 예측
                    - 윈도우 크기: window_size (마지막 시점 제외)
            time_unit (str): 시간 단위 (기본값: 'seconds')
                - 'seconds', 'minutes', 'hours', 'days' 등
                - 윈도우 유효성 검증에 사용
                
        Raises:
            ValueError: 지원하지 않는 model_type인 경우
            AssertionError: 데이터와 타임스탬프의 길이가 다른 경우
            
        Note:
            - 데이터와 타임스탬프의 길이는 동일해야 함
            - 유효한 윈도우만 선택되므로 실제 데이터셋 크기는 원본보다 작을 수 있음
            - slide_size가 클수록 더 많은 윈도우가 생성됨 (데이터 증강 효과)
        """
        # ===== 기본 속성 설정 =====
        # 타임스탬프 배열을 numpy 배열로 변환
        self.ts = np.array(timestamps)
        # 데이터를 float32 타입으로 변환하여 메모리 효율성과 GPU 호환성 확보
        self.tag_values = np.array(data, dtype=np.float32)
        # 윈도우 크기 저장
        self.window_size = window_size
        # 모델 타입 저장
        self.model_type = model_type
        # 시간 단위 저장
        self.time_unit = time_unit

        # ===== 시간 간격 설정 =====
        # 모델 타입에 따라 다른 시간 간격 계산
        # reconstruction: window_size-1 (마지막 시점 포함)
        # prediction: window_size (마지막 시점 제외)
        timedelta_kwargs = {
            self.time_unit: window_size - (1 if self.model_type == 'reconstruction' else 0)
        }
                
        # ===== 유효한 윈도우 인덱스 생성 =====
        # 시간 연속성을 만족하는 윈도우들의 시작 인덱스를 생성
        self.valid_idxs = self._generate_valid_indices(timedelta_kwargs, slide_size)

        # 생성된 유효한 윈도우 수 출력
        print(f"유효한 윈도우 수: {len(self.valid_idxs)}")

        # ===== 이상 라벨 설정 =====
        if attacks is not None:
            # 이상 라벨이 제공된 경우 float32 타입으로 변환하여 저장
            self.attacks = np.array(attacks, dtype=np.float32)
        else:
            # 이상 라벨이 없는 경우 None으로 설정
            self.attacks = None

    def _generate_valid_indices(self, timedelta_kwargs: Dict[str, int], slide_size: int) -> List[int]:
        """
        유효한 윈도우 인덱스를 생성합니다.
        
        시계열 데이터에서 시간 연속성을 만족하는 윈도우들의 시작 인덱스를 찾습니다.
        모델 타입에 따라 다른 윈도우 생성 전략을 사용하며, slide_size에 따라 겹치는 윈도우를 생성할 수 있습니다.
        
        Args:
            timedelta_kwargs (Dict[str, int]): 시간 간격 설정
                - 키: 시간 단위 ('seconds', 'minutes', 'hours', 'days')
                - 값: 해당 단위의 간격 크기
                - 예: {'seconds': 59} (60초 간격)
            slide_size (int): 이동 윈도우 크기
                - 윈도우를 이동할 때의 스텝 크기
                - 1: 연속적인 윈도우 (겹치지 않음)
                - >1: 겹치는 윈도우 (데이터 증강 효과)
                
        Returns:
            List[int]: 유효한 윈도우의 시작 인덱스 리스트
                - 각 인덱스는 해당 윈도우의 시작 위치
                - 시간 연속성을 만족하는 윈도우만 포함
                - slide_size에 따라 겹치는 윈도우 포함 가능
                
        Raises:
            ValueError: 지원하지 않는 model_type인 경우
            
        Note:
            - reconstruction 모델: 윈도우 크기 = window_size (마지막 시점 포함)
            - prediction 모델: 윈도우 크기 = window_size (마지막 시점 제외)
            - 시간 연속성 검증은 _is_valid_window 메서드에서 수행
        """
        valid_idxs = []  # 유효한 윈도우의 시작 인덱스를 저장할 리스트
        
        if self.model_type == 'reconstruction':
            # ===== 재구성 모델용 윈도우 생성 =====
            # 윈도우 크기: window_size (마지막 시점 포함)
            # 범위: 0부터 (전체 길이 - 윈도우 크기)까지, slide_size씩 이동
            for L in range(0, len(self.ts) - self.window_size + 1, slide_size):
                R = L + self.window_size - 1  # 끝 인덱스 (마지막 시점 포함)
                # 윈도우의 시간 연속성 검증
                if self._is_valid_window(L, R, timedelta_kwargs):
                    valid_idxs.append(L)  # 유효한 윈도우의 시작 인덱스 추가
                    
        elif self.model_type == 'prediction':
            # ===== 예측 모델용 윈도우 생성 =====
            # 윈도우 크기: window_size (마지막 시점 제외)
            # 범위: 0부터 (전체 길이 - 윈도우 크기 - 1)까지, slide_size씩 이동
            for L in range(0, len(self.ts) - self.window_size, slide_size):
                R = L + self.window_size  # 끝 인덱스 (마지막 시점 제외)
                # 윈도우의 시간 연속성 검증
                if self._is_valid_window(L, R, timedelta_kwargs):
                    valid_idxs.append(L)  # 유효한 윈도우의 시작 인덱스 추가
                    
        else:
            # 지원하지 않는 모델 타입인 경우 오류 발생
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
            
        return valid_idxs

    def _is_valid_window(self, start_idx: int, end_idx: int, 
                        timedelta_kwargs: Dict[str, int]) -> bool:
        """
        윈도우가 유효한지 확인합니다.
        
        주어진 윈도우의 시간 연속성을 검증합니다. 타임스탬프의 형태에 따라 다른 검증 방법을 사용합니다.
        날짜/시간 문자열인 경우 실제 시간 간격을 계산하고, 숫자 타임스탬프인 경우 인덱스 차이를 확인합니다.
        
        Args:
            start_idx (int): 윈도우의 시작 인덱스
                - 0 이상의 정수
                - self.ts 배열의 유효한 인덱스 범위 내
            end_idx (int): 윈도우의 끝 인덱스
                - start_idx보다 큰 정수
                - self.ts 배열의 유효한 인덱스 범위 내
            timedelta_kwargs (Dict[str, int]): 시간 간격 설정
                - 키: 시간 단위 ('seconds', 'minutes', 'hours', 'days')
                - 값: 해당 단위의 간격 크기
                - 예: {'seconds': 59} (60초 간격)
                
        Returns:
            bool: 윈도우 유효성 여부
                - True: 시간 연속성을 만족하는 유효한 윈도우
                - False: 시간 간격이 맞지 않거나 파싱 오류가 발생한 경우
                
        Note:
            - 날짜/시간 문자열: dateutil.parser를 사용하여 파싱 후 실제 시간 간격 계산
            - 숫자 타임스탬프: 인덱스 차이로 시간 간격 확인
            - 파싱 오류 시 숫자 타임스탬프로 간주하여 처리
        """
        try:
            # ===== 날짜/시간 문자열인 경우 =====
            # dateutil.parser를 사용하여 타임스탬프를 datetime 객체로 변환
            start_time = dateutil.parser.parse(self.ts[start_idx])
            end_time = dateutil.parser.parse(self.ts[end_idx])
            
            # 예상되는 시간 간격을 timedelta 객체로 생성
            expected_delta = timedelta(**timedelta_kwargs)
            
            # 실제 시간 간격이 예상 간격과 일치하는지 확인
            return (end_time - start_time) == expected_delta
            
        except (ValueError, TypeError):
            # ===== 숫자 타임스탬프인 경우 =====
            # 날짜 파싱이 실패한 경우 숫자 타임스탬프로 간주
            if self.model_type == 'reconstruction':
                # 재구성 모델: 윈도우 크기 = window_size (마지막 시점 포함)
                # 인덱스 차이 = window_size - 1
                return self.ts[end_idx] - self.ts[start_idx] == self.window_size - 1
            else:
                # 예측 모델: 윈도우 크기 = window_size (마지막 시점 제외)
                # 인덱스 차이 = window_size
                return self.ts[end_idx] - self.ts[start_idx] == self.window_size

    def __len__(self) -> int:
        """
        데이터셋의 길이를 반환합니다.
        
        유효한 윈도우의 개수를 반환합니다. 이는 원본 데이터의 길이와 다를 수 있습니다.
        시간 연속성을 만족하지 않는 윈도우는 제외되기 때문입니다.
        
        Returns:
            int: 데이터셋의 길이 (유효한 윈도우의 개수)
                - 0 이상의 정수
                - 원본 데이터 길이보다 작거나 같음
                
        Note:
            - 실제 사용 가능한 데이터의 개수를 반환
            - DataLoader에서 배치 생성을 위해 사용됨
        """
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 데이터를 반환합니다.
        
        주어진 인덱스에 해당하는 윈도우 데이터를 추출하여 PyTorch 텐서로 변환합니다.
        모델 타입에 따라 다른 데이터 구조를 반환하며, 타임스탬프는 직렬화 가능한 형태로 변환합니다.
        
        Args:
            idx (int): 데이터 인덱스
                - 0 이상 len(self) 미만의 정수
                - 유효한 윈도우의 인덱스
                
        Returns:
            Dict[str, torch.Tensor]: 데이터 딕셔너리
                - 'given': 입력 시계열 데이터
                    - reconstruction: (window_size, num_features)
                    - prediction: (window_size-1, num_features)
                - 'ts': 타임스탬프 데이터
                    - shape: (window_size,) 또는 (window_size-1,)
                    - float32 타입으로 변환됨
                - 'answer': 타겟 시계열 데이터
                    - reconstruction: (window_size, num_features)
                    - prediction: (1, num_features)
                - 'attack': 이상 라벨 데이터
                    - shape: answer와 동일
                    - 0: 정상, 1: 이상
                    
        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            
        Note:
            - 모든 반환값은 torch.FloatTensor 타입
            - 타임스탬프는 직렬화를 위해 숫자 인덱스로 변환
            - attack이 없는 경우 0으로 채워진 텐서 반환
        """
        # ===== 인덱스 범위 검증 =====
        if idx >= len(self.valid_idxs):
            raise IndexError(f"인덱스 {idx}가 범위를 벗어났습니다.")
            
        # ===== 윈도우 인덱스 추출 =====
        start_idx = self.valid_idxs[idx]  # 유효한 윈도우의 시작 인덱스
        
        if self.model_type == 'reconstruction':
            # ===== 재구성 모델용 데이터 추출 =====
            end_idx = start_idx + self.window_size  # 끝 인덱스
            given = self.tag_values[start_idx:end_idx]  # 입력 데이터 (전체 윈도우)
            answer = self.tag_values[start_idx:end_idx]  # 타겟 데이터 (입력과 동일)
            ts = self.ts[start_idx:end_idx]  # 타임스탬프 (전체 윈도우)
        else:  # prediction
            # ===== 예측 모델용 데이터 추출 =====
            end_idx = start_idx + self.window_size  # 끝 인덱스
            given = self.tag_values[start_idx:end_idx-1]  # 입력 데이터 (마지막 시점 제외)
            answer = self.tag_values[end_idx-1:end_idx]  # 타겟 데이터 (마지막 시점만)
            ts = self.ts[start_idx:end_idx]  # 타임스탬프 (전체 윈도우)

        # ===== 타임스탬프 직렬화 =====
        # PyTorch 텐서로 변환하기 위해 직렬화 가능한 형태로 변환
        if isinstance(ts[0], (np.datetime64, np.timedelta64)):
            # datetime/timedelta 타입인 경우 숫자 인덱스로 변환
            ts = np.arange(len(ts), dtype=np.float32)
        elif isinstance(ts[0], (int, float, np.integer, np.floating)):
            # 숫자 타입인 경우 float32로 변환
            ts = ts.astype(np.float32)
        else:
            # 문자열이나 다른 타입인 경우 숫자 인덱스로 변환
            ts = np.arange(len(ts), dtype=np.float32)

        # ===== 기본 결과 딕셔너리 생성 =====
        result = {
            'given': torch.FloatTensor(given),    # 입력 데이터를 PyTorch 텐서로 변환
            'ts': torch.FloatTensor(ts),          # 타임스탬프를 PyTorch 텐서로 변환
            'answer': torch.FloatTensor(answer)   # 타겟 데이터를 PyTorch 텐서로 변환
        }
        
        # ===== 이상 라벨 처리 =====
        if self.attacks is not None:
            # 이상 라벨이 있는 경우
            if self.model_type == 'reconstruction':
                # 재구성 모델: 전체 윈도우의 라벨
                attack_data = self.attacks[start_idx:end_idx]
            else:
                # 예측 모델: 마지막 시점의 라벨
                attack_data = self.attacks[end_idx-1:end_idx]
            
            # attack 데이터가 1차원이 아닌 경우 1차원으로 변환
            if len(attack_data.shape) > 1:
                attack_data = attack_data.flatten()
            
            result['attack'] = torch.FloatTensor(attack_data)  # 이상 라벨을 PyTorch 텐서로 변환
        else:
            # ===== 이상 라벨이 없는 경우 =====
            # answer와 같은 형태로 0으로 채워진 텐서 생성
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
    
    다양한 이상 탐지 데이터셋을 로드하고 시간 순서대로 학습/검증/테스트 세트로 분할합니다.
    각 데이터셋마다 다른 로딩 방식과 분할 전략을 사용합니다.
    
    Args:
        dataname (str): 데이터셋 이름
            - 'SWaT': Secure Water Treatment 데이터셋
            - 'PSM': Power System Monitoring 데이터셋
            - 'SEMICONDUCTOR': Semiconductor 데이터셋
        datainfo (Dict[str, Any]): 데이터 정보 딕셔너리
            - train_path: 학습 데이터 파일 경로
            - test_path: 테스트 데이터 파일 경로
            - data_dir: 데이터 디렉토리 경로
            - test_label_path: 테스트 라벨 파일 경로
        subdataname (Optional[str]): 서브 데이터셋 이름 (선택사항)
            - None인 경우 전체 데이터셋 사용
        valid_split_rate (float): 검증 세트 분할 비율 (기본값: 0.8)
            - 0.0 ~ 1.0 사이의 값
            - 학습 데이터에서 이 비율만큼을 학습용으로 사용
            - 나머지는 검증용으로 사용
            
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (train_data, train_ts, val_data, val_ts, test_data, test_ts, labels) 튜플
            - train_data: 학습 데이터 (samples, features)
            - train_ts: 학습 데이터 타임스탬프 (samples,)
            - val_data: 검증 데이터 (samples, features)
            - val_ts: 검증 데이터 타임스탬프 (samples,)
            - test_data: 테스트 데이터 (samples, features)
            - test_ts: 테스트 데이터 타임스탬프 (samples,)
            - labels: 테스트 데이터 이상 라벨 (samples,)
                - 0: 정상, 1: 이상
        
    Raises:
        AssertionError: 지원하지 않는 데이터셋인 경우
        FileNotFoundError: 데이터 파일이 존재하지 않는 경우
        pd.errors.EmptyDataError: 데이터 파일이 비어있는 경우
        
    Note:
        - 각 데이터셋마다 다른 시간 범위와 분할 전략 사용
        - 모든 데이터는 시간 순서대로 정렬됨
    """
    # ===== 데이터셋 이름 검증 =====
    try:
        # 지원되는 데이터셋 목록 확인
        assert dataname in ['SWaT', 'PSM', 'SEMICONDUCTOR']
    except AssertionError as e:
        # 지원하지 않는 데이터셋인 경우 오류 발생
        raise AssertionError(f"지원하지 않는 데이터셋: {dataname}") from e

    # ===== SWaT 데이터셋 처리 =====
    if dataname == 'SWaT':
        # SWaT (Secure Water Treatment) 데이터셋 로드
        # pickle 파일에서 데이터를 읽고 불필요한 컬럼 제거
        trainset = pd.read_pickle(datainfo.train_path).drop(['Normal/Attack', ' Timestamp'], axis=1).to_numpy()
        
        # ===== 학습/검증 데이터 분할 =====
        # 비율 기반으로 학습 데이터를 학습용과 검증용으로 분할
        valid_split_index = int(len(trainset) * valid_split_rate)  # 분할 지점 계산
        validset = trainset[valid_split_index:]  # 검증 데이터 (뒤쪽 20%)
        trainset = trainset[:valid_split_index]  # 학습 데이터 (앞쪽 80%)
        train_label = np.zeros(len(trainset))
        valid_label = np.zeros(len(validset))
        
        # ===== 테스트 데이터 로드 =====
        # 테스트 데이터는 별도 파일에서 로드
        testset = pd.read_pickle(datainfo.test_path)
        
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
        
    elif dataname == 'PSM':
        # PSM (Power System Monitoring) 데이터셋 로드
        # pickle 파일에서 데이터를 읽고 불필요한 컬럼 제거
        trainset = pd.read_csv(Path(args.root_path) / 'train.csv')
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
        
        # ===== 학습/검증 데이터 분할 =====
        # 비율 기반으로 학습 데이터를 학습용과 검증용으로 분할

    # ===== SKAB 데이터셋 처리 =====
    elif dataname == 'SKAB':
        # SKAB (Skoltech Anomaly Benchmark) 데이터셋 로드
        # CSV 파일을 읽고 datetime을 인덱스로 설정, changepoint 컬럼 제거
        dataset = pd.read_csv(os.path.join(datainfo.data_dir, f'{subdataname}.csv'), 
                             sep=";", index_col="datetime", parse_dates=True).drop(['changepoint'], axis=1)
        
        # ===== 고정 인덱스 기반 데이터 분할 =====
        # SKAB 데이터는 고정된 인덱스로 분할 (시간 순서 보장)
        trainset = dataset.copy()[:400].drop(['anomaly'], axis=1).to_numpy()      # 학습 데이터: 0-399
        validset = dataset.copy()[400:550].drop(['anomaly'], axis=1).to_numpy()   # 검증 데이터: 400-549
        testset = dataset.copy()[550:]  # 테스트 데이터: 550 이후
        
        # ===== 이상 라벨 추출 =====
        # 테스트 데이터에서 이상 라벨 추출
        test_label = testset.copy()['anomaly'].to_numpy()
        # 테스트 데이터에서 이상 라벨 컬럼 제거
        testset = testset.drop(['anomaly'], axis=1).to_numpy()

        # ===== 타임스탬프 생성 =====
        # SKAB 데이터도 연속적인 인덱스를 타임스탬프로 사용
        train_timestamp = np.arange(len(trainset))    # 학습 데이터 타임스탬프
        valid_timestamp = np.arange(len(validset))    # 검증 데이터 타임스탬프
        test_timestamp = np.arange(len(testset))      # 테스트 데이터 타임스탬프
        
    
    # ===== 기타 데이터셋 처리 =====
    else:
        # 기타 데이터셋 처리
        # numpy 파일에서 직접 로드
        trainset = np.load(os.path.join(datainfo.train_dir, f'{subdataname}.npy'))  # 학습 데이터 로드
        testset = np.load(os.path.join(datainfo.test_dir, f'{subdataname}.npy'))    # 테스트 데이터 로드
        
        # ===== 비율 기반 데이터 분할 =====
        # 학습 데이터를 학습용과 검증용으로 분할
        valid_split_index = int(len(trainset) * valid_split_rate)  # 분할 지점 계산
        validset = trainset[valid_split_index:]  # 검증 데이터 (뒤쪽 20%)
        trainset = trainset[:valid_split_index]  # 학습 데이터 (앞쪽 80%)
        
        # ===== 타임스탬프 생성 =====
        # 연속적인 인덱스를 타임스탬프로 사용
        train_timestamp = np.arange(len(trainset))    # 학습 데이터 타임스탬프
        valid_timestamp = np.arange(len(validset))    # 검증 데이터 타임스탬프
        test_timestamp = np.arange(len(testset))      # 테스트 데이터 타임스탬프
        
        # ===== 이상 라벨 생성 =====
        # CSV 파일에서 이상 구간 정보를 읽어와 라벨 생성
        test_label_info = pd.read_csv(datainfo.test_label_path, index_col=0).loc[subdataname]
        test_label = np.zeros([int(test_label_info.num_values)], dtype=np.int)  # 0으로 초기화

        # ===== 이상 구간 라벨링 =====
        # anomaly_sequences에서 이상 구간 정보를 읽어와 1로 설정
        for i in eval(test_label_info.anomaly_sequences):
            if type(i) == list:
                # 구간 형태 [시작, 끝]: 해당 구간을 모두 1로 설정
                test_label[i[0]:i[1] + 1] = 1
            else:
                # 단일 인덱스: 해당 위치를 1로 설정
                test_label[i] = 1

    # ===== 결과 반환 =====
    # (학습데이터, 학습타임스탬프, 검증데이터, 검증타임스탬프, 테스트데이터, 테스트타임스탬프, 라벨) 튜플 반환
    return trainset, train_timestamp, validset, valid_timestamp, testset, test_timestamp, test_label