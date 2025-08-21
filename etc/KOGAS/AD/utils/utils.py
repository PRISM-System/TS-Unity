import torch
import torch.nn as nn
import numpy as np
import os
import random
import sys
import time
import logging
from typing import Optional, Tuple, Any, Dict
from pathlib import Path


def set_seed(random_seed: int = 72) -> None:
    """
    재현 가능한 결과를 위한 랜덤 시드를 설정합니다.

    Args:
        random_seed: 랜덤 시드 번호 (기본값: 72)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def load_model(resume: int, logdir: str) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
    """
    저장된 모델을 로드합니다.
    
    Args:
        resume: 재학습 또는 테스트할 버전
        logdir: 모델을 로드할 버전 디렉토리

    Returns:
        weights: 저장된 모델 가중치
        start_epoch: 저장된 실험 버전의 마지막 에포크
        last_lr: 저장된 학습률
        best_metrics: 저장된 실험 버전의 최고 메트릭

    Raises:
        FileNotFoundError: .pth 파일을 찾을 수 없는 경우
        ValueError: 모델 파일 로드 중 오류가 발생한 경우
    """
    savedir = os.path.join(os.path.dirname(logdir), f'version{resume}')
    
    if not os.path.exists(savedir):
        raise FileNotFoundError(f"저장 디렉토리를 찾을 수 없습니다: {savedir}")
    
    # .pth 파일 찾기
    modelname = None
    for name in os.listdir(savedir):
        if '.pth' in name:
            modelname = name
            break
    
    if modelname is None:
        raise FileNotFoundError(f"{savedir}에서 .pth 파일을 찾을 수 없습니다")

    modelpath = os.path.join(savedir, modelname)
    print(f'모델 경로: {modelpath}')

    try:
        # 우선 안전 모드로 시도 (PyTorch 2.6+ 기본)
        from torch.serialization import add_safe_globals
        import numpy as np
        try:    
            loadfile = torch.load(modelpath, map_location='cpu', weights_only=True)
        except Exception:
            # 신뢰된 체크포인트일 때만 사용: legacy 포맷 호환
            try:
                loadfile = torch.load(modelpath, map_location='cpu', weights_only=False)
            except Exception:
                # 필요한 NumPy 글로벌 허용 후 재시도 (신뢰된 파일만)
                try:
                    add_safe_globals([np.core.multiarray.scalar])
                except Exception:
                    pass
                loadfile = torch.load(modelpath, map_location='cpu', weights_only=True)
        weights = loadfile['weight']
        start_epoch = loadfile['best_epoch']
        last_lr = loadfile['best_lr']
        best_metrics = loadfile['best_loss']
        
        return weights, start_epoch, last_lr, best_metrics
        
    except Exception as e:
        raise ValueError(f"모델 파일 로드 중 오류 발생: {str(e)}")


def version_build(logdir: str, is_train: bool, resume: Optional[int] = None) -> str:
    """
    n번째 버전 폴더를 생성합니다.

    Args:
        logdir: 로그 디렉토리
        is_train: 학습 여부
        resume: 재학습 또는 테스트할 버전

    Returns:
        로그 히스토리를 저장할 버전 디렉토리
    """
    logdir_path = Path(logdir)
    
    if not logdir_path.exists():
        logdir_path.mkdir(parents=True, exist_ok=True)

    if is_train and resume is None:
        # 새로운 버전 번호 생성
        existing_versions = [d for d in logdir_path.iterdir() 
                           if d.is_dir() and d.name.startswith('version')]
        version = len(existing_versions)
    else:
        version = resume if resume is not None else 0

    version_dir = logdir_path / f'version{version}'

    if is_train and resume is None:
        version_dir.mkdir(exist_ok=True)

    return str(version_dir)


# 전역 변수 (진행률 표시용)
_last_time = time.time()
_begin_time = _last_time


def progress_bar(current: int, total: int, name: str, 
                msg: Optional[str] = None, width: Optional[int] = None) -> None:
    """
    진행률을 표시하는 프로그레스 바를 출력합니다.
    
    Args:
        current: 현재 배치 인덱스
        total: 전체 데이터 길이
        name: 프로그레스 바 설명
        msg: 학습 모델 히스토리 (선택사항)
        width: 터미널 크기 (선택사항)
    """
    global _last_time, _begin_time
    
    # 터미널 너비 결정
    if width is None:
        try:
            _, term_width = os.popen('stty size', 'r').read().split()
            term_width = int(term_width)
        except (OSError, ValueError):
            term_width = 80  # 기본값
    else:
        term_width = width

    TOTAL_BAR_LENGTH = 65.0

    if current == 0:
        _begin_time = time.time()  # 새로운 바를 위해 리셋

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    # 프로그레스 바 출력
    sys.stdout.write(f'{name} [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    # 시간 정보 계산
    cur_time = time.time()
    step_time = cur_time - _last_time
    _last_time = cur_time
    tot_time = cur_time - _begin_time

    # 메시지 구성
    time_info = [f'  Step: {format_time(step_time)}', f' | Tot: {format_time(tot_time)}']
    if msg:
        time_info.append(f' | {msg}')

    msg_str = ''.join(time_info)
    sys.stdout.write(msg_str)
    
    # 공백으로 패딩
    padding = term_width - int(TOTAL_BAR_LENGTH) - len(msg_str) - 3
    sys.stdout.write(' ' * max(0, padding))

    # 바 중앙으로 이동
    backspace_count = term_width - int(TOTAL_BAR_LENGTH / 2) + 2
    sys.stdout.write('\b' * backspace_count)
    sys.stdout.write(f' {current + 1}/{total} ')

    # 줄바꿈 또는 캐리지 리턴
    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds: float) -> str:
    """
    초 단위 시간을 읽기 쉬운 형식으로 변환합니다.
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        포맷된 시간 문자열
    """
    if seconds < 0:
        return "0s"
        
    days = int(seconds // (3600 * 24))
    seconds -= days * 3600 * 24
    
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    
    secondsf = int(seconds)
    seconds -= secondsf
    
    millis = int(seconds * 1000)

    result = []
    if days > 0:
        result.append(f"{days}D")
    if hours > 0 and len(result) < 2:
        result.append(f"{hours}h")
    if minutes > 0 and len(result) < 2:
        result.append(f"{minutes}m")
    if secondsf > 0 and len(result) < 2:
        result.append(f"{secondsf}s")
    if millis > 0 and len(result) < 2:
        result.append(f"{millis}ms")
        
    return ''.join(result) if result else "0s"


class CheckPoint:
    """
    모델 체크포인트를 관리하는 클래스
    
    최고 성능 모델을 저장하고 조기 종료를 처리합니다.
    """
    
    def __init__(self, logdir: str, last_metrics: Optional[float] = None, 
                 metric_type: str = 'loss'):
        """
        Args:
            logdir: 체크포인트를 저장할 디렉토리
            last_metrics: 마지막 메트릭 값
            metric_type: 메트릭 타입 ('loss' 또는 'score')
        """
        self.logdir = logdir
        self.last_metrics = last_metrics
        self.metric_type = metric_type
        self.best_metrics = float('inf') if metric_type == 'loss' else float('-inf')
        self.best_epoch = 0
        self.best_lr = 0.0
        
        # 디렉토리 생성
        os.makedirs(logdir, exist_ok=True)

    def check(self, epoch: int, model: nn.Module, score: float, lr: float) -> bool:
        """
        현재 모델이 최고 성능인지 확인하고 저장 여부를 결정합니다.
        
        Args:
            epoch: 현재 에포크
            model: PyTorch 모델
            score: 현재 메트릭 점수
            lr: 현재 학습률
            
        Returns:
            모델이 저장되었는지 여부
        """
        is_better = False
        
        if self.metric_type == 'loss':
            if score < self.best_metrics:
                is_better = True
        else:  # score
            if score > self.best_metrics:
                is_better = True
                
        if is_better:
            self.best_metrics = score
            self.best_epoch = epoch
            self.best_lr = lr
            self.model_save(epoch, model, score, lr)
            
        return is_better

    def model_save(self, epoch: int, model: nn.Module, score: float, lr: float) -> None:
        """
        모델을 저장합니다.
        
        Args:
            epoch: 현재 에포크
            model: PyTorch 모델
            score: 현재 메트릭 점수
            lr: 현재 학습률
        """
        model_path = os.path.join(self.logdir, f'model_epoch_{epoch}.pth')
        
        # 모델 상태 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'lr': lr,
            'best_epoch': self.best_epoch,
            'best_lr': self.best_lr,
            'best_loss': self.best_metrics,
            'weight': model.state_dict()
        }, model_path)
        
        print(f"모델이 저장되었습니다: {model_path}")


def log_setting(logdir: str, log_name: str,
               formatter: str = '%(asctime)s|%(name)s|%(levelname)s:%(message)s') -> logging.Logger:
    """
    로깅 설정을 초기화합니다.
    
    Args:
        logdir: 로그 파일을 저장할 디렉토리
        log_name: 로그 파일 이름
        formatter: 로그 포맷터
        
    Returns:
        설정된 로거 객체
    """
    # 디렉토리 생성
    os.makedirs(logdir, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 설정
    log_file = os.path.join(logdir, f'{log_name}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter_obj = logging.Formatter(formatter)
    file_handler.setFormatter(formatter_obj)
    console_handler.setFormatter(formatter_obj)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
