import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
from typing import Any, Optional, Union, Tuple
import logging

dst = scipy.spatial.distance.euclidean

plt.switch_backend('agg')

logger = logging.getLogger(__name__)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, params: Any) -> None:
    """
    학습률을 조정합니다.

    Args:
        optimizer: PyTorch 옵티마이저
        epoch: 현재 에포크
        params: 학습률 조정 파라미터
    """
    if params.lradj == 'type1':
        lr_adjust = {epoch: params.lr * (0.5 ** ((epoch - 1) // 1))}
    elif params.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        logger.warning(f"알 수 없는 학습률 조정 타입: {params.lradj}")
        return
        
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f'학습률을 {lr}로 업데이트했습니다.')


class EarlyStopping:
    """
    조기 종료를 위한 클래스
    
    검증 손실이 개선되지 않을 때 학습을 조기 종료합니다.
    """
    
    def __init__(self, patience: int = 0, verbose: bool = False):
        """
        Args:
            patience: 개선되지 않는 에포크 수
            verbose: 상세 출력 여부
        """
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        self.early_stop = False

    def __call__(self, loss: float, model: torch.nn.Module) -> bool:
        """
        조기 종료 조건을 확인합니다.
        
        Args:
            loss: 현재 검증 손실
            model: PyTorch 모델
            
        Returns:
            조기 종료 여부
        """
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    logger.info('학습 과정이 조기 종료됩니다.')
                self.early_stop = True
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

    def validate(self, loss: float) -> bool:
        """
        조기 종료 조건을 확인합니다 (하위 호환성).
        
        Args:
            loss: 현재 검증 손실
            
        Returns:
            조기 종료 여부
        """
        return self.__call__(loss, None)


def check_graph(xs: np.ndarray, att: np.ndarray, piece: int = 1, 
                threshold: Optional[float] = None) -> plt.Figure:
    """
    이상 점수와 이상 라벨을 시각화합니다.

    Args:
        xs: 이상 점수 배열
        att: 이상 라벨 배열
        piece: 분할할 그림 수
        threshold: 이상 임계값 (선택사항)

    Returns:
        matplotlib Figure 객체
    """
    l = xs.shape[0]
    chunk = l // piece
    
    if piece == 1:
        fig, axs = plt.subplots(1, figsize=(20, 4))
        axs = [axs]
    else:
        fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        if piece == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.plot(xticks, xs[L:R], color='#0C090A')
        ymin, ymax = ax.get_ylim()
        ymin = 0
        ax.set_ylim(ymin, ymax)
        if len(xs[L:R]) > 0:
            ax.vlines(xticks[np.where(att[L:R] == 1)], ymin=ymin, ymax=ymax, color='#FED8B1',
                      alpha=0.6, label='true anomaly')
        ax.plot(xticks, xs[L:R], color='#0C090A', label='anomaly score')
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.8, label=f'threshold:{threshold:.4f}')
        ax.legend()

    return fig


def gap(data: np.ndarray, refs: Optional[np.ndarray] = None, 
        nrefs: int = 20, ks: range = range(1, 11)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gap 통계를 계산합니다.
    
    Args:
        data: 입력 데이터
        refs: 참조 데이터 (선택사항)
        nrefs: 참조 데이터 수
        ks: 클러스터 수 범위
        
    Returns:
        gap 통계와 표준 오차
    """
    shape = data.shape
    if refs is None:
        tops = data.flatten()
        tops = tops.reshape(1, -1)
        bottoms = np.zeros((1, shape[1]))
        for i in range(shape[1]):
            bottoms[0, i] = data[:, i].min()
        refs = np.random.uniform(bottoms, tops, (nrefs, shape[0], shape[1]))
    
    gaps = np.zeros((len(ks),))
    s = np.zeros((len(ks),))
    
    for i, k in enumerate(ks):
        # 실제 데이터 클러스터링
        centroids, labels = scipy.cluster.vq.kmeans2(data, k, iter=100)
        within = np.sum([np.sum([dst(data[j], centroids[labels[j]]) 
                               for j in range(len(data)) if labels[j] == i]) 
                       for i in range(k)])
        
        # 참조 데이터 클러스터링
        ref_within = np.zeros((nrefs,))
        for j in range(nrefs):
            ref_centroids, ref_labels = scipy.cluster.vq.kmeans2(refs[j], k, iter=100)
            ref_within[j] = np.sum([np.sum([dst(refs[j][l], ref_centroids[ref_labels[l]]) 
                                          for l in range(len(refs[j])) if ref_labels[l] == i]) 
                                  for i in range(k)])
        
        gaps[i] = np.log(np.mean(ref_within)) - np.log(within)
        s[i] = np.sqrt(np.mean((np.log(ref_within) - np.log(within) - gaps[i]) ** 2))
    
    return gaps, s