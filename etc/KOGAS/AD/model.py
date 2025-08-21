"""
Anomaly Detection Model Module

이 모듈은 이상 탐지 모델의 빌드, 학습, 테스트, 추론을 위한 클래스들을 제공합니다.
"""

from models import VTTPAT, VTTSAT, Transformer
from models import LSTM_AE, LSTM_VAE, USAD, DAGMM, AnomalyTransformer, OmniAnomaly
from utils.tools import EarlyStopping, adjust_learning_rate, check_graph
from utils.utils import load_model, CheckPoint, progress_bar
from utils.metrics import (
    anomaly_metric, bf_search, get_best_f1, get_adjusted_composite_metrics, 
    pot_eval, percentile_search, calc_seq, valid_search
)

import torch
import torch.nn as nn
from torch import optim
from einops import rearrange

import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import warnings
import numpy as np
import wandb
import json

warnings.filterwarnings('ignore')


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL divergence 손실을 계산합니다.
    
    Args:
        p: 첫 번째 확률 분포
        q: 두 번째 확률 분포
        
    Returns:
        KL divergence 손실
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class AnomalyDetectionModel:
    """
    이상 탐지 모델의 빌드, 학습, 테스트, 추론을 담당하는 클래스
    
    Attributes:
        args: 명령행 인수
        params: 모델 하이퍼파라미터
        savedir: 저장 디렉토리
        device: 사용할 디바이스 (CPU/GPU)
        model: PyTorch 모델
        init_weight: 초기 가중치 경로
    """
    
    def __init__(self, args: Any, params: Dict[str, Any], savedir: str):
        """
        Args:
            args: 명령행 인수
            params: 모델 하이퍼파라미터
            savedir: 저장 디렉토리
        """
        self.args = args
        self.params = params
        self.savedir = savedir
        
        # Logger 초기화 (먼저 수행)
        try:
            self.logger = logging.getLogger(__name__)
            # Logger가 설정되지 않은 경우를 대비한 fallback
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        except Exception:
            # Logger 초기화 실패 시 print 함수를 사용하는 fallback logger 생성
            class FallbackLogger:
                def info(self, msg): print(f"[INFO] {msg}")
                def warning(self, msg): print(f"[WARNING] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
                def debug(self, msg): print(f"[DEBUG] {msg}")
            self.logger = FallbackLogger()
        
        # 디바이스와 모델 초기화 (logger 초기화 후 수행)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.init_weight = args.init_weight
        
    def set_logger(self, logger: logging.Logger) -> None:
        """
        외부에서 logger를 설정합니다.
        
        Args:
            logger: 설정할 logger 객체
        """
        self.logger = logger

    def _build_model(self) -> nn.Module:
        """
        선택된 모델을 빌드합니다.
        
        Returns:
            빌드된 PyTorch 모델
        """
        model_dict = {
            'VTTSAT': VTTSAT,
            'VTTPAT': VTTPAT,
            'TF': Transformer,
            'AnomalyTransformer': AnomalyTransformer,
            'LSTM_AE': LSTM_AE,
            'LSTM_VAE': LSTM_VAE,
            'USAD': USAD,
            'DAGMM': DAGMM,
            'OmniAnomaly': OmniAnomaly,
        }
        
        if self.args.model not in model_dict:
            raise ValueError(f"지원하지 않는 모델: {self.args.model}")
            
        model = model_dict[self.args.model].Model(self.params).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            if hasattr(self, 'logger'):
                self.logger.info('멀티 GPU 사용')
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    def _acquire_device(self) -> torch.device:
        """
        사용할 디바이스를 결정합니다.
        
        Returns:
            사용할 디바이스 (CPU 또는 GPU)
        """
        if self.args.use_gpu:
            if not self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                if hasattr(self, 'logger'):
                    self.logger.info(f'GPU 사용: cuda:{self.args.gpu}')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device(f'cuda:{self.args.gpu}')
                if hasattr(self, 'logger'):
                    self.logger.info(f'멀티 GPU 사용: {self.args.devices}')
        else:
            device = torch.device('cpu')
            if hasattr(self, 'logger'):
                self.logger.info('CPU 사용')
            
        return device

    def _select_optimizer(self) -> optim.Optimizer:
        """
        옵티마이저를 선택합니다.
        
        Returns:
            선택된 옵티마이저
        """
        optimizer_dict = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        
        if self.params.optim not in optimizer_dict:
            raise ValueError(f"지원하지 않는 옵티마이저: {self.params.optim}")
            
        return optimizer_dict[self.params.optim](
            self.model.parameters(), 
            lr=self.params.lr
        )

    def _select_criterion(self) -> nn.Module:
        """
        손실 함수를 선택합니다.
        
        Returns:
            선택된 손실 함수
        """
        criterion_dict = {
            'mse': nn.MSELoss,
            'mae': nn.L1Loss
        }
        
        if self.args.loss not in criterion_dict:
            raise ValueError(f"지원하지 않는 손실 함수: {self.args.loss}")
            
        return criterion_dict[self.args.loss](reduction='none')
    
    def _compute_model_loss(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                           criterion: nn.Module, epoch: int = 0) -> Tuple[torch.Tensor, np.ndarray]:
        """
        모델별 손실을 계산합니다.
        
        Args:
            batch_x: 입력 데이터
            batch_y: 타겟 데이터
            criterion: 손실 함수
            epoch: 현재 에포크
            
        Returns:
            손실 텐서와 검증 점수
        """
        if self.args.model in ['VTTPAT', 'VTTSAT']:
            output, _ = self.model(batch_x)
            loss = criterion(output, batch_y)
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            
        elif self.args.model == 'LSTM_VAE':
            output = self.model(batch_x)
            loss = criterion(output[0], batch_y)
            kl_loss = output[1]
            loss = loss + kl_loss
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            
        elif self.args.model == 'USAD':
            output = self.model(batch_x)
            w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
            w2 = output[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
            w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
            
            loss1 = (1 / (epoch + 1) * torch.mean((batch_y - w1) ** 2, axis=2) + 
                    (1 - 1 / (epoch + 1)) * torch.mean((batch_y - w3) ** 2, axis=2))
            loss2 = (1 / (epoch + 1) * torch.mean((batch_y - w2) ** 2, axis=2) - 
                    (1 - 1 / (epoch + 1)) * torch.mean((batch_y - w3) ** 2, axis=2))
            
            loss = loss1 + loss2
            valid_score = loss.cpu().detach().numpy()
            epoch_loss1 = torch.mean(loss1)
            epoch_loss2 = torch.mean(loss2)
            loss = epoch_loss1 + epoch_loss2
            
        elif self.args.model == 'OmniAnomaly':
            y_pred, mu, logvar, hidden = self.model(batch_x, hidden if 'hidden' in locals() else None)
            MSE = criterion(y_pred, batch_y)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = MSE + self.model.beta * KLD
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            
        elif self.args.model == 'DAGMM':
            _, x_hat, z, gamma = self.model(batch_x)
            l1, l2 = criterion(x_hat, batch_x), criterion(gamma, batch_x)
            loss = l1 + l2
            valid_score = loss.cpu().detach().numpy()
            loss = torch.mean(loss)
            
        else:
            # 기본 재구성 모델들
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            
        return loss, valid_score
    
    def valid(self, valid_loader: Any, criterion: nn.Module, epoch: int = 0) -> Tuple[float, List[np.ndarray]]:
        """
        검증을 수행합니다.
        
        Args:
            valid_loader: 검증 데이터 로더
            criterion: 손실 함수
            epoch: 현재 에포크
            
        Returns:
            총 손실과 검증 점수 리스트
        """
        total_loss = []
        valid_score = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)
                
                loss, score = self._compute_model_loss(batch_x, batch_y, criterion, epoch)
                
                total_loss.append(loss.item())
                valid_score.append(score)
                
        return np.mean(total_loss), valid_score

    def train(self, train_loader: Any, valid_loader: Any, test_loader: Any, 
              alpha: float = 0.5, beta: float = 0.5) -> Dict[str, List[float]]:
        """
        모델 학습을 수행합니다.
        
        Args:
            train_loader: 학습 데이터 로더
            valid_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더
            alpha: USAD 모델용 가중치 파라미터
            beta: USAD 모델용 가중치 파라미터
            
        Returns:
            학습 히스토리 딕셔너리
        """
        time_now = time.time()

        # 모델 로드 (재학습 또는 초기 가중치)
        best_metrics = self._load_model_weights()
        
        # 체크포인트 설정
        ckp = CheckPoint(
            logdir=self.savedir,
            last_metrics=best_metrics,
            metric_type='loss'
        )

        # 옵티마이저 및 손실 함수 설정
        optimizers = self._setup_optimizers()
        criterion = self._select_criterion()

        # 조기 종료 설정
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 학습 히스토리 초기화
        history = {
            'train_loss': [],
            'validation_loss': []
        }

        # 에포크별 학습
        for epoch in range(self.args.epochs):
            # 학습 단계
            train_loss, train_score = self._train_epoch(
                train_loader, optimizers, criterion, epoch, alpha, beta
            )
            
            # 검증 단계
            valid_loss, valid_score = self.valid(valid_loader, criterion, epoch)
            
            # 히스토리 업데이트
            history['train_loss'].append(np.mean(train_loss))
            history['validation_loss'].append(valid_loss)
            
            # 체크포인트 저장
            is_better = ckp.check(epoch, self.model, valid_loss, self.params.lr)
            
            # 로깅
            self._log_training_progress(epoch, train_loss, valid_loss, time_now)
            
            # 조기 종료 확인
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("조기 종료!")
                break
                
            # 학습률 조정
            adjust_learning_rate(optimizers, epoch + 1, self.params)

        return history

    def _load_model_weights(self) -> Optional[float]:
        """모델 가중치를 로드합니다."""
        best_metrics = None
        
        if self.args.resume is not None:
            self.logger.info(f'버전 {self.args.resume}에서 재개')
            weights, start_epoch, saved_lr, best_metrics = load_model(
                resume=self.args.resume, logdir=self.savedir
            )
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)
                
        if self.args.init_weight is not None:
            # 신뢰된 가중치 파일만 사용하세요
            try:
                load_file = torch.load(self.args.init_weight, map_location='cpu', weights_only=True)
            except Exception:
                load_file = torch.load(self.args.init_weight, map_location='cpu', weights_only=False)
            self.model.load_state_dict(load_file['weight'])
            self.logger.info("초기 가중치 로드 완료")   
            
        return best_metrics

    def _setup_optimizers(self) -> Union[optim.Optimizer, Tuple[optim.Optimizer, optim.Optimizer]]:
        """옵티마이저를 설정합니다."""
        if self.args.model in ['USAD', 'AnomalyTransformer']:
            return self._select_optimizer(), self._select_optimizer()
        else:
            return self._select_optimizer()

    def _train_epoch(self, train_loader: Any, optimizers: Any, criterion: nn.Module, 
                    epoch: int, alpha: float, beta: float) -> Tuple[List[float], List[np.ndarray]]:
        """한 에포크의 학습을 수행합니다."""
        train_loss = []
        train_score = []
        
        self.model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            batch_x = batch['given'].float().to(self.device)
            batch_y = batch['answer'].float().to(self.device)
            
            # 모델별 학습 로직 실행
            loss, score = self._train_step(
                batch_x, batch_y, optimizers, criterion, epoch, alpha, beta
            )
            
            train_loss.append(loss)
            train_score.append(score)
            
        return train_loss, train_score

    def _train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   optimizers: Any, criterion: nn.Module, epoch: int, 
                   alpha: float, beta: float) -> Tuple[float, np.ndarray]:
        """한 배치의 학습을 수행합니다."""
        if self.args.model in ['VTTSAT', 'VTTPAT']:
            return self._train_vttsat_vttpat(batch_x, batch_y, optimizers, criterion)
        elif self.args.model == 'LSTM_VAE':
            return self._train_lstm_vae(batch_x, batch_y, optimizers, criterion)
        elif self.args.model == 'USAD':
            return self._train_usad(batch_x, batch_y, optimizers, criterion, epoch, alpha, beta)
        elif self.args.model == 'OmniAnomaly':
            return self._train_omnianomaly(batch_x, batch_y, optimizers, criterion)
        elif self.args.model == 'DAGMM':
            return self._train_dagmm(batch_x, batch_y, optimizers, criterion)
        elif self.args.model == 'AnomalyTransformer':
            return self._train_anomaly_transformer(batch_x, batch_y, optimizers, criterion)
        else:
            return self._train_default(batch_x, batch_y, optimizers, criterion)

    def _train_vttsat_vttpat(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                            optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """VTTSAT/VTTPAT 모델 학습"""
        optimizers.zero_grad()
        output, _ = self.model(batch_x)
        loss = criterion(output, batch_y)
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score

    def _train_lstm_vae(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                       optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """LSTM-VAE 모델 학습"""
        optimizers.zero_grad()
        output = self.model(batch_x)
        loss = criterion(output[0], batch_y)
        kl_loss = output[1]
        loss = loss + kl_loss
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score

    def _train_usad(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   optimizers: Tuple[optim.Optimizer, optim.Optimizer], criterion: nn.Module, 
                   epoch: int, alpha: float, beta: float) -> Tuple[float, np.ndarray]:
        """USAD 모델 학습"""
        model_optim1, model_optim2 = optimizers
        
        # 첫 번째 오토인코더 학습
        model_optim1.zero_grad()
        output = self.model(batch_x)
        w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        loss1 = (1 / (epoch + 1) * torch.mean((batch_y - w1) ** 2) + 
                (1 - 1 / (epoch + 1)) * torch.mean((batch_y - w3) ** 2))
        loss1.backward()
        model_optim1.step()
        
        # 두 번째 오토인코더 학습
        model_optim2.zero_grad()
        output = self.model(batch_x)
        w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w2 = output[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        loss2 = (1 / (epoch + 1) * torch.mean((batch_y - w2) ** 2) - 
                (1 - 1 / (epoch + 1)) * torch.mean((batch_y - w3) ** 2))
        loss2.backward()
        model_optim2.step()
        
        # 최종 손실 계산
        loss = alpha * criterion(w1, batch_x) + beta * criterion(w2, batch_x)
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        
        return loss.item(), score

    def _train_omnianomaly(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                          optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """OmniAnomaly 모델 학습"""
        optimizers.zero_grad()
        hidden = None
        y_pred, mu, logvar, hidden = self.model(batch_x, hidden if 'hidden' in locals() else None)
        MSE = criterion(y_pred, batch_y)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = MSE + self.model.beta * KLD
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score

    def _train_dagmm(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                    optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """DAGMM 모델 학습"""
        optimizers.zero_grad()
        _, x_hat, z, gamma = self.model(batch_x)
        l1, l2 = criterion(x_hat, batch_x), criterion(gamma, batch_x)
        loss = l1 + l2
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score

    def _train_anomaly_transformer(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                                 optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """AnomalyTransformer 모델 학습"""
        optimizers.zero_grad()
        output, series, prior, _ = self.model(batch_x)

        # Series loss 계산
        series_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           self.args.window_size)).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.window_size)).detach(),
                           series[u])))
        series_loss /= len(prior)

        # Reconstruction loss 계산
        rec_loss = criterion(output, batch_x)
        loss = rec_loss - self.params['k'] * series_loss
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward(retain_graph=True)
        optimizers.step()
        
        return loss.item(), score

    def _train_default(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                      optimizers: optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """기본 모델 학습"""
        optimizers.zero_grad()
        output = self.model(batch_x)
        loss = criterion(output, batch_y)
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score

    def _log_training_progress(self, epoch: int, train_loss: List[float], 
                             valid_loss: float, start_time: float) -> None:
        """학습 진행 상황을 로깅합니다."""
        epoch_time = time.time() - start_time
        avg_train_loss = np.mean(train_loss)
        
        self.logger.info(
            f'Epoch [{epoch + 1}/{self.args.epochs}] '
            f'Train Loss: {avg_train_loss:.7f} '
            f'Valid Loss: {valid_loss:.7f} '
            f'Time: {epoch_time:.2f}s'
        )

    def test(self, valid_loader: Any, test_loader: Any, 
             alpha: float = 0.5, beta: float = 0.5) -> Tuple[Dict[str, Any], float]:
        """
        모델 테스트를 수행합니다.
        
        Args:
            valid_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더
            alpha: USAD 모델용 가중치 파라미터
            beta: USAD 모델용 가중치 파라미터
            
        Returns:
            테스트 결과 딕셔너리와 AUC 점수
        """
        # 모델 로드 (테스트 모드)
        if self.args.resume is not None and not self.args.train:
            self.logger.info(f'버전 {self.args.resume}에서 모델 로드')
            weights, start_epoch, saved_lr, best_metrics = load_model(
                resume=self.args.resume, logdir=self.savedir
            )
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

        # 테스트 실행
        dist, attack, pred = self._run_test(test_loader, alpha, beta)
        
        # 검증 데이터로 임계값 설정
        _, valid_score = self.valid(valid_loader, self._select_criterion(), 0)
        valid_score = np.concatenate(valid_score).flatten()
        
        # 이상 탐지 메트릭 계산
        history, auc = self._calculate_anomaly_metrics(dist, attack)
        
        # 테스트 결과 저장
        self._save_test_results(dist, attack, pred, valid_score, history['threshold'][0])
        
        return history, auc

    def _run_test(self, test_loader: Any, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """테스트를 실행하여 거리, 공격 라벨, 예측값을 반환합니다."""
        dist = []
        attack = []
        pred = []
        criterion = self._select_criterion()
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)
                
                # 모델별 테스트 로직 실행
                batch_pred, batch_dist = self._test_step(
                    batch_x, batch_y, criterion, batch_idx, alpha, beta
                )
                
                pred.append(batch_pred)
                dist.append(batch_dist)
                attack.append(batch['attack'].reshape(-1, batch['attack'].shape[-1]).numpy())

        # 결과 결합
        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        pred = np.concatenate(pred)
        
        return dist, attack, pred

    def _test_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                  criterion: nn.Module, batch_idx: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """한 배치의 테스트를 수행합니다."""
        if self.args.model in ['VTTSAT', 'VTTPAT']:
            return self._test_vttsat_vttpat(batch_x, batch_y, criterion)
        elif self.args.model == 'LSTM_VAE':
            return self._test_lstm_vae(batch_x, batch_y, criterion)
        elif self.args.model == 'USAD':
            return self._test_usad(batch_x, batch_y, alpha, beta)
        elif self.args.model == 'OmniAnomaly':
            return self._test_omnianomaly(batch_x, batch_y, criterion, batch_idx)
        elif self.args.model == 'DAGMM':
            return self._test_dagmm(batch_x, batch_y, criterion)
        elif self.args.model == 'AnomalyTransformer':
            return self._test_anomaly_transformer(batch_x, batch_y, criterion)
        else:
            return self._test_default(batch_x, batch_y, criterion)

    def _test_vttsat_vttpat(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                           criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """VTTSAT/VTTPAT 모델 테스트"""
        predictions, _ = self.model.forward(batch_x)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_lstm_vae(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                      criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM-VAE 모델 테스트"""
        predictions = self.model.forward(batch_x)
        score = criterion(predictions[0], batch_y).cpu().detach().numpy()
        pred = predictions[0].cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_usad(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                  alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """USAD 모델 테스트"""
        predictions = self.model.forward(batch_x)
        w1 = predictions[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w2 = predictions[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        pred = (alpha * (batch_x - w1) + beta * (batch_x - w2)).cpu().detach().numpy()
        dist = (alpha * torch.mean((batch_x - w1) ** 2, axis=2) + 
               beta * torch.mean((batch_x - w2) ** 2, axis=2)).detach().cpu()
        return pred, dist

    def _test_omnianomaly(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                         criterion: nn.Module, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """OmniAnomaly 모델 테스트"""
        hidden = None
        predictions, _, _, hidden = self.model.forward(batch_x, hidden if batch_idx else None)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_dagmm(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """DAGMM 모델 테스트"""
        _, x_hat, _, _ = self.model.forward(batch_x)
        score = criterion(x_hat, batch_y).cpu().detach().numpy()
        pred = x_hat.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_anomaly_transformer(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                                criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """AnomalyTransformer 모델 테스트"""
        output, series, prior, _ = self.model(batch_x)
        pred = output.cpu().detach().numpy()
        
        loss = torch.mean(criterion(batch_x, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        temperature = 50
        
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.window_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        dist = cri.detach().cpu().numpy()
        
        return pred, dist

    def _test_default(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                     criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """기본 모델 테스트"""
        predictions = self.model.forward(batch_x)
        pred = predictions.cpu().detach().numpy()
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    
    def _calculate_anomaly_metrics(self, dist: np.ndarray, attack: np.ndarray, K_VALUES: List[int] = None) -> Tuple[Dict[str, Any], float]:
        """PA%K 메트릭만 계산합니다."""
        # PA%K 메트릭 계산
        history = {}
        pa_auc = self._calculate_pa_metrics(dist, attack, history, K_VALUES)
        
        return history, pa_auc

    def _calculate_pa_metrics(self, dist: np.ndarray, attack: np.ndarray, 
                            history: Dict[str, Any], K_VALUES: List[int] = None) -> float:
        """PA%K 메트릭을 계산합니다."""
        # 데이터 검증
        unique_classes = np.unique(attack)
        if len(unique_classes) == 1:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Only one class present in attack labels: {unique_classes[0]}")
                self.logger.warning("PA%K metrics cannot be calculated properly. Returning default values.")
            return 0.0
        
        # 기본 K_VALUES 설정
        if K_VALUES is None:
            K_VALUES = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        f1_values = []
        
        # 임계값 범위 출력
        start_threshold = np.percentile(dist, 90)
        end_threshold = np.percentile(dist, 99)
        if hasattr(self, 'logger'):
            self.logger.info(f'Threshold start: {start_threshold:.4f} end: {end_threshold:.4f}')
        
        # 각 K 값에 대해 메트릭 계산
        for k in K_VALUES:
            scores = dist.copy()
            metrics, threshold = self._calculate_k_metrics(scores, attack, k)
            
            f1, precision, recall, roc_auc = metrics
            f1_values.append(f1)
            
            # 결과 출력
            if hasattr(self, 'logger'):
                self.logger.info(f"K: {k} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} AUROC: {roc_auc:.4f}")
            
            # History에 저장
            self._update_history_metrics(history, k, precision, recall, f1, roc_auc, threshold)
            
            # WandB 로깅
            self._log_wandb_metrics(k, f1)
        
        # PA%K AUC 계산
        pa_auc = self._calculate_pa_auc(f1_values, K_VALUES)
        if hasattr(self, 'logger'):
            self.logger.info(f'PA%K AUC: {pa_auc}')
        
        return pa_auc

    def _calculate_k_metrics(self, scores: np.ndarray, attack: np.ndarray, k: int) -> Tuple[Tuple[float, float, float, float], float]:
        """특정 K 값에 대한 메트릭을 계산합니다."""
        try:
            [f1, precision, recall, _, _, _, _, roc_auc, _, _], threshold = bf_search(
                scores, attack,
                start=np.percentile(scores, 50),
                end=np.percentile(scores, 99),
                step_num=1000,
                K=k,
                verbose=False
            )
            return (f1, precision, recall, roc_auc), threshold
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                # ROC AUC를 계산할 수 없는 경우 기본값 반환
                if hasattr(self, 'logger'):
                    self.logger.warning(f"ROC AUC cannot be calculated for K={k} (only one class present). Using default values.")
                return (0.0, 0.0, 0.0, 0.0), np.percentile(scores, 90)
            else:
                # 다른 ValueError의 경우 재발생
                raise e

    def _update_history_metrics(self, history: Dict[str, Any], k: int, 
                              precision: float, recall: float, f1: float, roc_auc: float, threshold: float) -> None:
        """History 딕셔너리에 메트릭을 업데이트합니다."""
        history.setdefault(f'precision_{k}', []).append(precision)
        history.setdefault(f'recall_{k}', []).append(recall)
        history.setdefault(f'f1_{k}', []).append(f1)
        history.setdefault(f'roc_auc', []).append(roc_auc)
        history.setdefault('threshold', []).append(threshold)

    def _log_wandb_metrics(self, k: int, f1: float) -> None:
        """WandB에 메트릭을 로깅합니다."""
        if hasattr(self.args, 'log_to_wandb') and self.args.log_to_wandb:
            wandb.log({f"PA%{k:03d}": f1})

    def _calculate_pa_auc(self, f1_values: List[float], k_values: List[int]) -> float:
        """PA%K AUC를 계산합니다."""
        auc = 0
        for i in range(len(k_values) - 1):
            auc += 0.5 * (f1_values[i] + f1_values[i + 1]) * (int(k_values[i + 1]) - int(k_values[i]))
        auc /= 100
        return auc

    def inference(self, test_loader: Any, epoch: Optional[int] = None, 
                 valid_loader: Optional[Any] = None, alpha: float = 0.5, beta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        모델 추론을 수행합니다.
        
        Args:
            test_loader: 테스트 데이터 로더
            epoch: 학습 에포크 (선택사항)
            valid_loader: 검증 데이터 로더 (선택사항)
            alpha: USAD 모델용 가중치 파라미터
            beta: USAD 모델용 가중치 파라미터
            
        Returns:
            거리 점수와 공격 라벨
        """
        # 추론 실행
        dist, attack, pred, answer = self._run_inference(test_loader, alpha, beta)
        
        # 결과 저장
        self._save_inference_results(dist, attack, pred, answer, epoch, valid_loader)
        
        return dist, attack

    def _run_inference(self, test_loader: Any, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """추론을 실행하여 결과를 반환합니다."""
        dist = []
        attack = []
        pred = []
        answer = []
        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)
                
                # 모델별 추론 로직 실행
                batch_pred, batch_dist = self._inference_step(
                    batch_x, batch_y, criterion, batch_idx, alpha, beta
                )
                
                pred.append(batch_pred)
                dist.append(batch_dist)
                answer.append(batch_y.cpu().detach().numpy())
                attack.append(batch['attack'].reshape(-1, batch['attack'].shape[-1]).numpy())

        # 결과 결합
        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        pred = np.concatenate(pred)
        answer = np.concatenate(answer)
        
        return dist, attack, pred, answer

    def _inference_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                       criterion: nn.Module, batch_idx: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """한 배치의 추론을 수행합니다."""
        if self.args.model in ['VTTSAT', 'VTTPAT']:
            return self._inference_vttsat_vttpat(batch_x, batch_y, criterion)
        elif self.args.model == 'LSTM_VAE':
            return self._inference_lstm_vae(batch_x, batch_y, criterion)
        elif self.args.model == 'USAD':
            return self._inference_usad(batch_x, batch_y, alpha, beta)
        elif self.args.model == 'OmniAnomaly':
            return self._inference_omnianomaly(batch_x, batch_y, criterion, batch_idx)
        elif self.args.model == 'DAGMM':
            return self._inference_dagmm(batch_x, batch_y, criterion)
        elif self.args.model == 'AnomalyTransformer':
            return self._inference_anomaly_transformer(batch_x, batch_y, criterion)
        else:
            return self._inference_default(batch_x, batch_y, criterion)

    def _inference_vttsat_vttpat(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                               criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """VTTSAT/VTTPAT 모델 추론"""
        predictions, _ = self.model.forward(batch_x)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _inference_lstm_vae(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                          criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM-VAE 모델 추론"""
        predictions = self.model.forward(batch_x)
        score = criterion(predictions[0], batch_y).cpu().detach().numpy()
        pred = predictions[0].cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _inference_usad(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                       alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """USAD 모델 추론"""
        predictions = self.model.forward(batch_x)
        w1 = predictions[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w2 = predictions[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        dist = (alpha * torch.mean((batch_x - w1) ** 2, axis=2) + 
               beta * torch.mean((batch_x - w2) ** 2, axis=2)).detach().cpu()
        pred = (alpha * (batch_x - w1) + beta * (batch_x - w2)).cpu().detach().numpy()
        return pred, dist

    def _inference_omnianomaly(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                             criterion: nn.Module, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """OmniAnomaly 모델 추론"""
        hidden = None
        predictions, _, _, hidden = self.model.forward(batch_x, hidden if batch_idx else None)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _inference_dagmm(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                        criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """DAGMM 모델 추론"""
        _, x_hat, z, gamma = self.model.forward(batch_x)
        l1, l2 = criterion(x_hat, batch_y), criterion(gamma, batch_y)
        score = l1 + l2
        pred = x_hat.cpu().detach().numpy()
        dist = np.mean(score.cpu().detach().numpy(), axis=2)
        return pred, dist

    def _inference_anomaly_transformer(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                                     criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """AnomalyTransformer 모델 추론"""
        output, series, prior, _ = self.model(batch_x)
        loss = torch.mean(criterion(batch_x, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        temperature = 50
        
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.window_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        dist = cri.detach().cpu().numpy()
        pred = output.cpu().detach().numpy()
        
        return pred, dist

    def _inference_default(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                          criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """기본 모델 추론"""
        predictions = self.model.forward(batch_x)
        pred = predictions.cpu().detach().numpy()
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _save_test_results(self, dist: np.ndarray, attack: np.ndarray, 
                          pred: np.ndarray, valid_score: np.ndarray, 
                          threshold: Optional[float] = None) -> None:
        """
        테스트 결과를 저장합니다.
        
        Args:
            dist: 이상 점수 배열
            attack: 공격 라벨 배열
            pred: 예측값 배열
            valid_score: 검증 점수 배열
            threshold: 이상 탐지 임계값 (선택사항)
        """
        try:
            # 결과 저장 디렉토리 설정
            folder_path = os.path.join(self.savedir, 'test_results')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                if hasattr(self, 'logger'):
                    self.logger.info(f"테스트 결과 저장 디렉토리 생성: {folder_path}")
            
            # 결과 저장
            np.save(os.path.join(folder_path, 'test_dist.npy'), dist)
            np.save(os.path.join(folder_path, 'test_attack.npy'), attack)
            np.save(os.path.join(folder_path, 'test_pred.npy'), pred)
            np.save(os.path.join(folder_path, 'valid_score.npy'), valid_score)
            
            # 메트릭 정보 저장
            metrics_info = {
                'dist_shape': list(dist.shape),
                'attack_shape': list(attack.shape),
                'pred_shape': list(pred.shape),
                'valid_score_shape': list(valid_score.shape),
                'anomaly_ratio': float(np.mean(attack)),
                'dist_mean': float(np.mean(dist)),
                'dist_std': float(np.std(dist))
            }
            
            with open(os.path.join(folder_path, 'test_metrics_info.json'), 'w') as f:
                json.dump(metrics_info, f, indent=4)
            
            # 시각화 그래프 저장
            try:
                visual = check_graph(dist, attack, piece=4, threshold=threshold)
                visual.savefig(os.path.join(folder_path, 'test_visualization.png'), 
                              dpi=300, bbox_inches='tight')
                plt.close(visual)  # 메모리 해제
                if hasattr(self, 'logger'):
                    self.logger.info(f"테스트 시각화 저장 완료: {os.path.join(folder_path, 'test_visualization.png')}")
                    if threshold is not None:
                        self.logger.info(f"임계값 {threshold:.4f}로 시각화 완료")
            except Exception as viz_error:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"시각화 저장 중 오류 발생: {str(viz_error)}")
                else:
                    print(f"시각화 저장 중 오류 발생: {str(viz_error)}")
            
            if hasattr(self, 'logger'):
                self.logger.info(f"테스트 결과 저장 완료: {folder_path}")
                self.logger.info(f"거리 점수 shape: {dist.shape}, 평균: {np.mean(dist):.4f}")
                self.logger.info(f"공격 라벨 shape: {attack.shape}, 이상 비율: {np.mean(attack):.4f}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"테스트 결과 저장 중 오류 발생: {str(e)}")
            else:
                print(f"테스트 결과 저장 중 오류 발생: {str(e)}")

    def _save_inference_results(self, dist: np.ndarray, attack: np.ndarray, 
                               pred: np.ndarray, answer: np.ndarray, 
                               epoch: Optional[int] = None, valid_loader: Optional[Any] = None) -> None:
        """추론 결과를 저장합니다."""
        # 결과 저장 디렉토리 설정
        if epoch is not None:
            folder_path = os.path.join(self.savedir, 'results', f'epoch_{epoch}')
        else:
            folder_path = os.path.join(self.savedir, 'results')
            
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 결과 저장
        np.save(os.path.join(folder_path, f'dist.npy'), dist)
        np.save(os.path.join(folder_path, f'attack.npy'), attack)
        np.save(os.path.join(folder_path, f'pred.npy'), pred)        
        np.save(os.path.join(folder_path, f'answer.npy'), answer)
        
        # 검증 점수 저장 (있는 경우)
        if valid_loader is not None:
            _, valid_score = self.valid(valid_loader, self._select_criterion(), 0)
            valid_score = np.concatenate(valid_score).flatten()
            np.save(os.path.join(folder_path, f'valid_score.npy'), valid_score)
            
        # 시각화 그래프 저장
        visual = check_graph(dist, attack, piece=4)
        visual.savefig(os.path.join(folder_path, f'graph.png'))


# 하위 호환성을 위한 별칭
build_model = AnomalyDetectionModel