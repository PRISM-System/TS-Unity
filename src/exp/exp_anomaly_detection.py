import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from core.base_model import BaseAnomalyDetectionModel
import warnings
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        
    def _build_model(self) -> BaseAnomalyDetectionModel:
        # USAD 모델을 위한 차원 설정
        if self.args.model == 'USAD':
            # PSM 데이터셋의 차원에 맞춰 명시적으로 설정
            self.args.win_size = self.args.seq_len
            self.args.enc_in = self.args.enc_in
            self.args.feature_num = self.args.enc_in
            self.args.window_size = self.args.seq_len
            
            # 디버깅을 위한 로그 출력
            print(f"USAD 모델 차원 설정:")
            print(f"  win_size: {self.args.win_size}")
            print(f"  enc_in: {self.args.enc_in}")
            print(f"  feature_num: {self.args.feature_num}")
            print(f"  window_size: {self.args.window_size}")
            print(f"  seq_len: {self.args.seq_len}")
        
        module = self.model_dict.get(self.args.model)
        if module is None:
            raise ImportError(f"Could not load model module for '{self.args.model}'")
        model = module.Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model == 'USAD':
            # USAD 모델은 두 개의 optimizer가 필요
            model_optim1 = torch.optim.Adam(self.model.encoder.parameters(), lr=self.args.learning_rate)
            model_optim2 = torch.optim.Adam(self.model.decoder2.parameters(), lr=self.args.learning_rate)
            return (model_optim1, model_optim2)
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')  # Keep per-sample loss for anomaly scoring
        return criterion
    
    def _is_reconstruction_model(self) -> bool:
        """Check if the current model is reconstruction-based."""
        reconstruction_models = [
            'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'DAGMM',
            'AutoEncoder', 'VAE', 'LSTM_VAE', 'LSTM_AE'
        ]
        return self.args.model in reconstruction_models
    
    def _get_anomaly_scores(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get anomaly scores using appropriate method based on model type.
        
        Args:
            batch_x: Input batch data
            
        Returns:
            Tuple of (outputs, anomaly_scores)
        """
        if self._is_reconstruction_model():
            return self._reconstruction_based_scoring(batch_x)
        else:
            return self._prediction_based_scoring(batch_x)
    
    def _reconstruction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using reconstruction-based method."""
        try:
            # Try to use model's specific anomaly detection method
            if hasattr(self.model, 'detect_anomaly'):
                outputs, scores = self.model.detect_anomaly(batch_x)
            elif hasattr(self.model, 'get_anomaly_score'):
                outputs = self.model(batch_x)
                scores = self.model.get_anomaly_score(batch_x)
            else:
                # Fallback: calculate reconstruction error
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: simple reconstruction error
            try:
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores
    
    def _my_kl_loss(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
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
    
    def _prediction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using prediction-based method."""
        try:
            # For prediction-based detection, we need to:
            # 1. Make predictions for the next few steps
            # 2. Calculate prediction variance or error as anomaly score
            
            # Get prediction horizon (use config pred_len or default to 1)
            pred_len = getattr(self.args, 'pred_len', 1)
            
            # Make predictions
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(batch_x, pred_len)
            else:
                # Fallback: use model directly
                predictions = self.model(batch_x)
            
            # Calculate prediction error as anomaly score
            if predictions.dim() == 3:  # (batch, pred_len, features)
                # Use prediction variance across time steps as anomaly score
                scores = torch.var(predictions, dim=1, keepdim=True)
            else:
                # Use prediction magnitude as anomaly score
                scores = torch.mean(torch.abs(predictions), dim=-1, keepdim=True)
            
            # For prediction-based, outputs are the predictions
            outputs = predictions
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: use input variance as anomaly score
            try:
                scores = torch.var(batch_x, dim=1, keepdim=True)
                outputs = batch_x
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores
    
    def _train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   optimizers: Any, criterion: nn.Module, epoch: int, 
                   alpha: float = 0.5, beta: float = 0.5) -> Tuple[float, np.ndarray]:
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
                            optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
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
                       optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
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
                   optimizers: Any, criterion: nn.Module, 
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
                          optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
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
                    optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
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
                                 optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """AnomalyTransformer 모델 학습"""
        optimizers.zero_grad()
        output, series, prior, _ = self.model(batch_x)

        # Series loss 계산
        series_loss = 0.0
        window_size = getattr(self.args, 'window_size', 100)
        for u in range(len(prior)):
            prior_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, window_size)
            series_loss += (torch.mean(self._my_kl_loss(series[u], prior_normalized.detach())) + 
                           torch.mean(self._my_kl_loss(prior_normalized.detach(), series[u])))
        series_loss /= len(prior)

        # Reconstruction loss 계산
        rec_loss = criterion(output, batch_x)
        k = getattr(self.args, 'k', 0.1)
        loss = rec_loss - k * series_loss
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward(retain_graph=True)
        optimizers.step()
        
        return loss.item(), score

    def _train_default(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                      optimizers: Any, criterion: nn.Module) -> Tuple[float, np.ndarray]:
        """기본 모델 학습"""
        optimizers.zero_grad()
        output = self.model(batch_x)
        loss = criterion(output, batch_y)
        score = np.mean(loss.cpu().detach().numpy(), axis=2)
        loss = torch.mean(loss)
        loss.backward()
        optimizers.step()
        return loss.item(), score
    
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
                    (1 - 1 / (epoch +1)) * torch.mean((batch_y - w3) ** 2, axis=2))
            loss2 = (1 / (epoch + 1) * torch.mean((batch_y - w2) ** 2, axis=2) - 
                    (1 - 1 / (epoch + 1)) * torch.mean((batch_y - w3) ** 2, axis=2))
            
            loss = loss1 + loss2
            valid_score = loss.cpu().detach().numpy()
            epoch_loss1 = torch.mean(loss1)
            epoch_loss2 = torch.mean(loss2)
            loss = epoch_loss1 + epoch_loss2
            return loss, valid_score
            
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
            
        elif self.args.model == 'LSTM_AE':
            # LSTM-AE 모델의 손실 계산
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            return loss, valid_score
            
        else:
            # 기본 재구성 모델들
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            valid_score = np.mean(loss.cpu().detach().numpy(), axis=2)
            loss = torch.mean(loss)
            return loss, valid_score
    
    def _compute_model_loss_with_score(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                                      criterion: nn.Module, epoch: int = 1) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute model-specific loss and score for validation.
        
        Args:
            batch_x: Input batch data
            batch_y: Target batch data
            criterion: Loss function
            epoch: Current epoch (used for some models like USAD)
            
        Returns:
            Tuple of (loss tensor, score numpy array)
        """
        # Use the updated _compute_model_loss method
        loss, valid_score = self._compute_model_loss(batch_x, batch_y, criterion, epoch)
        return loss, valid_score
    
    def _compute_usad_loss(self, batch_x: torch.Tensor, criterion: nn.Module, epoch: int) -> torch.Tensor:
        """USAD-specific loss computation."""
        if hasattr(self.model, 'training_step'):
            # Use model's training step method
            loss1, loss2 = self.model.training_step(batch_x, epoch)
            return loss1 + loss2
        else:
            # Fallback to reconstruction loss
            outputs = self.model(batch_x)
            if isinstance(outputs, list):
                # USAD returns multiple outputs
                w1, w2 = outputs[0], outputs[1] if len(outputs) > 1 else outputs[0]
                loss = 0.5 * (criterion(w1, batch_x).mean() + criterion(w2, batch_x).mean())
            else:
                loss = criterion(outputs, batch_x).mean()
            return loss
    
    def _compute_anomaly_transformer_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """AnomalyTransformer-specific loss computation."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback to reconstruction loss
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) >= 4:
                output, series, prior, _ = outputs
                
                # Series loss 계산
                series_loss = 0.0
                for u in range(len(prior)):
                    # KL divergence loss for series and prior
                    prior_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.args.window_size)
                    series_loss += (torch.mean(self._my_kl_loss(series[u], prior_normalized.detach())) + 
                                   torch.mean(self._my_kl_loss(prior_normalized.detach(), series[u])))
                series_loss /= len(prior)
                
                # Reconstruction loss 계산
                rec_loss = criterion(output, batch_x)
                # k는 하이퍼파라미터로 설정 (기본값 0.1)
                k = getattr(self.args, 'k', 0.1)
                loss = rec_loss - k * series_loss
                return torch.mean(loss)
            else:
                loss = criterion(outputs, batch_x).mean()
                return loss
    
    def _compute_omnianomaly_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """OmniAnomaly-specific loss computation."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                x_recon, mu, logvar = outputs[0], outputs[1], outputs[2]
                rec_loss = criterion(x_recon, batch_x).mean()
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                beta = getattr(self.model, 'beta', 0.01)
                return rec_loss + beta * kl_loss
            else:
                return criterion(outputs, batch_x).mean()
    
    def _compute_lstm_vae_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """LSTM-VAE-specific loss computation."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, list) and len(outputs) >= 2:
                x_decoded, kl_loss = outputs[0], outputs[1]
                rec_loss = criterion(x_decoded, batch_x).mean()
                return rec_loss + kl_loss
            else:
                return criterion(outputs, batch_x).mean()
    
    def _compute_dagmm_loss(self, batch_x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """DAGMM-specific loss computation."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch_x, criterion)
        else:
            # Fallback implementation
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                _, x_hat = outputs[0], outputs[1]
                loss = criterion(x_hat, batch_x).mean()
            else:
                loss = criterion(outputs, batch_x).mean()
            return loss

    def vali(self, vali_data, vali_loader, criterion) -> Tuple[float, List[np.ndarray]]:
        """
        검증을 수행합니다.
        
        Args:
            vali_data: 검증 데이터셋
            vali_loader: 검증 데이터 로더
            criterion: 손실 함수
            
        Returns:
            총 손실과 검증 점수 리스트
        """
        total_loss = []
        valid_score = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                # For reconstruction models, target is the input itself
                if self._is_reconstruction_model():
                    batch_y = batch_x
                else:
                    batch_y = batch_y.float().to(self.device)
                
                # Use model-specific loss computation
                loss, score = self._compute_model_loss_with_score(batch_x, batch_y, criterion)
                
                total_loss.append(loss.item())
                valid_score.append(score)
                
        self.model.train()
        return np.mean(total_loss), valid_score

    def train(self) -> Dict[str, Any]:
        import pdb; pdb.set_trace()  # 디버깅을 위한 중단점
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.des)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        train_history = []
        val_history = []

        # Log detection method being used
        detection_method = "reconstruction-based" if self._is_reconstruction_model() else "prediction-based"
        self.logger.info(f"Using {detection_method} anomaly detection with model: {self.args.model}")

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # USAD 모델의 경우 optimizer가 tuple이므로 zero_grad를 각각 호출
                if isinstance(model_optim, tuple):
                    for opt in model_optim:
                        opt.zero_grad()
                else:
                    model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                # For reconstruction models, target is the input itself
                if self._is_reconstruction_model():
                    batch_y = batch_x
                else:
                    batch_y = batch_y.float().to(self.device)

                # Model-specific training step (backward와 step이 이미 포함됨)
                loss, score = self._train_step(batch_x, batch_y, model_optim, criterion, epoch + 1)
                train_loss.append(loss)

            train_loss = np.average(train_loss)
            vali_loss, vali_score = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_score = self.vali(test_data, test_loader, criterion)
            
            train_metrics = {'loss': train_loss}
            val_metrics = {'loss': vali_loss}
            
            train_history.append(train_metrics)
            val_history.append(val_metrics)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            # Learning rate 조정 (USAD 모델의 경우 두 optimizer 모두 조정)
            if isinstance(model_optim, tuple):
                for opt in model_optim:
                    adjust_learning_rate(opt, epoch + 1, self.args)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_model_path': best_model_path,
            'detection_method': detection_method
        }

    def test(self, alpha: float = 0.5, beta: float = 0.5) -> Dict[str, Any]:
        """
        모델 테스트를 수행합니다.
        
        Args:
            alpha: USAD 모델용 가중치 파라미터
            beta: USAD 모델용 가중치 파라미터
            
        Returns:
            테스트 결과 딕셔너리
        """
        test_data, test_loader = self._get_data(flag='test')
        
        if hasattr(self.args, 'test') and self.args.test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.des, 'checkpoint.pth')))

        # 테스트 실행
        dist, attack, pred = self._run_test(test_loader, alpha, beta)
        
        # 검증 데이터로 임계값 설정 (가능한 경우)
        try:
            vali_data, vali_loader = self._get_data(flag='val')
            _, valid_score = self.vali(vali_data, vali_loader, self._select_criterion())
            if isinstance(valid_score, list) and len(valid_score) > 0:
                valid_score = np.concatenate(valid_score).flatten()
            else:
                valid_score = None
        except:
            valid_score = None
        
        # 이상 탐지 메트릭 계산
        if hasattr(test_data, 'labels') or 'attack' in locals():
            labels = getattr(test_data, 'labels', None)
            if labels is None and 'attack' in locals():
                labels = attack
            
            if labels is not None:
                history, auc = self._calculate_anomaly_metrics(dist, labels)
                
                # PA%K metrics 계산
                pa_metrics = self._calculate_pa_metrics(dist, labels)
                
                result_dict = {
                    'auc': auc,
                    'pa_auc': pa_metrics['pa_auc'],
                    'anomaly_score_mean': np.mean(dist),
                    'anomaly_score_std': np.std(dist),
                    'detection_method': 'model_specific',
                    'method_info': {
                        'model_type': self.args.model,
                        'description': 'Uses model-specific anomaly detection logic'
                    }
                }
                
                # Add PA%K individual metrics
                result_dict.update(pa_metrics['metrics'])
                
                # 성능 지표 결과를 로그로 출력
                self._log_test_results(result_dict, pa_metrics, history)
                
                # Save results
                folder_path = './results/' + self.args.des + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                np.save(folder_path + 'anomaly_scores.npy', dist)
                np.save(folder_path + 'predictions.npy', pred)
                np.save(folder_path + 'labels.npy', labels)
                np.save(folder_path + 'pa_metrics.npy', pa_metrics)
                
                return result_dict
        
        # 기본 결과 반환
        folder_path = './results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path + 'anomaly_scores.npy', dist)
        np.save(folder_path + 'predictions.npy', pred)
        
        basic_result = {
            'anomaly_score_mean': np.mean(dist),
            'anomaly_score_std': np.std(dist),
            'detection_method': 'model_specific',
            'method_info': {
                'model_type': self.args.model,
                'description': 'Uses model-specific anomaly detection logic'
            }
        }
        
        # 기본 결과도 로그로 출력
        self._log_basic_test_results(basic_result)
        
        return basic_result

    def _run_test(self, test_loader: Any, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """테스트를 실행하여 거리, 공격 라벨, 예측값을 반환합니다."""
        dist = []
        attack = []
        pred = []
        criterion = self._select_criterion()
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # 모델별 테스트 로직 실행
                batch_pred, batch_dist = self._test_step(
                    batch_x, batch_y, criterion, i, alpha, beta
                )
                
                pred.append(batch_pred)
                dist.append(batch_dist)
                
                # 공격 라벨이 있는 경우 (PSM 데이터셋 등)
                if hasattr(batch_y, 'shape') and len(batch_y.shape) >= 2:
                    # 중앙점의 라벨 사용
                    center_idx = batch_y.shape[1] // 2
                    batch_attack = batch_y[:, center_idx].cpu().numpy()
                else:
                    batch_attack = batch_y.cpu().numpy()
                attack.append(batch_attack)

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
        predictions, _ = self.model(batch_x)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_lstm_vae(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                      criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """LSTM-VAE 모델 테스트"""
        predictions = self.model(batch_x)
        score = criterion(predictions[0], batch_y).cpu().detach().numpy()
        pred = predictions[0].cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_usad(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                  alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """USAD 모델 테스트"""
        predictions = self.model(batch_x)
        w1 = predictions[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        w2 = predictions[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
        pred = (alpha * (batch_x - w1) + beta * (batch_x - w2)).cpu().detach().numpy()
        dist = (alpha * torch.mean((batch_x - w1) ** 2, axis=2) + 
               beta * torch.mean((batch_x - w2) ** 2, axis=2)).detach().cpu().numpy()
        return pred, dist

    def _test_omnianomaly(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                         criterion: nn.Module, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """OmniAnomaly 모델 테스트"""
        hidden = None
        predictions, _, _, hidden = self.model(batch_x, hidden if batch_idx == 0 else None)
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        pred = predictions.cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _test_dagmm(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   criterion: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """DAGMM 모델 테스트"""
        _, x_hat, _, _ = self.model(batch_x)
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
                series_loss = self._my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss = self._my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += self._my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.window_size)).detach()) * temperature
                prior_loss += self._my_kl_loss(
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
        predictions = self.model(batch_x)
        pred = predictions.cpu().detach().numpy()
        score = criterion(predictions, batch_y).cpu().detach().numpy()
        dist = np.mean(score, axis=2)
        return pred, dist

    def _calculate_anomaly_metrics(self, dist: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """이상 탐지 메트릭을 계산합니다."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        try:
            auc = roc_auc_score(labels.flatten(), dist.flatten())
            ap = average_precision_score(labels.flatten(), dist.flatten())
        except ValueError as e:
            self.logger.warning(f"Could not calculate standard metrics: {e}")
            auc, ap = 0.0, 0.0
        
        history = {
            'threshold': [np.percentile(dist, 95)],  # 95th percentile as threshold
            'auc': auc,
            'ap': ap
        }
        
        return history, auc

    def _log_test_results(self, result_dict: Dict[str, Any], pa_metrics: Dict[str, Any], history: Dict[str, Any]) -> None:
        """테스트 결과를 로그로 출력합니다."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 ANOMALY DETECTION TEST RESULTS")
        self.logger.info("=" * 80)
        
        # 기본 정보
        self.logger.info(f"📊 Model: {self.args.model}")
        self.logger.info(f"📁 Dataset: {self.args.data}")
        self.logger.info(f"🔍 Detection Method: {result_dict['detection_method']}")
        self.logger.info(f"📈 Description: {result_dict['method_info']['description']}")
        
        # 기본 메트릭
        self.logger.info("-" * 50)
        self.logger.info("📊 BASIC METRICS")
        self.logger.info("-" * 50)
        self.logger.info(f"🎯 ROC-AUC: {result_dict['auc']:.6f}")
        self.logger.info(f"📊 Average Precision: {history.get('ap', 'N/A')}")
        self.logger.info(f"📈 PA%K AUC: {result_dict['pa_auc']:.6f}")
        self.logger.info(f"📊 Anomaly Score Mean: {result_dict['anomaly_score_mean']:.6f}")
        self.logger.info(f"📊 Anomaly Score Std: {result_dict['anomaly_score_std']:.6f}")
        
        # PA%K 개별 메트릭
        self.logger.info("-" * 50)
        self.logger.info("📊 PA%K INDIVIDUAL METRICS")
        self.logger.info("-" * 50)
        
        # K 값별로 그룹화하여 출력
        k_values = []
        for key in pa_metrics['metrics'].keys():
            if key.startswith('f1_'):
                k_values.append(int(key.split('_')[1]))
        
        k_values.sort()
        
        for k in k_values:
            f1_key = f'f1_{k}'
            precision_key = f'precision_{k}'
            recall_key = f'recall_{k}'
            auc_key = f'roc_auc_{k}'
            
            if f1_key in pa_metrics['metrics']:
                f1 = pa_metrics['metrics'][f1_key]
                precision = pa_metrics['metrics'].get(precision_key, 'N/A')
                recall = pa_metrics['metrics'].get(recall_key, 'N/A')
                auc = pa_metrics['metrics'].get(auc_key, 'N/A')
                
                # 문자열 값인 경우 포맷팅하지 않음
                if isinstance(f1, (int, float)) and isinstance(precision, (int, float)) and isinstance(recall, (int, float)) and isinstance(auc, (int, float)):
                    self.logger.info(f"🎯 PA%{k:02d}: F1={f1:.6f}, P={precision:.6f}, R={recall:.6f}, AUC={auc:.6f}")
                else:
                    self.logger.info(f"🎯 PA%{k:02d}: F1={f1}, P={precision}, R={recall}, AUC={auc}")
        
        # 임계값 정보
        if 'threshold' in history and history['threshold']:
            self.logger.info("-" * 50)
            self.logger.info("🔍 THRESHOLD INFORMATION")
            self.logger.info("-" * 50)
            self.logger.info(f"📊 95th Percentile Threshold: {history['threshold'][0]:.6f}")
        
        # 요약
        self.logger.info("-" * 50)
        self.logger.info("📋 SUMMARY")
        self.logger.info("-" * 50)
        self.logger.info(f"✅ Test completed successfully for {self.args.model}")
        self.logger.info(f"📁 Results saved to: ./results/{self.args.des}/")
        self.logger.info(f"🎯 Best PA%K F1: {max([pa_metrics['metrics'].get(f'f1_{k}', 0) for k in k_values]):.6f}")
        self.logger.info("=" * 80)
    
    def predict_single(self, input_data: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Make prediction for single input (mainly for prediction-based models).
        
        Args:
            input_data: Input tensor
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(input_data, num_steps)
            else:
                # Fallback: use model directly
                predictions = self.model(input_data)
        
        return predictions
    
    def detect_anomaly(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies in input data.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Anomaly scores tensor
        """
        self.model.eval()
        with torch.no_grad():
            outputs, scores = self._get_anomaly_scores(input_data)
        
        return scores
    
    def _calculate_pa_metrics(self, anomaly_scores: np.ndarray, labels: np.ndarray, 
                             K_VALUES: list = None) -> dict:
        """
        Calculate PA%K metrics following KOGAS methodology.
        
        Args:
            anomaly_scores: Anomaly scores array
            labels: Ground truth labels array
            K_VALUES: List of K values for PA%K calculation
            
        Returns:
            Dictionary containing PA%K metrics
        """
        if K_VALUES is None:
            K_VALUES = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        # Flatten arrays if needed
        scores_flat = anomaly_scores.flatten()
        labels_flat = labels.flatten()
        
        # Check if we have both classes
        unique_classes = np.unique(labels_flat)
        if len(unique_classes) == 1:
            self.logger.warning(f"Only one class present in labels: {unique_classes[0]}")
            return {'pa_auc': 0.0, 'metrics': {}}
        
        metrics = {}
        f1_values = []
        
        # Calculate threshold range
        start_threshold = np.percentile(scores_flat, 90)
        end_threshold = np.percentile(scores_flat, 99)
        self.logger.info(f'PA%K threshold range: {start_threshold:.4f} to {end_threshold:.4f}')
        
        # Calculate PA%K for each K value
        for k in K_VALUES:
            try:
                # Import the specific metric function if available
                try:
                    from utils.metrics import bf_search
                    result, threshold = bf_search(
                        scores_flat, labels_flat,
                        start=np.percentile(scores_flat, 50),
                        end=np.percentile(scores_flat, 99),
                        step_num=1000,
                        K=k,
                        verbose=False
                    )
                    f1, precision, recall, _, _, _, _, roc_auc, _, _ = result
                except ImportError:
                    # Fallback implementation
                    precision, recall, thresholds = precision_recall_curve(labels_flat, scores_flat)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    f1, precision, recall = f1_scores[best_idx], precision[best_idx], recall[best_idx]
                    roc_auc = roc_auc_score(labels_flat, scores_flat)
                    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                
                f1_values.append(f1)
                metrics[f'precision_{k}'] = precision
                metrics[f'recall_{k}'] = recall
                metrics[f'f1_{k}'] = f1
                metrics[f'roc_auc_{k}'] = roc_auc
                
                self.logger.info(f"PA%{k:02d}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate PA%{k} metrics: {e}")
                f1_values.append(0.0)
                metrics[f'f1_{k}'] = 0.0
        
        # Calculate PA%K AUC
        pa_auc = self._calculate_pa_auc(f1_values, K_VALUES)
        metrics['pa_auc'] = pa_auc
        
        self.logger.info(f'PA%K AUC: {pa_auc:.4f}')
        
        return {'pa_auc': pa_auc, 'metrics': metrics}
    
    def _calculate_pa_auc(self, f1_values: list, k_values: list) -> float:
        """Calculate PA%K AUC following KOGAS methodology."""
        auc = 0
        for i in range(len(k_values) - 1):
            auc += 0.5 * (f1_values[i] + f1_values[i + 1]) * (int(k_values[i + 1]) - int(k_values[i]))
        auc /= 100
        return auc