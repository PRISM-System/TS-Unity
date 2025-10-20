import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path
from einops import rearrange, reduce, repeat
from typing import Dict, Any, Tuple, Optional
from .BaseModel import BaseAnomalyDetectionModel
import numpy as np


class LSTMAutoencoder(BaseAnomalyDetectionModel):
    """
    LSTM 기반 오토인코더 모델
    
    인코더, 순환 모듈, 디코더로 구성된 이상 탐지 모델입니다.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super(LSTMAutoencoder, self).__init__(config)
        
        # Handle both old params dict format and new config format
        if hasattr(config, 'enc_in'):
            # New config format
            self.feature_num = config.enc_in
            self.batch_size = config.batch_size
            self.rnn_type = getattr(config, 'rnn_type', 'LSTM')
            self.rnn_inp_size = getattr(config, 'rnn_inp_size', 16)
            self.rnn_hid_size = getattr(config, 'rnn_hid_size', 16)
            self.nlayers = getattr(config, 'nlayers', 2)
            self.dropout = config.dropout
        else:
            # Old params dict format (for backward compatibility)
            self.feature_num = config.get('feature_num', 7)
            self.batch_size = config.get('batch_size', 32)
            self.rnn_type = config.get('rnn_type', 'LSTM')
            self.rnn_inp_size = config.get('rnn_inp_size', 16)
            self.rnn_hid_size = config.get('rnn_hid_size', 16)
            self.nlayers = config.get('nlayers', 2)
            self.dropout = config.get('dropout', 0.3)
        self.tie_weights = False
        self.res_connection = False
        self.return_hiddens = False
    
        # 드롭아웃 함수 정의
        self.drop = nn.Dropout(self.dropout)
        
        # 인코더
        self.encoder = nn.Linear(self.feature_num, self.rnn_inp_size)

        # RNN 모델 정의
        if self.rnn_type == 'LSTM':
            self.model = nn.LSTM(
                input_size=self.rnn_inp_size,
                hidden_size=self.rnn_hid_size,
                num_layers=self.nlayers,
                batch_first=True,
                dropout=self.dropout
            )
        elif self.rnn_type == 'GRU':
            self.model = nn.GRU(
                self.rnn_inp_size,
                self.rnn_hid_size,
                self.nlayers,
                dropout=self.dropout
            )
        else:
            raise NotImplementedError(f"지원하지 않는 RNN 타입: {self.rnn_type}")
        
        # 디코더
        self.decoder = nn.Linear(self.rnn_hid_size, self.feature_num)

        # 가중치 초기화
        self.init_weights()
        
    def init_weights(self) -> None:
        """모델 가중치를 초기화합니다."""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        RNN의 초기 숨겨진 상태를 초기화합니다.
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            초기화된 숨겨진 상태
        """
        device = next(self.parameters()).device
        if self.rnn_type == 'GRU':
            return torch.zeros(self.nlayers, batch_size, self.rnn_hid_size, device=device)
        elif self.rnn_type == 'LSTM':
            return (
                torch.zeros(self.nlayers, batch_size, self.rnn_hid_size, device=device),
                torch.zeros(self.nlayers, batch_size, self.rnn_hid_size, device=device)
            )
        else:
            raise ValueError(f'알 수 없는 RNN 타입: {self.rnn_type}. 유효한 옵션: "gru", "lstm"')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, feature_size)
            
        Returns:
            재구성된 출력 텐서 (batch_size, seq_len, feature_size)
        """
        B, S, F = x.size()  # [batch_size, seq_len, feature_size]
        hidden = self.init_hidden(B)
        
        # LSTM 임베딩
        emb = self.encoder(rearrange(x, 'batch seq feature -> (batch seq) feature'))
        emb = self.drop(emb)
        emb = rearrange(emb, '(batch seq) feature -> batch seq feature', batch=B)
        
        # LSTM 실행
        output, hidden = self.model(emb, hidden)
        
        output = self.drop(output) 
        decoded = self.decoder(rearrange(output, 'batch seq feature -> (batch seq) feature'))
        decoded = rearrange(decoded, '(batch seq) feature -> batch seq feature', batch=B)
        
        # 잔차 연결 (선택사항)
        if self.res_connection:
            decoded = decoded + x
            
        # 숨겨진 상태 반환 (선택사항)
        if self.return_hiddens:
            return decoded, hidden, output

        return decoded
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input data.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_size)
            
        Returns:
            Reconstructed output tensor (batch_size, seq_len, feature_size)
        """
        return self.forward(x)
    
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                   epoch: int = 1, **kwargs) -> Tuple[float, np.ndarray]:
        """
        LSTM-AE specific training step with gradient clipping.
        
        Args:
            batch_x: Input tensor
            batch_y: Target tensor
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss_value, anomaly_scores)
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self.forward(batch_x)
        
        # Compute loss
        loss = criterion(output, batch_y)
        
        # Check for NaN or Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf loss detected at epoch {epoch}")
            return float('inf'), np.zeros((batch_x.shape[0], batch_x.shape[1]))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for LSTM stability
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Calculate anomaly scores
        with torch.no_grad():
            reconstructed = self.reconstruct(batch_x)
            anomaly_scores = self._calculate_anomaly_scores(batch_x, reconstructed)
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
        
        return loss.item(), anomaly_scores
    
    def validation_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, 
                       criterion: nn.Module, **kwargs) -> Tuple[float, np.ndarray]:
        """
        Perform a single validation step.
        
        Args:
            batch_x: Input tensor
            batch_y: Target tensor
            criterion: Loss function
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (loss_value, anomaly_scores)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch_x)
            loss = criterion(output, batch_y)
            anomaly_scores = self._calculate_anomaly_scores(batch_x, output)
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
        
        return loss.item(), anomaly_scores

# Compatibility alias
Model = LSTMAutoencoder