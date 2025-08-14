import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path
from einops import rearrange, reduce, repeat
from typing import Dict, Any, Tuple, Optional


class LSTMAutoencoder(nn.Module):
    """
    LSTM 기반 오토인코더 모델
    
    인코더, 순환 모듈, 디코더로 구성된 이상 탐지 모델입니다.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Args:
            params: 모델 하이퍼파라미터
                - feature_num: 특성 수
                - batch_size: 배치 크기
                - rnn_type: RNN 타입 ('LSTM' 또는 'GRU')
                - rnn_inp_size: RNN 입력 크기
                - rnn_hid_size: RNN 숨겨진 크기
                - nlayers: RNN 레이어 수
                - dropout: 드롭아웃 비율
        """
        super(LSTMAutoencoder, self).__init__()
        self.feature_num = params['feature_num']
        self.batch_size = params['batch_size']
        self.rnn_type = params['rnn_type']
        self.rnn_inp_size = params['rnn_inp_size']
        self.rnn_hid_size = params['rnn_hid_size']
        self.nlayers = params['nlayers']
        self.dropout = params['dropout']
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
        if self.rnn_type == 'GRU':
            return torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda()
        elif self.rnn_type == 'LSTM':
            return (
                torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda(),
                torch.zeros(self.nlayers, batch_size, self.rnn_hid_size).cuda()
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


# 하위 호환성을 위한 별칭
Model = LSTMAutoencoder