import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.Embed import PositionalEmbedding, CausalConv1d
from layers.Attention import VariableTemporalAttention
from layers.Transformer_Enc import PreNorm
from typing import Dict, Any, Tuple, List, Optional


class VTTPAT(nn.Module):
    """
    VTTPAT (Variable and Temporal Transformer with Parallel Attention) 모델
    
    변수별 어텐션과 시간적 어텐션을 병렬로 결합한 트랜스포머 기반 이상 탐지 모델입니다.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Args:
            params: 모델 하이퍼파라미터
                - feature_num: 특성 수
                - n_layer: 트랜스포머 블록 수
                - n_head: 어텐션 헤드 수
                - hidden_size: 숨겨진 크기
                - attn_pdrop: 어텐션 드롭아웃 비율
                - resid_pdrop: 잔차 드롭아웃 비율
                - time_emb: 시간 임베딩 크기
        """
        super(VTTPAT, self).__init__()
        self.feature_num = params['feature_num']
        self.num_transformer_blocks = params['n_layer']
        self.num_heads = params['n_head']
        self.embedding_dims = params['hidden_size']
        self.attn_dropout = params['attn_pdrop']
        self.ff_dropout = params['resid_pdrop']
        self.time_emb = params['time_emb']

        # 위치 임베딩과 값 임베딩
        self.position_embedding = PositionalEmbedding(d_model=self.embedding_dims)
        self.value_embedding = nn.Linear(self.time_emb, self.embedding_dims)

        # 인과적 컨볼루션 레이어들
        self.causal_conv1 = CausalConv1d(
            self.feature_num,
            self.feature_num,
            kernel_size=4,
            dilation=1,
            groups=self.feature_num
        )
        self.causal_conv2 = CausalConv1d(
            self.feature_num,
            self.feature_num,
            kernel_size=8,
            dilation=2,
            groups=self.feature_num
        )
        self.causal_conv3 = CausalConv1d(
            self.feature_num,
            self.feature_num,
            kernel_size=16,
            dilation=3,
            groups=self.feature_num
        )

        # 트랜스포머 레이어들
        self.transformer_layers = nn.ModuleList([])

        for _ in range(self.num_transformer_blocks):
            self.transformer_layers.append(
                PreNorm(self.embedding_dims, VariableTemporalAttention(
                    self.embedding_dims,
                    heads=self.num_heads,
                    dim_head=self.embedding_dims,
                    dropout=self.attn_dropout
                ))
            )

        self.dropout = nn.Dropout(self.ff_dropout)

        # 출력 MLP 헤드
        self.mlp_head = nn.Linear(self.feature_num * self.embedding_dims, self.feature_num)

    def forward(self, x: torch.Tensor, use_attn: bool = False) -> Tuple[torch.Tensor, List]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, window_size, feature_num)
            use_attn: 어텐션 가중치 반환 여부
            
        Returns:
            출력 텐서와 어텐션 가중치 리스트
        """
        variable_attn_weights = []
        temporal_attn_weights = []
        b, w, f = x.shape

        # 인과적 컨볼루션 적용
        x = rearrange(x, 'b w f -> b f w')
        conv1 = self.causal_conv1(x)
        conv2 = self.causal_conv2(x)
        conv3 = self.causal_conv3(x)

        # 다중 스케일 특성 결합
        x = torch.stack([x, conv1, conv2, conv3], dim=-1)
        x = rearrange(x, 'b f w d -> b w f d')
        x = self.value_embedding(x)

        # 위치 임베딩 추가
        position_emb = self.position_embedding(x)
        position_emb = repeat(position_emb, 'b t d -> b t f d', f=f)
        x += position_emb
        x = self.dropout(x)
        h = x

        # 트랜스포머 레이어 통과
        for attn in self.transformer_layers:
            x, vweights, tweights = attn(h, use_attn=use_attn)
            h = x + h
            variable_attn_weights.append(vweights)
            temporal_attn_weights.append(tweights)

        # 출력 생성
        h = rearrange(h, 'b w f d -> b w (f d)')
        h = self.mlp_head(h)
        
        # 어텐션 가중치를 텐서로 변환
        if use_attn:
            variable_attn_weights = torch.stack(variable_attn_weights, dim=0)
            temporal_attn_weights = torch.stack(temporal_attn_weights, dim=0)
        
        return h, [variable_attn_weights, temporal_attn_weights]


# 하위 호환성을 위한 별칭
Model = VTTPAT