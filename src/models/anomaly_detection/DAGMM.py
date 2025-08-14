import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Any, Tuple


class DAGMM(nn.Module):
    """
    DAGMM (Deep Autoencoding Gaussian Mixture Model) 모델
    
    오토인코더와 가우시안 혼합 모델을 결합한 이상 탐지 모델입니다.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Args:
            params: 모델 하이퍼파라미터
                - window_size: 윈도우 크기
                - feature_num: 특성 수
                - hidden_size: 숨겨진 레이어 크기
                - latent_size: 잠재 공간 크기
        """
        super(DAGMM, self).__init__()

        self.beta = 0.01
        self.n_window = params['window_size']  # DAGMM w_size = 5
        self.n_feats = params['feature_num']
        self.n_hidden = params['hidden_size']
        self.n_latent = params['latent_size']

        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        
        # GMM 추정기
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        재구성 오차를 계산합니다.
        
        Args:
            x: 원본 입력
            x_hat: 재구성된 출력
            
        Returns:
            상대 유클리드 거리와 코사인 유사도
        """
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, window_size, feature_num)
            
        Returns:
            z_c: 잠재 벡터
            x_hat: 재구성된 출력
            z: 확장된 잠재 벡터 (재구성 오차 포함)
            gamma: GMM 가중치
        """
        b, w, f = x.shape
        
        # 인코더-디코더
        x_flat = rearrange(x, 'b w f -> b (w f)') 
        
        z_c = self.encoder(x_flat)
        x_hat = self.decoder(z_c)
        
        # 재구성 오차 계산
        rec_1, rec_2 = self.compute_reconstruction(x_flat, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        
        # GMM 가중치 추정
        gamma = self.estimate(z)
        
        # 형태 복원
        x_hat = rearrange(x_hat, 'b (w f) -> b w f', w=w) 
        gamma = rearrange(gamma, 'b (w f) -> b w f', w=w) 
        
        return z_c, x_hat, z, gamma


# 하위 호환성을 위한 별칭
Model = DAGMM