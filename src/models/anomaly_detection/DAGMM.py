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
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object or params dict with model parameters
        """
        super(DAGMM, self).__init__()

        # Handle both old params dict format and new config format
        if hasattr(config, 'win_size'):
            # New config format
            self.n_window = config.win_size
            self.n_feats = config.enc_in
            self.n_hidden = getattr(config, 'hidden_size', 128)
            self.n_latent = getattr(config, 'latent_size', 64)
        else:
            # Old params dict format (for backward compatibility)
            self.n_window = config.get('window_size', 5)
            self.n_feats = config.get('feature_num', 7)
            self.n_hidden = config.get('hidden_size', 128)
            self.n_latent = config.get('latent_size', 64)
            
        self.beta = 0.01

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
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input data.
        
        Args:
            x: Input tensor (batch_size, window_size, feature_num)
            
        Returns:
            Reconstructed output tensor (batch_size, window_size, feature_num)
        """
        _, x_hat, _, _ = self.forward(x)
        return x_hat
    
    def detect_anomaly(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies using DAGMM reconstruction error.
        
        Args:
            x: Input tensor (batch_size, window_size, feature_num)
            
        Returns:
            Tuple of (reconstructed_output, anomaly_scores)
        """
        z_c, x_hat, z, gamma = self.forward(x)
        
        # 재구성 오차를 이상 점수로 사용
        anomaly_scores = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        
        return x_hat, anomaly_scores
    
    def get_anomaly_score(self, x: torch.Tensor, method: str = 'mse') -> torch.Tensor:
        """
        Calculate anomaly scores using reconstruction error.
        
        Args:
            x: Input tensor (batch_size, window_size, feature_num)
            method: Scoring method ('mse', 'mae', 'combined')
            
        Returns:
            Anomaly scores tensor
        """
        reconstructed = self.reconstruct(x)
        
        if method == 'mse':
            return torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        elif method == 'mae':
            return torch.mean(torch.abs(x - reconstructed), dim=(1, 2))
        elif method == 'combined':
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            mae = torch.mean(torch.abs(x - reconstructed), dim=(1, 2))
            return 0.5 * (mse + mae)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def compute_loss(self, x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Compute training loss for DAGMM model.
        
        Args:
            x: Input tensor (batch_size, window_size, feature_num)
            criterion: Loss function
            
        Returns:
            Combined loss tensor
        """
        z_c, x_hat, z, gamma = self.forward(x)
        
        # 재구성 손실
        rec_loss = criterion(x_hat, x)
        
        # 추가적인 DAGMM 손실을 여기에 구현할 수 있습니다
        # (예: GMM 정규화 손실 등)
        
        return torch.mean(rec_loss)


# 하위 호환성을 위한 별칭
Model = DAGMM