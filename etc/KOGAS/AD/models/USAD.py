import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any


class Encoder(nn.Module):
    """
    USAD 모델의 인코더
    
    입력 데이터를 잠재 공간으로 인코딩합니다.
    """
    
    def __init__(self, in_size: int, latent_size: int):
        """
        Args:
            in_size: 입력 특성 수
            latent_size: 잠재 공간 크기
        """
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            w: 입력 텐서 (batch_size, in_size)
            
        Returns:
            인코딩된 잠재 벡터 (batch_size, latent_size)
        """
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    """
    USAD 모델의 디코더
    
    잠재 공간의 벡터를 원본 공간으로 디코딩합니다.
    """
    
    def __init__(self, latent_size: int, out_size: int):
        """
        Args:
            latent_size: 잠재 공간 크기
            out_size: 출력 특성 수
        """
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            z: 잠재 벡터 (batch_size, latent_size)
            
        Returns:
            디코딩된 출력 (batch_size, out_size)
        """
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class USADModel(nn.Module):
    """
    USAD (Unsupervised Anomaly Detection) 모델
    
    두 개의 오토인코더를 사용하여 이상을 탐지하는 모델입니다.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Args:
            params: 모델 하이퍼파라미터
                - window_size: 윈도우 크기
                - feature_num: 특성 수
                - hidden_size: 숨겨진 레이어 크기
        """
        super().__init__()
        
        # 입력과 출력 크기 계산
        w_size = params['window_size'] * params['feature_num']
        z_size = params['window_size'] * params['hidden_size']

        # 인코더와 디코더 초기화
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """
        순전파
        
        Args:
            batch: 입력 배치 (batch_size, window_size, feature_num)
            
        Returns:
            세 개의 재구성 출력 리스트 [w1, w2, w3]
        """
        # 배치를 2D로 변환
        batch_2d = batch.view(([batch.shape[0], batch.shape[1] * batch.shape[2]]))
        
        # 인코딩
        z = self.encoder(batch_2d)
        
        # 첫 번째 디코더로 재구성
        w1 = self.decoder1(z)
        
        # 두 번째 디코더로 재구성
        w2 = self.decoder2(z)
        
        # 첫 번째 재구성을 다시 인코딩하고 두 번째 디코더로 재구성
        w3 = self.decoder2(self.encoder(w1))

        return [w1, w2, w3]

    def training_step(self, batch: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        학습 단계에서의 손실 계산
        
        Args:
            batch: 입력 배치 (batch_size, window_size, feature_num)
            n: 현재 에포크 번호
            
        Returns:
            두 개의 손실 (loss1, loss2)
        """
        # 배치를 2D로 변환
        batch_2d = batch.view(([batch.shape[0], batch.shape[1] * batch.shape[2]]))
        
        # 인코딩
        z = self.encoder(batch_2d)
        
        # 첫 번째 디코더로 재구성
        w1 = self.decoder1(z)
        
        # 두 번째 디코더로 재구성
        w2 = self.decoder2(z)
        
        # 첫 번째 재구성을 다시 인코딩하고 두 번째 디코더로 재구성
        w3 = self.decoder2(self.encoder(w1))
        
        # 손실 계산
        loss1 = (1 / n * torch.mean((batch_2d - w1) ** 2) + 
                (1 - 1 / n) * torch.mean((batch_2d - w3) ** 2))
        loss2 = (1 / n * torch.mean((batch_2d - w2) ** 2) - 
                (1 - 1 / n) * torch.mean((batch_2d - w3) ** 2))
        
        return loss1, loss2

    def validation_step(self, batch: torch.Tensor, n: int) -> Dict[str, torch.Tensor]:
        """
        검증 단계에서의 손실 계산
        
        Args:
            batch: 입력 배치 (batch_size, window_size, feature_num)
            n: 현재 에포크 번호
            
        Returns:
            검증 손실 딕셔너리
        """
        # 배치를 2D로 변환
        batch_2d = batch.view(([batch.shape[0], batch.shape[1] * batch.shape[2]]))
        
        # 인코딩
        z = self.encoder(batch_2d)
        
        # 첫 번째 디코더로 재구성
        w1 = self.decoder1(z)
        
        # 두 번째 디코더로 재구성
        w2 = self.decoder2(z)
        
        # 첫 번째 재구성을 다시 인코딩하고 두 번째 디코더로 재구성
        w3 = self.decoder2(self.encoder(w1))
        
        # 손실 계산
        loss1 = (1 / n * torch.mean((batch_2d - w1) ** 2) + 
                (1 - 1 / n) * torch.mean((batch_2d - w3) ** 2))
        loss2 = (1 / n * torch.mean((batch_2d - w2) ** 2) - 
                (1 - 1 / n) * torch.mean((batch_2d - w3) ** 2))
        
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        검증 에포크 종료 시 평균 손실 계산
        
        Args:
            outputs: 검증 단계 출력 리스트
            
        Returns:
            평균 검증 손실 딕셔너리
        """
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch: int, result: Dict[str, float]) -> None:
        """
        에포크 종료 시 결과 출력
        
        Args:
            epoch: 에포크 번호
            result: 에포크 결과 딕셔너리
        """
        print(
            f"Epoch [{epoch}], val_loss1: {result['val_loss1']:.4f}, "
            f"val_loss2: {result['val_loss2']:.4f}"
        )


# 하위 호환성을 위한 별칭
Model = USADModel