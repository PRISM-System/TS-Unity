import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging


class BaseTimeSeriesModel(nn.Module, ABC):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.__class__.__name__,
            'total_params': self.get_model_size(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
    
    def freeze_layers(self, layers_to_freeze: list):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                self.logger.info(f"Froze layer: {name}")
                
    def unfreeze_layers(self, layers_to_unfreeze: list):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
                self.logger.info(f"Unfroze layer: {name}")


class BaseForecastingModel(BaseTimeSeriesModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        
    @abstractmethod
    def forecast(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forecast(x, *args, **kwargs)


class BaseAnomalyDetectionModel(BaseTimeSeriesModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.win_size = getattr(config, 'win_size', 100)
        self.detection_method = 'reconstruction'  # Default to reconstruction-based
        
    @abstractmethod
    def detect_anomaly(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_output, anomaly_scores)
        """
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for reconstruction."""
        return self.reconstruct(x, *args, **kwargs)
        
    @abstractmethod
    def reconstruct(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Reconstruct input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        raise NotImplementedError
        
    def get_anomaly_score(self, x: torch.Tensor, method: str = 'mse') -> torch.Tensor:
        """
        Calculate anomaly scores using reconstruction error.
        
        Args:
            x: Input tensor
            method: Scoring method ('mse', 'mae', 'combined')
            
        Returns:
            Anomaly scores tensor
        """
        reconstructed = self.reconstruct(x)
        
        if method == 'mse':
            return torch.mean((x - reconstructed) ** 2, dim=-1)
        elif method == 'mae': 
            return torch.mean(torch.abs(x - reconstructed), dim=-1)
        elif method == 'combined':
            mse = torch.mean((x - reconstructed) ** 2, dim=-1)
            mae = torch.mean(torch.abs(x - reconstructed), dim=-1)
            return 0.5 * (mse + mae)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def get_reconstruction_error(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                               reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate reconstruction error.
        
        Args:
            x: Original input tensor
            reconstructed: Reconstructed tensor
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error tensor
        """
        error = (x - reconstructed) ** 2
        
        if reduction == 'mean':
            return torch.mean(error, dim=-1)
        elif reduction == 'sum':
            return torch.sum(error, dim=-1)
        elif reduction == 'none':
            return error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")


class BaseImputationModel(BaseTimeSeriesModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.mask_rate = getattr(config, 'mask_rate', 0.25)
        
    @abstractmethod
    def impute(self, x: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.impute(x, mask, *args, **kwargs)


class BaseClassificationModel(BaseTimeSeriesModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.num_class = config.num_class
        
    @abstractmethod
    def classify(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.classify(x, *args, **kwargs)