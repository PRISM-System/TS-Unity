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
        
    @abstractmethod
    def detect_anomaly(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.detect_anomaly(x, *args, **kwargs)
        
    @abstractmethod
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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