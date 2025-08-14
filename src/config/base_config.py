"""
Configuration Management Module

This module provides configuration classes for different time series analysis tasks,
including base configuration, task-specific configurations, and utility methods.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import yaml
import json
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of supported task types."""
    LONG_TERM_FORECAST = 'long_term_forecast'
    SHORT_TERM_FORECAST = 'short_term_forecast'
    ANOMALY_DETECTION = 'anomaly_detection'
    IMPUTATION = 'imputation'
    CLASSIFICATION = 'classification'


class FeatureType(Enum):
    """Enumeration of feature types for forecasting tasks."""
    MULTIVARIATE = 'M'  # Multivariate predict multivariate
    UNIVARIATE = 'S'    # Univariate predict univariate
    MIXED = 'MS'        # Multivariate predict univariate


class EmbeddingType(Enum):
    """Enumeration of time feature encoding methods."""
    TIME_FEATURES = 'timeF'
    FIXED = 'fixed'
    LEARNED = 'learned'


class ActivationType(Enum):
    """Enumeration of activation functions."""
    GELU = 'gelu'
    RELU = 'relu'
    TANH = 'tanh'
    SWISH = 'swish'


class LossType(Enum):
    """Enumeration of loss functions."""
    MSE = 'MSE'
    MAE = 'MAE'
    HUBER = 'Huber'
    SMOOTH_L1 = 'SmoothL1'


class LRAdjustmentType(Enum):
    """Enumeration of learning rate adjustment strategies."""
    TYPE1 = 'type1'  # StepLR with gamma=0.5
    TYPE2 = 'type2'  # CosineAnnealingLR
    NONE = 'none'    # No adjustment


@dataclass
class BaseConfig:
    """Base configuration class for all time series analysis tasks."""
    
    # Task configuration
    task_name: str = field(default='long_term_forecast', metadata={'description': 'Task type'})
    is_training: bool = field(default=True, metadata={'description': 'Training mode flag'})
    model: str = field(default='Autoformer', metadata={'description': 'Model name'})
    
    # Data configuration
    data: str = field(default='ETTh1', metadata={'description': 'Dataset name'})
    root_path: str = field(default='./data/ETT/', metadata={'description': 'Data root directory'})
    data_path: str = field(default='ETTh1.csv', metadata={'description': 'Data file name'})
    features: str = field(default='M', metadata={'description': 'Feature type (M/S/MS)'})
    target: str = field(default='OT', metadata={'description': 'Target feature for S/MS tasks'})
    freq: str = field(default='h', metadata={'description': 'Time frequency encoding'})
    checkpoints: str = field(default='./checkpoints/', metadata={'description': 'Checkpoint directory'})
    
    # Sequence configuration
    seq_len: int = field(default=96, metadata={'description': 'Input sequence length'})
    label_len: int = field(default=48, metadata={'description': 'Label sequence length'})
    pred_len: int = field(default=96, metadata={'description': 'Prediction sequence length'})
    seasonal_patterns: Optional[str] = field(default=None, metadata={'description': 'Seasonal patterns'})
    
    # Model architecture
    enc_in: int = field(default=7, metadata={'description': 'Encoder input dimension'})
    dec_in: int = field(default=7, metadata={'description': 'Decoder input dimension'})
    c_out: int = field(default=7, metadata={'description': 'Output dimension'})
    d_model: int = field(default=512, metadata={'description': 'Model dimension'})
    n_heads: int = field(default=8, metadata={'description': 'Number of attention heads'})
    e_layers: int = field(default=2, metadata={'description': 'Number of encoder layers'})
    d_layers: int = field(default=1, metadata={'description': 'Number of decoder layers'})
    d_ff: int = field(default=2048, metadata={'description': 'Feed-forward dimension'})
    moving_avg: int = field(default=25, metadata={'description': 'Moving average window size'})
    factor: int = field(default=1, metadata={'description': 'Attention factor'})
    distil: bool = field(default=True, metadata={'description': 'Use distillation'})
    dropout: float = field(default=0.1, metadata={'description': 'Dropout rate'})
    embed: str = field(default='timeF', metadata={'description': 'Time embedding method'})
    activation: str = field(default='gelu', metadata={'description': 'Activation function'})
    output_attention: bool = field(default=False, metadata={'description': 'Output attention weights'})
    
    # Training configuration
    num_workers: int = field(default=10, metadata={'description': 'Data loader workers'})
    itr: int = field(default=1, metadata={'description': 'Number of iterations'})
    train_epochs: int = field(default=10, metadata={'description': 'Training epochs'})
    batch_size: int = field(default=32, metadata={'description': 'Batch size'})
    patience: int = field(default=3, metadata={'description': 'Early stopping patience'})
    learning_rate: float = field(default=0.0001, metadata={'description': 'Learning rate'})
    des: str = field(default='test', metadata={'description': 'Experiment description'})
    loss: str = field(default='MSE', metadata={'description': 'Loss function'})
    lradj: str = field(default='type1', metadata={'description': 'LR adjustment strategy'})
    use_amp: bool = field(default=False, metadata={'description': 'Use mixed precision'})
    
    # Hardware configuration
    use_gpu: bool = field(default=True, metadata={'description': 'Use GPU'})
    gpu: int = field(default=0, metadata={'description': 'GPU device ID'})
    gpu_type: str = field(default='cuda', metadata={'description': 'GPU type'})
    use_multi_gpu: bool = field(default=False, metadata={'description': 'Use multiple GPUs'})
    devices: str = field(default='0,1,2,3', metadata={'description': 'Multi-GPU device IDs'})
    seed: int = field(default=2021, metadata={'description': 'Random seed'})
    
    # Additional options
    use_dtw: bool = field(default=False, metadata={'description': 'Use DTW'})
    use_tqdm: bool = field(default=True, metadata={'description': 'Use progress bars'})
    use_wandb: bool = field(default=False, metadata={'description': 'Use Weights & Biases'})
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate task name
        if self.task_name not in [task.value for task in TaskType]:
            logger.warning(f"Unknown task name: {self.task_name}")
        
        # Validate feature type
        if self.features not in [ft.value for ft in FeatureType]:
            logger.warning(f"Unknown feature type: {self.features}")
        
        # Validate embedding type
        if self.embed not in [et.value for et in EmbeddingType]:
            logger.warning(f"Unknown embedding type: {self.embed}")
        
        # Validate activation type
        if self.activation not in [at.value for at in ActivationType]:
            logger.warning(f"Unknown activation type: {self.activation}")
        
        # Validate loss type
        if self.loss not in [lt.value for lt in LossType]:
            logger.warning(f"Unknown loss type: {self.loss}")
        
        # Validate learning rate adjustment
        if self.lradj not in [lrt.value for lrt in LRAdjustmentType]:
            logger.warning(f"Unknown LR adjustment type: {self.lradj}")
        
        # Validate numeric parameters
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        
        # Validate device configuration
        if self.use_multi_gpu and not self.use_gpu:
            raise ValueError("Multi-GPU requires GPU usage")
        
        logger.debug("Configuration validation completed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return {
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out': self.c_out,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_layers': self.d_layers,
            'd_ff': self.d_ff,
            'moving_avg': self.moving_avg,
            'factor': self.factor,
            'distil': self.distil,
            'dropout': self.dropout,
            'embed': self.embed,
            'activation': self.activation,
            'output_attention': self.output_attention
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training-specific parameters."""
        return {
            'num_workers': self.num_workers,
            'train_epochs': self.train_epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'lradj': self.lradj,
            'use_amp': self.use_amp
        }
    
    def get_data_params(self) -> Dict[str, Any]:
        """Get data-specific parameters."""
        return {
            'data': self.data,
            'root_path': self.root_path,
            'data_path': self.data_path,
            'features': self.features,
            'target': self.target,
            'freq': self.freq,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'seasonal_patterns': self.seasonal_patterns
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'BaseConfig':
        """Create configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from: {yaml_path}")
            return cls.from_dict(config_dict)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file {yaml_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {yaml_path}: {e}")
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'BaseConfig':
        """Create configuration from JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            logger.info(f"Loaded configuration from: {json_path}")
            return cls.from_dict(config_dict)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {json_path}: {e}")
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {yaml_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving configuration to {yaml_path}: {e}")
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {json_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving configuration to {json_path}: {e}")
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after update
        self._validate_config()
    
    def copy(self) -> 'BaseConfig':
        """Create a copy of the configuration."""
        return self.__class__(**self.to_dict())


@dataclass
class AnomalyDetectionConfig(BaseConfig):
    """Configuration for anomaly detection tasks."""
    
    task_name: str = field(default='anomaly_detection', init=False)
    anomaly_ratio: float = field(default=0.25, metadata={'description': 'Expected anomaly ratio'})
    win_size: int = field(default=100, metadata={'description': 'Detection window size'})
    threshold_method: str = field(default='percentile', metadata={'description': 'Threshold method'})
    
    def __post_init__(self):
        """Validate anomaly detection specific parameters."""
        super().__post_init__()
        
        if self.anomaly_ratio < 0 or self.anomaly_ratio > 1:
            raise ValueError("anomaly_ratio must be between 0 and 1")
        
        if self.win_size <= 0:
            raise ValueError("win_size must be positive")


@dataclass
class ForecastingConfig(BaseConfig):
    """Configuration for forecasting tasks."""
    
    task_name: str = field(default='long_term_forecast', init=False)
    inverse: bool = field(default=False, metadata={'description': 'Inverse transform'})
    scale: bool = field(default=True, metadata={'description': 'Scale data'})
    time_encoding: bool = field(default=True, metadata={'description': 'Use time encoding'})
    
    def __post_init__(self):
        """Validate forecasting specific parameters."""
        super().__post_init__()
        
        if self.pred_len > self.seq_len:
            logger.warning("pred_len is greater than seq_len, this may cause issues")


@dataclass
class ImputationConfig(BaseConfig):
    """Configuration for imputation tasks."""
    
    task_name: str = field(default='imputation', init=False)
    mask_rate: float = field(default=0.25, metadata={'description': 'Masking rate'})
    mask_method: str = field(default='random', metadata={'description': 'Masking method'})
    
    def __post_init__(self):
        """Validate imputation specific parameters."""
        super().__post_init__()
        
        if self.mask_rate < 0 or self.mask_rate > 1:
            raise ValueError("mask_rate must be between 0 and 1")


@dataclass
class ClassificationConfig(BaseConfig):
    """Configuration for classification tasks."""
    
    task_name: str = field(default='classification', init=False)
    num_class: int = field(default=10, metadata={'description': 'Number of classes'})
    class_weights: Optional[List[float]] = field(default=None, metadata={'description': 'Class weights'})
    
    def __post_init__(self):
        """Validate classification specific parameters."""
        super().__post_init__()
        
        if self.num_class <= 1:
            raise ValueError("num_class must be greater than 1")
        
        if self.class_weights is not None and len(self.class_weights) != self.num_class:
            raise ValueError("class_weights length must match num_class")