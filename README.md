# TS-Unity: Time Series Unity Framework

A comprehensive and modular framework for time series analysis including forecasting, anomaly detection, imputation, and classification tasks. The framework provides a unified interface for various state-of-the-art models and easy experimentation.

## 🚀 Features

- **Multiple Tasks**: Supports long-term forecasting, short-term forecasting, anomaly detection, imputation, and classification
- **State-of-the-art Models**: Includes 40+ models like Autoformer, Transformer, TimesNet, PatchTST, and more
- **Modular Design**: Clean separation of concerns with base classes and interfaces
- **Configuration Management**: YAML/JSON based configuration system with validation
- **Comprehensive Logging**: Built-in experiment tracking and monitoring
- **Extensible Architecture**: Easy to add new models and tasks
- **Type Safety**: Full type hints and validation throughout the codebase
- **Error Handling**: Robust error handling and validation

## 📁 Framework Structure

```
src/
├── config/                 # Configuration management
│   ├── base_config.py      # Base configuration classes with validation
│   └── __init__.py
├── core/                   # Core framework components
│   ├── base_model.py       # Base model interfaces
│   ├── base_trainer.py     # Enhanced training pipeline with metrics
│   ├── pipeline.py         # Refactored main training pipeline
│   └── __init__.py
├── data_provider/          # Data loading and preprocessing
│   ├── data_factory.py     # Data factory with unified interface
│   ├── data_loader.py      # Dataset implementations
│   └── __init__.py
├── exp/                    # Experiment implementations
│   ├── exp_basic.py        # Base experiment class
│   ├── exp_long_term_forecasting.py
│   ├── exp_short_term_forecasting.py
│   ├── exp_anomaly_detection.py
│   ├── exp_imputation.py
│   ├── exp_classification.py
│   └── __init__.py
├── models/                 # Model implementations
│   ├── forecasting/        # Forecasting models
│   ├── anomaly_detection/  # Anomaly detection models
│   └── __init__.py
├── layers/                 # Neural network layers
├── utils/                  # Utility functions
│   ├── logger.py          # Comprehensive logging system
│   ├── metrics.py         # Evaluation metrics
│   ├── anomaly_detection_metrics.py  # Refactored metrics with classes
│   ├── tools.py           # Helper functions
│   └── __init__.py
└── main.py                # Enhanced entry point with error handling
```

## 🔧 Installation

```bash
git clone <repository-url>
cd TS-Unity
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### Using Command Line

```bash
# Long-term forecasting with Autoformer
python src/main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model Autoformer \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --train_epochs 10

# Anomaly detection with AnomalyTransformer
python src/main.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --model AnomalyTransformer \
    --data PSM \
    --seq_len 100 \
    --train_epochs 10
```

### Using Configuration Files

```bash
# Using YAML configuration
python src/main.py --config_file configs/example_forecasting.yaml

# Using JSON configuration
python src/main.py --config_file configs/example_anomaly.json
```

### Programmatic Usage

```python
from config.base_config import ForecastingConfig
from core.pipeline import TrainingPipeline

# Create configuration
config = ForecastingConfig(
    task_name='long_term_forecast',
    model='Autoformer',
    data='ETTh1',
    seq_len=96,
    pred_len=96,
    train_epochs=10
)

# Create and run pipeline
pipeline = TrainingPipeline(config, use_wandb=False)
results = pipeline.run_training()
```

## 🆕 Recent Improvements (v2.0)

### Code Quality & Structure
- **Complete Type Hints**: Added comprehensive type annotations throughout the codebase
- **Class-based Organization**: Refactored utility functions into logical classes
- **Enhanced Error Handling**: Improved exception handling and validation
- **Better Documentation**: Comprehensive docstrings and inline documentation

### Configuration Management
- **Validation**: Added parameter validation with meaningful error messages
- **Enums**: Introduced enums for better type safety (TaskType, FeatureType, etc.)
- **Metadata**: Added metadata fields for better documentation
- **Flexible Loading**: Enhanced YAML/JSON loading with error handling

### Training Pipeline
- **Task Registry**: Centralized task management with registry pattern
- **Metrics Tracking**: Enhanced training metrics and validation
- **Checkpoint Management**: Improved checkpoint saving/loading
- **Resource Management**: Better GPU memory management and cleanup

### Metrics & Evaluation
- **Organized Metrics**: Grouped related metrics into logical classes
- **Constants**: Replaced magic numbers with named constants
- **Backward Compatibility**: Maintained existing function interfaces
- **Enhanced Logging**: Better progress tracking and debugging

## 📊 Supported Tasks

### 1. Long-term Forecasting
- **Models**: Autoformer, Transformer, TimesNet, iTransformer, Koopa, TiDE, FreTS
- **Datasets**: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, ILI
- **Features**: Multivariate (M), Univariate (S), Mixed (MS)

### 2. Short-term Forecasting
- **Models**: Same as long-term forecasting
- **Use Case**: Short-horizon predictions (1-24 steps ahead)

### 3. Anomaly Detection
- **Models**: AnomalyTransformer, OmniAnomaly, USAD, DAGMM
- **Datasets**: PSM, MSL, SMAP, SMD, SWaT, WADI
- **Metrics**: F1-score, Precision, Recall, AUROC, AUPRC

### 4. Imputation
- **Models**: BRITS, SAITS, Transformer, TimesNet
- **Datasets**: Same as forecasting datasets
- **Features**: Random masking, structured masking

### 5. Classification
- **Models**: Transformer, TimesNet, TCN, ResNet
- **Datasets**: UCR, UEA, HAR, SleepEDF
- **Metrics**: Accuracy, F1-score, Precision, Recall

## 🛠️ Configuration

### Base Configuration
```python
@dataclass
class BaseConfig:
    # Task configuration
    task_name: str = 'long_term_forecast'
    is_training: bool = True
    model: str = 'Autoformer'
    
    # Data configuration
    data: str = 'ETTh1'
    seq_len: int = 96
    pred_len: int = 96
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    
    # Training configuration
    train_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0001
```

### Task-Specific Configurations
```python
@dataclass
class ForecastingConfig(BaseConfig):
    inverse: bool = False
    scale: bool = True
    time_encoding: bool = True

@dataclass
class AnomalyDetectionConfig(BaseConfig):
    anomaly_ratio: float = 0.25
    win_size: int = 100
    threshold_method: str = 'percentile'
```

## 🔍 Metrics & Evaluation

### Forecasting Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **MSPE**: Mean Squared Percentage Error

### Anomaly Detection Metrics
```python
from utils.anomaly_detection_metrics import (
    AnomalyMetrics, PointMetrics, ThresholdOptimization,
    SequenceMetrics, AdvancedMetrics
)

# Basic metrics
mae = AnomalyMetrics.mae(predictions, targets)
mse = AnomalyMetrics.mse(predictions, targets)

# Point-wise metrics
f1, precision, recall, tp, tn, fp, fn = PointMetrics.calc_point2point(
    predictions, targets
)

# Threshold optimization
best_metrics, best_threshold = ThresholdOptimization.bf_search(
    scores, labels, start=0.1, end=0.9, step_num=100
)
```

## 🚀 Advanced Usage

### Custom Model Integration
```python
from core.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your model implementation
        
    def forward(self, x):
        # Forward pass implementation
        pass
```

### Custom Trainer
```python
from core.base_trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        # Custom training logic
        pass
        
    def validate_epoch(self, epoch: int) -> ValidationMetrics:
        # Custom validation logic
        pass
```

### Custom Configuration
```python
@dataclass
class CustomConfig(BaseConfig):
    custom_param: str = 'default_value'
    
    def __post_init__(self):
        super().__post_init__()
        # Custom validation
        if self.custom_param == 'invalid':
            raise ValueError("Invalid custom_param value")
```

## 📝 Logging & Monitoring

### Built-in Logging
```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Use framework logger
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.warning("Low learning rate detected")
logger.error("Training failed")
```

### Weights & Biases Integration
```bash
python src/main.py --use_wandb --task_name long_term_forecast
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models/
python -m pytest tests/test_metrics/
python -m pytest tests/test_config/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and documentation
4. Add tests for new functionality
5. Submit a pull request

### Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling and validation
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original model implementations from various research papers
- PyTorch community for the excellent deep learning framework
- Contributors and users of the framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**TS-Unity v2.0** - A modern, type-safe, and well-structured time series analysis framework.
