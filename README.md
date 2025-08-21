# TS-Unity: Time Series Unified Framework

A comprehensive framework for time series analysis including forecasting, anomaly detection, imputation, and classification.

## Features

- **Multi-Task Support**: Forecasting, Anomaly Detection, Imputation, Classification
- **Multiple Models**: Autoformer, Transformer, Informer, and more
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust error handling and validation
- **Real-Time Inference**: Streaming inference with sliding window support
- **Batch Inference**: Process entire datasets efficiently
- **REST API**: HTTP endpoints for easy integration
- **Modular Design**: Clean, maintainable architecture

## Framework Structure

```
src/
├── config/
│   └── base_config.py          # Enhanced configuration management
├── core/
│   ├── base_trainer.py         # Enhanced base trainer
│   └── pipeline.py             # Multi-task inference pipeline
├── models/                     # Model implementations
├── utils/
│   └── anomaly_detection_metrics.py  # Refactored metrics
├── api/
│   └── inference_server.py     # FastAPI inference server
├── scripts/
│   ├── realtime_inference.py   # Real-time inference demo
│   ├── batch_inference.py      # Batch inference demo
│   ├── anomaly_detection_inference.py  # Anomaly detection demo
│   └── api_client_example.py   # API client example
└── main.py                     # Enhanced main entry point
```

## Installation

```bash
git clone https://github.com/your-username/TS-Unity.git
cd TS-Unity
pip install -r requirements.txt
```

## REST API Server

TS-Unity now provides a REST API server for easy integration with other systems and applications.

### Starting the API Server

```bash
# Start the server
python src/api/inference_server.py --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python src/api/inference_server.py --reload
```

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Load Model
```bash
POST /load_model
{
    "task_type": "anomaly_detection",
    "checkpoint_path": "/path/to/checkpoint.pth",
    "config_overrides": {
        "seq_len": 100,
        "enc_in": 7
    }
}
```

#### Forecasting Inference
```bash
POST /forecast
{
    "task_type": "forecasting",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...],
    "num_steps": 24,
    "window_size": 100,
    "stride": 10
}
```

#### Anomaly Detection
```bash
POST /detect_anomalies
{
    "task_type": "anomaly_detection",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...],
    "threshold": 0.8,
    "window_size": 100,
    "stride": 10
}
```

#### Generic Inference
```bash
POST /inference
{
    "task_type": "imputation",
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...],
    "window_size": 100,
    "stride": 10
}
```

#### File-based Inference
```bash
POST /inference/file?file_path=data.csv&task_type=forecasting&num_steps=24
```

#### Model Information
```bash
GET /model/info
```

#### Unload Model
```bash
DELETE /model
```

### API Client Example

```python
from src.scripts.api_client_example import TSUnityAPIClient

# Initialize client
client = TSUnityAPIClient("http://localhost:8000")

# Load model
client.load_model("anomaly_detection", "/path/to/checkpoint.pth")

# Run anomaly detection
result = client.run_anomaly_detection(
    data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...],
    threshold=0.8
)

print(f"Anomalies detected: {result['anomalies_detected']}")
print(f"Anomaly rate: {result['anomaly_rate']:.2f}%")
```

### Interactive API Documentation

Once the server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Real-Time Inference

TS-Unity supports real-time streaming inference for live time series data.

### Python API Usage

```python
from src.core.pipeline import InferencePipeline, RealTimeInferenceManager
from src.config.base_config import ForecastingConfig

# Create configuration
config = ForecastingConfig(
    task_name='short_term_forecast',
    model='Autoformer',
    seq_len=100,
    enc_in=7,
    dec_in=7,
    c_out=7,
    pred_len=24
)

# Initialize inference pipeline
pipeline = InferencePipeline(config, checkpoint_path='model.pth')

# Add real-time data points
pipeline.add_data_point([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

# Get predictions
predictions = pipeline.predict_next(num_steps=12)
```

### Command Line Usage

```bash
# Real-time inference demo
python src/scripts/realtime_inference.py --checkpoint_path model.pth --mode interactive

# Simulate streaming data
python src/scripts/realtime_inference.py --checkpoint_path model.pth --mode simulation

# Stream from CSV file
python src/scripts/realtime_inference.py --checkpoint_path model.pth --mode csv --input_file data.csv
```

## Batch Inference

TS-Unity supports efficient batch inference for processing large datasets.

### Python API Usage

```python
# Single sequence inference
predictions = pipeline.predict_batch(sequence_data, num_steps=24)

# Multiple sequences
batch_predictions = pipeline.predict_batch(batch_data, num_steps=24)

# Sliding window on long series
results = pipeline.predict_with_sliding_window(
    long_series, 
    window_size=100, 
    stride=10, 
    num_steps=24
)

# File-based inference
results = pipeline.predict_from_file(
    'data.csv', 
    num_steps=24, 
    output_path='predictions.csv'
)
```

### Command Line Usage

```bash
# Single sequence inference
python src/scripts/batch_inference.py --checkpoint_path model.pth --mode single

# Batch inference
python src/scripts/batch_inference.py --checkpoint_path model.pth --mode batch --batch_size 10

# Sliding window inference
python src/scripts/batch_inference.py --checkpoint_path model.pth --mode sliding --data_length 1000

# File-based inference
python src/scripts/batch_inference.py --checkpoint_path model.pth --mode file --input_file data.csv

# Performance benchmark
python src/scripts/batch_inference.py --checkpoint_path model.pth --mode benchmark
```

## Anomaly Detection

TS-Unity now supports anomaly detection inference with dedicated methods and analysis.

### Anomaly Detection Approaches

TS-Unity supports two different approaches to anomaly detection:

#### 1. **Reconstruction-based Anomaly Detection**
- **Models**: AnomalyTransformer, OmniAnomaly, USAD, DAGMM, AutoEncoder, VAE
- **Method**: Compares input with reconstructed output
- **Anomaly Score**: Reconstruction error (higher = more anomalous)
- **Best for**: Detecting structural anomalies, pattern changes, distribution shifts
- **How it works**: The model learns to reconstruct normal patterns, and anomalies have higher reconstruction errors

#### 2. **Prediction-based Anomaly Detection**
- **Models**: Autoformer, Transformer, TimesNet, and other forecasting models
- **Method**: Analyzes prediction patterns and variance
- **Anomaly Score**: Prediction variance or error (higher = more anomalous)
- **Best for**: Detecting temporal anomalies, trend changes, unexpected patterns
- **How it works**: Uses forecasting models to predict future values and detects anomalies based on prediction confidence

### Automatic Method Selection

The system automatically selects the appropriate detection method based on the model type:

```python
# Reconstruction-based detection (e.g., AnomalyTransformer)
config = AnomalyDetectionConfig(
    task_name='anomaly_detection',
    model='AnomalyTransformer',  # Reconstruction model
    seq_len=100,
    enc_in=7
)

# Prediction-based detection (e.g., Autoformer)
config = ForecastingConfig(
    task_name='anomaly_detection',
    model='Autoformer',  # Forecasting model
    seq_len=100,
    enc_in=7
)
```

### Enhanced Experiment Classes

The experiment classes now support both approaches:

#### **Exp_Anomaly_Detection**
- **Reconstruction-based**: Uses `_reconstruction_based_scoring()` for models like AnomalyTransformer
- **Prediction-based**: Uses `_prediction_based_scoring()` for forecasting models
- **Automatic selection**: Detects model type and applies appropriate method
- **Method information**: Saves detection method details in results

#### **Exp_Long_Term_Forecast & Exp_Short_Term_Forecast**
- **Forecasting**: Primary functionality for time series prediction
- **Anomaly detection**: Secondary functionality using prediction-based approach
- **Dual purpose**: Can be used for both forecasting and anomaly detection
- **Flexible input**: Handles various model architectures (Linear, TST, Transformer-based)

### Usage Examples

```python
# Using forecasting model for anomaly detection
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

exp = Exp_Long_Term_Forecast(args)
exp.model.load_state_dict(torch.load('forecasting_model.pth'))

# Anomaly detection using prediction-based method
anomaly_scores = exp.detect_anomaly(input_data)

# Regular forecasting
predictions = exp.predict_single(input_data, num_steps=24)
```

### Python API Usage

```python
from src.config.base_config import AnomalyDetectionConfig

# Create anomaly detection configuration
config = AnomalyDetectionConfig(
    task_name='anomaly_detection',
    model='AnomalyTransformer',
    seq_len=100,
    enc_in=7,
    dec_in=7,
    c_out=7,
    anomaly_ratio=0.1,
    win_size=100
)

# Initialize inference pipeline
pipeline = InferencePipeline(config, checkpoint_path='anomaly_model.pth')

# Run anomaly detection
anomaly_scores = pipeline.predict_batch(sequence_data, num_steps=1)

# Analyze results
print(f"Anomaly scores shape: {anomaly_scores.shape}")
```

### Command Line Usage

```bash
# Single sequence anomaly detection
python src/scripts/anomaly_detection_inference.py --checkpoint_path anomaly_model.pth --mode single

# Batch anomaly detection
python src/scripts/anomaly_detection_inference.py --checkpoint_path anomaly_model.pth --mode batch --batch_size 10

# Sliding window anomaly detection
python src/scripts/anomaly_detection_inference.py --checkpoint_path anomaly_model.pth --mode sliding --data_length 1000

# File-based anomaly detection
python src/scripts/anomaly_detection_inference.py --checkpoint_path anomaly_model.pth --mode file --input_file data.csv

# Performance benchmark
python src/scripts/anomaly_detection_inference.py --checkpoint_path anomaly_model.pth --mode benchmark
```

## Data Format Requirements

### Input Data
- **Real-time**: Single data points as 1D arrays
- **Batch**: 2D arrays (time_steps, features) or 3D arrays (batch_size, time_steps, features)
- **File**: CSV, NPY, or NPZ files

### Output Data
- **Forecasting**: Predictions for specified number of steps ahead
- **Anomaly Detection**: Anomaly scores for each time step
- **Imputation**: Imputed values
- **Classification**: Class probabilities

## Performance Tips

### Real-Time Inference
- Use appropriate buffer size for your use case
- Consider batch processing for multiple data points
- Monitor memory usage with large buffers

### Batch Inference
- Use sliding window for very long time series
- Adjust stride based on your requirements
- Consider parallel processing for large batches

### API Usage
- Keep models loaded for multiple requests
- Use appropriate batch sizes for your data
- Monitor API response times

## Examples

See the `src/scripts/` directory for comprehensive examples:
- `realtime_inference.py`: Real-time streaming examples
- `batch_inference.py`: Batch processing examples
- `anomaly_detection_inference.py`: Anomaly detection examples
- `api_client_example.py`: API usage examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**TS-Unity v2.0** - Enhanced with type safety, error handling, real-time inference, batch processing, and REST API support.
