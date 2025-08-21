"""
Training Pipeline Module

This module provides a comprehensive training pipeline for time series analysis tasks,
supporting multiple task types and models with configurable training parameters.
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional, Type, Union, List
from pathlib import Path
import argparse
import json
import logging
from dataclasses import dataclass
import time
from collections import deque

from config.base_config import (
    BaseConfig, ForecastingConfig, AnomalyDetectionConfig, 
    ImputationConfig, ClassificationConfig
)
from core.base_trainer import BaseTrainer, ForecastingTrainer
from data_provider.data_factory import DataFactory
from utils.logger import ExperimentLogger
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for different task types."""
    name: str
    exp_class: Type
    config_class: Type[BaseConfig]


class TaskRegistry:
    """Registry for available tasks and their configurations."""
    
    TASKS = {
        'long_term_forecast': TaskConfig(
            name='long_term_forecast',
            exp_class=Exp_Long_Term_Forecast,
            config_class=ForecastingConfig
        ),
        'short_term_forecast': TaskConfig(
            name='short_term_forecast',
            exp_class=Exp_Short_Term_Forecast,
            config_class=ForecastingConfig
        ),
        'anomaly_detection': TaskConfig(
            name='anomaly_detection',
            exp_class=Exp_Anomaly_Detection,
            config_class=AnomalyDetectionConfig
        ),
        'imputation': TaskConfig(
            name='imputation',
            exp_class=Exp_Imputation,
            config_class=ImputationConfig
        ),
        'classification': TaskConfig(
            name='classification',
            exp_class=Exp_Classification,
            config_class=ClassificationConfig
        )
    }
    
    @classmethod
    def get_task_config(cls, task_name: str) -> Optional[TaskConfig]:
        """Get task configuration by name."""
        return cls.TASKS.get(task_name)
    
    @classmethod
    def get_available_tasks(cls) -> list:
        """Get list of available task names."""
        return list(cls.TASKS.keys())
    
    @classmethod
    def get_config_class(cls, task_name: str) -> Optional[Type[BaseConfig]]:
        """Get configuration class for a task."""
        task_config = cls.get_task_config(task_name)
        return task_config.config_class if task_config else None


class InferenceData:
    """Container for inference input data."""
    data: np.ndarray  # Shape: (seq_len, num_features)
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class InferencePipeline:
    """Real-time inference pipeline for streaming time series data."""
    
    def __init__(self, config: BaseConfig, checkpoint_path: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.experiment = None
        self.model = None
        self.data_buffer = deque(maxlen=config.seq_len)
        self.is_initialized = False
        self.task_type = config.task_name
        
        self._setup_inference()
    
    def _setup_inference(self) -> None:
        """Setup the inference components."""
        logger.info(f"Setting up inference pipeline for task: {self.task_type}")
        
        # Create experiment
        self.experiment = self._create_experiment()
        
        # Load model if checkpoint provided
        if self.checkpoint_path:
            self._load_model()
        
        # Initialize data buffer
        self._initialize_buffer()
        
        self.is_initialized = True
        logger.info("Inference pipeline setup completed")
    
    def _create_experiment(self):
        """Create experiment instance for inference."""
        task_config = TaskRegistry.get_task_config(self.task_type)
        
        if task_config is None:
            raise ValueError(f"Unsupported task: {self.task_type}")
        
        return task_config.exp_class(self.config)
    
    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Load model state
        self.experiment.model.load_state_dict(checkpoint['model_state_dict'])
        self.experiment.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _initialize_buffer(self) -> None:
        """Initialize data buffer with zeros."""
        # Initialize with zeros or historical data if available
        buffer_shape = (self.config.seq_len, self.config.enc_in)
        self.data_buffer.extend([np.zeros(self.config.enc_in) for _ in range(self.config.seq_len)])
        
        logger.info(f"Data buffer initialized with shape: {buffer_shape}")
    
    def add_data_point(self, data_point: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Add a new data point to the inference buffer.
        
        Args:
            data_point: New data point with shape (num_features,)
            timestamp: Optional timestamp for the data point
        """
        if not self.is_initialized:
            raise RuntimeError("Inference pipeline not initialized")
        
        if data_point.shape[0] != self.config.enc_in:
            raise ValueError(f"Data point shape {data_point.shape} doesn't match expected {self.config.enc_in}")
        
        # Add to buffer
        self.data_buffer.append(data_point)
        
        logger.debug(f"Added data point: {data_point[:3]}... (timestamp: {timestamp})")
    
    def add_batch_data(self, data_batch: np.ndarray, timestamps: Optional[List[float]] = None) -> None:
        """
        Add multiple data points to the inference buffer.
        
        Args:
            data_batch: Batch of data points with shape (num_points, num_features)
            timestamps: Optional list of timestamps
        """
        if not self.is_initialized:
            raise RuntimeError("Inference pipeline not initialized")
        
        if data_batch.shape[1] != self.config.enc_in:
            raise ValueError(f"Data batch shape {data_batch.shape} doesn't match expected features {self.config.enc_in}")
        
        for i, data_point in enumerate(data_batch):
            timestamp = timestamps[i] if timestamps else None
            self.add_data_point(data_point, timestamp)
    
    def predict_next(self, num_steps: int = 1) -> np.ndarray:
        """
        Predict next values using current buffer data.
        
        Returns:
            Predictions with shape (num_steps, num_features) for forecasting
            or anomaly scores with shape (num_steps, num_features) for anomaly detection
        """
        if not self.is_initialized:
            raise RuntimeError("Inference pipeline not initialized")
        
        if len(self.data_buffer) < self.config.seq_len:
            raise ValueError(f"Insufficient data in buffer. Need {self.config.seq_len}, have {len(self.data_buffer)}")
        
        # Prepare input data
        input_data = self._prepare_input_data()
        
        # Make prediction based on task type
        if self.task_type in ['long_term_forecast', 'short_term_forecast']:
            return self._forecast_predict(input_data, num_steps)
        elif self.task_type == 'anomaly_detection':
            return self._anomaly_detect(input_data)
        elif self.task_type == 'imputation':
            return self._impute_data(input_data)
        elif self.task_type == 'classification':
            return self._classify_data(input_data)
        else:
            raise ValueError(f"Unsupported task type for prediction: {self.task_type}")
    
    def _forecast_predict(self, input_data: torch.Tensor, num_steps: int) -> np.ndarray:
        """Make forecasting predictions."""
        with torch.no_grad():
            predictions = self.experiment.predict_single(input_data, num_steps)
        
        logger.info(f"Generated forecasting predictions for {num_steps} steps ahead")
        return predictions
    
    def _anomaly_detect(self, input_data: torch.Tensor) -> np.ndarray:
        """Perform anomaly detection using appropriate method based on model type."""
        with torch.no_grad():
            # Determine detection method based on model type
            if self._is_reconstruction_model():
                return self._reconstruction_based_detection(input_data)
            else:
                return self._prediction_based_detection(input_data)
    
    def _is_reconstruction_model(self) -> bool:
        """Check if the current model is reconstruction-based."""
        reconstruction_models = [
            'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'DAGMM',
            'AutoEncoder', 'VAE', 'LSTM_VAE', 'LSTM_AE'
        ]
        return self.config.model in reconstruction_models
    
    def _reconstruction_based_detection(self, input_data: torch.Tensor) -> np.ndarray:
        """Perform reconstruction-based anomaly detection."""
        try:
            # Try to use model's specific anomaly detection method
            if hasattr(self.experiment, 'detect_anomaly'):
                anomaly_scores = self.experiment.detect_anomaly(input_data)
            elif hasattr(self.experiment.model, 'detect_anomaly'):
                anomaly_scores = self.experiment.model.detect_anomaly(input_data)
            else:
                # Fallback: calculate reconstruction error
                reconstructed = self.experiment.model(input_data)
                anomaly_scores = torch.mean(torch.abs(input_data - reconstructed), dim=-1)
            
            logger.info("Reconstruction-based anomaly detection completed")
            return anomaly_scores.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Reconstruction-based detection failed: {e}, using fallback method")
            # Fallback: simple reconstruction error
            try:
                reconstructed = self.experiment.model(input_data)
                anomaly_scores = torch.mean(torch.abs(input_data - reconstructed), dim=-1)
                return anomaly_scores.cpu().numpy()
            except Exception as e2:
                logger.error(f"Fallback reconstruction detection also failed: {e2}")
                # Return dummy scores
                return np.zeros((input_data.shape[0], 1))
    
    def _prediction_based_detection(self, input_data: torch.Tensor) -> np.ndarray:
        """Perform prediction-based anomaly detection using forecasting models."""
        try:
            # For prediction-based detection, we need to:
            # 1. Make predictions for the next few steps
            # 2. Compare predictions with actual values (if available)
            # 3. Calculate prediction error as anomaly score
            
            # Get prediction horizon (use config pred_len or default to 1)
            pred_len = getattr(self.config, 'pred_len', 1)
            
            # Make predictions
            if hasattr(self.experiment, 'predict_single'):
                predictions = self.experiment.predict_single(input_data, pred_len)
            else:
                # Fallback: use model directly
                predictions = self.experiment.model(input_data)
            
            # Calculate prediction error as anomaly score
            # For prediction-based detection, we typically use the variance of predictions
            # or the difference between predicted and actual values
            
            if predictions.dim() == 3:  # (batch, pred_len, features)
                # Use prediction variance across time steps as anomaly score
                anomaly_scores = torch.var(predictions, dim=1, keepdim=True)
            else:
                # Use prediction magnitude as anomaly score
                anomaly_scores = torch.mean(torch.abs(predictions), dim=-1, keepdim=True)
            
            logger.info("Prediction-based anomaly detection completed")
            return anomaly_scores.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Prediction-based detection failed: {e}, using fallback method")
            # Fallback: use input variance as anomaly score
            try:
                anomaly_scores = torch.var(input_data, dim=1, keepdim=True)
                return anomaly_scores.cpu().numpy()
            except Exception as e2:
                logger.error(f"Fallback prediction detection also failed: {e2}")
                # Return dummy scores
                return np.zeros((input_data.shape[0], 1))
    
    def _impute_data(self, input_data: torch.Tensor) -> np.ndarray:
        """Perform data imputation."""
        with torch.no_grad():
            if hasattr(self.experiment, 'impute'):
                imputed_data = self.experiment.impute(input_data)
            else:
                # Fallback: return original data
                imputed_data = input_data
        
        logger.info("Data imputation completed")
        return imputed_data.cpu().numpy()
    
    def _classify_data(self, input_data: torch.Tensor) -> np.ndarray:
        """Perform time series classification."""
        with torch.no_grad():
            if hasattr(self.experiment, 'classify'):
                classifications = self.experiment.classify(input_data)
            else:
                # Fallback: try to get model output
                try:
                    classifications = self.experiment.model(input_data)
                except Exception as e:
                    logger.warning(f"Could not perform classification: {e}")
                    # Return dummy classifications
                    classifications = torch.zeros(input_data.shape[0], self.config.num_class)
        
        logger.info("Time series classification completed")
        return classifications.cpu().numpy()
    
    def predict_streaming(self, new_data: np.ndarray, num_steps: int = 1) -> np.ndarray:
        """
        Stream new data and get predictions immediately.
        
        Args:
            new_data: New data point or batch
            num_steps: Number of steps to predict ahead (for forecasting)
            
        Returns:
            Predictions or anomaly scores based on task type
        """
        # Add new data
        if new_data.ndim == 1:
            self.add_data_point(new_data)
        else:
            self.add_batch_data(new_data)
        
        # Generate predictions
        return self.predict_next(num_steps)
    
    def predict_batch(self, input_data: np.ndarray, num_steps: int = 1) -> np.ndarray:
        """
        Predict from complete input time series data without modifying buffer.
        
        Args:
            input_data: Complete input time series with shape (seq_len, num_features) or (batch_size, seq_len, num_features)
            num_steps: Number of steps to predict ahead (for forecasting)
            
        Returns:
            Predictions or anomaly scores based on task type
        """
        if not self.is_initialized:
            raise RuntimeError("Inference pipeline not initialized")
        
        # Validate input shape
        if input_data.ndim == 2:
            # Single sequence: (seq_len, num_features) -> (1, seq_len, num_features)
            input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
        elif input_data.ndim == 3:
            # Batch of sequences: (batch_size, seq_len, num_features)
            pass
        else:
            raise ValueError(f"Input data must be 2D (seq_len, num_features) or 3D (batch_size, seq_len, num_features), got {input_data.ndim}D")
        
        # Validate sequence length
        if input_data.shape[1] != self.config.seq_len:
            raise ValueError(f"Input sequence length {input_data.shape[1]} doesn't match expected {self.config.seq_len}")
        
        # Validate feature count
        if input_data.shape[2] != self.config.enc_in:
            raise ValueError(f"Input feature count {input_data.shape[2]} doesn't match expected {self.config.enc_in}")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data)
        
        # Make prediction based on task type
        if self.task_type in ['long_term_forecast', 'short_term_forecast']:
            predictions = self._forecast_predict(input_tensor, num_steps)
        elif self.task_type == 'anomaly_detection':
            # For anomaly detection, num_steps is ignored and we use appropriate detection method
            predictions = self._anomaly_detect(input_tensor)
        elif self.task_type == 'imputation':
            predictions = self._impute_data(input_tensor)
        elif self.task_type == 'classification':
            predictions = self._classify_data(input_tensor)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        logger.info(f"Batch prediction completed for {input_data.shape[0]} sequences")
        return predictions
    
    def predict_from_file(self, file_path: str, num_steps: int = 1, 
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load time series data from file and make predictions.
        
        Args:
            file_path: Path to input data file (CSV, NPZ, etc.)
            num_steps: Number of steps to predict ahead (for forecasting)
            output_path: Optional path to save results
            
        Returns:
            Dictionary containing predictions and metadata
        """
        import pandas as pd
        
        logger.info(f"Loading data from: {file_path}")
        
        # Load data based on file extension
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            data = df.values
        elif file_ext == '.npy':
            data = np.load(file_path)
        elif file_ext == '.npz':
            npz_data = np.load(file_path)
            # Assume first array is the data
            data = npz_data[npz_data.files[0]]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Validate data dimensions
        if data.ndim == 1:
            # Single time series: reshape to (seq_len, 1)
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            # Multiple time series: (time_steps, features)
            pass
        else:
            raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D")
        
        # Check if we have enough data
        if data.shape[0] < self.config.seq_len:
            raise ValueError(f"Data length {data.shape[0]} is less than required sequence length {self.config.seq_len}")
        
        # Process data in sliding windows
        predictions = []
        input_windows = []
        
        for i in range(data.shape[0] - self.config.seq_len + 1):
            # Extract window
            window = data[i:i + self.config.seq_len]
            input_windows.append(window)
            
            # Make prediction based on task type
            if self.task_type in ['long_term_forecast', 'short_term_forecast']:
                pred = self.predict_batch(window, num_steps)
            else:
                # For non-forecasting tasks (anomaly detection, imputation, classification), 
                # num_steps is ignored and we use appropriate method
                pred = self.predict_batch(window, 1)
            predictions.append(pred)
        
        # Convert to numpy array
        predictions = np.array(predictions)
        input_windows = np.array(input_windows)
        
        logger.info(f"Generated predictions shape: {predictions.shape}")
        
        # Save results if output path provided
        results = {
            'predictions': predictions,
            'input_windows': input_windows,
            'num_windows': len(predictions),
            'num_steps': num_steps if self.task_type in ['long_term_forecast', 'short_term_forecast'] else 1,
            'input_shape': data.shape,
            'task_type': self.task_type,
            'detection_method': self._get_detection_method_info() if self.task_type == 'anomaly_detection' else None
        }
        
        if output_path:
            self._save_predictions(results, output_path)
            results['output_path'] = output_path
        
        return results
    
    def _get_detection_method_info(self) -> Dict[str, Any]:
        """Get information about the anomaly detection method being used."""
        if self.task_type != 'anomaly_detection':
            return None
        
        is_reconstruction = self._is_reconstruction_model()
        method_info = {
            'method': 'reconstruction_based' if is_reconstruction else 'prediction_based',
            'model_type': self.config.model,
            'description': 'Uses reconstruction error to detect anomalies' if is_reconstruction else 'Uses prediction variance to detect anomalies'
        }
        
        if is_reconstruction:
            method_info['approach'] = 'Compares input with reconstructed output'
        else:
            method_info['approach'] = 'Analyzes prediction patterns and variance'
        
        return method_info
    
    def _save_predictions(self, results: Dict[str, Any], output_path: str) -> None:
        """Save prediction results to file."""
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save predictions based on task type
        if output_path.suffix.lower() == '.csv':
            if self.task_type == 'anomaly_detection':
                # For anomaly detection, save scores
                pred_df = pd.DataFrame(results['predictions'].reshape(-1, results['predictions'].shape[-1]))
                pred_df.columns = [f'anomaly_score_{i}' for i in range(pred_df.shape[1])]
            elif self.task_type == 'classification':
                # For classification, save class probabilities
                pred_df = pd.DataFrame(results['predictions'].reshape(-1, results['predictions'].shape[-1]))
                pred_df.columns = [f'class_prob_{i}' for i in range(pred_df.shape[1])]
            else:
                # For forecasting, save predictions
                pred_df = pd.DataFrame(results['predictions'].reshape(-1, results['predictions'].shape[-1]))
                pred_df.columns = [f'pred_{i}' for i in range(pred_df.shape[1])]
            
            pred_df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.npy':
            np.save(output_path, results['predictions'])
        elif output_path.suffix.lower() == '.npz':
            np.savez_compressed(output_path, **results)
        else:
            # Default to numpy
            np.save(output_path, results['predictions'])
        
        logger.info(f"Results saved to: {output_path}")
    
    def predict_with_sliding_window(self, input_data: np.ndarray, 
                                   window_size: Optional[int] = None,
                                   stride: int = 1, num_steps: int = 1) -> Dict[str, Any]:
        """
        Make predictions using sliding window approach.
        
        Args:
            input_data: Input time series data (time_steps, features)
            window_size: Window size (defaults to config.seq_len)
            stride: Stride between windows
            num_steps: Number of steps to predict ahead (for forecasting)
            
        Returns:
            Dictionary containing predictions and window information
        """
        if window_size is None:
            window_size = self.config.seq_len
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        
        # Validate input
        if input_data.shape[0] < window_size:
            raise ValueError(f"Input length {input_data.shape[0]} is less than window size {window_size}")
        
        if input_data.shape[1] != self.config.enc_in:
            raise ValueError(f"Input features {input_data.shape[1]} don't match expected {self.config.enc_in}")
        
        # Generate windows
        windows = []
        window_indices = []
        
        for i in range(0, input_data.shape[0] - window_size + 1, stride):
            window = input_data[i:i + window_size]
            windows.append(window)
            window_indices.append((i, i + window_size))
        
        # Convert to numpy array
        windows = np.array(windows)
        
        # Make predictions for all windows based on task type
        if self.task_type in ['long_term_forecast', 'short_term_forecast']:
            predictions = self.predict_batch(windows, num_steps)
        else:
            # For non-forecasting tasks (anomaly detection, imputation, classification), 
            # num_steps is ignored and we use appropriate method
            predictions = self.predict_batch(windows, 1)
        
        results = {
            'predictions': predictions,
            'input_windows': windows,
            'window_indices': window_indices,
            'num_windows': len(windows),
            'window_size': window_size,
            'stride': stride,
            'num_steps': num_steps if self.task_type in ['long_term_forecast', 'short_term_forecast'] else 1,
            'task_type': self.task_type,
            'detection_method': self._get_detection_method_info() if self.task_type == 'anomaly_detection' else None
        }
        
        logger.info(f"Sliding window prediction completed: {len(windows)} windows")
        return results
    
    def _prepare_input_data(self) -> torch.Tensor:
        """Prepare input data for model inference."""
        # Convert buffer to numpy array
        buffer_array = np.array(list(self.data_buffer))
        
        # Reshape for model input: (batch_size=1, seq_len, num_features)
        input_data = buffer_array.reshape(1, self.config.seq_len, self.config.enc_in)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data)
        
        return input_tensor
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            'buffer_size': len(self.data_buffer),
            'required_size': self.config.seq_len,
            'is_ready': len(self.data_buffer) >= self.config.seq_len,
            'buffer_shape': (len(self.data_buffer), self.config.enc_in) if self.data_buffer else None
        }
    
    def reset_buffer(self) -> None:
        """Reset the data buffer."""
        self.data_buffer.clear()
        self._initialize_buffer()
        logger.info("Data buffer reset")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.experiment:
            del self.experiment
        torch.cuda.empty_cache()
        logger.info("Inference pipeline closed")


class RealTimeInferenceManager:
    """Manager for real-time inference operations."""
    
    def __init__(self, config: BaseConfig, checkpoint_path: str):
        """
        Initialize the real-time inference manager.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.inference_pipeline = None
        self.is_running = False
        
        self._setup_manager()
    
    def _setup_manager(self) -> None:
        """Setup the inference manager."""
        logger.info("Setting up real-time inference manager...")
        
        self.inference_pipeline = InferencePipeline(self.config, self.checkpoint_path)
        
        logger.info("Real-time inference manager setup completed")
    
    def start_streaming(self) -> None:
        """Start the streaming inference process."""
        if self.is_running:
            logger.warning("Streaming already running")
            return
        
        self.is_running = True
        logger.info("Started real-time streaming inference")
    
    def stop_streaming(self) -> None:
        """Stop the streaming inference process."""
        if not self.is_running:
            logger.warning("Streaming not running")
            return
        
        self.is_running = False
        logger.info("Stopped real-time streaming inference")
    
    def process_streaming_data(self, data_stream: np.ndarray, 
                             timestamps: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Process streaming data and return predictions.
        
        Args:
            data_stream: Stream of data points with shape (num_points, num_features)
            timestamps: Optional timestamps for each data point
            
        Returns:
            List of predictions for each data point
        """
        if not self.is_running:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
        
        predictions = []
        
        for i, data_point in enumerate(data_stream):
            timestamp = timestamps[i] if timestamps else None
            
            # Add data and get prediction
            pred = self.inference_pipeline.predict_streaming(data_point, num_steps=1)
            predictions.append(pred)
            
            logger.debug(f"Processed data point {i+1}/{len(data_stream)}")
        
        return predictions
    
    def get_realtime_predictions(self, data_point: np.ndarray, 
                                num_steps: int = 1) -> np.ndarray:
        """
        Get real-time predictions for a single data point.
        
        Args:
            data_point: Single data point with shape (num_features,)
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions
        """
        return self.inference_pipeline.predict_streaming(data_point, num_steps)
    
    def predict_batch_data(self, batch_data: np.ndarray, num_steps: int = 1) -> np.ndarray:
        """
        Make predictions for a batch of time series data.
        
        Args:
            batch_data: Batch of time series data with shape (batch_size, seq_len, num_features)
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions for all sequences
        """
        return self.inference_pipeline.predict_batch(batch_data, num_steps)
    
    def predict_from_file(self, file_path: str, num_steps: int = 1, 
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data from file and make predictions.
        
        Args:
            file_path: Path to input data file
            num_steps: Number of steps to predict ahead
            output_path: Optional path to save results
            
        Returns:
            Dictionary containing predictions and metadata
        """
        return self.inference_pipeline.predict_from_file(file_path, num_steps, output_path)
    
    def predict_with_sliding_window(self, input_data: np.ndarray, 
                                  window_size: Optional[int] = None,
                                  stride: int = 1, num_steps: int = 1) -> Dict[str, Any]:
        """
        Make predictions using sliding window approach.
        
        Args:
            input_data: Input time series data
            window_size: Window size for sliding
            stride: Stride between windows
            num_steps: Number of steps to predict ahead
            
        Returns:
            Dictionary containing predictions and window information
        """
        return self.inference_pipeline.predict_with_sliding_window(
            input_data, window_size, stride, num_steps
        )
    
    def close(self) -> None:
        """Clean up resources."""
        if self.inference_pipeline:
            self.inference_pipeline.close()
        logger.info("Real-time inference manager closed")


class TrainingPipeline:
    """Main training pipeline for time series analysis tasks."""
    
    def __init__(self, config: BaseConfig, use_wandb: bool = False):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.use_wandb = use_wandb
        self.logger: Optional[ExperimentLogger] = None
        
        self._setup_pipeline()
    
    def _setup_pipeline(self) -> None:
        """Setup the pipeline components."""
        self._set_seed(self.config.seed)
        self._setup_experiment()
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        logger.info(f"Random seed set to: {seed}")
    
    def _setup_experiment(self) -> None:
        """Setup experiment logging and configuration."""
        experiment_name = self._generate_experiment_name()
        
        self.logger = ExperimentLogger(
            name=experiment_name,
            log_dir=f"{self.config.checkpoints}/logs",
            use_wandb=self.use_wandb,
            wandb_project="ts-unity",
            wandb_config=self.config.to_dict()
        )
        
        self.logger.log_config(self.config.to_dict())
        self.logger.log_system_info()
        
        logger.info(f"Experiment setup complete: {experiment_name}")
    
    def _generate_experiment_name(self) -> str:
        """Generate experiment name from configuration."""
        return f"{self.config.model}_{self.config.task_name}_{self.config.data}_{self.config.des}"
    
    def create_experiment(self):
        """Create experiment instance based on task configuration."""
        task_config = TaskRegistry.get_task_config(self.config.task_name)
        
        if task_config is None:
            raise ValueError(f"Unsupported task: {self.config.task_name}. "
                           f"Available tasks: {TaskRegistry.get_available_tasks()}")
        
        logger.info(f"Creating experiment for task: {self.config.task_name}")
        return task_config.exp_class(self.config)
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the training process.
        
        Returns:
            Dictionary containing training results
        """
        exp = self.create_experiment()
        self.logger.log_model_info(exp.model)
        
        if not self.config.is_training:
            logger.info("Training mode disabled, running test only")
            return self._run_test(exp)
        
        results = {}
        logger.info(f"Starting training for {self.config.itr} iterations")
        
        for iteration in range(self.config.itr):
            logger.info(f"Training iteration {iteration + 1}/{self.config.itr}")
            
            # Training phase
            exp.train()
            
            # Testing phase
            test_results = self._run_test(exp)
            results[f'iteration_{iteration + 1}'] = test_results
            
            # Cleanup
            torch.cuda.empty_cache()
        
        # Log final results
        self.logger.log_metrics(test_results, prefix="final_test_")
        logger.info("Training completed successfully")
        
        return results
    
    def _run_test(self, exp) -> Dict[str, Any]:
        """Run testing phase."""
        logger.info("Running test phase")
        test_results = exp.test()
        self.logger.log_metrics(test_results, prefix="test_")
        return test_results
    
    def run_prediction(self) -> Dict[str, Any]:
        """
        Run prediction phase.
        
        Returns:
            Dictionary containing prediction results
        """
        exp = self.create_experiment()
        
        logger.info("Running prediction phase")
        pred_results = exp.predict()
        
        return pred_results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.logger:
            self.logger.close()
        logger.info("Pipeline closed")


class ConfigManager:
    """Manages configuration creation and validation."""
    
    @staticmethod
    def create_config_from_args(args: argparse.Namespace) -> BaseConfig:
        """Create configuration object from command line arguments."""
        config_class = TaskRegistry.get_config_class(args.task_name)
        
        if config_class is None:
            raise ValueError(f"Invalid task name: {args.task_name}")
        
        # Filter out None values
        config_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        return config_class(**config_dict)
    
    @staticmethod
    def setup_args() -> argparse.Namespace:
        """Setup command line argument parser."""
        parser = argparse.ArgumentParser(description='Time Series Analysis Pipeline')
        
        # Task configuration
        parser.add_argument(
            '--task_name', type=str, required=True, default='long_term_forecast',
            choices=TaskRegistry.get_available_tasks(),
            help='Task name for time series analysis'
        )
        parser.add_argument(
            '--is_training', type=int, required=True, default=1,
            help='Training mode (1: training, 0: testing only)'
        )
        
        parser.add_argument(
            '--is_inference', type=int, required=True, default=0,
            help='Inference mode (1: inference, 0: training or testing)'
        )
        
        # Model configuration
        parser.add_argument(
            '--model', type=str, required=True, default='Autoformer',
            help='Model name for time series analysis'
        )
        
        # Data configuration
        parser.add_argument(
            '--data', type=str, required=True, default='ETTh1',
            help='Dataset type'
        )
        parser.add_argument(
            '--root_path', type=str, default='./data/ETT/',
            help='Root path of the data file'
        )
        parser.add_argument(
            '--data_path', type=str, default='ETTh1.csv',
            help='Data file name'
        )
        parser.add_argument(
            '--features', type=str, default='M',
            choices=['M', 'S', 'MS'],
            help='Forecasting task type: M (multivariate), S (univariate), MS (multivariate->univariate)'
        )
        parser.add_argument(
            '--target', type=str, default='OT',
            help='Target feature for S or MS tasks'
        )
        parser.add_argument(
            '--freq', type=str, default='h',
            help='Frequency for time features encoding'
        )
        parser.add_argument(
            '--checkpoints', type=str, default='./checkpoints/',
            help='Location of model checkpoints'
        )
        
        # Sequence configuration
        parser.add_argument(
            '--seq_len', type=int, default=96,
            help='Input sequence length'
        )
        parser.add_argument(
            '--label_len', type=int, default=48,
            help='Start token length'
        )
        parser.add_argument(
            '--pred_len', type=int, default=96,
            help='Prediction sequence length'
        )
        parser.add_argument(
            '--seasonal_patterns', type=str, default='Monthly',
            help='Seasonal patterns for M4 dataset'
        )
        
        # Model architecture
        parser.add_argument(
            '--enc_in', type=int, default=7,
            help='Encoder input size'
        )
        parser.add_argument(
            '--dec_in', type=int, default=7,
            help='Decoder input size'
        )
        parser.add_argument(
            '--c_out', type=int, default=7,
            help='Output size'
        )
        parser.add_argument(
            '--d_model', type=int, default=512,
            help='Dimension of model'
        )
        parser.add_argument(
            '--n_heads', type=int, default=8,
            help='Number of attention heads'
        )
        parser.add_argument(
            '--e_layers', type=int, default=2,
            help='Number of encoder layers'
        )
        parser.add_argument(
            '--d_layers', type=int, default=1,
            help='Number of decoder layers'
        )
        parser.add_argument(
            '--d_ff', type=int, default=2048,
            help='Dimension of feed-forward network'
        )
        parser.add_argument(
            '--moving_avg', type=int, default=25,
            help='Window size of moving average'
        )
        parser.add_argument(
            '--factor', type=int, default=1,
            help='Attention factor'
        )
        parser.add_argument(
            '--distil', action='store_false', default=True,
            help='Whether to use distilling in encoder'
        )
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='Dropout rate'
        )
        parser.add_argument(
            '--embed', type=str, default='timeF',
            choices=['timeF', 'fixed', 'learned'],
            help='Time features encoding method'
        )
        parser.add_argument(
            '--activation', type=str, default='gelu',
            help='Activation function'
        )
        parser.add_argument(
            '--output_attention', action='store_true',
            help='Whether to output attention in encoder'
        )
        
        # Training configuration
        parser.add_argument(
            '--num_workers', type=int, default=10,
            help='Data loader number of workers'
        )
        parser.add_argument(
            '--itr', type=int, default=1,
            help='Number of experiment iterations'
        )
        parser.add_argument(
            '--train_epochs', type=int, default=10,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch_size', type=int, default=32,
            help='Training batch size'
        )
        parser.add_argument(
            '--patience', type=int, default=3,
            help='Early stopping patience'
        )
        parser.add_argument(
            '--learning_rate', type=float, default=0.0001,
            help='Learning rate'
        )
        parser.add_argument(
            '--des', type=str, default='test',
            help='Experiment description'
        )
        parser.add_argument(
            '--loss', type=str, default='MSE',
            help='Loss function'
        )
        parser.add_argument(
            '--lradj', type=str, default='type1',
            help='Learning rate adjustment strategy'
        )
        parser.add_argument(
            '--use_amp', action='store_true', default=False,
            help='Use automatic mixed precision training'
        )
        
        # Hardware configuration
        parser.add_argument(
            '--use_gpu', type=bool, default=True,
            help='Whether to use GPU'
        )
        parser.add_argument(
            '--gpu', type=int, default=0,
            help='GPU device ID'
        )
        parser.add_argument(
            '--gpu_type', type=str, default='cuda',
            help='GPU type (cuda, mps)'
        )
        parser.add_argument(
            '--use_multi_gpu', action='store_true', default=False,
            help='Use multiple GPUs'
        )
        parser.add_argument(
            '--devices', type=str, default='0,1,2,3',
            help='Device IDs for multi-GPU setup'
        )
        parser.add_argument(
            '--seed', type=int, default=2021,
            help='Random seed'
        )
        
        # Additional options
        parser.add_argument(
            '--use_dtw', type=bool, default=False,
            help='Use DTW or not'
        )
        parser.add_argument(
            '--use_tqdm', type=bool, default=True,
            help='Use tqdm progress bar'
        )
        parser.add_argument(
            '--use_wandb', action='store_true', default=False,
            help='Use Weights & Biases for logging'
        )
        parser.add_argument(
            '--config_file', type=str,
            help='Path to configuration file'
        )
        
        # Inference specific arguments
        parser.add_argument(
            '--checkpoint_path', type=str,
            help='Path to model checkpoint for inference'
        )
        parser.add_argument(
            '--input_data', type=str,
            help='Path to input data file for inference (CSV format)'
        )
        parser.add_argument(
            '--output_path', type=str,
            help='Path to save inference results'
        )
        
        args = parser.parse_args()
        
        # Handle configuration file
        if args.config_file:
            config = BaseConfig.from_yaml(args.config_file)
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None:
                    setattr(config, key, value)
            return config
        
        return ConfigManager.create_config_from_args(args)


def main() -> None:
    """Main entry point for the training pipeline."""
    try:
        args = ConfigManager.setup_args()
        
        if args.is_inference:
            # Inference mode
            if not args.checkpoint_path:
                raise ValueError("Checkpoint path is required for inference mode")
            
            logger.info("Starting inference mode...")
            
            # Create inference pipeline
            inference_pipeline = InferencePipeline(args, args.checkpoint_path)
            
            if args.input_data:
                # Batch inference from file
                results = run_batch_inference(inference_pipeline, args.input_data, args.output_path)
                logger.info(f"Batch inference completed. Results saved to: {args.output_path}")
            else:
                # Interactive inference mode
                run_interactive_inference(inference_pipeline)
                
            inference_pipeline.close()
            
        elif args.is_training:
            # Training mode
            pipeline = TrainingPipeline(args, use_wandb=args.use_wandb)
            results = pipeline.run_training()
            logger.info(f"Training completed successfully. Results: {results}")
            pipeline.close()
        else:
            # Prediction mode
            pipeline = TrainingPipeline(args, use_wandb=args.use_wandb)
            results = pipeline.run_prediction()
            logger.info(f"Prediction completed successfully. Results: {results}")
            pipeline.close()
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


def run_batch_inference(inference_pipeline: InferencePipeline, 
                       input_file: str, output_file: str) -> Dict[str, Any]:
    """Run batch inference on data from file."""
    import pandas as pd
    
    # Load input data
    df = pd.read_csv(input_file)
    data = df.values
    
    # Run inference
    predictions = []
    for i in range(len(data)):
        pred = inference_pipeline.predict_streaming(data[i], num_steps=1)
        predictions.append(pred.flatten())
    
    # Save results
    results_df = pd.DataFrame(predictions, columns=[f'pred_{i}' for i in range(predictions[0].shape[0])])
    results_df.to_csv(output_file, index=False)
    
    return {'predictions': predictions, 'output_file': output_file}


def run_interactive_inference(inference_pipeline: InferencePipeline) -> None:
    """Run interactive inference mode for real-time data input."""
    logger.info("Interactive inference mode started. Enter data points (comma-separated values):")
    logger.info(f"Expected format: {inference_pipeline.config.enc_in} features")
    logger.info("Type 'quit' to exit, 'status' to check buffer status")
    
    try:
        while True:
            user_input = input("Enter data point: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                status = inference_pipeline.get_buffer_status()
                logger.info(f"Buffer status: {status}")
                continue
            elif user_input.lower() == 'reset':
                inference_pipeline.reset_buffer()
                logger.info("Buffer reset")
                continue
            
            try:
                # Parse input data
                data_point = np.array([float(x.strip()) for x in user_input.split(',')])
                
                if data_point.shape[0] != inference_pipeline.config.enc_in:
                    logger.error(f"Expected {inference_pipeline.config.enc_in} features, got {data_point.shape[0]}")
                    continue
                
                # Get prediction
                prediction = inference_pipeline.predict_streaming(data_point, num_steps=1)
                logger.info(f"Prediction: {prediction.flatten()}")
                
            except ValueError as e:
                logger.error(f"Invalid input format: {e}")
                logger.info("Please enter comma-separated numeric values")
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                
    except KeyboardInterrupt:
        logger.info("Interactive inference stopped by user")
    
    logger.info("Interactive inference mode ended")


if __name__ == '__main__':
    main()