#!/usr/bin/env python3
"""
TS-Unity Inference API Client Example

This script demonstrates how to use the TS-Unity inference API
for forecasting and anomaly detection.
"""

import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time


class TSUnityAPIClient:
    """Client for TS-Unity Inference API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, task_type: str, checkpoint_path: str, 
                   config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a model for inference.
        
        Args:
            task_type: Type of task (forecasting, anomaly_detection, etc.)
            checkpoint_path: Path to model checkpoint
            config_overrides: Optional configuration overrides
            
        Returns:
            Response from the API
        """
        payload = {
            "task_type": task_type,
            "checkpoint_path": checkpoint_path
        }
        
        if config_overrides:
            payload["config_overrides"] = config_overrides
        
        response = self.session.post(f"{self.base_url}/load_model", json=payload)
        response.raise_for_status()
        return response.json()
    
    def run_forecasting(self, data: List[List[float]], num_steps: int,
                       window_size: int = None, stride: int = 1) -> Dict[str, Any]:
        """
        Run forecasting inference.
        
        Args:
            data: Time series data as 2D array (time_steps, features)
            num_steps: Number of steps to predict ahead
            window_size: Optional window size for sliding window
            stride: Stride for sliding window
            
        Returns:
            Forecasting results
        """
        payload = {
            "task_type": "forecasting",
            "data": data,
            "num_steps": num_steps,
            "window_size": window_size,
            "stride": stride
        }
        
        response = self.session.post(f"{self.base_url}/forecast", json=payload)
        response.raise_for_status()
        return response.json()
    
    def run_anomaly_detection(self, data: List[List[float]], 
                             threshold: float = None,
                             window_size: int = None, stride: int = 1) -> Dict[str, Any]:
        """
        Run anomaly detection inference.
        
        Args:
            data: Time series data as 2D array (time_steps, features)
            threshold: Optional anomaly threshold
            window_size: Optional window size for sliding window
            stride: Stride for sliding window
            
        Returns:
            Anomaly detection results
        """
        payload = {
            "task_type": "anomaly_detection",
            "data": data,
            "threshold": threshold,
            "window_size": window_size,
            "stride": stride
        }
        
        response = self.session.post(f"{self.base_url}/detect_anomalies", json=payload)
        response.raise_for_status()
        return response.json()
    
    def run_generic_inference(self, task_type: str, data: List[List[float]],
                             num_steps: int = 1, window_size: int = None, 
                             stride: int = 1) -> Dict[str, Any]:
        """
        Run generic inference for any task type.
        
        Args:
            task_type: Type of task
            data: Time series data
            num_steps: Number of steps (for forecasting)
            window_size: Optional window size
            stride: Stride for sliding window
            
        Returns:
            Inference results
        """
        payload = {
            "task_type": task_type,
            "data": data,
            "num_steps": num_steps,
            "window_size": window_size,
            "stride": stride
        }
        
        response = self.session.post(f"{self.base_url}/inference", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model."""
        response = self.session.delete(f"{self.base_url}/model")
        response.raise_for_status()
        return response.json()
    
    def run_file_inference(self, file_path: str, task_type: str, 
                           num_steps: int = 1, output_path: str = None) -> Dict[str, Any]:
        """
        Run inference on data from file.
        
        Args:
            file_path: Path to input file
            task_type: Type of task
            num_steps: Number of steps (for forecasting)
            output_path: Optional output path
            
        Returns:
            File inference results
        """
        params = {
            "file_path": file_path,
            "task_type": task_type,
            "num_steps": num_steps
        }
        
        if output_path:
            params["output_path"] = output_path
        
        response = self.session.post(f"{self.base_url}/inference/file", params=params)
        response.raise_for_status()
        return response.json()


def generate_sample_data(seq_len: int = 100, num_features: int = 7) -> List[List[float]]:
    """Generate sample time series data."""
    # Create synthetic time series data
    time_steps = np.arange(seq_len)
    
    data = []
    for t in range(seq_len):
        row = []
        for f in range(num_features):
            if f == 0:
                # Sine wave
                value = np.sin(time_steps[t] * 0.1) * 10
            elif f == 1:
                # Linear trend with noise
                value = time_steps[t] * 0.5 + np.random.normal(0, 1)
            elif f == 2:
                # Exponential decay
                value = 20 * np.exp(-time_steps[t] * 0.02)
            elif f == 3:
                # Random walk
                value = np.cumsum(np.random.normal(0, 0.5))[t] if t > 0 else 0
            else:
                # Random noise
                value = np.random.normal(0, 2)
            
            row.append(float(value))
        data.append(row)
    
    return data


def generate_anomaly_data(seq_len: int = 100, num_features: int = 7) -> List[List[float]]:
    """Generate sample time series data with anomalies."""
    # Start with normal data
    data = generate_sample_data(seq_len, num_features)
    
    # Add anomalies
    for i in range(num_features):
        # Add spike anomaly
        data[20][i] += 15
        # Add drop anomaly
        data[60][i] -= 12
        # Add trend change
        for t in range(40, seq_len):
            data[t][i] += 20
    
    return data


def demo_forecasting(client: TSUnityAPIClient):
    """Demonstrate forecasting inference."""
    print("\n" + "="*50)
    print("FORECASTING DEMO")
    print("="*50)
    
    # Generate sample data
    print("Generating sample time series data...")
    data = generate_sample_data(seq_len=100, num_features=7)
    print(f"Generated data: {len(data)} time steps, {len(data[0])} features")
    
    # Run forecasting
    print("\nRunning forecasting inference...")
    start_time = time.time()
    
    try:
        result = client.run_forecasting(
            data=data,
            num_steps=24,  # Predict 24 steps ahead
            window_size=None,  # No sliding window
            stride=1
        )
        
        inference_time = time.time() - start_time
        
        print(f"✅ Forecasting completed in {inference_time:.4f}s")
        print(f"Task type: {result['task_type']}")
        print(f"Data shape: {result['data_shape']}")
        print(f"Number of steps: {result['num_steps']}")
        print(f"Forecast horizon: {result['forecast_horizon']}")
        print(f"Predictions shape: {len(result['predictions'])} x {len(result['predictions'][0])}")
        
        # Show sample predictions
        print("\nSample predictions (first 3 time steps, first 3 features):")
        for i in range(min(3, len(result['predictions']))):
            print(f"  t+{i+1}: {result['predictions'][i][:3]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Forecasting failed: {e}")
        return None


def demo_anomaly_detection(client: TSUnityAPIClient):
    """Demonstrate anomaly detection inference."""
    print("\n" + "="*50)
    print("ANOMALY DETECTION DEMO")
    print("="*50)
    
    # Generate sample data with anomalies
    print("Generating sample time series data with anomalies...")
    data = generate_anomaly_data(seq_len=100, num_features=7)
    print(f"Generated data: {len(data)} time steps, {len(data[0])} features")
    
    # Run anomaly detection
    print("\nRunning anomaly detection inference...")
    start_time = time.time()
    
    try:
        result = client.run_anomaly_detection(
            data=data,
            threshold=None,  # Use automatic threshold
            window_size=None,  # No sliding window
            stride=1
        )
        
        inference_time = time.time() - start_time
        
        print(f"✅ Anomaly detection completed in {inference_time:.4f}s")
        print(f"Task type: {result['task_type']}")
        print(f"Data shape: {result['data_shape']}")
        print(f"Anomalies detected: {result['anomalies_detected']}")
        print(f"Anomaly rate: {result['anomaly_rate']:.2f}%")
        print(f"Threshold used: {result['threshold_used']:.4f}")
        print(f"Anomaly scores shape: {len(result['anomaly_scores'])} x {len(result['anomaly_scores'][0])}")
        
        # Display detection method information if available
        if 'detection_method' in result and result['detection_method']:
            print("\n" + "="*40)
            print("DETECTION METHOD INFORMATION")
            print("="*40)
            method_info = result['detection_method']
            print(f"Method: {method_info.get('method', 'Unknown')}")
            print(f"Model Type: {method_info.get('model_type', 'Unknown')}")
            print(f"Description: {method_info.get('description', 'No description')}")
            print(f"Approach: {method_info.get('approach', 'No approach info')}")
            print("="*40)
        
        # Show sample anomaly scores
        print("\nSample anomaly scores (first 3 time steps, first 3 features):")
        for i in range(min(3, len(result['anomaly_scores']))):
            print(f"  t={i}: {result['anomaly_scores'][i][:3]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Anomaly detection failed: {e}")
        return None


def demo_sliding_window(client: TSUnityAPIClient):
    """Demonstrate sliding window inference."""
    print("\n" + "="*50)
    print("SLIDING WINDOW DEMO")
    print("="*50)
    
    # Generate longer time series
    print("Generating longer time series data...")
    data = generate_sample_data(seq_len=500, num_features=7)
    print(f"Generated data: {len(data)} time steps, {len(data[0])} features")
    
    # Run sliding window inference
    print("\nRunning sliding window inference...")
    start_time = time.time()
    
    try:
        result = client.run_forecasting(
            data=data,
            num_steps=12,  # Predict 12 steps ahead
            window_size=100,  # Use sliding window
            stride=10  # Stride of 10
        )
        
        inference_time = time.time() - start_time
        
        print(f"✅ Sliding window inference completed in {inference_time:.4f}s")
        print(f"Task type: {result['task_type']}")
        print(f"Data shape: {result['data_shape']}")
        print(f"Number of steps: {result['num_steps']}")
        print(f"Window size: {result['metadata']['window_size']}")
        print(f"Stride: {result['metadata']['stride']}")
        print(f"Predictions shape: {len(result['predictions'])} x {len(result['predictions'][0])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Sliding window inference failed: {e}")
        return None


def main():
    """Main function to run the API client demo."""
    print("🚀 TS-Unity Inference API Client Demo")
    print("="*60)
    
    # Initialize client
    client = TSUnityAPIClient("http://localhost:8000")
    
    try:
        # Check API health
        print("Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Model loaded: {health['model_loaded']}")
        
        if not health['model_loaded']:
            print("\n⚠️  No model loaded. Please load a model first using the API.")
            print("   You can use the /load_model endpoint or run the server with a pre-loaded model.")
            print("\nExample model loading:")
            print("   POST /load_model")
            print("   {")
            print('     "task_type": "forecasting",')
            print('     "checkpoint_path": "/path/to/your/checkpoint.pth"')
            print("   }")
            return
        
        # Get model info
        print("\nGetting model information...")
        model_info = client.get_model_info()
        print(f"✅ Model loaded: {model_info['model']}")
        print(f"   Task type: {model_info['task_type']}")
        print(f"   Sequence length: {model_info['sequence_length']}")
        print(f"   Input features: {model_info['input_features']}")
        print(f"   Output features: {model_info['output_features']}")
        
        # Run demos based on task type
        if model_info['task_type'] in ['long_term_forecast', 'short_term_forecast']:
            demo_forecasting(client)
            demo_sliding_window(client)
        elif model_info['task_type'] == 'anomaly_detection':
            demo_anomaly_detection(client)
        else:
            print(f"\n⚠️  Task type {model_info['task_type']} not yet implemented in demo")
        
        print("\n" + "="*60)
        print("🎉 Demo completed successfully!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("   Make sure the server is running on http://localhost:8000")
        print("\nTo start the server:")
        print("   python src/api/inference_server.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise
    
    finally:
        # Clean up
        try:
            client.unload_model()
            print("\n🧹 Model unloaded")
        except:
            pass


if __name__ == "__main__":
    main()
