#!/usr/bin/env python3
"""
Anomaly Detection Inference Script for TS-Unity

This script demonstrates how to use the InferencePipeline for anomaly detection
on time series data.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import List, Optional, Dict, Any
import time

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from config.base_config import AnomalyDetectionConfig
from core.pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetectionDemo:
    """Demo class for anomaly detection inference."""
    
    def __init__(self, config: AnomalyDetectionConfig, checkpoint_path: str):
        """
        Initialize the demo.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.inference_pipeline = None
        
        self._setup_inference()
    
    def _setup_inference(self) -> None:
        """Setup the inference components."""
        logger.info("Setting up anomaly detection inference demo...")
        
        self.inference_pipeline = InferencePipeline(self.config, self.checkpoint_path)
        
        logger.info("Anomaly detection inference demo setup completed")
    
    def demo_single_sequence(self) -> None:
        """Demo anomaly detection on a single time series sequence."""
        logger.info("Running single sequence anomaly detection demo...")
        
        # Generate sample sequence data
        seq_len = self.config.seq_len
        num_features = self.config.enc_in
        
        # Create synthetic time series data with some anomalies
        sequence_data = self._generate_synthetic_sequence_with_anomalies(seq_len, num_features)
        
        logger.info(f"Generated sequence data shape: {sequence_data.shape}")
        
        # Show sample data
        print(f"Sample data (first 3 time steps, first 3 features):")
        print(sequence_data[:3, :3])
        
        # Perform anomaly detection
        start_time = time.time()
        anomaly_scores = self.inference_pipeline.predict_batch(sequence_data, num_steps=1)
        inference_time = time.time() - start_time
        
        logger.info(f"Single sequence anomaly detection completed in {inference_time:.4f}s")
        logger.info(f"Anomaly scores shape: {anomaly_scores.shape}")
        logger.info(f"Sample anomaly scores: {anomaly_scores[0, 0, :3]}...")
        
        # Display detection method information
        self._display_detection_method_info()
        
        # Analyze results
        self._analyze_anomaly_scores(anomaly_scores)
        
        return anomaly_scores
    
    def demo_batch_sequences(self, batch_size: int = 10) -> None:
        """Demo anomaly detection on multiple time series sequences."""
        logger.info(f"Running batch anomaly detection demo with {batch_size} sequences...")
        
        # Generate batch of sequences
        seq_len = self.config.seq_len
        num_features = self.config.enc_in
        
        batch_data = np.array([
            self._generate_synthetic_sequence_with_anomalies(seq_len, num_features)
            for _ in range(batch_size)
        ])
        
        logger.info(f"Generated batch data shape: {batch_data.shape}")
        
        # Perform anomaly detection
        start_time = time.time()
        anomaly_scores = self.inference_pipeline.predict_batch(batch_data, num_steps=1)
        inference_time = time.time() - start_time
        
        logger.info(f"Batch anomaly detection completed in {inference_time:.4f}s")
        logger.info(f"Anomaly scores shape: {anomaly_scores.shape}")
        logger.info(f"Average detection time per sequence: {inference_time/batch_size:.4f}s")
        
        # Analyze results
        self._analyze_anomaly_scores(anomaly_scores)
        
        return anomaly_scores
    
    def demo_sliding_window(self, data_length: int = 1000, stride: int = 10) -> None:
        """Demo sliding window anomaly detection on long time series."""
        logger.info(f"Running sliding window anomaly detection demo on {data_length} time steps...")
        
        # Generate long time series with anomalies
        num_features = self.config.enc_in
        long_series = self._generate_synthetic_sequence_with_anomalies(data_length, num_features)
        
        logger.info(f"Generated long series shape: {long_series.shape}")
        
        # Apply sliding window
        start_time = time.time()
        results = self.inference_pipeline.predict_with_sliding_window(
            long_series, stride=stride, num_steps=1
        )
        inference_time = time.time() - start_time
        
        logger.info(f"Sliding window anomaly detection completed in {inference_time:.4f}s")
        logger.info(f"Number of windows: {results['num_windows']}")
        logger.info(f"Anomaly scores shape: {results['anomaly_scores'].shape}")
        logger.info(f"Average time per window: {inference_time/results['num_windows']:.4f}s")
        
        # Analyze results
        self._analyze_anomaly_scores(results['anomaly_scores'])
        
        return results
    
    def demo_file_inference(self, input_file: str, output_file: str) -> None:
        """Demo anomaly detection on data loaded from file."""
        logger.info(f"Running file-based anomaly detection demo on: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            logger.warning(f"Input file {input_file} not found, creating sample data...")
            self._create_sample_csv_with_anomalies(input_file)
        
        # Run inference
        start_time = time.time()
        results = self.inference_pipeline.predict_from_file(
            input_file, num_steps=1, output_path=output_file
        )
        inference_time = time.time() - start_time
        
        logger.info(f"File-based anomaly detection completed in {inference_time:.4f}s")
        logger.info(f"Results: {results}")
        
        # Analyze results
        self._analyze_anomaly_scores(results['predictions'])
        
        return results
    
    def _generate_synthetic_sequence_with_anomalies(self, seq_len: int, num_features: int) -> np.ndarray:
        """Generate synthetic time series sequence with anomalies."""
        # Create base patterns
        time_steps = np.arange(seq_len)
        
        # Generate different patterns for each feature
        sequence = np.zeros((seq_len, num_features))
        
        for i in range(num_features):
            # Different pattern for each feature
            if i == 0:
                # Sine wave with anomalies
                sequence[:, i] = np.sin(time_steps * 0.1) * 10
                # Add anomalies at specific points
                sequence[20:25, i] += 15  # Spike anomaly
                sequence[60:65, i] -= 12  # Drop anomaly
            elif i == 1:
                # Linear trend with noise
                sequence[:, i] = time_steps * 0.5 + np.random.normal(0, 1, seq_len)
                # Add trend change anomaly
                sequence[40:, i] += 20
            elif i == 2:
                # Exponential decay with anomalies
                sequence[:, i] = 20 * np.exp(-time_steps * 0.02)
                # Add level shift anomaly
                sequence[70:80, i] += 25
            elif i == 3:
                # Random walk with outliers
                sequence[:, i] = np.cumsum(np.random.normal(0, 0.5, seq_len))
                # Add outlier anomalies
                sequence[30, i] += 50
                sequence[80, i] -= 40
            else:
                # Random noise with occasional spikes
                sequence[:, i] = np.random.normal(0, 2, seq_len)
                # Add spike anomalies
                sequence[45:50, i] += 20
                sequence[85:90, i] += 15
        
        return sequence
    
    def _create_sample_csv_with_anomalies(self, file_path: str) -> None:
        """Create sample CSV file with anomalies for testing."""
        logger.info(f"Creating sample CSV file with anomalies: {file_path}")
        
        # Generate sample data
        seq_len = 200
        num_features = self.config.enc_in
        
        sample_data = self._generate_synthetic_sequence_with_anomalies(seq_len, num_features)
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(num_features)]
        df = pd.DataFrame(sample_data, columns=columns)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Sample CSV with anomalies created: {seq_len} rows, {num_features} features")
    
    def _analyze_anomaly_scores(self, anomaly_scores: np.ndarray) -> None:
        """Analyze and display anomaly detection results."""
        logger.info("Analyzing anomaly detection results...")
        
        # Flatten scores for analysis
        flat_scores = anomaly_scores.flatten()
        
        # Basic statistics
        logger.info(f"Anomaly scores statistics:")
        logger.info(f"  - Min: {np.min(flat_scores):.4f}")
        logger.info(f"  - Max: {np.max(flat_scores):.4f}")
        logger.info(f"  - Mean: {np.mean(flat_scores):.4f}")
        logger.info(f"  - Std: {np.std(flat_scores):.4f}")
        
        # Percentile analysis
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(flat_scores, p)
            logger.info(f"  - {p}th percentile: {value:.4f}")
        
        # Threshold-based analysis (example thresholds)
        thresholds = [np.mean(flat_scores) + 2*np.std(flat_scores),  # 2-sigma
                     np.percentile(flat_scores, 95),                  # 95th percentile
                     np.percentile(flat_scores, 99)]                  # 99th percentile
        
        for i, threshold in enumerate(thresholds):
            anomalies = np.sum(flat_scores > threshold)
            anomaly_rate = anomalies / len(flat_scores) * 100
            logger.info(f"  - Threshold {i+1} ({threshold:.4f}): {anomalies} anomalies ({anomaly_rate:.2f}%)")
    
    def benchmark_performance(self, data_sizes: List[int] = [100, 500, 1000]) -> Dict[str, Any]:
        """Benchmark anomaly detection performance with different data sizes."""
        logger.info("Running anomaly detection performance benchmark...")
        
        results = {}
        num_features = self.config.enc_in
        
        for size in data_sizes:
            logger.info(f"Benchmarking with data size: {size}")
            
            # Generate data
            data = self._generate_synthetic_sequence_with_anomalies(size, num_features)
            
            # Measure inference time
            start_time = time.time()
            anomaly_scores = self.inference_pipeline.predict_batch(data, num_steps=1)
            inference_time = time.time() - start_time
            
            results[size] = {
                'data_shape': data.shape,
                'anomaly_scores_shape': anomaly_scores.shape,
                'inference_time': inference_time,
                'throughput': size / inference_time  # sequences per second
            }
            
            logger.info(f"  Size {size}: {inference_time:.4f}s, Throughput: {results[size]['throughput']:.2f} seq/s")
        
        return results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.inference_pipeline:
            self.inference_pipeline.close()
        logger.info("Anomaly detection inference demo closed")

    def _display_detection_method_info(self) -> None:
        """Display information about the anomaly detection method being used."""
        print("\n" + "="*50)
        print("ANOMALY DETECTION METHOD")
        print("="*50)
        
        # Check if it's reconstruction or prediction based
        if hasattr(self.inference_pipeline, '_is_reconstruction_model'):
            is_reconstruction = self.inference_pipeline._is_reconstruction_model()
            
            if is_reconstruction:
                print("🔍 Method: Reconstruction-based Anomaly Detection")
                print("   - Model type: Reconstruction model (Autoencoder, AnomalyTransformer, etc.)")
                print("   - Approach: Compares input with reconstructed output")
                print("   - Anomaly score: Reconstruction error (higher = more anomalous)")
                print("   - Best for: Detecting structural anomalies, pattern changes")
            else:
                print("🔮 Method: Prediction-based Anomaly Detection")
                print("   - Model type: Forecasting model (Autoformer, Transformer, etc.)")
                print("   - Approach: Analyzes prediction patterns and variance")
                print("   - Anomaly score: Prediction variance or error (higher = more anomalous)")
                print("   - Best for: Detecting temporal anomalies, trend changes")
        else:
            print("⚠️  Could not determine detection method")
        
        print(f"   - Model: {self.config.model}")
        print(f"   - Sequence length: {self.config.seq_len}")
        print(f"   - Input features: {self.config.enc_in}")
        print("="*50)


def main():
    """Main function for anomaly detection inference demo."""
    parser = argparse.ArgumentParser(description='Anomaly Detection Inference Demo')
    
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='Path to trained anomaly detection model checkpoint'
    )
    parser.add_argument(
        '--mode', type=str, default='single',
        choices=['single', 'batch', 'sliding', 'file', 'benchmark'],
        help='Inference mode'
    )
    parser.add_argument(
        '--input_file', type=str,
        help='Input file for file mode'
    )
    parser.add_argument(
        '--output_file', type=str, default='anomaly_scores.csv',
        help='Output file for results'
    )
    parser.add_argument(
        '--batch_size', type=int, default=10,
        help='Batch size for batch mode'
    )
    parser.add_argument(
        '--data_length', type=int, default=1000,
        help='Data length for sliding window mode'
    )
    parser.add_argument(
        '--stride', type=int, default=10,
        help='Stride for sliding window mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration for anomaly detection
        config = AnomalyDetectionConfig(
            task_name='anomaly_detection',
            model='AnomalyTransformer',  # or your preferred anomaly detection model
            seq_len=100,
            enc_in=7,
            dec_in=7,
            c_out=7,
            anomaly_ratio=0.1,
            win_size=100
        )
        
        # Create demo
        demo = AnomalyDetectionDemo(config, args.checkpoint_path)
        
        # Run based on mode
        if args.mode == 'single':
            demo.demo_single_sequence()
        elif args.mode == 'batch':
            demo.demo_batch_sequences(args.batch_size)
        elif args.mode == 'sliding':
            demo.demo_sliding_window(args.data_length, args.stride)
        elif args.mode == 'file':
            if not args.input_file:
                args.input_file = 'sample_anomaly_data.csv'
            demo.demo_file_inference(args.input_file, args.output_file)
        elif args.mode == 'benchmark':
            demo.benchmark_performance([100, 500, 1000])
        
        demo.close()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()
