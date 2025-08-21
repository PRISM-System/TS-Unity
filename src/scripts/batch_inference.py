#!/usr/bin/env python3
"""
Batch Inference Script for TS-Unity

This script demonstrates how to use the InferencePipeline for batch inference
on complete time series data without streaming.
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

from config.base_config import ForecastingConfig
from core.pipeline import InferencePipeline, RealTimeInferenceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchInferenceDemo:
    """Demo class for batch inference."""
    
    def __init__(self, config: ForecastingConfig, checkpoint_path: str):
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
        logger.info("Setting up batch inference demo...")
        
        self.inference_pipeline = InferencePipeline(self.config, self.checkpoint_path)
        
        logger.info("Batch inference demo setup completed")
    
    def demo_single_sequence(self, num_steps: int = 24) -> None:
        """Demo inference on a single time series sequence."""
        logger.info("Running single sequence inference demo...")
        
        # Generate sample sequence data
        seq_len = self.config.seq_len
        num_features = self.config.enc_in
        
        # Create synthetic time series data
        sequence_data = self._generate_synthetic_sequence(seq_len, num_features)
        
        logger.info(f"Generated sequence data shape: {sequence_data.shape}")
        
        # Make prediction
        start_time = time.time()
        predictions = self.inference_pipeline.predict_batch(sequence_data, num_steps)
        inference_time = time.time() - start_time
        
        logger.info(f"Single sequence prediction completed in {inference_time:.4f}s")
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Sample prediction: {predictions[0, 0, :3]}...")
        
        return predictions
    
    def demo_batch_sequences(self, batch_size: int = 10, num_steps: int = 24) -> None:
        """Demo inference on multiple time series sequences."""
        logger.info(f"Running batch inference demo with {batch_size} sequences...")
        
        # Generate batch of sequences
        seq_len = self.config.seq_len
        num_features = self.config.enc_in
        
        batch_data = np.array([
            self._generate_synthetic_sequence(seq_len, num_features)
            for _ in range(batch_size)
        ])
        
        logger.info(f"Generated batch data shape: {batch_data.shape}")
        
        # Make predictions
        start_time = time.time()
        predictions = self.inference_pipeline.predict_batch(batch_data, num_steps)
        inference_time = time.time() - start_time
        
        logger.info(f"Batch inference completed in {inference_time:.4f}s")
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Average prediction time per sequence: {inference_time/batch_size:.4f}s")
        
        return predictions
    
    def demo_sliding_window(self, data_length: int = 1000, stride: int = 10, 
                           num_steps: int = 24) -> None:
        """Demo sliding window inference on long time series."""
        logger.info(f"Running sliding window demo on {data_length} time steps...")
        
        # Generate long time series
        num_features = self.config.enc_in
        long_series = self._generate_synthetic_sequence(data_length, num_features)
        
        logger.info(f"Generated long series shape: {long_series.shape}")
        
        # Apply sliding window
        start_time = time.time()
        results = self.inference_pipeline.predict_with_sliding_window(
            long_series, stride=stride, num_steps=num_steps
        )
        inference_time = time.time() - start_time
        
        logger.info(f"Sliding window inference completed in {inference_time:.4f}s")
        logger.info(f"Number of windows: {results['num_windows']}")
        logger.info(f"Predictions shape: {results['predictions'].shape}")
        logger.info(f"Average time per window: {inference_time/results['num_windows']:.4f}s")
        
        return results
    
    def demo_file_inference(self, input_file: str, output_file: str, 
                           num_steps: int = 24) -> None:
        """Demo inference on data loaded from file."""
        logger.info(f"Running file inference demo on: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            logger.warning(f"Input file {input_file} not found, creating sample data...")
            self._create_sample_csv(input_file)
        
        # Run inference
        start_time = time.time()
        results = self.inference_pipeline.predict_from_file(
            input_file, num_steps, output_file
        )
        inference_time = time.time() - start_time
        
        logger.info(f"File inference completed in {inference_time:.4f}s")
        logger.info(f"Results: {results}")
        
        return results
    
    def _generate_synthetic_sequence(self, seq_len: int, num_features: int) -> np.ndarray:
        """Generate synthetic time series sequence."""
        # Create base patterns
        time_steps = np.arange(seq_len)
        
        # Generate different patterns for each feature
        sequence = np.zeros((seq_len, num_features))
        
        for i in range(num_features):
            # Different pattern for each feature
            if i == 0:
                # Sine wave
                sequence[:, i] = np.sin(time_steps * 0.1) * 10
            elif i == 1:
                # Linear trend
                sequence[:, i] = time_steps * 0.5
            elif i == 2:
                # Exponential decay
                sequence[:, i] = 20 * np.exp(-time_steps * 0.02)
            elif i == 3:
                # Random walk
                sequence[:, i] = np.cumsum(np.random.normal(0, 0.5, seq_len))
            else:
                # Random noise
                sequence[:, i] = np.random.normal(0, 2, seq_len)
        
        return sequence
    
    def _create_sample_csv(self, file_path: str) -> None:
        """Create sample CSV file for testing."""
        logger.info(f"Creating sample CSV file: {file_path}")
        
        # Generate sample data
        seq_len = 200
        num_features = self.config.enc_in
        
        sample_data = self._generate_synthetic_sequence(seq_len, num_features)
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(num_features)]
        df = pd.DataFrame(sample_data, columns=columns)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Sample CSV created with {seq_len} rows and {num_features} features")
    
    def benchmark_performance(self, data_sizes: List[int] = [100, 500, 1000], 
                            num_steps: int = 24) -> Dict[str, Any]:
        """Benchmark inference performance with different data sizes."""
        logger.info("Running performance benchmark...")
        
        results = {}
        num_features = self.config.enc_in
        
        for size in data_sizes:
            logger.info(f"Benchmarking with data size: {size}")
            
            # Generate data
            data = self._generate_synthetic_sequence(size, num_features)
            
            # Measure inference time
            start_time = time.time()
            predictions = self.inference_pipeline.predict_batch(data, num_steps)
            inference_time = time.time() - start_time
            
            results[size] = {
                'data_shape': data.shape,
                'predictions_shape': predictions.shape,
                'inference_time': inference_time,
                'throughput': size / inference_time  # sequences per second
            }
            
            logger.info(f"  Size {size}: {inference_time:.4f}s, Throughput: {results[size]['throughput']:.2f} seq/s")
        
        return results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.inference_pipeline:
            self.inference_pipeline.close()
        logger.info("Batch inference demo closed")


def main():
    """Main function for batch inference demo."""
    parser = argparse.ArgumentParser(description='Batch Inference Demo')
    
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='Path to trained model checkpoint'
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
        '--output_file', type=str, default='batch_predictions.csv',
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
    parser.add_argument(
        '--num_steps', type=int, default=24,
        help='Number of prediction steps'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = ForecastingConfig(
            task_name='long_term_forecast',
            model='Autoformer',
            seq_len=96,
            pred_len=args.num_steps,
            enc_in=7,
            dec_in=7,
            c_out=7
        )
        
        # Create demo
        demo = BatchInferenceDemo(config, args.checkpoint_path)
        
        # Run based on mode
        if args.mode == 'single':
            demo.demo_single_sequence(args.num_steps)
        elif args.mode == 'batch':
            demo.demo_batch_sequences(args.batch_size, args.num_steps)
        elif args.mode == 'sliding':
            demo.demo_sliding_window(args.data_length, args.stride, args.num_steps)
        elif args.mode == 'file':
            if not args.input_file:
                args.input_file = 'sample_data.csv'
            demo.demo_file_inference(args.input_file, args.output_file, args.num_steps)
        elif args.mode == 'benchmark':
            demo.benchmark_performance([100, 500, 1000], args.num_steps)
        
        demo.close()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()
