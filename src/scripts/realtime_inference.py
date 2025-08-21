#!/usr/bin/env python3
"""
Real-time Inference Script for TS-Unity

This script demonstrates how to use the InferencePipeline for real-time
time series forecasting with streaming data input.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import List, Optional, Dict, Any

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


class RealTimeInferenceDemo:
    """Demo class for real-time inference."""
    
    def __init__(self, config: ForecastingConfig, checkpoint_path: str):
        """
        Initialize the demo.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.inference_manager = None
        
        self._setup_inference()
    
    def _setup_inference(self) -> None:
        """Setup the inference components."""
        logger.info("Setting up real-time inference demo...")
        
        self.inference_manager = RealTimeInferenceManager(self.config, self.checkpoint_path)
        self.inference_manager.start_streaming()
        
        logger.info("Real-time inference demo setup completed")
    
    def simulate_streaming_data(self, num_points: int = 100, 
                              interval: float = 1.0) -> None:
        """
        Simulate streaming data for demonstration.
        
        Args:
            num_points: Number of data points to simulate
            interval: Time interval between data points (seconds)
        """
        logger.info(f"Starting streaming simulation with {num_points} points, {interval}s interval")
        
        # Generate synthetic data
        for i in range(num_points):
            # Simulate multivariate time series data
            data_point = self._generate_synthetic_data_point(i)
            timestamp = time.time()
            
            # Get real-time prediction
            prediction = self.inference_manager.get_realtime_predictions(data_point, num_steps=1)
            
            # Log results
            logger.info(f"Point {i+1}: Input={data_point[:3]}..., Prediction={prediction.flatten()[:3]}...")
            
            # Wait for next interval
            time.sleep(interval)
        
        logger.info("Streaming simulation completed")
    
    def _generate_synthetic_data_point(self, index: int) -> np.ndarray:
        """Generate synthetic data point for demonstration."""
        # Simple sine wave pattern with noise
        base_value = np.sin(index * 0.1) * 10
        noise = np.random.normal(0, 0.5, self.config.enc_in)
        
        # Create multivariate data
        data_point = np.array([base_value + noise[i] for i in range(self.config.enc_in)])
        
        return data_point
    
    def process_csv_stream(self, csv_file: str, output_file: str) -> None:
        """
        Process streaming data from CSV file.
        
        Args:
            csv_file: Path to input CSV file
            output_file: Path to save results
        """
        logger.info(f"Processing CSV stream from: {csv_file}")
        
        # Read CSV in chunks for streaming simulation
        chunk_size = 10
        results = []
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_data = chunk.values
            
            # Process each row in the chunk
            for i, row in enumerate(chunk_data):
                data_point = row.astype(float)
                
                # Get prediction
                prediction = self.inference_manager.get_realtime_predictions(data_point, num_steps=1)
                
                # Store results
                results.append({
                    'input': data_point.tolist(),
                    'prediction': prediction.flatten().tolist(),
                    'timestamp': time.time()
                })
                
                logger.debug(f"Processed row {len(results)}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        logger.info(f"Results saved to: {output_file}")
    
    def interactive_mode(self) -> None:
        """Run interactive mode for manual data input."""
        logger.info("Interactive mode started. Enter data points manually:")
        logger.info(f"Expected format: {self.config.enc_in} comma-separated values")
        logger.info("Commands: 'quit', 'status', 'reset', 'help'")
        
        try:
            while True:
                user_input = input("\nEnter data point: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'status':
                    status = self.inference_manager.inference_pipeline.get_buffer_status()
                    logger.info(f"Buffer status: {status}")
                    continue
                elif user_input.lower() == 'reset':
                    self.inference_manager.inference_pipeline.reset_buffer()
                    logger.info("Buffer reset")
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                try:
                    # Parse input data
                    data_point = np.array([float(x.strip()) for x in user_input.split(',')])
                    
                    if data_point.shape[0] != self.config.enc_in:
                        logger.error(f"Expected {self.config.enc_in} features, got {data_point.shape[0]}")
                        continue
                    
                    # Get prediction
                    prediction = self.inference_manager.get_realtime_predictions(data_point, num_steps=1)
                    logger.info(f"Prediction: {prediction.flatten()}")
                    
                except ValueError as e:
                    logger.error(f"Invalid input format: {e}")
                    logger.info("Please enter comma-separated numeric values")
                except Exception as e:
                    logger.error(f"Error during inference: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Interactive mode stopped by user")
        
        logger.info("Interactive mode ended")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
Available commands:
- Enter comma-separated values (e.g., 1.2,3.4,5.6,7.8,9.0,10.1,11.2)
- 'quit': Exit interactive mode
- 'status': Show buffer status
- 'reset': Reset data buffer
- 'help': Show this help message

Data format: {num_features} comma-separated numeric values
        """.format(num_features=self.config.enc_in)
        
        print(help_text)
    
    def close(self) -> None:
        """Clean up resources."""
        if self.inference_manager:
            self.inference_manager.close()
        logger.info("Real-time inference demo closed")


def main():
    """Main function for real-time inference demo."""
    parser = argparse.ArgumentParser(description='Real-time Inference Demo')
    
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--mode', type=str, default='interactive',
        choices=['interactive', 'simulate', 'csv'],
        help='Inference mode'
    )
    parser.add_argument(
        '--input_csv', type=str,
        help='Input CSV file for csv mode'
    )
    parser.add_argument(
        '--output_csv', type=str, default='inference_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--num_points', type=int, default=100,
        help='Number of points for simulation mode'
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Interval between data points for simulation mode (seconds)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = ForecastingConfig(
            task_name='long_term_forecast',
            model='Autoformer',
            seq_len=96,
            pred_len=24,
            enc_in=7,
            dec_in=7,
            c_out=7
        )
        
        # Create demo
        demo = RealTimeInferenceDemo(config, args.checkpoint_path)
        
        # Run based on mode
        if args.mode == 'interactive':
            demo.interactive_mode()
        elif args.mode == 'simulate':
            demo.simulate_streaming_data(args.num_points, args.interval)
        elif args.mode == 'csv':
            if not args.input_csv:
                raise ValueError("Input CSV file is required for csv mode")
            demo.process_csv_stream(args.input_csv, args.output_csv)
        
        demo.close()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()
