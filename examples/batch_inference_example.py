#!/usr/bin/env python3
"""
Batch Inference Example for TS-Unity

This example demonstrates how to use the InferencePipeline for batch inference
on complete time series data without streaming.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config.base_config import ForecastingConfig
from core.pipeline import InferencePipeline


def main():
    """Main example function."""
    print("🚀 TS-Unity Batch Inference Example")
    print("=" * 50)
    
    # Create configuration
    config = ForecastingConfig(
        task_name='long_term_forecast',
        model='Autoformer',
        seq_len=96,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1
    )
    
    print(f"Configuration created:")
    print(f"  - Sequence length: {config.seq_len}")
    print(f"  - Prediction length: {config.pred_len}")
    print(f"  - Input features: {config.enc_in}")
    print(f"  - Output features: {config.c_out}")
    
    # Note: In a real scenario, you would provide an actual checkpoint path
    checkpoint_path = "./checkpoints/best_model.pth"
    
    print(f"\n📁 Checkpoint path: {checkpoint_path}")
    print("⚠️  Note: This is a demo. In production, use a real trained model checkpoint.")
    
    try:
        # Initialize inference pipeline
        print("\n🔧 Initializing inference pipeline...")
        inference_pipeline = InferencePipeline(config, checkpoint_path=None)  # None for demo
        
        print("✅ Inference pipeline initialized successfully!")
        
        # Example 1: Single sequence inference
        print("\n" + "="*30)
        print("📊 Example 1: Single Sequence Inference")
        print("="*30)
        
        # Generate sample sequence
        seq_len = config.seq_len
        num_features = config.enc_in
        
        # Create synthetic time series data
        sequence_data = generate_synthetic_sequence(seq_len, num_features)
        print(f"Generated sequence data shape: {sequence_data.shape}")
        
        # Show sample data
        print(f"Sample data (first 3 time steps, first 3 features):")
        print(sequence_data[:3, :3])
        
        # Demonstrate prediction (would fail without real model)
        print("\n🔮 Attempting single sequence prediction...")
        try:
            predictions = inference_pipeline.predict_batch(sequence_data, num_steps=3)
            print(f"✅ Predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"❌ Prediction failed (expected without real model): {e}")
        
        # Example 2: Batch inference
        print("\n" + "="*30)
        print("📦 Example 2: Batch Inference")
        print("="*30)
        
        # Generate batch of sequences
        batch_size = 5
        batch_data = np.array([
            generate_synthetic_sequence(seq_len, num_features)
            for _ in range(batch_size)
        ])
        
        print(f"Generated batch data shape: {batch_data.shape}")
        print(f"Batch contains {batch_size} sequences")
        
        # Demonstrate batch prediction
        print("\n🔮 Attempting batch prediction...")
        try:
            batch_predictions = inference_pipeline.predict_batch(batch_data, num_steps=3)
            print(f"✅ Batch predictions shape: {batch_predictions.shape}")
        except Exception as e:
            print(f"❌ Batch prediction failed (expected without real model): {e}")
        
        # Example 3: Sliding window inference
        print("\n" + "="*30)
        print("🪟 Example 3: Sliding Window Inference")
        print("="*30)
        
        # Generate longer time series
        long_series_length = 200
        long_series = generate_synthetic_sequence(long_series_length, num_features)
        print(f"Generated long series shape: {long_series.shape}")
        
        # Demonstrate sliding window
        print("\n🔮 Attempting sliding window inference...")
        try:
            sliding_results = inference_pipeline.predict_with_sliding_window(
                long_series, stride=20, num_steps=3
            )
            print(f"✅ Sliding window results:")
            print(f"   - Number of windows: {sliding_results['num_windows']}")
            print(f"   - Predictions shape: {sliding_results['predictions'].shape}")
        except Exception as e:
            print(f"❌ Sliding window inference failed (expected without real model): {e}")
        
        # Clean up
        inference_pipeline.close()
        print("\n🧹 Cleanup completed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("This is expected in demo mode without a real model checkpoint.")
    
    print("\n" + "=" * 50)
    print("🎯 Batch Inference Usage Summary:")
    print("1. predict_batch(): Single sequence or batch of sequences")
    print("2. predict_with_sliding_window(): Long time series with sliding windows")
    print("3. predict_from_file(): Load data from CSV/NPY/NPZ files")
    print("\n📚 See README.md for detailed usage instructions!")


def generate_synthetic_sequence(seq_len: int, num_features: int) -> np.ndarray:
    """Generate synthetic time series sequence for demonstration."""
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


if __name__ == '__main__':
    main()
