#!/usr/bin/env python3
"""
Real-time Inference Example for TS-Unity

This example demonstrates how to use the InferencePipeline for real-time
time series forecasting with streaming data.
"""

import sys
import os
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
    print("🚀 TS-Unity Real-time Inference Example")
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
        
        # Show buffer status
        status = inference_pipeline.get_buffer_status()
        print(f"\n📊 Buffer status: {status}")
        
        # Simulate adding some data points
        print("\n📥 Adding sample data points...")
        
        for i in range(5):
            # Generate sample data point
            data_point = np.random.randn(config.enc_in)
            timestamp = time.time()
            
            print(f"  Point {i+1}: {data_point[:3]}... (timestamp: {timestamp:.2f})")
            
            # Add to pipeline
            inference_pipeline.add_data_point(data_point, timestamp)
            
            # Small delay to simulate real-time
            time.sleep(0.1)
        
        # Show updated buffer status
        status = inference_pipeline.get_buffer_status()
        print(f"\n📊 Updated buffer status: {status}")
        
        # Demonstrate prediction (would fail without real model)
        print("\n🔮 Attempting prediction...")
        try:
            predictions = inference_pipeline.predict_next(num_steps=3)
            print(f"✅ Predictions: {predictions.shape}")
        except Exception as e:
            print(f"❌ Prediction failed (expected without real model): {e}")
        
        # Clean up
        inference_pipeline.close()
        print("\n🧹 Cleanup completed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("This is expected in demo mode without a real model checkpoint.")
    
    print("\n" + "=" * 50)
    print("🎯 To use this in production:")
    print("1. Train a model using the training pipeline")
    print("2. Save the model checkpoint")
    print("3. Use the checkpoint path in InferencePipeline")
    print("4. Feed real-time data through add_data_point()")
    print("5. Get predictions using predict_next()")
    print("\n📚 See README.md for detailed usage instructions!")


if __name__ == '__main__':
    main()
