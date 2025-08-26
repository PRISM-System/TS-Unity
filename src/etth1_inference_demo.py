#!/usr/bin/env python3
"""
ETTh1 Dataset Inference Demo

This script demonstrates how to:
1. Load a trained checkpoint for ETTh1 forecasting
2. Load a subset of ETTh1 data
3. Perform inference using the loaded model
4. Visualize and analyze the results
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from core.pipeline import InferencePipeline
from config.base_config import ForecastingConfig
from data_provider.data_factory import _ETTDataset


def load_etth1_subset(data_path: str, start_idx: int = 0, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a subset of ETTh1 data for inference.
    
    Args:
        data_path: Path to ETTh1.csv file
        start_idx: Starting index for the subset
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (data, timestamps)
    """
    print(f"Loading ETTh1 data subset from {data_path}")
    print(f"Start index: {start_idx}, Number of samples: {num_samples}")
    
    # Load the CSV file
    df = pd.read_csv(data_path)
    
    # Extract timestamps and data
    timestamps = pd.to_datetime(df.iloc[:, 0])
    data = df.iloc[:, 1:].values.astype(np.float32)  # Skip timestamp column
    
    print(f"Full dataset shape: {data.shape}")
    print(f"Features: {list(df.columns[1:])}")
    
    # Extract subset
    end_idx = min(start_idx + num_samples, len(data))
    subset_data = data[start_idx:end_idx]
    subset_timestamps = timestamps[start_idx:end_idx]
    
    print(f"Subset data shape: {subset_data.shape}")
    print(f"Subset time range: {subset_timestamps.iloc[0]} to {subset_timestamps.iloc[-1]}")
    
    return subset_data, subset_timestamps


def create_inference_config() -> ForecastingConfig:
    """
    Create inference configuration based on the trained model.
    
    Returns:
        ForecastingConfig object
    """
    config = ForecastingConfig(
        task_name='long_term_forecast',
        model='DLinear',
        data='ETTh1',
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=7,
        dec_in=7,
        c_out=7,
        features='M',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        use_gpu=True,
        gpu=0
    )
    
    print(f"Inference config created:")
    print(f"  Model: {config.model}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Prediction length: {config.pred_len}")
    print(f"  Input features: {config.enc_in}")
    print(f"  Output features: {config.c_out}")
    
    return config


def perform_inference(pipeline: InferencePipeline, 
                     data: np.ndarray, 
                     timestamps: pd.Series,
                     num_steps: int = 96) -> Dict[str, Any]:
    """
    Perform inference using the loaded pipeline.
    
    Args:
        pipeline: Loaded inference pipeline
        data: Input data array
        timestamps: Corresponding timestamps
        num_steps: Number of steps to predict
        
    Returns:
        Dictionary containing inference results
    """
    print(f"\nPerforming inference with {num_steps} prediction steps...")
    
    # Method 1: Single sequence prediction
    print("1. Single sequence prediction...")
    if len(data) >= pipeline.config.seq_len:
        # Take the last sequence for prediction
        input_seq = data[-pipeline.config.seq_len:]
        single_pred = pipeline.predict_batch(input_seq, num_steps=num_steps)
        print(f"   Input shape: {input_seq.shape}")
        print(f"   Prediction shape: {single_pred.shape}")
        print(f"   Prediction range: {single_pred.min():.4f} to {single_pred.max():.4f}")
    else:
        print("   Not enough data for single sequence prediction")
        single_pred = None
    
    # Method 2: Sliding window prediction
    print("2. Sliding window prediction...")
    if len(data) >= pipeline.config.seq_len:
        window_size = pipeline.config.seq_len
        stride = max(1, window_size // 4)  # Use 1/4 of window size as stride
        
        sliding_results = pipeline.predict_with_sliding_window(
            data, 
            window_size=window_size, 
            stride=stride, 
            num_steps=num_steps
        )
        
        print(f"   Number of windows: {sliding_results['num_windows']}")
        print(f"   Predictions shape: {sliding_results['predictions'].shape}")
        print(f"   Input windows shape: {sliding_results['input_windows'].shape}")
    else:
        print("   Not enough data for sliding window prediction")
        sliding_results = None
    
    return {
        'single_prediction': single_pred,
        'sliding_predictions': sliding_results,
        'input_data': data,
        'timestamps': timestamps
    }


def visualize_results(results: Dict[str, Any], save_path: str = None):
    """
    Visualize the inference results.
    
    Args:
        results: Dictionary containing inference results
        save_path: Optional path to save the plot
    """
    print("\nVisualizing results...")
    
    data = results['input_data']
    timestamps = results['timestamps']
    single_pred = results['single_prediction']
    sliding_preds = results['sliding_predictions']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ETTh1 Inference Results', fontsize=16)
    
    # Plot 1: Input data (first few features)
    ax1 = axes[0, 0]
    num_features_to_plot = min(3, data.shape[1])
    for i in range(num_features_to_plot):
        ax1.plot(timestamps, data[:, i], label=f'Feature {i+1}', alpha=0.7)
    ax1.set_title('Input Data (First 3 Features)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Single prediction
    if single_pred is not None:
        ax2 = axes[0, 1]
        # Create future timestamps for prediction
        last_time = timestamps.iloc[-1]
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=single_pred.shape[1],
            freq='H'
        )
        
        for i in range(min(3, single_pred.shape[2])):
            ax2.plot(future_times, single_pred[0, :, i], 
                    label=f'Predicted Feature {i+1}', linestyle='--', marker='o')
        ax2.set_title('Single Sequence Prediction')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sliding window predictions (first feature)
    if sliding_preds is not None:
        ax3 = axes[1, 0]
        predictions = sliding_preds['predictions']
        window_indices = sliding_preds['window_indices']
        
        # Plot predictions for the first feature
        for i, (start_idx, end_idx) in enumerate(window_indices):
            if i < len(predictions):
                pred_times = timestamps[start_idx:end_idx + sliding_preds['num_steps']]
                if len(pred_times) == predictions[i].shape[1]:
                    ax3.plot(pred_times, predictions[i, 0, :, 0], 
                            alpha=0.5, label=f'Window {i+1}' if i < 3 else "")
        
        ax3.set_title('Sliding Window Predictions (First Feature)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        if len(window_indices) <= 3:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prediction statistics
    ax4 = axes[1, 1]
    if sliding_preds is not None:
        predictions = sliding_preds['predictions']
        # Calculate prediction statistics across all windows
        all_preds = predictions.reshape(-1, predictions.shape[-1])
        
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        feature_names = [f'F{i+1}' for i in range(all_preds.shape[1])]
        x_pos = np.arange(len(feature_names))
        
        ax4.bar(x_pos, mean_pred, yerr=std_pred, capsize=5, alpha=0.7)
        ax4.set_title('Prediction Statistics Across Windows')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Mean Value ± Std')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(feature_names)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the ETTh1 inference demo."""
    print("=" * 60)
    print("ETTh1 Dataset Inference Demo")
    print("=" * 60)
    
    # Configuration
    data_path = "../datasets/ETT-small/ETTh1.csv"
    checkpoint_path = "../checkpoints/DLinear_ETTh1_96_96/checkpoint.pth"
    
    # Check if files exist
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please ensure the ETTh1.csv file exists in the datasets/ETT-small/ directory")
        return
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("Please ensure the checkpoint.pth file exists in the checkpoints/DLinear_ETTh1_96_96/ directory")
        return
    
    try:
        # Step 1: Create inference configuration
        config = create_inference_config()
        
        # Step 2: Load inference pipeline
        print(f"\nLoading inference pipeline from: {checkpoint_path}")
        pipeline = InferencePipeline(config, checkpoint_path)
        print("Pipeline loaded successfully!")
        
        # Step 3: Load ETTh1 data subset
        data, timestamps = load_etth1_subset(
            data_path=data_path,
            start_idx=1000,  # Start from 1000th sample
            num_samples=2000  # Load 2000 samples
        )
        
        # Step 4: Perform inference
        results = perform_inference(pipeline, data, timestamps, num_steps=96)
        
        # Step 5: Visualize results
        output_dir = Path("../results/inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "etth1_inference_results.png"
        
        visualize_results(results, save_path=str(save_path))
        
        # Step 6: Print summary statistics
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Input data shape: {data.shape}")
        print(f"Time range: {timestamps.iloc[0]} to {timestamps.iloc[-1]}")
        
        if results['single_prediction'] is not None:
            pred = results['single_prediction']
            print(f"Single prediction shape: {pred.shape}")
            print(f"Prediction range: {pred.min():.4f} to {pred.max():.4f}")
            print(f"Prediction mean: {pred.mean():.4f}")
            print(f"Prediction std: {pred.std():.4f}")
        
        if results['sliding_predictions'] is not None:
            sliding = results['sliding_predictions']
            print(f"Sliding window predictions: {sliding['num_windows']} windows")
            print(f"Total predictions: {sliding['predictions'].shape}")
        
        print(f"\nResults saved to: {save_path}")
        print("Inference demo completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
