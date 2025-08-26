#!/usr/bin/env python3
"""
Simple script to run USAD anomaly detection on PSM dataset
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from config.base_config import AnomalyDetectionConfig
from exp.exp_anomaly_detection import Exp_Anomaly_Detection

def load_psm_test_data():
    """Load PSM test data and labels"""
    test_data_path = Path('./datasets/PSM/test.csv')
    test_label_path = Path('./datasets/PSM/test_label.csv')
    
    if not test_data_path.exists():
        print(f"Test data not found: {test_data_path}")
        return None, None
    
    if not test_label_path.exists():
        print(f"Test labels not found: {test_label_path}")
        return None, None
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    test_labels = pd.read_csv(test_label_path)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    return test_data, test_labels

def main():
    # Load configuration
    config = AnomalyDetectionConfig.from_yaml('configs/usad_psm_anomaly_detection.yaml')
    
    # Set additional required fields
    config.task_name = 'anomaly_detection'
    config.is_training = True
    config.model = 'USAD'
    config.data = 'PSM'
    
    print(f"Configuration loaded: {config}")
    
    # Create experiment
    exp = Exp_Anomaly_Detection(config)
    
    # Run training
    print("Starting training...")
    exp.train()
    
    # Load test data
    print("\nLoading test data...")
    test_data, test_labels = load_psm_test_data()
    
    if test_data is not None and test_labels is not None:
        # Run testing with actual test data
        print("Starting testing with PSM test dataset...")
        
        # Convert test data to numpy array (exclude timestamp column if exists)
        if 'timestamp' in test_data.columns:
            test_data_np = test_data.drop('timestamp', axis=1).values
        else:
            test_data_np = test_data.values
        
        # Convert labels to numpy array
        if 'timestamp' in test_labels.columns:
            test_labels_np = test_labels.drop('timestamp', axis=1).values.flatten()
        else:
            test_labels_np = test_labels.values.flatten()
        
        print(f"Test data shape (numpy): {test_data_np.shape}")
        print(f"Test labels shape (numpy): {test_labels_np.shape}")
        
        # Run anomaly detection on test data
        print("Running anomaly detection on test data...")
        exp.test(test_data_np, test_labels_np)
        
        # Display results summary
        print("\n" + "="*50)
        print("ANOMALY DETECTION RESULTS SUMMARY")
        print("="*50)
        
        # Check if results were saved
        results_dir = Path('./src/results/test/')
        if results_dir.exists():
            print(f"Results saved to: {results_dir}")
            
            # List result files
            result_files = list(results_dir.glob('*.npy'))
            for file in result_files:
                print(f"  - {file.name}")
                
            # Load and display some key metrics
            try:
                if (results_dir / 'anomaly_scores.npy').exists():
                    anomaly_scores = np.load(results_dir / 'anomaly_scores.npy')
                    print(f"\nAnomaly scores shape: {anomaly_scores.shape}")
                    print(f"Anomaly scores range: {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")
                    print(f"Mean anomaly score: {anomaly_scores.mean():.4f}")
                
                if (results_dir / 'labels.npy').exists():
                    labels = np.load(results_dir / 'labels.npy')
                    print(f"\nGround truth labels shape: {labels.shape}")
                    print(f"Number of anomalies: {np.sum(labels)}")
                    print(f"Anomaly ratio: {np.mean(labels):.4f}")
                
                if (results_dir / 'metrics.npy').exists():
                    metrics = np.load(results_dir / 'metrics.npy', allow_pickle=True).item()
                    print(f"\nDetection metrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.4f}")
                        
            except Exception as e:
                print(f"Error loading results: {e}")
        else:
            print("Results directory not found. Check if test() method saves results properly.")
    
    print("\nAnomaly detection completed!")

if __name__ == '__main__':
    main()
