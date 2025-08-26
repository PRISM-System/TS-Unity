#!/usr/bin/env python3
import numpy as np

print("=== IMPROVED USAD PA METRICS ANALYSIS ===")

try:
    pa_metrics = np.load('src/results/test/pa_metrics.npy', allow_pickle=True).item()
    print("PA Metrics:")
    print(f"  PA-AUC: {pa_metrics['pa_auc']:.4f}")
    
    print("\nDetailed metrics:")
    for k, v in pa_metrics['metrics'].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
except Exception as e:
    print(f"Error loading results: {e}")
