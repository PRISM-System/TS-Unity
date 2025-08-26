#!/bin/bash

# Main script to run all DLinear ETTh1 forecasting experiments
# This script trains and tests all prediction length variants

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "DLinear ETTh1 Long-term Forecasting Experiments"
echo "=========================================="
echo ""

# Change to the scripts directory
cd scripts/long_term_forecasting/ETTh1

# Make all scripts executable
chmod +x DLinear_ETTh1_96_96.sh
chmod +x DLinear_ETTh1_96_192.sh
chmod +x DLinear_ETTh1_96_336.sh
chmod +x DLinear_ETTh1_96_720.sh
chmod +x DLinear_ETTh1_test.sh

echo "Starting training experiments..."
echo ""

# Train 96-96 model
echo "1. Training DLinear ETTh1 96-96 model..."
./DLinear_ETTh1_96_96.sh
echo ""

# Train 96-192 model
echo "2. Training DLinear ETTh1 96-192 model..."
./DLinear_ETTh1_96_192.sh
echo ""

# Train 96-336 model
echo "3. Training DLinear ETTh1 96-336 model..."
./DLinear_ETTh1_96_336.sh
echo ""

# Train 96-720 model
echo "4. Training DLinear ETTh1 96-720 model..."
./DLinear_ETTh1_96_720.sh
echo ""

echo "All training experiments completed!"
echo ""

echo "Starting testing experiments..."
echo ""

# Test all models
echo "Testing all trained models..."
./DLinear_ETTh1_test.sh
echo ""

echo "=========================================="
echo "All DLinear ETTh1 experiments completed!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "- Training logs: ../checkpoints/logs/"
echo "- Model checkpoints: ../checkpoints/DLinear_long_term_forecast_ETTh1_*/"
echo "- Test results: ../results/test/"
echo ""
echo "Check the logs for detailed performance metrics."
