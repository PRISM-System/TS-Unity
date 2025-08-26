#!/bin/bash

# LSTM-AE PSM Anomaly Detection Script
# This script performs anomaly detection using LSTM-AE model on PSM dataset

export CUDA_VISIBLE_DEVICES=0

model_name=LSTM_AE
task_name=anomaly_detection

echo "Starting LSTM-AE PSM anomaly detection experiment..."

# Training phase
echo "Phase 1: Training LSTM-AE model on PSM dataset..."
python -u main.py \
  --task_name $task_name \
  --is_training 1 \
  --is_inference 0 \
  --model $model_name \
  --data PSM \
  --root_path ./datasets/PSM/ \
  --data_path train.csv \
  --features M \
  --seq_len 100 \
  --label_len 50 \
  --pred_len 1 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 128 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --dropout 0.1 \
  --embed timeF \
  --activation gelu \
  --num_workers 8 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 5 \
  --learning_rate 0.001 \
  --des LSTM_AE_PSM_anomaly_detection \
  --itr 1

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
    exit 1
fi

# Testing phase
echo "Phase 2: Testing LSTM-AE model on PSM test dataset..."
python -u main.py \
  --task_name $task_name \
  --is_training 0 \
  --is_inference 0 \
  --model $model_name \
  --data PSM \
  --root_path ./datasets/PSM/ \
  --data_path test.csv \
  --features M \
  --seq_len 100 \
  --label_len 50 \
  --pred_len 1 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 128 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --dropout 0.1 \
  --embed timeF \
  --activation gelu \
  --num_workers 8 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 5 \
  --learning_rate 0.001 \
  --des LSTM_AE_PSM_anomaly_detection \
  --itr 1

if [ $? -eq 0 ]; then
    echo "Testing completed successfully!"
else
    echo "Testing failed!"
    exit 1
fi

echo "LSTM-AE PSM anomaly detection experiment completed!"
echo ""
echo "Results Summary:"
echo "================"
echo "Model: LSTM-AE"
echo "Dataset: PSM"
echo "Task: Anomaly Detection"
echo "Training epochs: 10"
echo "Sequence length: 100"
echo "Input features: 25"
echo ""
echo "Checkpoint saved to: checkpoints/LSTM_AE_PSM_anomaly_detection/checkpoint.pth"
echo "Training logs: checkpoints/logs/LSTM_AE_anomaly_detection_PSM_LSTM_AE_PSM_anomaly_detection_*.log"
echo "Configuration: checkpoints/logs/LSTM_AE_anomaly_detection_PSM_LSTM_AE_PSM_anomaly_detection_config.json"
echo "Metrics: checkpoints/logs/LSTM_AE_anomaly_detection_PSM_LSTM_AE_PSM_anomaly_detection_metrics_history.json"
