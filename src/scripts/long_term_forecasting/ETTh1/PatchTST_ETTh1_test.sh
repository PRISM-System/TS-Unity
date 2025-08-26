#!/bin/bash

# PatchTST ETTh1 Testing Script
# This script tests all trained PatchTST models on ETTh1 dataset

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

echo "Starting PatchTST ETTh1 testing experiments..."

# Test 96-96 model
echo "Testing PatchTST ETTh1 96-96 model..."
python -u main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --is_inference 0 \
  --model $model_name \
  --data ETTh1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --checkpoints ../checkpoints/PatchTST_long_term_forecast_ETTh1_PatchTST_ETTh1_96_96/ \
  --des 'PatchTST_ETTh1_96_96_test'

echo "96-96 model testing completed!"

# Test 96-192 model
echo "Testing PatchTST ETTh1 96-192 model..."
python -u main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --is_inference 0 \
  --model $model_name \
  --data ETTh1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --checkpoints ../checkpoints/PatchTST_long_term_forecast_ETTh1_PatchTST_ETTh1_96_192/ \
  --des 'PatchTST_ETTh1_96_192_test'

echo "96-192 model testing completed!"

# Test 96-336 model
echo "Testing PatchTST ETTh1 96-336 model..."
python -u main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --is_inference 0 \
  --model $model_name \
  --data ETTh1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --checkpoints ../checkpoints/PatchTST_long_term_forecast_ETTh1_PatchTST_ETTh1_96_336/ \
  --des 'PatchTST_ETTh1_96_336_test'

echo "96-336 model testing completed!"

# Test 96-720 model
echo "Testing PatchTST ETTh1 96-720 model..."
python -u main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --is_inference 0 \
  --model $model_name \
  --data ETTh1 \
  --root_path ../datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --checkpoints ../checkpoints/PatchTST_long_term_forecast_ETTh1_PatchTST_ETTh1_96_720/ \
  --des 'PatchTST_ETTh1_96_720_test'

echo "96-720 model testing completed!"

echo "All PatchTST ETTh1 testing experiments completed!"
