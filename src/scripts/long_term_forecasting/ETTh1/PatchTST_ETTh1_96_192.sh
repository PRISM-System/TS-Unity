#!/bin/bash

# PatchTST ETTh1 Long-term Forecasting Experiment
# Sequence Length: 96, Prediction Length: 192

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

echo "Starting PatchTST ETTh1 96-192 forecasting experiment..."

python -u main.py \
  --task_name long_term_forecast \
  --is_training 1 \
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
  --d_model 128 \
  --n_heads 8 \
  --d_ff 256 \
  --moving_avg 25 \
  --dropout 0.1 \
  --embed timeF \
  --activation gelu \
  --num_workers 8 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 5 \
  --learning_rate 0.0001 \
  --des 'PatchTST_ETTh1_96_192' \
  --itr 1

echo "PatchTST ETTh1 96-192 forecasting experiment completed!"
