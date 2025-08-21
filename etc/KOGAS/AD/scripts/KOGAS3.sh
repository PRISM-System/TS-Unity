#!/bin/bash

#  models=(LSTM_AE LSTM_VAE OmniAnomaly Transformer USAD DAGMM TF AnomalyTransformer VTTSAT VTTPAT)
models=(LSTM_VAE)
# subdataname=(kogas_a kogas_b kogas_c)
subdataname=(kogas_b kogas_c)


## KOGAS3
for sub in ${subdataname[@]}
do 
  for model in ${models[@]}
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname KOGAS3 \
    --subdataname $sub \
    --window_size 24 \
    --slide_size 12 \
    --resume 2
  done
done 