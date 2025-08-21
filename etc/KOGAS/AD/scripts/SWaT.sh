#!/bin/bash
 
#  models=(LSTM_AE LSTM_VAE OmniAnomaly Transformer USAD DAGMM TF AnomalyTransformer VTTSAT VTTPAT)
 models=(LSTM_AE)
 
 ## SWaT
 for model in ${models[@]}
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname SWaT
 done