#!/bin/bash
 
#  models=(LSTM_AE LSTM_VAE OmniAnomaly Transformer USAD DAGMM TF AnomalyTransformer VTTSAT VTTPAT)
 models=(LSTM_VAE VTTPAT)
 subdataname=(kogas_c)
 
 ## SWaT
 for model in ${models[@]}
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname KOGAS \
   --subdataname $subdataname \
   --window_size 24 \
   --slide_size 12
 done