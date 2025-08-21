#!/bin/bash
 
#  models=(LSTM_AE LSTM_VAE OmniAnomaly Transformer USAD DAGMM TF AnomalyTransformer VTTSAT VTTPAT)
 models=(LSTM_VAE VTTPAT)
 subdataname=kogas_c
 train_path=data/KOGAS/kogas_a.csv
 test_path=data/KOGAS/kogas_c.csv
 ## SWaT
 for model in ${models[@]}
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname KOGAS2 \
   --subdataname $subdataname \
   --window_size 24 \
   --slide_size 12 \
   --train_path $train_path \
   --test_path $test_path
 done