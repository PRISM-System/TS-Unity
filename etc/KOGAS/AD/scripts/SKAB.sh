#!/bin/bash
 
 # models=(LSTM_AE LSTM_VAE OmniAnomaly Transformer USAD DAGMM TF AnomalyTransformer VTTSAT VTTPAT)
 models=(OmniAnomaly TF AnomalyTransformer VTTSAT VTTPAT)
 
 ## SKAB
 
 data_paths=../../data/SKAB
 subfolders=(valve1 valve2 other)
 
 
 for model in ${models[@]}
 do
   for subfolder in ${subfolders[@]}
   do
     folder_path="$data_paths/$subfolder"
     
     # 폴더가 없으면 건너뛰기
     if [ ! -d "$folder_path" ]; then
       echo "Skipping: Subfolder $folder_path does not exist"
       continue
     fi
 
     for file in "$folder_path"/*.csv
     do
       # CSV 파일이 없으면 건너뛰기
       if [ ! -f "$file" ]; then
         continue
       fi
 
       filename=${file##*/}
       filename_without_ext=${filename%.csv}
       subdataname="$subfolder/$filename_without_ext"
       echo "Processing file: $file with subdataname: $subdataname"
       
       python main.py \
       --train \
       --test \
       --model $model \
       --dataname SKAB \
       --window_size 10 \
       --slide_size 1 \
       --subdataname $subdataname
     done
   done
 done